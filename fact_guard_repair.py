import json
import os
import re
from typing import Any, Dict

from ollama import chat
from config import MODEL


FACT_GUARD_REPAIR_VERSION = "fact-guard-repair-v1.6.2-platform-repair-re-import"

NUM_THREADS = max(1, int(os.getenv("FACT_GUARD_NUM_THREADS", "16")))
NUM_CTX = max(4096, int(os.getenv("FACT_GUARD_NUM_CTX", "8192")))
NUM_BATCH = max(64, int(os.getenv("FACT_GUARD_NUM_BATCH", "512")))
REPAIR_CTX = max(8192, int(os.getenv("FACT_GUARD_REPAIR_CTX", "8192")))
REPAIR_TOKENS = max(280, int(os.getenv("FACT_GUARD_REPAIR_TOKENS", "420")))


_ARTICLE_FORMAT = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "h1": {"type": "string"},
        "paragraphs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": {"type": "string"},
        },
    },
    "required": ["title", "description", "h1", "paragraphs"],
}



def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object returned.")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])

    raise ValueError("Incomplete JSON object.")


def _schema_ok(article: Dict[str, Any]) -> bool:
    if not isinstance(article, dict):
        return False

    required = ("title", "description", "h1", "paragraphs")
    if any(key not in article for key in required):
        return False

    for key in ("title", "description", "h1"):
        if not isinstance(article[key], str):
            return False

    paragraphs = article["paragraphs"]
    if not isinstance(paragraphs, list) or not 1 <= len(paragraphs) <= 20:
        return False

    for paragraph in paragraphs:
        if not isinstance(paragraph, str) or not paragraph.strip():
            return False

    return True



def _normalize_issue_type(issue_type: str) -> str:
    """Normalize harmless spelling/format variants emitted by the audits."""
    value = str(issue_type or "").strip().casefold()
    value = value.replace("-", "_").replace(" ", "_")
    aliases = {
        "unsupported_record": "unsupported_superlative",
        "unsupported_superlative_claim": "unsupported_superlative",
        "scope_exaggeration": "scope_inflation",
        "exaggerated_scope": "scope_inflation",
        "unsupported_causal_claim": "causal_claim",
        "unsupported_motive": "causal_claim",
        "unsupported_connection": "causal_claim",
        "wrong_quote": "unsupported_quote",
        "quote_attribution_error": "unsupported_quote",
        "wrong_date": "event_date_mismatch",
        "date_mismatch": "event_date_mismatch",
        "wrong_event_date": "event_date_mismatch",
        "wrong_event_status": "event_status_mismatch",
        "wrong_status": "event_status_mismatch",
        "status_mismatch": "event_status_mismatch",
        "wrong_number": "wrong_fact",
        "unsupported_number": "wrong_fact",
        "wrong_location": "wrong_fact",
        "wrong_name": "wrong_fact",
        "wrong_team": "wrong_entity_attribution",
        "wrong_organization": "wrong_entity_attribution",
        "wrong_person": "wrong_entity_attribution",
        "wrong_show": "wrong_entity_attribution",
        "wrong_character": "wrong_entity_attribution",
        "wrong_team_attribution": "wrong_entity_attribution",
        "wrong_org_attribution": "wrong_entity_attribution",
        "cross_day_conflation": "cross_event_conflation",
        "cross_match_conflation": "cross_event_conflation",
        "cross_week_conflation": "cross_event_conflation",
    }
    return aliases.get(value, value)


def _supported_repair(issue: Dict[str, Any]) -> bool:
    """
    Validate that every HIGH/MEDIUM Fact Guard blocker has a safe repair path.

    The repair taxonomy is intentionally closed.  Detection may report richer
    explanations, but an unknown issue type must never be silently sent to the
    LLM repair prompt.

    Repair classes:
      1) SOURCE-CORRECTABLE:
         wrong_entity_attribution / wrong_role / wrong_amount
         may be corrected only when the source excerpt directly establishes
         the correction.

      2) DELETE-ONLY:
         the safest action is to remove the unsupported or contradictory
         claim rather than invent a replacement.

      3) TEMPORAL DELETE-ONLY:
         remove the smallest claim that incorrectly joins event instances or
         assigns an unsupported date/status.
    """
    issue_type = _normalize_issue_type(issue.get("type", ""))

    severity_ok = (
        str(issue.get("severity", "")).strip().upper()
        in {"HIGH", "MEDIUM"}
    )

    common_ok = (
        severity_ok
        and bool(str(issue.get("claim", "")).strip())
        and bool(str(issue.get("reason", "")).strip())
    )
    if not common_ok:
        return False

    source_correctable = {
        "wrong_entity_attribution",
        "wrong_role",
        "wrong_amount",
        "wrong_temperature",
    }

    delete_only = {
        "unsupported_claim",
        "unsupported_superlative",
        "scope_inflation",
        "unsupported_quote",
        "wrong_fact",
        "wrong_platform",
        "causal_claim",
        "event_status_mismatch",
        "event_date_mismatch",
        "cross_event_conflation",
        "cross_round_conflation",
        "cross_fact_temporal_consistency",
        "cross_day_conflation",
        "cross_match_conflation",
        "cross_week_conflation",
    }

    if issue_type in source_correctable:
        return bool(str(issue.get("source_excerpt", "")).strip())

    if issue_type in delete_only:
        return True

    return False


def repair(
    article: Dict[str, Any],
    source: str,
    guard_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform one consolidated targeted repair for all currently reported HIGH/MEDIUM blocking issues.

    Safety rules:
    - One or more HIGH/MEDIUM blocking issues may exist.
    - The source material is the ONLY factual authority.
    - Repair ONLY the reported blocking issues.
    - Never introduce new facts.
    - Never infer a missing date, status, entity, role, event, number or quote.
    - wrong_entity_attribution may be corrected only from direct
      source-supported evidence.
    - unsupported_claim is delete-only.
    - event_date_mismatch is delete-only and includes unsupported temporal
      framing such as "next week", "tomorrow", "is set to air", or similar
      future/past assertions when the source establishes a different state.
    - For event_date_mismatch, do NOT replace the bad temporal assertion with
      the current date, reference date, publication date, trend date, feed
      date, or any inferred date.
    - Preserve every other supported claim and preserve uncertainty/status.
    - The caller MUST run Fact Guard again after this function returns.
    """
    if not isinstance(article, dict):
        raise ValueError("Article must be a JSON object.")

    if not isinstance(guard_result, dict):
        raise ValueError("Fact Guard result must be a JSON object.")

    blocking = [
        x for x in guard_result.get("issues", [])
        if isinstance(x, dict)
        and str(x.get("severity", "")).strip().upper() in {"HIGH", "MEDIUM"}
    ]

    if not blocking:
        raise ValueError("Fact Guard repair requires at least one blocking issue.")

    unsupported = [x for x in blocking if not _supported_repair(x)]
    if unsupported:
        types = ", ".join(sorted({
            str(x.get("type", "")).strip() or "unknown"
            for x in unsupported
        }))
        raise ValueError(f"Unsupported Fact Guard repair type(s): {types}")

    issue_rules_parts = []
    for index, issue in enumerate(blocking, 1):
        raw_issue_type = str(issue.get("type", "")).strip()
        issue_type = _normalize_issue_type(raw_issue_type)

        if issue_type in {"wrong_entity_attribution", "wrong_role"}:
            rules = (
                "- SOURCE-CORRECTABLE ONLY: correct the entity/role only when the "
                "source excerpt directly establishes the correct entity or role.\n"
                "- If the source does not establish the correction, DELETE the "
                "incorrect attribution/role instead of guessing.\n"
                "- Do not transfer an action, quote, status, deadline, injury, "
                "appointment, signing, departure, or decision between entities "
                "merely because they are mentioned nearby."
            )
        elif issue_type == "wrong_amount":
            rules = (
                "- SOURCE-CORRECTABLE ONLY: correct the amount only when the source "
                "excerpt directly supports the exact corrected amount.\n"
                "- Preserve the exact source-supported amount and currency.\n"
                "- Do not calculate, estimate, round, convert currency, or infer "
                "a different amount.\n"
                "- If the correction cannot be established directly, DELETE the "
                "smallest amount-bearing claim instead."
            )
        elif issue_type == "wrong_platform":
            rules = (
                "- DELETE-ONLY: remove the unsupported platform attribution "
                "unless the issue/source excerpt directly establishes a safe correction.\n"
                "- Never substitute a platform from outside knowledge.\n"
                "- Preserve surrounding source-supported reporting where possible."
            )
        elif issue_type in {
            "unsupported_claim",
            "wrong_fact",
            "causal_claim",
            "unsupported_quote",
            "scope_inflation",
            "unsupported_superlative",
        }:
            rules = (
                "- DELETE-ONLY: remove the unsupported, exaggerated, misattributed, "
                "or contradicted claim, using the smallest sentence/phrase that "
                "can be removed safely.\n"
                "- Do not soften it into another unsupported assertion.\n"
                "- Do not replace it with outside knowledge, an inferred fact, a "
                "different quote, a guessed number, or a broader/narrower claim.\n"
                "- For unsupported_quote, remove the unsupported quoted wording "
                "and preserve surrounding source-supported reporting where possible.\n"
                "- For scope_inflation, remove only the unsupported universal/"
                "nationwide scope assertion rather than inventing a narrower scope."
            )
        elif issue_type == "event_date_mismatch":
            rules = (
                "- DELETE-ONLY: remove the unsupported event date or temporal "
                "assertion that conflicts with the source.\n"
                "- Do not replace it with the publication date, reference date, "
                "current date, feed date, or an inferred date.\n"
                "- Preserve the surrounding event wording when it remains "
                "source-supported.\n"
                "- Do not turn a completed event into an upcoming event or vice versa."
            )
        elif issue_type == "event_status_mismatch":
            rules = (
                "- DELETE-ONLY: remove the smallest claim that assigns the event "
                "the wrong status/state.\n"
                "- Do not infer or substitute the opposite status unless the issue "
                "itself explicitly provides a source-supported correction and the "
                "correction is unambiguous.\n"
                "- Preserve uncertainty and all independently supported context."
            )
        elif issue_type in {
            "cross_event_conflation",
            "cross_round_conflation",
            "cross_fact_temporal_consistency",
            "cross_day_conflation",
            "cross_match_conflation",
            "cross_week_conflation",
        }:
            rules = (
                "- DELETE-ONLY: remove the smallest claim or sentence that "
                "incorrectly combines facts from different event instances.\n"
                "- Do not rewrite the chronology or merge the events.\n"
                "- Do not substitute another date, result, amount, status, or "
                "event unless that replacement is explicitly part of the reported "
                "issue and directly supported by the source excerpt.\n"
                "- Preserve surrounding claims that remain independently supported "
                "and belong to the main event."
            )
        else:
            raise ValueError(
                f"Unsupported Fact Guard repair type after normalization: "
                f"{raw_issue_type!r} -> {issue_type!r}"
            )

        issue_rules_parts.append(f"ISSUE {index} ({issue_type}):\n{rules}")

    issue_rules = "\n\n".join(issue_rules_parts)

    prompt = f"""
You are TrendCurrent's targeted Fact Guard repair engine.

Repair ONE already-generated article after an independent Fact Guard found
one or more blocking factual errors.

SOURCE MATERIAL is the ONLY factual authority.

GLOBAL SAFETY RULES:
- Fix ONLY the listed issue.
- Preserve all other supported article content.
- Do NOT regenerate the article from scratch.
- Do NOT add any new fact, context, motive, date, number, quote, event,
  entity, role, or status.
- Do NOT use outside knowledge.
- Do NOT infer missing information.
- Preserve uncertainty when the source is uncertain.
- Keep the article in its existing language.
- Issue types may be normalized only for selecting a safe repair rule; the actual issue claim, reason, and source excerpt remain the authoritative repair target.
- Return ONLY valid article JSON matching the required schema.
- The universal article schema uses paragraphs only. Do not return intro, sections, FAQ, or other legacy fields.

ISSUE-SPECIFIC RULES:
{issue_rules}

FACT GUARD ISSUES:
{json.dumps(blocking, ensure_ascii=False, indent=2)}

SOURCE MATERIAL:
{source}

ARTICLE:
{json.dumps(article, ensure_ascii=False, indent=2)}
"""

    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0,
            "top_p": 0.85,
            "top_k": 40,
            "num_ctx": REPAIR_CTX,
            "num_predict": REPAIR_TOKENS,
            "num_batch": NUM_BATCH,
            "num_thread": NUM_THREADS,
        },
        format=_ARTICLE_FORMAT,
    )

    raw = response.message.content or ""
    repaired = _extract_json_object(raw)

    if not _schema_ok(repaired):
        raise ValueError("Fact Guard repair returned invalid article schema.")

    return repaired
