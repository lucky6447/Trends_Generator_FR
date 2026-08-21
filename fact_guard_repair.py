import json
import os
from typing import Any, Dict

from ollama import chat
from config import MODEL


FACT_GUARD_REPAIR_VERSION = "fact-guard-repair-v1.3-multi-issue"

NUM_THREADS = max(1, int(os.getenv("FACT_GUARD_NUM_THREADS", "16")))
NUM_CTX = max(4096, int(os.getenv("FACT_GUARD_NUM_CTX", "8192")))
NUM_BATCH = max(64, int(os.getenv("FACT_GUARD_NUM_BATCH", "512")))
REPAIR_CTX = max(4096, int(os.getenv("FACT_GUARD_REPAIR_CTX", "4096")))
REPAIR_TOKENS = max(280, int(os.getenv("FACT_GUARD_REPAIR_TOKENS", "420")))


_ARTICLE_FORMAT = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "h1": {"type": "string"},
        "intro": {"type": "string"},
        "sections": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["title", "text"],
            },
        },
    },
    "required": ["title", "description", "h1", "intro", "sections"],
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

    required = ("title", "description", "h1", "intro", "sections")
    if any(key not in article for key in required):
        return False

    for key in ("title", "description", "h1", "intro"):
        if not isinstance(article[key], str):
            return False

    sections = article["sections"]
    if not isinstance(sections, list) or not 1 <= len(sections) <= 5:
        return False

    for section in sections:
        if not isinstance(section, dict):
            return False
        if not isinstance(section.get("title"), str):
            return False
        if not isinstance(section.get("text"), str):
            return False

    return True


def _supported_repair(issue: Dict[str, Any]) -> bool:
    """
    v1.2 supports three narrowly-scoped production failure modes:

    - wrong_entity_attribution:
      correct only when the source excerpt directly supports the
      correct attribution.

    - unsupported_claim:
      delete-only repair. The unsupported claim is removed rather than
      replaced with an inferred or guessed fact.

    - event_date_mismatch:
      delete-only repair. Remove the unsupported date or temporal assertion
      without inventing, substituting, or inferring another date/status.

    No other repair types are accepted.
    """
    issue_type = str(issue.get("type", "")).strip().casefold()

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

    if issue_type in {"wrong_entity_attribution", "wrong_role"}:
        return bool(str(issue.get("source_excerpt", "")).strip())

    if issue_type in {"unsupported_claim", "event_date_mismatch"}:
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
        issue_type = str(issue.get("type", "")).strip().casefold()

        if issue_type in {"wrong_entity_attribution", "wrong_role"}:
            rules = (
                "- Correct the entity/role only when the source excerpt directly "
                "establishes the correct entity or role.\n"
                "- If the source does not establish the correction, DELETE the "
                "incorrect attribution/role instead of guessing."
            )
        elif issue_type == "unsupported_claim":
            rules = (
                "- DELETE-ONLY: remove the unsupported claim or smallest sentence "
                "containing it.\n"
                "- Do not soften, paraphrase, reinterpret or replace it."
            )
        elif issue_type == "event_date_mismatch":
            rules = (
                "- DELETE-ONLY: remove the unsupported date/temporal assertion.\n"
                "- Do not replace it with an inferred date or status.\n"
                "- Preserve surrounding supported event wording."
            )
        else:
            raise ValueError(f"Unsupported Fact Guard repair type: {issue_type}")

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
- Return ONLY valid article JSON matching the required schema.

ISSUE-SPECIFIC RULES:
{issue_rules}

FACT GUARD ISSUE:
{json.dumps(issue, ensure_ascii=False, indent=2)}

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
