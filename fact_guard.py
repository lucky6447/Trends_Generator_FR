import json
import os
import re
import sys
from typing import Any, Dict, List

from ollama import chat
from config import MODEL


# ============================================================
# TrendCurrent FACT GUARD
# External post-generation factual validation layer.
#
# IMPORTANT:
#   This file does NOT modify ollama_client.py.
#
# Input:
#   source material + generated article
#
# Output:
#   PASS / FLAG with concrete factual issues.
#
# Design:
#   1) deterministic checks for obvious contradictions
#   2) independent Ollama audit for semantic/source-grounding issues
#   3) no automatic article rewriting
# ============================================================

FACT_GUARD_VERSION = "fact-guard-v1.1.1-temporal"

NUM_THREADS = max(1, int(os.getenv("FACT_GUARD_NUM_THREADS", "16")))
NUM_CTX = max(4096, int(os.getenv("FACT_GUARD_NUM_CTX", "8192")))
NUM_BATCH = max(64, int(os.getenv("FACT_GUARD_NUM_BATCH", "512")))
AUDIT_TOKENS = max(300, int(os.getenv("FACT_GUARD_AUDIT_TOKENS", "520")))

print(f"[FACT GUARD] {FACT_GUARD_VERSION}")


# ============================================================
# JSON / text helpers
# ============================================================

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


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _article_text(article: Dict[str, Any]) -> str:
    parts = [
        article.get("title", ""),
        article.get("description", ""),
        article.get("h1", ""),
        article.get("intro", ""),
    ]

    for section in article.get("sections", []):
        if isinstance(section, dict):
            parts.append(section.get("title", ""))
            parts.append(section.get("text", ""))

    return "\n".join(str(x) for x in parts if x)


def _source_contains(source: str, text: str) -> bool:
    text_norm = _normalize(text)
    return bool(text_norm) and text_norm in _normalize(source)


# ============================================================

def _temporal_consistency_signals(article: Dict[str, Any]) -> Dict[str, bool]:
    """
    Detect whether a focused cross-fact temporal audit is warranted.

    This is only a trigger. It does not decide factual correctness.
    """
    text_norm = _normalize(_article_text(article))

    temporal_patterns = [
        r"\bround\s+\d+\b", r"\bround\b",
        r"\bweek\s+\d+\b", r"\bmatchday\s+\d+\b", r"\bday\s+\d+\b",
        r"\bopening round\b", r"\bsecond round\b", r"\bthird round\b",
        r"\bfinal round\b", r"\bquarter[- ]final\b", r"\bsemi[- ]final\b",
        r"\bfinal\b", r"\btoday\b", r"\byesterday\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b",
        r"\b[a-z]+\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]

    score_patterns = [
        r"\b(?:score|scored|shot|fired|carded|finished|recorded)\b.{0,45}\b\d{1,3}\b",
        r"\b\d{1,3}\s*(?:under|over)\b",
        r"\b\d{1,3}[-–]\d{1,3}\b",
        r"\b\d{1,3}\b.{0,12}\b(?:goals?|points?|runs?|yards?|strokes?|shots?)\b",
    ]

    relation_patterns = [
        r"\bin contrast\b", r"\bwhile\b", r"\bcompared with\b",
        r"\bcompared to\b", r"\bversus\b", r"\bvs\.?\b",
        r"\bwhereas\b", r"\bmeanwhile\b", r"\bbut\b",
        r"\bafter\b", r"\bbefore\b", r"\bsame (?:day|round|week|match)\b",
    ]

    return {
        "temporal": any(re.search(p, text_norm) for p in temporal_patterns),
        "score": any(re.search(p, text_norm) for p in score_patterns),
        "relation": any(re.search(p, text_norm) for p in relation_patterns),
    }


# Deterministic checks
# ============================================================

def _deterministic_checks(source: str, article: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    These checks are deliberately conservative.

    They do NOT attempt to prove that every sentence is true.
    They catch obvious situations where the article contains a
    suspiciously strong claim or a contradiction that can be
    demonstrated directly from the supplied source.
    """
    issues: List[Dict[str, Any]] = []
    text = _article_text(article)
    source_norm = _normalize(source)
    text_norm = _normalize(text)

    def add(severity: str, issue_type: str, claim: str, reason: str):
        issues.append({
            "severity": severity,
            "type": issue_type,
            "claim": claim,
            "reason": reason,
            "evidence": [],
            "deterministic": True,
        })

    # --------------------------------------------------------
    # Absolute / superlative claims
    # --------------------------------------------------------
    superlative_patterns = [
        (r"\blowest\b.{0,50}\brecord\b", "lowest on record"),
        (r"\bhighest\b.{0,50}\brecord\b", "highest on record"),
        (r"\brecord[- ]breaking\b", "record-breaking"),
        (r"\brecord\b.{0,50}\bintensit", "record intensity"),
        (r"\bunprecedented\b", "unprecedented"),
        (r"\bhistoric(?:al)?\b.{0,40}\bintens", "historic intensity"),
        (r"\ball[- ]time\b", "all-time"),
        (r"\bfirst[- ]ever\b", "first-ever"),
        (r"\bnever before\b", "never before"),
    ]

    for pattern, label in superlative_patterns:
        if re.search(pattern, text_norm):
            # Only flag automatically when the source itself does not
            # contain a comparable explicit formulation.
            if not re.search(pattern, source_norm):
                add(
                    "HIGH",
                    "unsupported_superlative",
                    label,
                    "The article makes a strong superlative/record claim that "
                    "is not explicitly present in the supplied source."
                )

    # --------------------------------------------------------
    # Nationwide / universal availability claims
    # --------------------------------------------------------
    availability_patterns = [
        r"\bnationwide\b",
        r"\bacross the country\b",
        r"\bthroughout the country\b",
        r"\bin all cinemas\b",
        r"\bat cinemas across\b",
        r"\beverywhere\b",
    ]

    for pattern in availability_patterns:
        if re.search(pattern, text_norm) and not re.search(pattern, source_norm):
            add(
                "HIGH",
                "scope_inflation",
                re.search(pattern, text_norm).group(0),
                "The article broadens the source to a nationwide/universal scope "
                "that is not explicitly supported."
            )

    # --------------------------------------------------------
    # Causal / motive language
    # --------------------------------------------------------
    causal_patterns = [
        r"\bbecause of\b",
        r"\bdue to\b",
        r"\bcaused by\b",
        r"\bcausing\b",
        r"\bled to\b",
        r"\bresulted in\b",
        r"\bconnected to\b",
        r"\blinked to\b",
        r"\bbehind (?:his|her|their) death\b",
        r"\bpossible connection\b",
    ]

    # Do not automatically reject every causal phrase. The LLM audit
    # determines whether the specific causal claim is actually supported.
    # We only mark it for the semantic audit.
    if any(re.search(pattern, text_norm) for pattern in causal_patterns):
        add(
            "REVIEW",
            "causal_claim",
            "One or more causal/connection claims detected.",
            "Causal or motive language requires explicit source support and "
            "must not be inferred merely from chronology or context."
        )

    # --------------------------------------------------------
    # Quotation / attribution guard
    # --------------------------------------------------------
    quote_chars = ['"', "“", "”", "„", "«", "»"]
    quoted_segments = []

    for match in re.finditer(r'"([^"]{8,300})"|“([^”]{8,300})”|„([^“]{8,300})“|«([^»]{8,300})»', text):
        quoted = next((g for g in match.groups() if g), "")
        if quoted:
            quoted_segments.append(quoted)

    for quoted in quoted_segments:
        if not _source_contains(source, quoted):
            add(
                "HIGH",
                "unsupported_quote",
                quoted,
                "The quoted wording cannot be traced directly to the supplied source."
            )

    # --------------------------------------------------------
    # Obvious role/title substitutions
    #
    # Keep this deliberately narrow: role/title consistency is handled
    # by the independent audit, but we explicitly mark it as a required
    # high-priority comparison. We do not maintain a hard-coded person list.
    # --------------------------------------------------------

    role_patterns = [
        r"\\bmanager\\b", r"\\bcoach\\b", r"\\bhead coach\\b",
        r"\\bplayer\\b", r"\\bmidfielder\\b", r"\\bdefender\\b",
        r"\\bforward\\b", r"\\bpresident\\b", r"\\bceo\\b",
        r"\\bowner\\b", r"\\bminister\\b", r"\\bmayor\\b",
    ]
    article_has_role_claim = any(re.search(p, text_norm) for p in role_patterns)

    if article_has_role_claim:
        add(
            "REVIEW",
            "role_attribution_check",
            "Explicit person/entity role or title detected.",
            "The independent audit must verify that each named person's role/title "
            "matches the source and is not merely inferred from team/entity association."
        )

    # --------------------------------------------------------
    # Current-state / temporal consistency
    #
    # A historical event must not be silently presented as a new/current
    # event. The semantic audit is responsible for deciding the actual
    # chronology from the supplied source.
    # --------------------------------------------------------

    temporal_patterns = [
        r"\\b(joined|joins|signed|signs|moved|moves|transferred|transfer|departed|leaves|left)\\b",
        r"\\b(on|since|from|as of)\\s+\\d{1,2}\\s+[a-z]+\\s+\\d{4}\\b",
        r"\\b(on|since|from|as of)\\s+[a-z]+\\s+\\d{1,2},\\s+\\d{4}\\b",
        r"\\b(\\d{4}-\\d{2}-\\d{2})\\b",
        r"\\b(first|debut|latest|current|currently|today|yesterday)\\b",
    ]

    if any(re.search(p, text_norm) for p in temporal_patterns):
        add(
            "REVIEW",
            "temporal_state_check",
            "Explicit transfer/event/current-state claim detected.",
            "The independent audit must reconstruct the event chronology from the "
            "source and must not treat an old event as a new/current event."
        )


    # Cross-fact temporal relationship trigger. Individual facts can each be
    # true while their combined temporal relationship is false.
    temporal_signals = _temporal_consistency_signals(article)
    if (
        temporal_signals["temporal"]
        and temporal_signals["score"]
        and temporal_signals["relation"]
    ):
        add(
            "REVIEW",
            "cross_fact_temporal_consistency",
            "Multiple time-bound facts are connected by a comparative or "
            "relational statement.",
            "The focused temporal audit must verify that the connected facts "
            "belong to the same relevant event state rather than merely being "
            "individually true."
        )

    return issues


# ============================================================
# Independent semantic audit
# ============================================================

_AUDIT_FORMAT = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "type": {"type": "string"},
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                    "source_excerpt": {"type": "string"},
                },
                "required": [
                    "severity",
                    "type",
                    "claim",
                    "reason",
                    "source_excerpt",
                ],
            },
        },
    },
    "required": ["passed", "issues"],
}


def _audit_prompt(source: str, article: Dict[str, Any]) -> str:
    return f"""
You are an independent factual validator for TrendCurrent.

Your job is NOT to rewrite the article.
Your job is to determine whether the ARTICLE contains material factual
claims that are unsupported or contradicted by the SOURCE MATERIAL.

SOURCE MATERIAL is the ONLY factual authority.

STRICT RULES:
- Do not use outside knowledge.
- Do not assume that a claim is true because it sounds plausible.
- Do not require identical wording.
- Normal journalistic paraphrasing is allowed.
- Flag a claim when the source does not support it.
- Flag wrong names, roles, dates, numbers, locations, event status,
  attribution, quotations, causal claims and exaggerated scope.
- Reconstruct the chronology of important events before judging them.
- When two or more factual claims are linked by "while", "in contrast",
  "compared with", "versus", "whereas", "meanwhile", or similar wording,
  verify the RELATION between the facts, not only each fact in isolation.
- For time-bound results/statistics, bind each fact to its relevant event,
  round, matchday, week, date, or status whenever the source provides it.
  Do not combine a true fact from one temporal state with a true fact from
  another temporal state as if they occurred in the same state.
- A fact such as "Player A scored 61" and "Player B scored 74" can both be
  source-supported while the sentence connecting them is still materially
  wrong if 61 belongs to Round 2 and 74 belongs to Round 1.
- Treat cross-round/cross-day/cross-match/cross-week conflation as HIGH when
  the source explicitly establishes the conflicting temporal assignments.
  Use REVIEW when the source does not contain enough temporal information.
- For every person mentioned with a job/team role, verify the ROLE itself,
  not merely that the person is associated with the club/company/entity.
- Treat PLAYER vs MANAGER/COACH and similar role substitutions as material
  factual errors when the source establishes the person's actual role.
- For transfers, appointments, departures, signings and debuts, determine
  the event date and the person's state at the article's claimed time.
  An old/historical transfer must NOT be accepted as a new/current transfer
  merely because the same person and destination still appear in the source.
- Be especially careful with:
  * "record", "lowest/highest ever", "unprecedented", "first-ever"
  * "nationwide", "all cinemas", "everywhere"
  * causes, motives or connections inferred from chronology
  * people whose job title/role may have been changed
  * old information presented as current
  * historical transfers/events presented as newly announced
   * results/statistics from different rounds, dates, matchdays or weeks
     accidentally combined into one comparison or same-day narrative
  * reported/expected/planned information presented as confirmed
- A contextual sentence is NOT automatically wrong merely because it is
  not word-for-word in the source. Flag it only when it makes a factual
  claim that the source cannot reasonably support.
- If a claim is uncertain, prefer severity REVIEW rather than HIGH.
- HIGH means a clear material factual error or unsupported concrete claim.
- A clearly wrong person-role attribution is HIGH.
- A clearly wrong event chronology, such as an old transfer presented as a
  new/current transfer, is HIGH.
- REVIEW means the wording requires human/secondary verification.
- Ignore style, grammar and harmless editorial wording.

For every issue, provide a short source excerpt that directly supports
your conclusion. If no source excerpt can be identified, leave it empty
and explain why.

Return ONLY JSON.

Format:
{{
  "passed": true,
  "issues": []
}}

or:

{{
  "passed": false,
  "issues": [
    {{
      "severity": "HIGH",
      "type": "wrong_role",
      "claim": "example claim",
      "reason": "The source identifies the person differently.",
      "source_excerpt": "short exact excerpt"
    }}
  ]
}}

SOURCE MATERIAL:
{source}

ARTICLE:
{json.dumps(article, ensure_ascii=False)}
"""



_TEMPORAL_AUDIT_FORMAT = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "type": {"type": "string"},
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                    "source_excerpt": {"type": "string"},
                },
                "required": [
                    "severity", "type", "claim", "reason", "source_excerpt"
                ],
            },
        },
    },
    "required": ["passed", "issues"],
}


def _temporal_audit_prompt(source: str, article: Dict[str, Any]) -> str:
    return f"""
You are a focused temporal-consistency validator for TrendCurrent.

Your ONLY task is to detect factual errors caused by combining individually
true facts that belong to different temporal/event states.

SOURCE MATERIAL is the ONLY factual authority.

STRICT RULES:
- Do not use outside knowledge.
- Do not rewrite the article.
- Do not reject normal paraphrasing.
- Bind each relevant result/statistic/event to the source-supported event,
  round, matchday, week, date, or status whenever available.
- Inspect whether the ARTICLE connects facts as if they belong to the same
  temporal state.
- Pay special attention to "while", "in contrast", "compared with",
  "versus", "whereas", "meanwhile", "but", and similar relational wording.
- A fact being individually true is NOT enough. The relationship between
  connected facts must also be source-supported.
- Example:
  Source: A = 61 in Round 2; B = 74 in Round 1; B = 70 in Round 2.
  Article: "A shot 61, while B struggled with 74."
  This is HIGH if the wording presents 61 and 74 as results from the same
  relevant round/day.
- Do NOT infer a same-round relationship merely because facts appear in the
  same source or article.
- If the article explicitly identifies different rounds/dates, that is fine.
- If the source lacks enough temporal information to decide, use REVIEW.
- HIGH requires a clear temporal contradiction established by the source.
- Ignore style and grammar.

For every issue, provide a short exact source excerpt supporting the
temporal conclusion. If none can be identified, leave it empty.

Return ONLY JSON:
{{
  "passed": true,
  "issues": []
}}

or:
{{
  "passed": false,
  "issues": [
    {{
      "severity": "HIGH",
      "type": "cross_round_conflation",
      "claim": "short description of the connected claim",
      "reason": "why the article combines different temporal states",
      "source_excerpt": "short exact excerpt"
    }}
  ]
}}

SOURCE MATERIAL:
{source}

ARTICLE:
{json.dumps(article, ensure_ascii=False)}
"""


def _ollama_temporal_audit(source: str, article: Dict[str, Any]) -> Dict[str, Any]:
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": _temporal_audit_prompt(source, article)}],
        options={
            "temperature": 0.0,
            "top_p": 0.85,
            "top_k": 40,
            "num_ctx": NUM_CTX,
            "num_predict": AUDIT_TOKENS,
            "num_batch": NUM_BATCH,
            "num_thread": NUM_THREADS,
        },
        format=_TEMPORAL_AUDIT_FORMAT,
    )

    raw = response.message.content or ""
    result = _extract_json_object(raw)

    if not isinstance(result, dict):
        raise ValueError("Temporal Fact Guard audit returned invalid JSON.")

    issues = result.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    clean = []
    for item in issues:
        if not isinstance(item, dict):
            continue

        claim = str(item.get("claim", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not claim or not reason:
            continue

        severity = str(item.get("severity", "REVIEW")).upper()
        if severity not in {"HIGH", "MEDIUM", "LOW", "REVIEW"}:
            severity = "REVIEW"

        clean.append({
            "severity": severity,
            "type": str(
                item.get("type", "cross_fact_temporal_consistency")
            ).strip() or "cross_fact_temporal_consistency",
            "claim": claim,
            "reason": reason,
            "source_excerpt": str(item.get("source_excerpt", "")).strip(),
            "deterministic": False,
            "audit_layer": "focused_temporal",
        })

    return {
        "passed": bool(result.get("passed", False)) and not clean,
        "issues": clean,
    }


def _ollama_audit(source: str, article: Dict[str, Any]) -> Dict[str, Any]:
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": _audit_prompt(source, article)}],
        options={
            "temperature": 0.0,
            "top_p": 0.85,
            "top_k": 40,
            "num_ctx": NUM_CTX,
            "num_predict": AUDIT_TOKENS,
            "num_batch": NUM_BATCH,
            "num_thread": NUM_THREADS,
        },
        format=_AUDIT_FORMAT,
    )

    raw = response.message.content or ""
    result = _extract_json_object(raw)

    if not isinstance(result, dict):
        raise ValueError("Fact Guard audit returned invalid JSON.")

    issues = result.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    clean = []

    for item in issues:
        if not isinstance(item, dict):
            continue

        claim = str(item.get("claim", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not claim or not reason:
            continue

        severity = str(item.get("severity", "REVIEW")).upper()
        if severity not in {"HIGH", "MEDIUM", "LOW", "REVIEW"}:
            severity = "REVIEW"

        clean.append({
            "severity": severity,
            "type": str(item.get("type", "unsupported_claim")).strip()
                     or "unsupported_claim",
            "claim": claim,
            "reason": reason,
            "source_excerpt": str(item.get("source_excerpt", "")).strip(),
            "deterministic": False,
        })

    return {
        "passed": bool(result.get("passed", False)) and not clean,
        "issues": clean,
    }


# ============================================================
# Result normalization
# ============================================================

def validate(source: str, article: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(article, dict):
        raise ValueError("Article must be a JSON object.")

    deterministic = _deterministic_checks(source, article)

    # The semantic audit remains independent. It is not told about
    # deterministic findings, so the two layers can cross-check each other.
    semantic = _ollama_audit(source, article)

    # Focused temporal audit is intentionally narrow: it runs only when the
    # article contains temporal context + result/statistic + relational wording.
    temporal_signals = _temporal_consistency_signals(article)
    temporal = {"issues": []}
    if (
        temporal_signals["temporal"]
        and temporal_signals["score"]
        and temporal_signals["relation"]
    ):
        temporal = _ollama_temporal_audit(source, article)


    all_issues = (
        deterministic
        + semantic.get("issues", [])
        + temporal.get("issues", [])
    )

    # REVIEW alone does not automatically reject an article.
    # HIGH / MEDIUM are publication blockers.
    blocking = [
        issue for issue in all_issues
        if issue.get("severity") in {"HIGH", "MEDIUM"}
    ]

    status = "FLAG" if blocking else "PASS"

    return {
        "fact_guard_version": FACT_GUARD_VERSION,
        "status": status,
        "passed": status == "PASS",
        "blocking_issues": len(blocking),
        "review_items": len([
            x for x in all_issues if x.get("severity") == "REVIEW"
        ]),
        "issues": all_issues,
    }


# ============================================================
# CLI
# ============================================================

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: python fact_guard.py source.txt article.json",
            file=sys.stderr,
        )
        return 2

    source_path = sys.argv[1]
    article_path = sys.argv[2]

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    article = _load_json(article_path)
    result = validate(source, article)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
