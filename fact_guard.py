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

FACT_GUARD_VERSION = "fact-guard-v1.0"

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
    # We do not maintain a hard-coded list of people. Instead, when
    # the source contains an explicit title immediately around a name,
    # the independent semantic audit will compare the article against it.
    # --------------------------------------------------------

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
- Be especially careful with:
  * "record", "lowest/highest ever", "unprecedented", "first-ever"
  * "nationwide", "all cinemas", "everywhere"
  * causes, motives or connections inferred from chronology
  * people whose job title/role may have been changed
  * old information presented as current
  * reported/expected/planned information presented as confirmed
- A contextual sentence is NOT automatically wrong merely because it is
  not word-for-word in the source. Flag it only when it makes a factual
  claim that the source cannot reasonably support.
- If a claim is uncertain, prefer severity REVIEW rather than HIGH.
- HIGH means a clear material factual error or unsupported concrete claim.
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

    all_issues = deterministic + semantic.get("issues", [])

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
