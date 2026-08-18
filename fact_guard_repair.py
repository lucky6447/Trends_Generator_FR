import json
import os
from typing import Any, Dict

from ollama import chat
from config import MODEL


FACT_GUARD_REPAIR_VERSION = "fact-guard-repair-v1.0"

NUM_THREADS = max(1, int(os.getenv("FACT_GUARD_NUM_THREADS", "16")))
NUM_CTX = max(4096, int(os.getenv("FACT_GUARD_NUM_CTX", "8192")))
NUM_BATCH = max(64, int(os.getenv("FACT_GUARD_NUM_BATCH", "512")))
REPAIR_TOKENS = max(300, int(os.getenv("FACT_GUARD_REPAIR_TOKENS", "520")))


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
    v1.0 intentionally supports only the production-proven
    wrong_entity_attribution failure mode.
    """
    return (
        str(issue.get("type", "")).strip().casefold()
        == "wrong_entity_attribution"
        and str(issue.get("severity", "")).strip().upper() in {"HIGH", "MEDIUM"}
        and bool(str(issue.get("claim", "")).strip())
        and bool(str(issue.get("reason", "")).strip())
        and bool(str(issue.get("source_excerpt", "")).strip())
    )


def repair(article: Dict[str, Any], source: str, guard_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform exactly one targeted repair.

    Safety rules:
    - Only wrong_entity_attribution is repairable in v1.0.
    - Exactly one blocking issue must exist.
    - The source material is the only factual authority.
    - The repair may not introduce new facts.
    - The caller MUST run Fact Guard again after this function returns.
    """
    if not isinstance(article, dict):
        raise ValueError("Article must be a JSON object.")

    if not isinstance(guard_result, dict):
        raise ValueError("Fact Guard result must be a JSON object.")

    blocking = [
        x for x in guard_result.get("issues", [])
        if isinstance(x, dict)
        and str(x.get("severity", "")).upper() in {"HIGH", "MEDIUM"}
    ]

    if len(blocking) != 1:
        raise ValueError(
            "fact-guard-repair-v1.0 requires exactly one blocking issue."
        )

    issue = blocking[0]

    if not _supported_repair(issue):
        raise ValueError(
            f"Unsupported Fact Guard repair type: {issue.get('type', '')}"
        )

    prompt = f"""
You are TrendCurrent's targeted Fact Guard repair engine.

Repair ONE already-generated article after an independent Fact Guard found
one wrong_entity_attribution error.

SOURCE MATERIAL is the ONLY factual authority.

CRITICAL RULES:
- Fix ONLY the listed wrong_entity_attribution issue.
- Preserve all other supported article content.
- Do NOT regenerate the article from scratch.
- Do NOT add any new fact, context, motive, date, number, quote or event.
- Do NOT use outside knowledge.
- If the source excerpt identifies the correct entity, replace the wrong
  entity attribution with the exact entity supported by the source.
- If the correct attribution cannot be established directly from the source
  material, DELETE the incorrect attribution/claim instead of guessing.
- Do not transfer an action, statement, role or status between entities based
  on proximity alone.
- Preserve uncertainty and status.
- Keep the article in its existing language.
- Return ONLY valid article JSON.

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
            "num_ctx": NUM_CTX,
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
