import json
import re
import os
from ollama import chat
from config import MODEL


def _clean(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_json(content):
    raw = str(content or "")
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("repetition guard returned no JSON object")
    return json.loads(raw[start:end + 1])


def validate(article, evidence_lock):
    """Hard post-generation gate for semantic paragraph redundancy."""
    paragraphs = article.get("paragraphs", []) if isinstance(article, dict) else []
    paragraphs = [_clean(p) for p in paragraphs if _clean(p)]
    facts = evidence_lock.get("facts", []) if isinstance(evidence_lock, dict) else []
    facts = [_clean(f) for f in facts if _clean(f)]

    if not paragraphs:
        return {
            "status": "REJECT",
            "reason": "no article paragraphs",
            "paragraph_count": 0,
            "paragraphs_with_new_information": 0,
            "redundant_paragraphs": 0,
            "repeated_information_units": 0,
            "repeated_pairs": [],
            "unique_information_units": 0,
            "information_density": 0.0,
        }

    fact_lines = "\n".join(f"F{i}: {f}" for i, f in enumerate(facts, 1))
    para_lines = "\n".join(f"P{i}: {p}" for i, p in enumerate(paragraphs, 1))

    prompt = f"""
You are TrendCurrent's strict post-generation repetition quality gate.

Your job is NOT to judge style, quality, grammar, or whether the article could be richer.
Judge ONLY whether paragraphs repeat information already established earlier.

A paragraph PASSes only if it adds at least one substantive, source-supported information unit
that was not already established in an earlier paragraph.

Important:
- Different wording of the same fact is repetition.
- Repeating the same event, outcome, cause, reaction, number, decision, or consequence is repetition.
- Sharing a person/place/event is NOT repetition if the paragraph adds a genuinely new fact.
- Attribution alone is NOT new information.
- A necessary short transition is acceptable only when it does not create a separate paragraph.
- Do NOT penalize a short factual article for being short.
- Do NOT demand extra detail beyond the evidence.
- Use the locked facts as the factual universe.
- Be conservative: reject only when a paragraph materially repeats earlier information instead of advancing the story.

LOCKED FACTS:
{fact_lines}

ARTICLE PARAGRAPHS:
{para_lines}

Return ONLY JSON in this exact shape:
{{
  "status":"PASS" or "REJECT",
  "paragraphs_with_new_information": 0,
  "redundant_paragraphs": 0,
  "repeated_information_units": 0,
  "repeated_pairs":[[1,3]],
  "unique_information_units": 0,
  "information_density": 0.0,
  "reason":"brief factual explanation"
}}

Rules for metrics:
- paragraphs_with_new_information = number of paragraphs that add substantive new information.
- redundant_paragraphs = paragraphs whose substantive content is materially already covered earlier.
- repeated_information_units = count of distinct repeated substantive information units, not repeated words.
- repeated_pairs = paragraph number pairs where the later paragraph materially repeats the earlier one.
- unique_information_units = distinct substantive information units actually conveyed by the article.
- information_density = unique_information_units / paragraph_count, rounded to 2 decimals.
- If every paragraph advances the story with new information, PASS.
- If one or more later paragraphs mainly restate earlier information, REJECT.
"""

    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "top_p": 0.85, "top_k": 40, "num_ctx": max(4096, int(os.getenv("OLLAMA_NUM_CTX", "6144"))), "num_predict": 240},
        format={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["PASS", "REJECT"]},
                "paragraphs_with_new_information": {"type": "integer"},
                "redundant_paragraphs": {"type": "integer"},
                "repeated_information_units": {"type": "integer"},
                "repeated_pairs": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}},
                "unique_information_units": {"type": "integer"},
                "information_density": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": [
                "status", "paragraphs_with_new_information", "redundant_paragraphs",
                "repeated_information_units", "repeated_pairs",
                "unique_information_units", "information_density", "reason",
            ],
        },
    )
    result = _extract_json(getattr(getattr(response, "message", None), "content", ""))

    status = str(result.get("status", "")).strip().upper()
    if status not in {"PASS", "REJECT"}:
        raise ValueError("repetition guard returned invalid status")

    paragraph_count = len(paragraphs)
    new_count = max(0, min(paragraph_count, int(result.get("paragraphs_with_new_information", 0))))
    redundant = max(0, min(paragraph_count, int(result.get("redundant_paragraphs", 0))))
    repeated_units = max(0, int(result.get("repeated_information_units", 0)))
    unique_units = max(0, int(result.get("unique_information_units", 0)))

    pairs = result.get("repeated_pairs", [])
    clean_pairs = []
    if isinstance(pairs, list):
        for pair in pairs:
            if isinstance(pair, list) and len(pair) == 2:
                try:
                    a, b = int(pair[0]), int(pair[1])
                    if 1 <= a < b <= paragraph_count:
                        clean_pairs.append([a, b])
                except Exception:
                    pass

    density = round(unique_units / paragraph_count, 2) if paragraph_count else 0.0
    reason = _clean(result.get("reason", ""))[:500]

    # Hard safety rule: any redundant later paragraph or repeated information unit blocks publication.
    if new_count < paragraph_count or redundant > 0 or repeated_units > 0 or clean_pairs:
        status = "REJECT"

    return {
        "status": status,
        "reason": reason or ("paragraph redundancy detected" if status == "REJECT" else "paragraphs add new information"),
        "paragraph_count": paragraph_count,
        "paragraphs_with_new_information": new_count,
        "redundant_paragraphs": redundant,
        "repeated_information_units": repeated_units,
        "repeated_pairs": clean_pairs,
        "unique_information_units": unique_units,
        "information_density": density,
    }
