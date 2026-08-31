"""
TrendCurrent Story Coherence Layer
----------------------------------
Isolated editorial boundary between verified evidence extraction and article
generation.

Purpose:
    Prevent multiple independent stories from being treated as one article
    merely because they share an entity, keyword, publisher, category or topic.

Design:
    - Uses the already retrieved news and already provenance-verified facts.
    - Selects ONE concrete story anchor from the supplied source headlines.
    - Classifies every locked fact as PRIMARY, SUPPORTING or EXCLUDED.
    - Only PRIMARY + SUPPORTING facts are returned to the writer.
    - Fails closed on malformed/ambiguous model output.
    - Does not alter discovery, scoring, freshness, evidence extraction,
      evidence density, Fact Guard, or article-length logic.
"""

import json
import os
import time
import re

from ollama import chat
from config import MODEL


STORY_COHERENCE_TOKENS = max(
    300, int(os.getenv("OLLAMA_STORY_COHERENCE_TOKENS", "500"))
)
STORY_COHERENCE_CTX = max(
    4096, int(os.getenv("OLLAMA_STORY_COHERENCE_CTX", "6144"))
)


_FORMAT = {
    "type": "object",
    "properties": {
        "story_anchor": {"type": "string"},
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact_id": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": ["PRIMARY", "SUPPORTING", "EXCLUDED"],
                    },
                },
                "required": ["fact_id", "classification"],
            },
        },
    },
    "required": ["story_anchor", "classifications"],
}


def _compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _extract_json_object(text):
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
        raise ValueError("Story coherence returned no JSON object.")

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

    raise ValueError("Story coherence returned incomplete JSON.")


def _source_titles(news):
    titles = []
    for index, item in enumerate(news or [], 1):
        title = str(item.get("title", "")).strip()
        if title:
            titles.append({"source_id": f"A{index}", "title": title})
    return titles


def _build_prompt(topic, news, facts, story_anchor_hint=""):
    sources = _source_titles(news)

    source_block = "\n".join(
        f'{item["source_id"]}: {item["title"]}'
        for item in sources
    )

    fact_block = "\n".join(
        f'{item.get("id", "")}: {item.get("fact", "")} '
        f'[provenance={item.get("excerpt", "")}]'
        for item in facts
    )

    return f"""
You are TrendCurrent's story-coherence editor.

Your ONLY job is to separate ONE concrete news story from other independent
stories that may appear in the same Google News evidence set.

RAW DISCOVERY TOPIC:
{topic}

UPSTREAM STORY ANCHOR HINT:
{story_anchor_hint or "(none supplied)"}

SOURCE HEADLINES:
{source_block}

VERIFIED FACTS:
{fact_block}

CORE RULE:
Choose exactly ONE concrete event/development represented by the evidence.
The raw discovery topic may be broad, ambiguous, a person name, surname,
company, place, show, sport, country or other entity. It is NOT itself the
story.

UPSTREAM ANCHOR RULE:
- If an UPSTREAM STORY ANCHOR HINT is supplied, it defines the story identity.
- Do NOT switch to a different person, event, programme, incident or development.
- Use the supplied hint as the intended story scope even when other source
  headlines mention the same entity or keyword.
- "story_anchor" MUST be copied EXACTLY from one supplied source headline.
- The selected source headline should represent the SAME concrete story as the
  upstream hint. Do not invent or rewrite the anchor.
- If no supplied source headline can represent the upstream story safely, fail
  rather than selecting an unrelated story.

For every verified fact classify it:

PRIMARY:
- Directly describes, confirms, develops or materially explains the same
  concrete event/development as the selected story anchor.

SUPPORTING:
- Directly helps understand that same event/development.
- It may involve a different entity, location or consequence, but it must
  belong to the same developing story/event chain.
- It must not require starting a separate news story to explain it.

EXCLUDED:
- A separate event involving the same person/entity/keyword/category/place.
- A different match, programme, transfer, incident, weather system, political
  development or entertainment story.
- Generic background that is not directly needed to understand the selected
  event.
- Anything that only looks related because it shares a surname, keyword,
  publisher, topic, category or broad subject.

IMPORTANT EXAMPLES:
- "Sadiq Khan + Oxford Street" and "Salman Khan + Bollywood" are separate.
- A US strike on Iran, Iranian retaliation, affected regional bases and a
  directly resulting Strait of Hormuz development may belong to one developing
  story when the evidence explicitly connects them.
- Several tropical systems are NOT one story merely because they affect the
  same region.

CONSERVATIVE RULE:
When classification is uncertain, use EXCLUDED.
It is better to produce a shorter article from fewer coherent facts than to
combine independent stories.

PROVENANCE:
Do not alter fact IDs. Return every input fact ID exactly once.
Do not invent IDs.

Return ONLY JSON:
{{
  "story_anchor": "EXACT SOURCE HEADLINE",
  "classifications": [
    {{"fact_id":"F1","classification":"PRIMARY"}}
  ]
}}
"""


def _call(prompt):
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0,
            "top_p": 0.85,
            "top_k": 40,
            "num_ctx": STORY_COHERENCE_CTX,
            "num_predict": STORY_COHERENCE_TOKENS,
        },
        format=_FORMAT,
    )
    return _extract_json_object(response.message.content or "")


def _validate_and_filter(result, topic, news, evidence, story_anchor_hint=""):
    if not isinstance(result, dict):
        raise ValueError("Story coherence result is not an object.")

    facts = evidence.get("facts", [])
    if not isinstance(facts, list) or not facts:
        raise ValueError("Story coherence received no facts.")

    valid_ids = [str(item.get("id", "")).strip() for item in facts]
    valid_ids = [item for item in valid_ids if item]

    source_titles = [item["title"] for item in _source_titles(news)]
    anchor = str(result.get("story_anchor", "")).strip()

    if not anchor or anchor not in source_titles:
        raise ValueError(
            "Story coherence returned a story anchor that is not an exact "
            "supplied source headline."
        )

    # If upstream supplied a concrete discovery headline, require the selected
    # source headline to represent that same story. We use conservative token
    # overlap only as an identity sanity check; the LLM still performs the
    # semantic fact classification.
    hint = " ".join(str(story_anchor_hint or "").split()).strip()
    if hint:
        def _identity_tokens(text):
            return {
                token.casefold()
                for token in re.findall(r"[A-Za-z0-9À-ÿ']+", text or "")
                if len(token) >= 4
            }
        hint_tokens = _identity_tokens(hint)
        anchor_tokens = _identity_tokens(anchor)
        if hint_tokens:
            overlap = len(hint_tokens & anchor_tokens) / max(1, len(hint_tokens))
            if overlap < 0.35:
                raise ValueError(
                    "Story coherence selected a source headline that is not "
                    "sufficiently aligned with the upstream story anchor hint."
                )

    raw = result.get("classifications")
    if not isinstance(raw, list):
        raise ValueError("Story coherence classifications are not a list.")

    seen = set()
    classification_map = {}

    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Story coherence classification item is invalid.")

        fact_id = str(item.get("fact_id", "")).strip()
        classification = str(item.get("classification", "")).strip().upper()

        if fact_id not in valid_ids:
            raise ValueError(
                f"Story coherence returned unknown fact id: {fact_id}"
            )
        if fact_id in seen:
            raise ValueError(
                f"Story coherence returned duplicate fact id: {fact_id}"
            )
        if classification not in {"PRIMARY", "SUPPORTING", "EXCLUDED"}:
            raise ValueError(
                f"Story coherence returned invalid classification for {fact_id}: "
                f"{classification}"
            )

        seen.add(fact_id)
        classification_map[fact_id] = classification

    missing = [fact_id for fact_id in valid_ids if fact_id not in seen]
    if missing:
        raise ValueError(
            "Story coherence did not classify every fact: "
            + ", ".join(missing)
        )

    selected = []
    excluded = []

    for fact in facts:
        fact_id = str(fact.get("id", "")).strip()
        classification = classification_map[fact_id]
        copied = dict(fact)
        copied["story_classification"] = classification

        if classification in {"PRIMARY", "SUPPORTING"}:
            selected.append(copied)
        else:
            excluded.append(copied)

    if not selected:
        raise ValueError(
            "Story coherence excluded every fact; candidate is not safely "
            "usable as a coherent story."
        )

    filtered = dict(evidence)
    filtered["facts"] = selected
    filtered["excluded_facts"] = excluded
    filtered["story_anchor"] = anchor
    filtered["story_coherence_version"] = "v1.1"

    # primary_group remains for compatibility with the existing evidence
    # contract. Recompute it only from the selected facts.
    group_counts = {}
    for fact in selected:
        group = str(fact.get("group", "")).strip()
        if group:
            group_counts[group] = group_counts.get(group, 0) + 1

    if group_counts:
        filtered["primary_group"] = max(
            group_counts,
            key=group_counts.get,
        )

    print(
        f"[STORY COHERENCE] PASS | anchor={anchor} | "
        f"selected={len(selected)} | excluded={len(excluded)}"
    )

    for fact in excluded:
        print(
            f"[STORY COHERENCE] EXCLUDED | "
            f"{fact.get('id', '')} | {fact.get('fact', '')}"
        )

    return filtered


def _retry_missing_classifications(result, topic, news, facts, missing_ids, story_anchor_hint=""):
    """Retry only missing fact IDs against the already selected story anchor."""
    fact_map = {str(item.get("id", "")).strip(): item for item in facts}
    retry_facts = [fact_map[fid] for fid in missing_ids if fid in fact_map]
    if not retry_facts:
        raise ValueError("Story coherence missing fact IDs could not be retried.")

    prompt = f"""
You are repairing a TrendCurrent story-coherence classification.

UPSTREAM STORY ANCHOR HINT:
{story_anchor_hint or "(none supplied)"}

LOCKED STORY ANCHOR:
{str(result.get("story_anchor", "")).strip()}

Classify ONLY these missing fact IDs against the locked story.
Do not change the story anchor. Do not invent IDs.

FACTS:
{chr(10).join(str(x.get("id", "")) + ": " + str(x.get("fact", "")) + " [provenance=" + str(x.get("excerpt", "")) + "]" for x in retry_facts)}

Return ONLY JSON:
{{
  "classifications": [
    {{"fact_id":"F1","classification":"PRIMARY"}}
  ]
}}

Allowed classifications: PRIMARY, SUPPORTING, EXCLUDED.
When uncertain, use EXCLUDED.
"""
    repaired = _call(prompt)
    raw = repaired.get("classifications")
    if not isinstance(raw, list):
        raise ValueError("Story coherence retry classifications are not a list.")

    merged = dict(result)
    existing = list(result.get("classifications") or [])
    seen = {str(x.get("fact_id", "")).strip() for x in existing if isinstance(x, dict)}

    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Story coherence retry classification item is invalid.")
        fid = str(item.get("fact_id", "")).strip()
        cls = str(item.get("classification", "")).strip().upper()
        if fid not in missing_ids:
            raise ValueError(f"Story coherence retry returned unexpected fact id: {fid}")
        if fid in seen:
            raise ValueError(f"Story coherence retry returned duplicate fact id: {fid}")
        if cls not in {"PRIMARY", "SUPPORTING", "EXCLUDED"}:
            raise ValueError(f"Story coherence retry returned invalid classification for {fid}: {cls}")
        existing.append({"fact_id": fid, "classification": cls})
        seen.add(fid)

    merged["classifications"] = existing
    return merged


def filter_evidence(evidence, topic, news, story_anchor_hint=""):
    """
    Apply the isolated story-coherence boundary to an already verified
    evidence lock.

    This function is intentionally fail-closed: if the coherence model cannot
    produce a complete, provenance-safe classification, the candidate must be
    rejected instead of falling back to mixed evidence.
    """
    started = time.perf_counter()

    if not isinstance(evidence, dict):
        raise ValueError("Story coherence received invalid evidence.")

    facts = evidence.get("facts", [])
    if not isinstance(facts, list) or not facts:
        raise ValueError("Story coherence received empty evidence.")

    if not isinstance(news, list) or not news:
        raise ValueError("Story coherence received no source headlines.")

    prompt = _build_prompt(
        str(topic or "").strip(),
        news,
        facts,
        story_anchor_hint=story_anchor_hint,
    )

    print(
        f"[STORY COHERENCE] START | topic={str(topic or '').strip()} | "
        f"anchor_hint={str(story_anchor_hint or '').strip()} | "
        f"facts={len(facts)} | sources={len(news)}"
    )

    result = _call(prompt)

    # If the model omitted one or more fact IDs, retry only those IDs.
    valid_ids = [
        str(item.get("id", "")).strip()
        for item in facts
        if str(item.get("id", "")).strip()
    ]
    returned_ids = {
        str(item.get("fact_id", "")).strip()
        for item in (result.get("classifications") or [])
        if isinstance(item, dict)
    }
    missing_ids = [fid for fid in valid_ids if fid not in returned_ids]
    if missing_ids:
        print(
            f"[STORY COHERENCE] CLASSIFICATION RETRY | "
            f"missing={','.join(missing_ids)}"
        )
        result = _retry_missing_classifications(
            result,
            topic,
            news,
            facts,
            missing_ids,
            story_anchor_hint=story_anchor_hint,
        )

    filtered = _validate_and_filter(
        result,
        topic,
        news,
        evidence,
        story_anchor_hint=story_anchor_hint,
    )

    print(
        f"[STORY COHERENCE] TOTAL | "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )

    return filtered
