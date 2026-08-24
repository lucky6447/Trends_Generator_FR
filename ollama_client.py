import json
import os
import re
import time
from ollama import chat
from config import MODEL, LANGUAGE


# ============================================================
# TrendCurrent UNIVERSAL FACT-LOCK PIPELINE
# Balanced rewrite
#
# SOURCE
#   -> compact evidence extraction
#   -> primary-event lock
#   -> article generation
#   -> focused audit
#   -> delete-first repair (only if needed)
#   -> final audit
#
# Goals:
#   * source-grounded without being needlessly rigid
#   * compact Ollama output so CPU inference does not run for minutes
#   * no cross-event article construction
#   * no invented facts during repair
#   * language-independent
# ============================================================

PIPELINE_VERSION = "universal-fact-lock-v2.3.2-compact-evidence-json-fixed2"

# IMPORTANT: Do not force a CPU thread count by default.
# Ollama can auto-detect the runner's optimal thread count.
# Set OLLAMA_NUM_THREADS explicitly only if benchmarking proves a fixed value
# is faster on the production machine.
_OLLAMA_THREADS_RAW = os.getenv("OLLAMA_NUM_THREADS", "").strip()
NUM_THREADS = (
    max(1, int(_OLLAMA_THREADS_RAW))
    if _OLLAMA_THREADS_RAW
    else None
)

NUM_CTX = max(4096, int(os.getenv("OLLAMA_NUM_CTX", "6144")))

# IMPORTANT: Do not force num_batch=512 by default.
# Keep an explicit override available for controlled benchmarking.
_OLLAMA_BATCH_RAW = os.getenv("OLLAMA_NUM_BATCH", "").strip()
NUM_BATCH = (
    max(32, int(_OLLAMA_BATCH_RAW))
    if _OLLAMA_BATCH_RAW
    else None
)

# The old extractor asked the model for facts + quotes + groups at once.
# That made a 500-token ceiling very easy to hit.  The balanced extractor
# keeps one compact fact record and a small number of records.
EVIDENCE_CHUNK_CHARS = max(
    7000, int(os.getenv("OLLAMA_EVIDENCE_CHUNK_CHARS", "14000"))
)
EVIDENCE_TOKENS = max(
    280, int(os.getenv("OLLAMA_EVIDENCE_TOKENS", "320"))
)
EVIDENCE_MAX_FACTS = max(
    4, int(os.getenv("OLLAMA_EVIDENCE_MAX_FACTS", "6"))
)

ARTICLE_TOKENS = max(
    420, int(os.getenv("OLLAMA_ARTICLE_TOKENS", "520"))
)
AUDIT_TOKENS = max(
    120, int(os.getenv("OLLAMA_AUDIT_TOKENS", "180"))
)
REPAIR_TOKENS = max(
    280, int(os.getenv("OLLAMA_REPAIR_TOKENS", "420"))
)

MIN_ARTICLE_WORDS = max(
    0, int(os.getenv("OLLAMA_MIN_ARTICLE_WORDS", "0"))
)

# Evidence-aware minimum. This prevents tiny articles when the ledger is rich,
# while allowing genuinely short stories to remain short instead of padding.
ADAPTIVE_MIN_1_2 = max(0, int(os.getenv("OLLAMA_MIN_1_2_FACTS", "45")))
ADAPTIVE_MIN_3_4 = max(0, int(os.getenv("OLLAMA_MIN_3_4_FACTS", "65")))
ADAPTIVE_MIN_5_PLUS = max(0, int(os.getenv("OLLAMA_MIN_5_PLUS_FACTS", "85")))
MAX_SECTIONS = 6

# Evidence is deliberately sequential on CPU.
EVIDENCE_PARALLEL = os.getenv(
    "OLLAMA_EVIDENCE_PARALLEL", "0"
).lower() not in {"0", "false", "no", "off"}

# One retry is allowed only for malformed/truncated evidence JSON.
# The retry uses a smaller output contract, not another huge prompt.
EVIDENCE_RETRY_TOKENS = max(
    EVIDENCE_TOKENS, int(os.getenv("OLLAMA_EVIDENCE_RETRY_TOKENS", "360"))
)

print(f"[TrendCurrent PIPELINE] {PIPELINE_VERSION}")


# ============================================================
# JSON helpers
# ============================================================

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
        raise ValueError("Ollama returned no JSON object.")

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

    raise ValueError("Ollama returned incomplete JSON.")


def _compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _call(
    prompt,
    *,
    temperature=0.0,
    num_predict=500,
    num_thread=None,
    response_format="json",
):
    started = time.perf_counter()
    # None means "do not send num_thread", allowing Ollama to auto-detect.
    threads = (
        max(1, int(num_thread))
        if num_thread is not None
        else NUM_THREADS
    )
    batch = NUM_BATCH

    print(
        f"[TIMER] Ollama START | predict={num_predict} | temp={temperature} "
        f"| prompt_chars={len(prompt)}"
        + (f" | threads={threads}" if threads is not None else " | threads=AUTO")
        + (f" | batch={batch}" if batch is not None else " | batch=AUTO")
    )

    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": temperature,
            "top_p": 0.85,
            "top_k": 40,
            "num_ctx": NUM_CTX,
            "num_predict": num_predict,
        },
        "format": response_format,
    }

    # Only send runner knobs when explicitly configured.
    if batch is not None:
        kwargs["options"]["num_batch"] = batch
    if threads is not None:
        kwargs["options"]["num_thread"] = threads

    response = chat(**kwargs)
    raw = response.message.content or ""
    elapsed = time.perf_counter() - started

    timing = {}
    for name in (
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
        "total_duration",
        "prompt_eval_count",
        "eval_count",
    ):
        value = getattr(response, name, None)
        if value is not None:
            timing[name] = value

    eval_ns = timing.get("eval_duration")
    eval_count = timing.get("eval_count")
    tok_s = None
    if isinstance(eval_ns, (int, float)) and eval_ns > 0:
        if isinstance(eval_count, (int, float)):
            tok_s = eval_count / (eval_ns / 1_000_000_000)

    prompt_eval_ns = timing.get("prompt_eval_duration")
    prompt_eval_count = timing.get("prompt_eval_count")
    prompt_tok_s = None
    if isinstance(prompt_eval_ns, (int, float)) and prompt_eval_ns > 0:
        if isinstance(prompt_eval_count, (int, float)):
            prompt_tok_s = prompt_eval_count / (prompt_eval_ns / 1_000_000_000)

    def _sec(value):
        return value / 1_000_000_000 if isinstance(value, (int, float)) else None

    load_s = _sec(timing.get("load_duration"))
    prompt_s = _sec(timing.get("prompt_eval_duration"))
    eval_s = _sec(timing.get("eval_duration"))

    print(
        f"[TIMER] Ollama END   | elapsed={elapsed:.2f}s "
        f"| response_chars={len(raw)}"
        + (f" | load={load_s:.2f}s" if load_s is not None else "")
        + (f" | prompt_eval={prompt_s:.2f}s" if prompt_s is not None else "")
        + (f" | eval={eval_s:.2f}s" if eval_s is not None else "")
        + (f" | prompt_tokens={prompt_eval_count}" if prompt_eval_count is not None else "")
        + (f" | output_tokens={eval_count}" if eval_count is not None else "")
        + (f" | prompt_tok_s={prompt_tok_s:.2f}" if prompt_tok_s is not None else "")
        + (f" | eval_tok_s={tok_s:.2f}" if tok_s is not None else "")
    )

    try:
        return _extract_json_object(raw)
    except Exception as exc:
        raise ValueError(
            f"Invalid Ollama JSON: {exc}; response={raw[:1000]!r}"
        ) from exc


# ============================================================
# Source splitting
# ============================================================

def _split_source(source):
    """
    Preserve ARTICLE blocks when present.  If no ARTICLE markers exist,
    split only when necessary.
    """
    text = (source or "").strip()
    if not text:
        return [""]

    marker = re.compile(r"(?m)^\s*ARTICLE\s+\d+\s*$")
    matches = list(marker.finditer(text))

    if len(matches) < 2:
        if len(text) <= EVIDENCE_CHUNK_CHARS:
            return [text]
        return [
            text[i:i + EVIDENCE_CHUNK_CHARS].strip()
            for i in range(0, len(text), EVIDENCE_CHUNK_CHARS)
            if text[i:i + EVIDENCE_CHUNK_CHARS].strip()
        ]

    prefix = text[:matches[0].start()].strip()
    blocks = []

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[match.start():end].strip()
        if block:
            blocks.append(block)

    chunks = []
    current = []
    current_len = len(prefix)

    for block in blocks:
        if len(block) > EVIDENCE_CHUNK_CHARS:
            if current:
                chunks.append(
                    (prefix + "\n\n" if prefix else "")
                    + "\n\n".join(current)
                )
                current = []
                current_len = len(prefix)

            for i in range(0, len(block), EVIDENCE_CHUNK_CHARS):
                piece = block[i:i + EVIDENCE_CHUNK_CHARS].strip()
                if piece:
                    chunks.append(piece)
            continue

        extra = len(block) + (2 if current else 0)
        if current and current_len + extra > EVIDENCE_CHUNK_CHARS:
            chunks.append(
                (prefix + "\n\n" if prefix else "")
                + "\n\n".join(current)
            )
            current = [block]
            current_len = len(prefix) + len(block)
        else:
            current.append(block)
            current_len += extra

    if current:
        chunks.append(
            (prefix + "\n\n" if prefix else "")
            + "\n\n".join(current)
        )

    return chunks or [text]


# ============================================================
# Evidence extraction
# ============================================================

_EVIDENCE_FORMAT = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "f": {"type": "string"},
                    "x": {"type": "string"},
                },
                "required": ["f", "x"],
            },
        },
    },
    "required": ["facts"],
}


def _sentence_index_source(source):
    """
    Add deterministic sentence IDs to the source.

    The model selects IDs; Python constructs the evidence excerpt directly
    from the original source. This removes the fragile LLM-generated-excerpt
    provenance step that was causing repeated 'excerpt not found in source'
    rejections.
    """
    text = (source or "").strip()
    if not text:
        return "", {}

    # Keep article headers/metadata as part of sentences.
    # Split on sentence-ending punctuation followed by whitespace, while also
    # handling closing quotes/brackets before the whitespace. Do not require
    # the next character to be uppercase: this is important for multilingual
    # sources, headlines, abbreviations and quoted text.
    parts = re.split(
        r'(?<=[.!?])(?:["”»’\'\)\]]+)?\s+',
        text
    )
    parts = [p.strip() for p in parts if p.strip()]

    # If the source has long blocks without punctuation, still expose them.
    mapping = {}
    indexed = []
    for i, part in enumerate(parts, 1):
        sid = f"S{i}"
        mapping[sid] = part
        indexed.append(f"[{sid}] {part}")

    return "\n".join(indexed), mapping


def _evidence_prompt(source, max_facts=None):
    limit = max_facts or EVIDENCE_MAX_FACTS
    indexed_source, sentence_map = _sentence_index_source(source)

    valid_ids = list(sentence_map.keys())
    valid_id_text = ", ".join(valid_ids)

    return f"""
You are TrendCurrent's source-evidence extractor.

Read ONLY the SOURCE MATERIAL. Build evidence for ONE publishable news story.

PRIMARY-EVENT RULE:
- Identify ONE coherent main story/event in the supplied material.
- Return facts ONLY from that same story/event.
- If the material contains multiple ARTICLE blocks or separate news stories sharing a broad keyword, choose ONE story and ignore the others.
- Never combine two different events just because they concern the same person, keyword, programme, team, place or general subject.
- Prefer facts that belong to the same concrete event/article cluster.
- Extract as many distinct useful facts as the SAME story supports, up to {limit}.
- Target 4-6 facts when the SAME story contains enough material.
- Cover different factual dimensions when available: what happened, who/where, timing, numbers, status, and other directly relevant details.
- Do not stop after the first fact merely because it is sufficient to identify the story.
- Do not manufacture facts to reach a count.

FACT RULES:
- Each fact must be explicitly supported by one source sentence.
- "x" MUST be one of these VALID SENTENCE IDs: {valid_id_text}
- Never invent or alter a sentence ID.
- "f" must be a concise factual statement supported by that sentence.
- Do NOT generate excerpts, source names, dates or status fields separately.
- Do not infer motives, causes, consequences or significance.
- Do not use outside knowledge.
- Preserve names, roles, dates, numbers and certainty exactly.
- Never transfer attributes between named entities.
- Return ONLY JSON.

Use exactly this compact JSON shape:
{{"facts":[{{"f":"fact","x":"S1"}}]}}

SOURCE MATERIAL:
{indexed_source}
"""



def _evidence_expansion_prompt(source):
    indexed_source, sentence_map = _sentence_index_source(source)
    valid_id_text = ", ".join(sentence_map.keys())
    return f"""
Extract the MAIN EVENT from this source and build a compact evidence ledger.

Return ONLY JSON:
{{"facts":[{{"f":"fact","x":"S1"}}]}}

RULES:
- Extract 4-{EVIDENCE_MAX_FACTS} distinct facts if supported; use fewer only when the source truly contains fewer.
- ALL returned facts must belong to ONE coherent event/story.
- If several ARTICLE blocks or separate stories appear in the source, choose one main story and ignore unrelated stories that merely share a keyword.
- Do not combine separate programmes, broadcasts, people, matches, incidents or other events.
- Do not repeat the same fact in different wording.
- Cover different useful details: event, people/entities, timing, numbers, status, location or other directly relevant facts.
- Every fact must be explicitly supported by one sentence.
- "x" must be one of these VALID SENTENCE IDs: {valid_id_text}
- Never invent or alter an ID.
- No excerpts, source names, dates or status fields outside "f".
- No outside knowledge.
- Return ONLY JSON.

SOURCE:
{indexed_source}
"""



def _evidence_retry_prompt(source):
    indexed_source, sentence_map = _sentence_index_source(source)
    valid_id_text = ", ".join(sentence_map.keys())

    return f"""
Extract the MAIN EVENT from this source.

Return ONLY compact JSON:
{{"facts":[{{"f":"fact","x":"S1"}}]}}

RULES:
- Return as many distinct facts as the source supports, up to {EVIDENCE_MAX_FACTS}.
- Prefer 4-6 facts when the source contains enough material.
- ALL facts must belong to ONE coherent main event/story.
- If multiple ARTICLE blocks or separate stories share a keyword, choose one story only and do not mix them.
- Cover different useful factual dimensions instead of repeating the same point.
- All facts must belong to the same main event.
- "x" must be one of these VALID SENTENCE IDs: {valid_id_text}
- Never invent or alter a sentence ID.
- "f" must be a concise supported fact.
- Do not generate excerpts, source names, dates or status fields.
- Do not invent facts or use outside knowledge.

SOURCE:
{indexed_source}
"""



def _source_excerpt_supported(source, excerpt):
    source_norm = re.sub(r"\s+", " ", (source or "")).strip().casefold()
    excerpt_norm = re.sub(r"\s+", " ", (excerpt or "")).strip().casefold()
    if not source_norm or not excerpt_norm:
        return False
    return excerpt_norm in source_norm


def _normalize_evidence(data, source_material=None):
    if not isinstance(data, dict):
        raise ValueError("Evidence response is not an object.")

    raw_facts = data.get("facts", [])
    if not isinstance(raw_facts, list):
        raw_facts = []

    _, sentence_map = _sentence_index_source(source_material or "")
    clean = []
    seen = set()

    for item in raw_facts:
        if not isinstance(item, dict):
            continue

        fact = str(item.get("f", "")).strip()
        sentence_id = str(item.get("x", "")).strip()

        if not fact or not sentence_id:
            continue

        excerpt = sentence_map.get(sentence_id, "").strip()
        if not excerpt:
            print(
                f"[PIPELINE] Evidence sentence rejected: unknown source id {sentence_id}"
            )
            continue

        key = (fact.lower(), sentence_id.lower())
        if key in seen:
            continue
        seen.add(key)

        clean.append({
            "id": f"F{len(clean) + 1}",
            "group": "G1",
            "fact": fact,
            "excerpt": excerpt[:160],
            "source": "",
            "date": "",
            "status": "",
        })

        if len(clean) >= EVIDENCE_MAX_FACTS:
            break

    if not clean:
        raise ValueError("Evidence extraction produced no usable facts.")

    return {
        "primary_group": "G1",
        "facts": clean[:EVIDENCE_MAX_FACTS],
    }



def _extract_evidence(source):
    started = time.perf_counter()
    chunks = _split_source(source)

    print(
        f"[TIMER] Evidence extraction START | source_chars={len(source or '')} "
        f"| chunks={len(chunks)}"
    )

    maps = []

    for index, chunk in enumerate(chunks, 1):
        try:
            data = _call(
                _evidence_prompt(chunk),
                temperature=0.0,
                num_predict=EVIDENCE_TOKENS,
                num_thread=NUM_THREADS,
                response_format=_EVIDENCE_FORMAT,
            )
            maps.append(_normalize_evidence(data, source_material=chunk))
        except ValueError as exc:
            # Retry only malformed/truncated JSON. Provenance no longer depends
            # on a model-written excerpt, so a valid response is not thrown away
            # because of wording differences.
            if "Invalid Ollama JSON" not in str(exc):
                raise

            print(f"[PIPELINE] Evidence JSON retry | chunk={index}")
            data = _call(
                _evidence_retry_prompt(chunk),
                temperature=0.0,
                num_predict=EVIDENCE_RETRY_TOKENS,
                num_thread=NUM_THREADS,
                response_format=_EVIDENCE_FORMAT,
            )
            maps.append(_normalize_evidence(data, source_material=chunk))

    facts = []
    seen = set()

    # If extraction is suspiciously sparse despite a large source, do one compact
    # expansion pass over the same chunks. This preserves the compact JSON contract
    # while preventing a rich source from collapsing to a single fact.
    if len(maps) and sum(len(x.get("facts", [])) for x in maps) <= 1 and len(source or "") >= 7000:
        print("[PIPELINE] Evidence sparse | requesting compact fact expansion...")
        maps = []
        for index, chunk in enumerate(chunks, 1):
            data = _call(
                _evidence_expansion_prompt(chunk),
                temperature=0.0,
                num_predict=EVIDENCE_TOKENS,
                num_thread=NUM_THREADS,
                response_format=_EVIDENCE_FORMAT,
            )
            maps.append(_normalize_evidence(data, source_material=chunk))

    for chunk_no, data in enumerate(maps, 1):
        group = f"C{chunk_no}-{data['primary_group']}"

        for item in data["facts"]:
            fact = dict(item)
            fact["group"] = group

            key = (
                fact["fact"].lower(),
                fact["excerpt"].lower(),
            )
            if key in seen:
                continue

            seen.add(key)
            fact["id"] = f"F{len(facts) + 1}"
            facts.append(fact)

    if not facts:
        raise ValueError("Evidence extraction produced no usable facts.")

    # Keep the strongest evidence across all chunks instead of discarding
    # an entire chunk just because another chunk produced more facts.
    # Facts are already constrained to the same main event by the extractor.
    locked = facts[:EVIDENCE_MAX_FACTS]

    group_counts = {}
    for fact in locked:
        group_counts[fact["group"]] = group_counts.get(fact["group"], 0) + 1

    primary_group = max(
        group_counts,
        key=group_counts.get,
        default="C1-G1",
    )

    evidence = {
        "primary_group": primary_group,
        "facts": locked,
    }

    print(
        f"[PERF] Evidence ready | facts={len(evidence['facts'])} "
        f"| primary_group={primary_group}"
    )
    print(
        f"[TIMER] Evidence extraction TOTAL | "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )

    return evidence


# ============================================================
# Article schema
# ============================================================

def _schema_ok(article):
    if not isinstance(article, dict):
        return False

    required = ("title", "description", "h1", "intro", "sections")
    if any(key not in article for key in required):
        return False

    for key in ("title", "description", "h1", "intro"):
        if not isinstance(article[key], str):
            return False

    sections = article["sections"]
    if not isinstance(sections, list) or not 1 <= len(sections) <= MAX_SECTIONS:
        return False

    for section in sections:
        if not isinstance(section, dict):
            return False
        if not isinstance(section.get("title"), str):
            return False
        if not isinstance(section.get("text"), str):
            return False

    return True


def _word_count(article):
    values = [
        article.get("title", ""),
        article.get("description", ""),
        article.get("h1", ""),
        article.get("intro", ""),
    ]

    for section in article.get("sections", []):
        values.append(section.get("title", ""))
        values.append(section.get("text", ""))

    return len(" ".join(values).split())


# ============================================================
# Article generation
# ============================================================

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
            "maxItems": MAX_SECTIONS,
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
    "required": [
        "title",
        "description",
        "h1",
        "intro",
        "sections",
    ],
}


def _article_prompt(evidence):
    return f"""
Write a clear TrendCurrent news article in {LANGUAGE}.

Use ONLY the LOCKED EVIDENCE below.

BALANCED FACT RULES:
- Stay on the one locked event.
- Treat every LOCKED EVIDENCE fact as belonging to the same event; never merge it with another event merely because names, topics or keywords overlap.
- You may naturally paraphrase supported facts.
- You may combine facts when they clearly describe the same event.
- Do not add outside facts.
- Do not invent dates, numbers, roles, locations, causes or motives.
- ENTITY ATTRIBUTE LOCK: A person's role, position, job title or other identity-defining
  attribute may be stated only when that attribute is explicitly present in LOCKED EVIDENCE.
  If it is absent from LOCKED EVIDENCE, do not guess it. If LOCKED EVIDENCE explicitly
  gives a different attribute, never substitute another one.
- ENTITY-TO-ENTITY LOCK: Never transfer a role, organization type, action, status or relationship
  from one named entity to another. In particular, do not infer that a person or fictional character
  is a company/production house/organization because a nearby sentence names a production company.
- Keep each factual attribute attached only to the exact entity for which the evidence states it.
- Preserve uncertainty and status.
- Do not turn a report into a confirmed fact.
- Do not use unsupported quotes.
- Do not add generic filler or speculation.
- Titles, descriptions and headings must also stay factual.
- Let article length be determined by the amount and richness of the locked evidence.
- For limited evidence, keep the article concise and complete.
- For richer evidence, develop the article enough to cover the distinct relevant facts,
  developments and factual dimensions available in the locked evidence.
- Do not impose a fixed or target word count on richer stories.
- Stop when the relevant evidence has been adequately covered.
- Never add filler, repetition or unsupported detail to increase length.
- Use the available supporting facts naturally. If 3 or more locked facts are available, develop the story across the intro and sections so the article explains the distinct supported details rather than reducing the story to one or two sentences.
- For 3-4 facts, normally use at least 2 sections when the material supports it.
- For richer evidence, use as many sections as are genuinely useful for presenting distinct supported facts and developments clearly.
- Do not create sections merely to increase article length.
- If the evidence genuinely contains only one or two facts, a shorter article is acceptable.
Return ONLY the required JSON.

LOCKED EVIDENCE:
{_compact(evidence)}
"""


def _generate_article(evidence):
    article = _call(
        _article_prompt(evidence),
        temperature=0.04,
        num_predict=ARTICLE_TOKENS,
        num_thread=NUM_THREADS,
        response_format=_ARTICLE_FORMAT,
    )

    if not _schema_ok(article):
        raise ValueError("Article generator returned invalid schema.")

    return article


# ============================================================
# Audit
# ============================================================

_AUDIT_FORMAT = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "severity",
                    "claim",
                    "reason",
                    "evidence_ids",
                ],
            },
        },
    },
    "required": ["passed", "errors"],
}


def _audit_prompt(article, evidence):
    return f"""
You are the final factual quality check for a TrendCurrent article.

Compare the ARTICLE only with the LOCKED EVIDENCE.

Pass normal journalistic paraphrasing when it preserves the evidence.
Flag only material factual problems:
- unsupported fact
- wrong date or number
- wrong person/role/location
- wrong or contradictory entity attribute (including a person's role, position or job title)
- changed status or certainty
- unsupported quote/attribution
- unsupported causal claim
- mixing a separate event

Do NOT require identical wording.
Do NOT use outside knowledge.
Do NOT flag harmless wording differences.

Return JSON:
{{"passed":true,"errors":[]}}
or
{{"passed":false,"errors":[{{"severity":"","claim":"","reason":"","evidence_ids":["F1"]}}]}}

LOCKED EVIDENCE:
{_compact(evidence)}

ARTICLE:
{_compact(article)}
"""


def _audit(article, evidence):
    result = _call(
        _audit_prompt(article, evidence),
        temperature=0.0,
        num_predict=AUDIT_TOKENS,
        num_thread=NUM_THREADS,
        response_format=_AUDIT_FORMAT,
    )

    if not isinstance(result, dict):
        raise ValueError("Auditor returned invalid JSON.")

    passed = bool(result.get("passed", False))
    errors = result.get("errors", [])

    if not isinstance(errors, list):
        errors = []

    clean = []
    for error in errors:
        if not isinstance(error, dict):
            continue

        claim = str(error.get("claim", "")).strip()
        reason = str(error.get("reason", "")).strip()

        if not claim or not reason:
            continue

        ids = error.get("evidence_ids", [])
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list):
            ids = []

        clean.append({
            "severity": str(
                error.get("severity", "MAJOR")
            ).upper(),
            "claim": claim,
            "reason": reason,
            "evidence_ids": [
                str(x) for x in ids if str(x).strip()
            ],
        })

    return {
        "passed": passed and not clean,
        "errors": clean,
    }


# ============================================================
# Repair
# ============================================================

_REPAIR_FORMAT = _ARTICLE_FORMAT


def _repair(article, evidence, audit):
    return _call(
        f"""
Repair this article using ONLY the LOCKED EVIDENCE and AUDIT.

Rules:
- Fix only the listed factual problems.
- Delete unsupported material instead of inventing a replacement.
- If an entity attribute conflicts with LOCKED EVIDENCE, delete the incorrect attribute or
  replace it only with the exact supported attribute from LOCKED EVIDENCE.
- Preserve supported material.
- Do not add new facts.
- Do not change dates, numbers, roles, status or certainty except to correct a listed
  contradiction using the exact LOCKED EVIDENCE.
- Do not create quotes or attribution.
- Keep the article in {LANGUAGE}.
- Return only the article JSON.

AUDIT:
{_compact(audit)}

LOCKED EVIDENCE:
{_compact(evidence)}

ARTICLE:
{_compact(article)}
""",
        temperature=0.0,
        num_predict=REPAIR_TOKENS,
        num_thread=NUM_THREADS,
        response_format=_REPAIR_FORMAT,
    )


# ============================================================
# Final deterministic checks
# ============================================================

def _sanitize_article(article):
    if not _schema_ok(article):
        raise ValueError("Final article schema is invalid.")

    clean = {
        "title": article["title"].strip(),
        "description": article["description"].strip(),
        "h1": article["h1"].strip(),
        "intro": article["intro"].strip(),
        "sections": [],
    }

    for section in article["sections"]:
        title = section["title"].strip()
        text = section["text"].strip()

        if not title or not text:
            continue

        clean["sections"].append({
            "title": title,
            "text": text,
        })

    if not clean["sections"]:
        raise ValueError("Article has no usable sections.")

    return clean


def _audit_or_raise(article, evidence, label):
    audit = _audit(article, evidence)

    print(
        f"[PIPELINE] {label}: "
        f"{'PASS' if audit['passed'] else 'FAIL'}"
    )

    if not audit["passed"]:
        print(
            json.dumps(
                audit,
                ensure_ascii=False,
                indent=2,
            )
        )

    return audit


# ============================================================
# Public API
# ============================================================

def extract_evidence(source):
    """Public compatibility wrapper for the evidence extractor."""
    return _extract_evidence(source)


def _adaptive_min_words(evidence):
    count = len(evidence.get("facts", [])) if isinstance(evidence, dict) else 0
    if MIN_ARTICLE_WORDS > 0:
        return MIN_ARTICLE_WORDS
    if count <= 2:
        return ADAPTIVE_MIN_1_2
    if count <= 4:
        return ADAPTIVE_MIN_3_4
    return ADAPTIVE_MIN_5_PLUS


def generate(prompt, retries=0, evidence=None):
    """
    Balanced universal pipeline.

    Normal:
        evidence -> article -> audit

    If audit fails:
        evidence -> article -> audit -> one repair -> final audit

    There is deliberately no:
        - recursive repair
        - article regeneration loop
        - filler regeneration
        - web recovery
        - second competing auditor

    The supplied source remains the factual authority.
    """

    pipeline_start = time.perf_counter()

    print(
        f"[TIMER] PIPELINE START | prompt_chars={len(prompt or '')}"
    )

    if evidence is not None:
        print(
            f"[PIPELINE] Reusing preflight evidence lock | "
            f"facts={len(evidence.get("facts", []))}"
        )
    else:
        print("[PIPELINE] Building balanced evidence lock...")
        evidence = _extract_evidence(prompt)

    print("[PIPELINE] Generating evidence-locked article...")
    article = _generate_article(evidence)
    article = _sanitize_article(article)

    print(
        f"[PIPELINE] Article generated | "
        f"words={_word_count(article)}"
    )

    print("[PIPELINE] Focused factual audit...")
    audit = _audit_or_raise(
        article,
        evidence,
        "Initial audit",
    )

    if audit["passed"]:
        words = _word_count(article)
        minimum = _adaptive_min_words(evidence)
        if words < minimum:
            raise ValueError(
                f"Article is factually clean but too short "
                f"({words} words; minimum {minimum} for {len(evidence['facts'])} evidence facts)."
            )
        print("[PIPELINE] FACT CHECK PASSED")
        print(
            f"[TIMER] PIPELINE TOTAL | "
            f"elapsed={time.perf_counter() - pipeline_start:.2f}s"
        )
        return article

    print("[PIPELINE] Delete-first factual repair...")
    repair_started = time.perf_counter()

    repaired = _repair(
        article,
        evidence,
        audit,
    )

    print(
        f"[TIMER] Repair stage complete | "
        f"elapsed={time.perf_counter() - repair_started:.2f}s"
    )

    if not _schema_ok(repaired):
        raise ValueError("Repair returned invalid article schema.")

    repaired = _sanitize_article(repaired)

    print("[PIPELINE] Final focused audit...")
    final_audit = _audit_or_raise(
        repaired,
        evidence,
        "Final audit",
    )

    if not final_audit["passed"]:
        print("[UNIVERSAL FACT CHECK FAILED AFTER REPAIR]")
        raise ValueError(
            "Article failed final source-grounded validation."
        )

    words = _word_count(repaired)
    minimum = _adaptive_min_words(evidence)

    if words < minimum:
        raise ValueError(
            f"Article is factually clean but too short "
            f"({words} words; minimum {minimum} for {len(evidence['facts'])} evidence facts)."
        )

    print("[PIPELINE] FACT CHECK PASSED AFTER REPAIR")
    print(
        f"[TIMER] PIPELINE TOTAL | "
        f"elapsed={time.perf_counter() - pipeline_start:.2f}s"
    )

    return repaired
