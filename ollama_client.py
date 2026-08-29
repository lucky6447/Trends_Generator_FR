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

PIPELINE_VERSION = "universal-fact-lock-v2.3.3-compact-evidence-json-fixed3-source-id-retry"

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
    420, int(os.getenv("OLLAMA_EVIDENCE_TOKENS", "700"))
)
EVIDENCE_MAX_FACTS = max(
    4, int(os.getenv("OLLAMA_EVIDENCE_MAX_FACTS", "12"))
)

ARTICLE_TOKENS = max(
    520, int(os.getenv("OLLAMA_ARTICLE_TOKENS", "900"))
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
ADAPTIVE_MIN_1_2 = max(0, int(os.getenv("OLLAMA_MIN_1_2_FACTS", "0")))
ADAPTIVE_MIN_3_4 = max(0, int(os.getenv("OLLAMA_MIN_3_4_FACTS", "0")))
ADAPTIVE_MIN_5_PLUS = max(0, int(os.getenv("OLLAMA_MIN_5_PLUS_FACTS", "0")))

# Evidence is deliberately sequential on CPU.
EVIDENCE_PARALLEL = os.getenv(
    "OLLAMA_EVIDENCE_PARALLEL", "0"
).lower() not in {"0", "false", "no", "off"}

# Sparse evidence expansion is expensive on CPU because it re-runs Ollama
# over every evidence chunk. Keep it disabled by default; enable only for
# controlled benchmarking with OLLAMA_EVIDENCE_EXPANSION=1.
EVIDENCE_EXPANSION = os.getenv(
    "OLLAMA_EVIDENCE_EXPANSION", "0"
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
    Build deterministic source-aware sentence IDs.

    ARTICLE blocks are indexed independently (A1-S1, A1-S2, A2-S1...).
    Title/source/published metadata is kept visible for scope, but is NOT
    exposed as factual sentence evidence. Facts must come from Summary/Full
    Article text so a sensational or conflicting headline cannot override
    the body evidence.
    """
    text = (source or "").strip()
    if not text:
        return "", {}

    article_marker = re.compile(r"(?m)^\s*ARTICLE\s+(\d+)\s*$")
    matches = list(article_marker.finditer(text))

    def split_sentences(block):
        parts = re.split(
            r'(?<=[.!?])(?:["”»’\'\)\]]+)?\s+',
            block,
        )
        return [p.strip() for p in parts if p.strip()]

    mapping = {}
    indexed_parts = []

    if matches:
        prefix = text[:matches[0].start()].strip()
        if prefix:
            indexed_parts.append(prefix)

        for pos, match in enumerate(matches):
            article_no = match.group(1)
            block_end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
            block = text[match.start():block_end].strip()

            lines = block.splitlines()
            header = lines[0].strip() if lines else f"ARTICLE {article_no}"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            indexed_parts.append(f"[A{article_no}-HEADER] {header}")

            # Only expose factual content sections. Metadata is scope only.
            factual_lines = []
            in_fact = False
            for line in body.splitlines():
                stripped = line.strip()
                low = stripped.casefold()
                if low in {"summary:", "full article:"}:
                    in_fact = True
                    continue
                if stripped == "---":
                    in_fact = False
                    continue
                if in_fact and stripped:
                    factual_lines.append(stripped)

            factual_text = "\n".join(factual_lines).strip()
            if not factual_text:
                # Conservative fallback: if the source block has no explicit
                # section labels, use its body, excluding obvious metadata lines.
                kept = []
                for line in body.splitlines():
                    stripped = line.strip()
                    low = stripped.casefold()
                    if not stripped or low.startswith(("title:", "source:", "published:")):
                        continue
                    kept.append(stripped)
                factual_text = "\n".join(kept).strip()

            sentences = split_sentences(factual_text)
            for sentence_no, sentence in enumerate(sentences, 1):
                sid = f"A{article_no}-S{sentence_no}"
                mapping[sid] = sentence
                indexed_parts.append(f"[{sid}] {sentence}")
            indexed_parts.append("---")

        return "\n".join(indexed_parts), mapping

    # Single unstructured source: preserve legacy S1... IDs.
    sentences = split_sentences(text)
    for i, sentence in enumerate(sentences, 1):
        sid = f"S{i}"
        mapping[sid] = sentence
        indexed_parts.append(f"[{sid}] {sentence}")
    return "\n".join(indexed_parts), mapping

def _evidence_prompt(source, max_facts=None):
    limit = max_facts or EVIDENCE_MAX_FACTS
    indexed_source, sentence_map = _sentence_index_source(source)

    valid_ids = list(sentence_map.keys())
    valid_id_text = ", ".join(valid_ids)

    return f"""
You are TrendCurrent's source-evidence extractor.

Read ALL of the SOURCE MATERIAL before deciding which facts to return.

PRIMARY-EVENT LOCK:
- The supplied material has already been prefiltered to one coherent story cluster.
- Treat the cluster as ONE story unless a sentence is clearly unrelated.
- Do NOT reject the cluster merely because articles are repetitive versions of the same story.
- Do NOT switch to a different event just because another entity or topic appears in one article.
- Return facts from the SAME concrete event/story only.

EVIDENCE COVERAGE:
- Build the strongest possible evidence ledger from the SAME story.
- Extract EVERY distinct, directly supported, useful fact you can find, up to {limit}.
- When the source supports 8 or more distinct useful facts, you MUST return at least 8 facts.
- Do NOT stop at 4-6 facts when additional distinct useful facts are present.
- Do NOT stop after one fact.
- Do NOT stop after identifying the main event.
- Prefer facts covering different dimensions when available: event/action, people/entities, opponent/location, date/status, score/number, qualification/stage, and other concrete developments.
- Avoid duplicate facts that merely repeat the same point.
- If fewer than 4 distinct facts are genuinely supported by the entire cluster, return all supported facts and no invented facts.

FACT RULES:
- Every fact MUST be explicitly supported by one source sentence.
- "x" MUST be one of these VALID SENTENCE IDs: {valid_id_text}
- Never invent, alter, or guess a sentence ID.
- "f" must be a concise factual statement supported by that sentence.
- Do NOT generate excerpts, source names, dates or status fields separately.
- Do not infer motives, causes, consequences, significance, reputation, strength, expectations or likely outcomes.
- Do not use outside knowledge.
- Preserve names, roles, dates, numbers and certainty exactly.
- Never transfer attributes between named entities.
- Prefer a concrete source-supported fact over generic background wording.
- Do not select a numeric or other materially conflicting claim merely from a headline or metadata.
- When the factual bodies of sources conflict on a number, date, status or attribution and the conflict is not explicitly resolved, omit the disputed detail rather than choosing one by guesswork.
- Return ONLY JSON.

IMPORTANT OUTPUT REQUIREMENT:
- Before returning JSON, silently review the ENTIRE SOURCE MATERIAL for additional distinct supported facts.
- Do not return only the first or most obvious fact when additional supported facts are present.
- The target is 8-12 facts whenever the source material supports that many; this is a coverage target, not a requirement to invent or pad facts.

Use exactly this compact JSON shape:
{{"facts":[{{"f":"fact","x":"A1-S1"}}]}}

SOURCE MATERIAL:
{indexed_source}
"""



def _evidence_expansion_prompt(source):
    indexed_source, sentence_map = _sentence_index_source(source)
    valid_id_text = ", ".join(sentence_map.keys())
    return f"""
Extract the MAIN EVENT from this source and build a compact evidence ledger.

Return ONLY JSON:
{{"facts":[{{"f":"fact","x":"A1-S1"}}]}}

RULES:
- Extract 8-{EVIDENCE_MAX_FACTS} distinct facts when the source materially supports that many; use fewer only when the entire source genuinely contains fewer useful facts.
- ALL returned facts must belong to ONE coherent event/story.
- If several ARTICLE blocks or separate stories appear in the source, choose one main story and ignore unrelated stories that merely share a keyword.
- Do not combine separate programmes, broadcasts, people, matches, incidents or other events.
- Do not repeat the same fact in different wording.
- Cover different useful details: event, people/entities, timing, numbers, status, location or other directly relevant facts.
- Every fact must be explicitly supported by one sentence from one source block.
- When ARTICLE blocks are present, the provenance ID must identify both the source article and sentence (for example A2-S3).
- "x" must be one of these VALID SENTENCE IDs: {valid_id_text}
- Do not interpret SOURCE S2, ARTICLE 2, or any source label as a sentence ID.
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
{{"facts":[{{"f":"fact","x":"A1-S1"}}]}}

RULES:
- Return as many distinct facts as the source supports, up to {EVIDENCE_MAX_FACTS}.
- Prefer 8-{EVIDENCE_MAX_FACTS} facts when the source contains enough material; do not stop at 4-6 merely because that is sufficient to summarize the headline.
- ALL facts must belong to ONE coherent main event/story.
- If multiple ARTICLE blocks or separate stories share a keyword, choose one story only and do not mix them.
- Cover different useful factual dimensions instead of repeating the same point.
- All facts must belong to the same main event.
- "x" must be one of these VALID SENTENCE IDs: {valid_id_text}
- Do not interpret SOURCE S2, ARTICLE 2, or any source label as a sentence ID.
- Never invent or alter a sentence ID.
- "f" must be a concise supported fact.
- Do not generate excerpts, source names, dates or status fields.
- Do not invent facts or use outside knowledge.

SOURCE:
{indexed_source}
"""



def _evidence_invalid_id_retry_prompt(source, invalid_ids):
    """
    Retry evidence extraction when Ollama returns a provenance ID that does not
    exist in the deterministic sentence map.

    This is a narrow recovery path for model ID hallucination. It does not
    reinterpret or remap an invalid ID to another sentence, because doing so
    could attach a correct fact to the wrong source evidence.
    """
    indexed_source, sentence_map = _sentence_index_source(source)
    valid_id_text = ", ".join(sentence_map.keys())
    invalid_id_text = ", ".join(sorted(set(invalid_ids)))

    return f"""
You are retrying TrendCurrent's source-evidence extraction because the previous
response used invalid provenance IDs: {invalid_id_text}.

Return ONLY compact JSON:
{{"facts":[{{"f":"fact","x":"A1-S1"}}]}}

STRICT PROVENANCE RULES:
- Read the ENTIRE SOURCE MATERIAL again.
- Every returned fact MUST be explicitly supported by one source sentence.
- "x" MUST be one of these exact VALID SENTENCE IDs: {valid_id_text}
- SOURCE S2 / ARTICLE 2 are source labels, not sentence IDs.
- NEVER invent an ID.
- NEVER reuse an ID from memory or from a previous response.
- NEVER change an ID's number or format.
- If a fact cannot be tied confidently to one of the valid IDs, omit that fact.
- Return as many distinct supported facts as possible, up to {EVIDENCE_MAX_FACTS}.
- Keep all facts within the same main event/story.
- Do not infer motives, causes, significance, outcomes or outside facts.
- Do not generate excerpts, source names, dates or status fields separately.

SOURCE MATERIAL:
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
    invalid_ids = []

    for item in raw_facts:
        if not isinstance(item, dict):
            continue

        fact = str(item.get("f", "")).strip()
        sentence_id = str(item.get("x", "")).strip()

        if not fact or not sentence_id:
            continue

        excerpt = sentence_map.get(sentence_id, "").strip()
        if not excerpt:
            invalid_ids.append(sentence_id)
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

    # Never silently remap an invalid provenance ID. A wrong remap could make
    # an otherwise correct fact appear source-supported by the wrong sentence.
    if invalid_ids:
        ids = ", ".join(sorted(set(invalid_ids)))
        raise ValueError(f"Evidence returned unknown source ids: {ids}")

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
            message = str(exc)

            # Narrow retry for malformed/truncated JSON.
            if "Invalid Ollama JSON" in message:
                print(f"[PIPELINE] Evidence JSON retry | chunk={index}")
                data = _call(
                    _evidence_retry_prompt(chunk),
                    temperature=0.0,
                    num_predict=EVIDENCE_RETRY_TOKENS,
                    num_thread=NUM_THREADS,
                    response_format=_EVIDENCE_FORMAT,
                )
                maps.append(_normalize_evidence(data, source_material=chunk))
                continue

            # Narrow retry for model-generated provenance IDs that do not exist
            # in the deterministic sentence map. Do NOT silently remap IDs.
            if "Evidence returned unknown source ids:" in message:
                invalid_ids = [
                    item.strip()
                    for item in message.split(":", 1)[1].split(",")
                    if item.strip()
                ]
                print(
                    f"[PIPELINE] Evidence provenance retry | chunk={index} "
                    f"| invalid_ids={','.join(invalid_ids)}"
                )
                data = _call(
                    _evidence_invalid_id_retry_prompt(chunk, invalid_ids),
                    temperature=0.0,
                    num_predict=EVIDENCE_RETRY_TOKENS,
                    num_thread=NUM_THREADS,
                    response_format=_EVIDENCE_FORMAT,
                )
                maps.append(_normalize_evidence(data, source_material=chunk))
                continue

            raise

    facts = []
    seen = set()

    # If extraction is suspiciously sparse despite a large source, do one compact
    # expansion pass over the same chunks. This preserves the compact JSON contract
    # while preventing a rich source from collapsing to too few facts.
    initial_fact_count = sum(len(x.get("facts", [])) for x in maps)
    if (
        EVIDENCE_EXPANSION
        and len(maps)
        and initial_fact_count <= 2
        and len(source or "") >= 7000
    ):
        print(
            f"[PIPELINE] Evidence sparse | initial_facts={initial_fact_count} "
            f"| source_chars={len(source or '')} | requesting compact fact expansion..."
        )
        expanded_maps = []
        for index, chunk in enumerate(chunks, 1):
            data = _call(
                _evidence_expansion_prompt(chunk),
                temperature=0.0,
                num_predict=EVIDENCE_TOKENS,
                num_thread=NUM_THREADS,
                response_format=_EVIDENCE_FORMAT,
            )
            expanded_maps.append(_normalize_evidence(data, source_material=chunk))

        # Preserve the initial extraction and add any new valid facts from the
        # expansion pass. The existing deduplication/limit logic below remains
        # the single final lock mechanism.
        maps.extend(expanded_maps)

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

    # Keep all provenance-verified facts across all chunks. Each chunk is already
    # bounded by EVIDENCE_MAX_FACTS, so this preserves coverage without a second
    # global truncation that could silently discard relevant evidence.
    locked = facts

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

    required = ("title", "description", "h1", "paragraphs")
    if any(key not in article for key in required):
        return False

    for key in ("title", "description", "h1"):
        if not isinstance(article[key], str):
            return False

    paragraphs = article["paragraphs"]
    if not isinstance(paragraphs, list) or not paragraphs:
        return False

    for paragraph in paragraphs:
        if not isinstance(paragraph, str):
            return False
        if not paragraph.strip():
            return False

    return True


def _word_count(article):
    values = [
        article.get("title", ""),
        article.get("description", ""),
        article.get("h1", ""),
    ]

    for paragraph in article.get("paragraphs", []):
        values.append(paragraph)

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
        "paragraphs": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
    },
    "required": [
        "title",
        "description",
        "h1",
        "paragraphs",
    ],
}



def _article_prompt(evidence):
    return f"""
Write a clear TrendCurrent news article in {LANGUAGE}.

LANGUAGE LOCK:
- The output language is {LANGUAGE}.
- TITLE, DESCRIPTION, H1 and EVERY paragraph MUST be written in {LANGUAGE}.
- The LOCKED EVIDENCE may be in another language. Translate/paraphrase only supported facts.
- Proper names and official names may remain in their original form.

SOURCE-LOCKED FACTUAL RULES:
- Use ONLY the LOCKED EVIDENCE below. It is the complete and closed factual universe.
- Do not use outside knowledge, memory, source headlines, publication metadata or topic wording as evidence.
- Every material sentence must be directly supported by the locked evidence. If a detail is not explicit, OMIT it.
- Never invent, infer, strengthen or embellish facts.

ENTITY AND ATTRIBUTION LOCK:
- Never infer or transfer a person's role, title, position, employer, nationality, relationship or responsibility.
- Keep every role, action, status and relationship attached only to the exact entity supported by evidence.
- Never transfer an action from an organization to an individual or from one individual to another.
- Never add a location unless the evidence explicitly establishes it for that exact event.
- Never add an activity or characterization such as romantic, controversial, major or historic unless explicitly supported.
- Attribute announcements, decisions, prices, actions and statements only to the exact person or organization named in the evidence.

EVENT AND TIME LOCK:
- Preserve the exact event status: scheduled != completed; announced != implemented; proposed/intended/predicted != completed; reported != confirmed.
- Never describe a historical event as current or recent unless the evidence explicitly supports that status.
- Publication/update date is NOT the event date. Never infer event date, weekday, timing or recency from publication metadata.
- If the evidence does not establish an exact event date, do not invent one.
- If multiple events from different dates are present, preserve chronology and do not merge them into one current event.
- Do not use recently, today, currently, this week or latest unless explicitly supported by evidence.

NUMBERS AND CLAIM STRENGTH:
- Preserve names, numbers, prices, dates, scores, percentages and certainty exactly.
- Never calculate or derive a new factual number.
- Never upgrade a weaker claim into a stronger claim.
- If evidence conflicts, do not guess or reconcile it; use only uncontested information or state the material conflict.

COVERAGE:
- Use the distinct, relevant verified facts available in the locked evidence.
- Do not stop after only the headline-level fact when additional relevant evidence exists.
- Prefer another distinct verified fact over repeating an existing one.
- Every paragraph must add a distinct supported fact or development.
- Do not pad, speculate, manufacture context or repeat facts to increase length.
- Article length follows the amount of useful verified evidence.

STYLE:
- Natural, fluent {LANGUAGE}; professional, clear, objective and precise.
- No clickbait, speculation, filler or unsupported conclusions.
- Do not mention publisher/source names unless attribution itself is an essential verified fact.
- Write ONE coherent article about ONE concrete story.

HEADLINE:
- Maximum 10 words AND 65 characters.
- TITLE and H1 must be identical.
- Use only the core verified entity and core verified development.
- Do not add facts not present in locked evidence.

FINAL ENTITLEMENT CHECK:
Before returning JSON, silently check every sentence: identify the exact locked fact supporting it; verify every person, role, action, location and attribution; verify event date/status separately from publication date; and remove anything unsupported, inferred, stronger, newer or more specific than the evidence.

Return ONLY the required JSON.

LOCKED EVIDENCE:
{_compact(evidence)}
"""


def _generate_article(evidence):
    fact_count = len(evidence.get("facts", [])) if isinstance(evidence, dict) else 0
    # Give richer evidence a larger output budget without imposing a target length.
    dynamic_tokens = max(ARTICLE_TOKENS, min(1400, 420 + fact_count * 80))
    article = _call(
        _article_prompt(evidence),
        temperature=0.04,
        num_predict=dynamic_tokens,
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
- materially incomplete coverage: a relevant, non-redundant locked fact is omitted

COVERAGE RULE:
- Treat the LOCKED EVIDENCE facts as the complete verified evidence inventory for this article.
- The article does not need to repeat every fact verbatim, but every materially relevant, non-redundant fact should be represented in the article.
- If a locked fact is genuinely relevant to the selected event and is absent from the article, report it as a HIGH error with type "omitted_relevant_fact".
- Do not flag a fact that is redundant with another covered fact or not useful to the reader.

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
- For omitted_relevant_fact issues, add the missing fact using ONLY the cited LOCKED EVIDENCE.
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
        "paragraphs": [],
    }

    for paragraph in article["paragraphs"]:
        text = paragraph.strip()
        if text:
            clean["paragraphs"].append(text)

    if not clean["paragraphs"]:
        raise ValueError("Article has no usable paragraphs.")

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
