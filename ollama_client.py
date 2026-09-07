import json
import os
import re
import time
import difflib
from ollama import chat
from config import MODEL, LANGUAGE
import generator_monitor as monitor


# ============================================================
# TrendCurrent UNIVERSAL FACT-LOCK PIPELINE
# Balanced rewrite
#
# SOURCE
#   -> compact evidence extraction
#   -> primary-event lock
#   -> article generation
#   -> production Fact Guard (owned by generate.py)
#   -> deterministic repetition/language gates
#
# Goals:
#   * source-grounded without being needlessly rigid
#   * compact Ollama output so CPU inference does not run for minutes
#   * no cross-event article construction
#   * language-independent
# ============================================================

PIPELINE_VERSION = "universal-fact-lock-v2.8.1-coverage-first-no-repair-fail-closed-discovery-evidence-optimized"

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
    7000, int(os.getenv("OLLAMA_EVIDENCE_CHUNK_CHARS", "18000"))
)
EVIDENCE_TOKENS = max(
    420, int(os.getenv("OLLAMA_EVIDENCE_TOKENS", "700"))
)
EVIDENCE_MAX_FACTS = max(
    4, min(12, int(os.getenv("OLLAMA_EVIDENCE_MAX_FACTS", "12")))
)

# Only a small set of facts are mandatory for article coverage. The remaining
# verified facts stay available as supporting evidence but do not become a
# checklist the writer must mechanically reproduce.
CORE_FACTS_MAX = max(1, min(6, int(os.getenv("OLLAMA_CORE_FACTS_MAX", "6"))))

# Controlled writer A/B test: optionally expose bounded source context to the
# writer while keeping LOCKED EVIDENCE as the only factual authority.
WRITER_SOURCE_CONTEXT = os.getenv("OLLAMA_WRITER_SOURCE_CONTEXT", "0").strip() == "1"
WRITER_SOURCE_CONTEXT_CHARS = max(4000, int(os.getenv("OLLAMA_WRITER_SOURCE_CONTEXT_CHARS", "12000")))

ARTICLE_TOKENS = max(
    700, int(os.getenv("OLLAMA_ARTICLE_TOKENS", "1200"))
)
AUDIT_TOKENS = max(
    120, int(os.getenv("OLLAMA_AUDIT_TOKENS", "180"))
)
# No deterministic article-length floor.
# A factual, concise article must not be rejected merely because it is short.

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
    stage=None,
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

    if stage:
        try:
            monitor.ollama_timing(
                stage,
                elapsed_seconds=round(elapsed, 3),
                load_duration=timing.get("load_duration"),
                prompt_eval_duration=timing.get("prompt_eval_duration"),
                eval_duration=timing.get("eval_duration"),
                total_duration=timing.get("total_duration"),
                prompt_eval_count=timing.get("prompt_eval_count"),
                eval_count=timing.get("eval_count"),
                prompt_tok_s=prompt_tok_s,
                eval_tok_s=tok_s,
                num_predict=num_predict,
                prompt_chars=len(prompt),
                threads=threads,
                batch=batch,
            )
        except Exception:
            # Telemetry must never affect generation.
            pass

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
    Preserve ARTICLE blocks when present. If no ARTICLE markers exist,
    split only when necessary.

    For a moderately large source that still fits safely inside the configured
    context, keep it as ONE evidence chunk. This avoids an unnecessary second
    Ollama inference for payloads just above the legacy 14k boundary while
    preserving every source character and every provenance-bearing sentence.
    """
    text = (source or "").strip()
    if not text:
        return [""]

    # The normal chunk size remains unchanged. A single larger chunk is allowed
    # only for payloads that stay below a conservative 18k-character ceiling.
    # This is a performance optimization, not a content reduction.
    effective_chunk_chars = EVIDENCE_CHUNK_CHARS
    if len(text) <= 18000:
        effective_chunk_chars = len(text)


    marker = re.compile(r"(?m)^\s*ARTICLE\s+\d+\s*$")
    matches = list(marker.finditer(text))

    if len(matches) < 2:
        if len(text) <= effective_chunk_chars:
            return [text]
        return [
            text[i:i + effective_chunk_chars].strip()
            for i in range(0, len(text), effective_chunk_chars)
            if text[i:i + effective_chunk_chars].strip()
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
        if len(block) > effective_chunk_chars:
            if current:
                chunks.append(
                    (prefix + "\n\n" if prefix else "")
                    + "\n\n".join(current)
                )
                current = []
                current_len = len(prefix)

            for i in range(0, len(block), effective_chunk_chars):
                piece = block[i:i + effective_chunk_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        extra = len(block) + (2 if current else 0)
        if current and current_len + extra > effective_chunk_chars:
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
- Extract distinct, directly supported, useful facts from the SAME story, up to {limit}.
- Prefer the strongest/core facts first: the main development, key entities/actions, event status/time, and other facts needed to understand the story.
- Additional useful details may follow as supporting facts.
- Do not manufacture facts merely to reach a count.
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
- RESULT / SCORE ATTRIBUTION — HARD LOCK:
- A score or result must NEVER be used to infer which team, player or side won.
- Never assume that the first number belongs to the first named team, or that the higher
  number belongs to the first named team.
- Never infer winner, loser, winning side or "in favor of" attribution from score ordering,
  team ordering, sentence position, headline wording, convention or outside knowledge.
- State the score itself only when the source explicitly supports that score.
- State which side won or lost only when the source sentence explicitly establishes that
  result or explicitly links the result to the named side.
- If the score is explicit but the winner attribution is not explicit, keep the score as a
  score-only fact and omit the winner/loser attribution.
- If winner, loser or result attribution is ambiguous or cannot be established directly
  from the source sentence, omit that attribution rather than guessing.
- Return ONLY JSON.

IMPORTANT OUTPUT REQUIREMENT:
- Before returning JSON, silently review the ENTIRE SOURCE MATERIAL for additional distinct supported facts.
- Do not return only the first or most obvious fact when additional supported facts are present.
- Return as many distinct useful facts as the source genuinely supports, but keep the strongest/core facts first. The writer will require only the strongest core facts and may use the rest as supporting evidence.

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
- Extract the strongest distinct facts the source genuinely supports, up to {EVIDENCE_MAX_FACTS}.
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
- RESULT / SCORE ATTRIBUTION — HARD LOCK:
- A score or result must NEVER be used to infer which team, player or side won.
- Never assume that the first number belongs to the first named team, or that the higher
  number belongs to the first named team.
- Never infer winner, loser, winning side or "in favor of" attribution from score ordering,
  team ordering, sentence position, headline wording, convention or outside knowledge.
- State the score itself only when the source explicitly supports that score.
- State which side won or lost only when the source sentence explicitly establishes that
  result or explicitly links the result to the named side.
- If the score is explicit but the winner attribution is not explicit, keep the score as a
  score-only fact and omit the winner/loser attribution.
- If winner, loser or result attribution is ambiguous or cannot be established directly
  from the source sentence, omit that attribution rather than guessing.
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
- Prefer broad factual coverage when the source supports it, but never pad the ledger just to reach a count.
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
- RESULT / SCORE ATTRIBUTION — HARD LOCK:
- A score or result must NEVER be used to infer which team, player or side won.
- Never assume that the first number belongs to the first named team, or that the higher
  number belongs to the first named team.
- Never infer winner, loser, winning side or "in favor of" attribution from score ordering,
  team ordering, sentence position, headline wording, convention or outside knowledge.
- State the score itself only when the source explicitly supports that score.
- State which side won or lost only when the source sentence explicitly establishes that
  result or explicitly links the result to the named side.
- If the score is explicit but the winner attribution is not explicit, keep the score as a
  score-only fact and omit the winner/loser attribution.
- If winner, loser or result attribution is ambiguous or cannot be established directly
  from the source sentence, omit that attribution rather than guessing.

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
- RESULT / SCORE ATTRIBUTION — HARD LOCK:
- A score or result must NEVER be used to infer which team, player or side won.
- Never assume that the first number belongs to the first named team, or that the higher
  number belongs to the first named team.
- Never infer winner, loser, winning side or "in favor of" attribution from score ordering,
  team ordering, sentence position, headline wording, convention or outside knowledge.
- State the score itself only when the source explicitly supports that score.
- State which side won or lost only when the source sentence explicitly establishes that
  result or explicitly links the result to the named side.
- If the score is explicit but the winner attribution is not explicit, keep the score as a
  score-only fact and omit the winner/loser attribution.
- If winner, loser or result attribution is ambiguous or cannot be established directly
  from the source sentence, omit that attribution rather than guessing.

SOURCE MATERIAL:
{indexed_source}
"""



def _fact_tokens(text):
    """Normalize fact text into conservative lexical tokens for lineage checks."""
    text = re.sub(r"[^\w\s]", " ", (text or "").casefold(), flags=re.UNICODE)
    return [t for t in re.split(r"\s+", text) if len(t) > 1]


def _fact_pair_similarity(a, b):
    """Return conservative lexical similarity signals for two extracted facts."""
    ta = _fact_tokens(a)
    tb = _fact_tokens(b)
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / max(1, len(sa | sb))
    sequence = difflib.SequenceMatcher(None, a.casefold(), b.casefold()).ratio()
    containment = (a.casefold() in b.casefold()) or (b.casefold() in a.casefold())
    return jaccard, sequence, containment


def _fact_lineage_candidates(facts):
    """Build only conservative candidate pairs; do not merge on weak overlap."""
    candidates = []
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            a = str(facts[i].get("fact", "")).strip()
            b = str(facts[j].get("fact", "")).strip()
            if not a or not b:
                continue
            jaccard, sequence, containment = _fact_pair_similarity(a, b)
            shared_tokens = len(set(ta := _fact_tokens(a)) & set(tb := _fact_tokens(b)))
            # Candidate generation is intentionally broader than the merge rule.
            # The semantic judge decides whether shared wording is actually the same
            # information unit; weak pairs are excluded to keep CPU/LLM cost bounded.
            if (jaccard >= 0.25 and shared_tokens >= 2) or sequence >= 0.55 or (containment and jaccard >= 0.50):
                candidates.append({"i": i, "j": j, "jaccard": round(jaccard, 3), "sequence": round(sequence, 3), "containment": containment})
    return candidates


_FACT_LINEAGE_FORMAT = {
    "type": "object",
    "properties": {
        "merge_pairs": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
        }
    },
    "required": ["merge_pairs"],
}


def _fact_lineage_prompt(facts, candidates):
    lines = []
    for idx, fact in enumerate(facts):
        lines.append(f"F{idx + 1}: {fact.get('fact', '')}")
    candidate_text = ", ".join(f"F{c['i'] + 1}/F{c['j'] + 1}" for c in candidates)
    return f"""
You are TrendCurrent's fact-lineage judge.

Determine which candidate fact pairs express the SAME underlying information unit.
This is NOT an event/story similarity task. Two facts may concern the same event but
must remain separate when the second adds a distinct verifiable detail.

MERGE only when:
- the second fact is a paraphrase, restatement, or narrower wording of the same core claim;
- no materially new person, action, number, date, location, status, rule, result, or other
  independently useful detail is introduced.

DO NOT MERGE when the facts describe different dimensions of the event, even if they share
most names or wording. For example, a missed penalty and the rule violation that caused
the retake are separate information units.

Candidate pairs: {candidate_text or 'none'}

ALL FACTS:
{chr(10).join(lines)}

Return ONLY JSON:
{{"merge_pairs":[[0,1]]}}
Use zero-based fact indexes. Return an empty list when no candidate pair should merge.
"""


def _deduplicate_evidence_facts(facts):
    """Collapse repeated evidence claims while preserving genuinely new details."""
    if len(facts) < 2:
        return facts, {"raw_facts": len(facts), "unique_information_units": len(facts), "merged_facts": 0, "merge_pairs": []}

    candidates = _fact_lineage_candidates(facts)
    accepted_pairs = []
    if candidates:
        try:
            data = _call(
                _fact_lineage_prompt(facts, candidates),
                temperature=0.0,
                num_predict=220,
                num_thread=NUM_THREADS,
                response_format=_FACT_LINEAGE_FORMAT,
            )
            raw_pairs = data.get("merge_pairs", []) if isinstance(data, dict) else []
            candidate_set = {(c["i"], c["j"]) for c in candidates}
            for pair in raw_pairs:
                if not isinstance(pair, list) or len(pair) != 2:
                    continue
                try:
                    i, j = int(pair[0]), int(pair[1])
                except (TypeError, ValueError):
                    continue
                if i > j:
                    i, j = j, i
                if (i, j) in candidate_set and (i, j) not in accepted_pairs:
                    accepted_pairs.append((i, j))
        except Exception as exc:
            # Fail open: lineage is an evidence-quality enhancement, never a reason
            # to discard otherwise provenance-valid evidence when the judge is unavailable.
            print(f"[FACT LINEAGE] judge unavailable | keeping extracted facts | error={exc}")

    parent = list(range(len(facts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in accepted_pairs:
        union(i, j)

    families = {}
    for idx in range(len(facts)):
        families.setdefault(find(idx), []).append(idx)

    merged = []
    lineage_pairs = []
    for members in families.values():
        representative = dict(facts[members[0]])
        if len(members) > 1:
            representative["lineage_members"] = [facts[i]["id"] for i in members]
            representative["lineage_source_excerpts"] = [facts[i].get("excerpt", "") for i in members if facts[i].get("excerpt")]
            for i in members[1:]:
                lineage_pairs.append([facts[members[0]]["id"], facts[i]["id"]])
        merged.append(representative)

    # Re-number locked facts after clustering; provenance is retained in lineage_members.
    for idx, fact in enumerate(merged, 1):
        fact["id"] = f"F{idx}"

    stats = {
        "raw_facts": len(facts),
        "unique_information_units": len(merged),
        "merged_facts": len(facts) - len(merged),
        "merge_pairs": lineage_pairs,
        "candidate_pairs": len(candidates),
    }
    return merged, stats



# ============================================================
# SUBSTANTIVE STORY VALUE GATE
# ============================================================
# This is a pre-generation editorial-value gate.
#
# Purpose:
#   Reject candidates that are technically factual and source-grounded but do
#   not contain a concrete news development or a useful reader takeaway.
#
# This is deliberately NOT:
#   - a word-count gate
#   - a source-count gate
#   - a fact-count gate
#   - a popularity/virality gate
##
# A small story can PASS when the evidence contains a concrete development.
# A large evidence set can FAIL when it is mostly generic commentary, "interest"
# statements, tipster/promotional framing, recycled context, or other content
# that gives the reader little substantive news.
SUBSTANTIVE_VALUE_TOKENS = max(
    120, int(os.getenv("OLLAMA_SUBSTANTIVE_VALUE_TOKENS", "180"))
)
SUBSTANTIVE_VALUE_MIN_CONFIDENCE = max(
    70, min(100, int(os.getenv("OLLAMA_SUBSTANTIVE_VALUE_MIN_CONFIDENCE", "80")))
)

_SUBSTANTIVE_VALUE_FORMAT = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["PASS", "REJECT"],
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "concrete_development": {"type": "boolean"},
        "reader_value": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "verdict",
        "confidence",
        "concrete_development",
        "reader_value",
        "reason",
    ],
}


def _substantive_value_prompt(evidence):
    facts = evidence.get("facts", []) if isinstance(evidence, dict) else []
    lines = []

    for idx, item in enumerate(facts[:EVIDENCE_MAX_FACTS], 1):
        if not isinstance(item, dict):
            continue

        fact = str(item.get("fact", "")).strip()
        excerpt = str(item.get("excerpt", "")).strip()

        if not fact:
            continue

        # The excerpt is included as supporting context so the judge can
        # distinguish concrete reporting from generic paraphrase. It remains
        # evidence only; it is never used to add outside knowledge.
        if excerpt:
            excerpt = re.sub(r"\s+", " ", excerpt)[:420]
            lines.append(f"F{idx}: {fact}\nSOURCE SENTENCE: {excerpt}")
        else:
            lines.append(f"F{idx}: {fact}")

    facts_text = "\n\n".join(lines) or "No locked evidence facts available."

    return f"""
You are TrendCurrent's pre-generation substantive story value editor.

Your job is NOT to judge writing quality, popularity, SEO, article length, or
whether the topic is globally important.

Your job is to decide whether the LOCKED EVIDENCE contains enough substantive
news value to justify spending production capacity on a standalone news article.

Return exactly one verdict:
- PASS = the evidence contains a concrete news development and gives a reader
  a useful factual takeaway.
- REJECT = the evidence is technically factual but substantively empty, generic,
  meta-level, promotional, repetitive, or lacks a concrete development.

IMPORTANT DISTINCTION:
A small or niche story may PASS. It does NOT need to be a major national or
global event. A single concrete decision, result, appointment, incident,
announcement, legal development, measurable change, discovery, death, or other
specific development can be enough when the evidence clearly tells the reader
what actually happened.

REJECT examples:
- "X is attracting attention."
- "Fans are interested in X."
- "A tipster discussed X's prospects."
- "X is considered a contender."
- "An expert gave insights" when the actual useful conclusion/details are absent.
- Generic background or statements about why a person/event is important.
- Rephrasing that an article/source discussed a topic without a concrete new
  development.
- Evidence that mostly says that something is being discussed, watched,
  expected, or talked about, without establishing what actually happened or
  changed.

PASS examples:
- A concrete result, decision, ruling, appointment, announcement, incident,
  policy change, discovery, transaction, measurable development, or confirmed
  event with enough factual detail for a reader to understand the development.
- A niche/local story with a real event or change, even if it is not widely
  significant.

RULES:
1. Judge ONLY the locked evidence below. Do not use outside knowledge.
2. Do not reject merely because the story is niche, short, local, cultural,
   entertainment-related, business-related, or otherwise not a major headline.
3. Do not require a minimum number of facts or sources.
4. Do not require a particular word count.
5. Do not confuse "same story" with "valuable story"; this gate assumes the
   evidence has already passed story concentration.
6. A factual opinion/analysis story may PASS only when the evidence contains a
   concrete underlying development that makes the analysis newsworthy.
7. If the evidence explicitly lacks the actual details of the supposed
   development and mainly describes interest, expectations, commentary, or
   coverage itself, REJECT it.
8. Do not infer significance, motives, causality, future outcomes, or importance
   that is not present in the evidence.
9. Be conservative about empty content, but do not impose a "big news only"
   standard.
10. Confidence must reflect how clearly the evidence supports the decision.

Return ONLY this JSON shape:
{{
  "verdict":"PASS",
  "confidence":95,
  "concrete_development":true,
  "reader_value":true,
  "reason":"brief evidence-grounded reason"
}}

LOCKED EVIDENCE:
{facts_text}
"""


def substantive_story_value_gate(evidence):
    """
    Fail closed before article generation when locked evidence lacks
    substantive news value.

    Returns the normalized decision on PASS. Raises ValueError on REJECT or
    unavailable/invalid judge output so the candidate never reaches generation.
    """
    if not isinstance(evidence, dict):
        raise ValueError("Substantive Story Value Gate requires an evidence object.")

    facts = evidence.get("facts", [])
    if not isinstance(facts, list) or not facts:
        raise ValueError("Substantive Story Value Gate rejected empty evidence.")

    started = time.perf_counter()

    try:
        result = _call(
            _substantive_value_prompt(evidence),
            temperature=0.0,
            num_predict=SUBSTANTIVE_VALUE_TOKENS,
            num_thread=NUM_THREADS,
            response_format=_SUBSTANTIVE_VALUE_FORMAT,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(
            f"[SUBSTANTIVE STORY VALUE] UNAVAILABLE | "
            f"elapsed={elapsed:.2f}s | error={exc}"
        )
        raise ValueError(
            f"Substantive Story Value Gate unavailable: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("Substantive Story Value Gate returned invalid JSON.")

    verdict = str(result.get("verdict", "")).strip().upper()
    try:
        confidence = max(0, min(100, int(result.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0

    concrete_development = bool(result.get("concrete_development", False))
    reader_value = bool(result.get("reader_value", False))
    reason = str(result.get("reason", "")).strip()[:500]

    # Fail closed: a PASS is only valid when the judge explicitly identifies
    # both a concrete development and a useful reader takeaway with adequate
    # confidence. This prevents vague low-confidence approvals.
    passed = (
        verdict == "PASS"
        and confidence >= SUBSTANTIVE_VALUE_MIN_CONFIDENCE
        and concrete_development
        and reader_value
    )

    if passed:
        print(
            f"[SUBSTANTIVE STORY VALUE] PASS | "
            f"confidence={confidence} | concrete_development=true | "
            f"reader_value=true | {reason}"
        )
        return {
            "verdict": "PASS",
            "confidence": confidence,
            "concrete_development": True,
            "reader_value": True,
            "reason": reason,
        }

    if verdict == "REJECT":
        rejection_reason = reason or "evidence lacks substantive news value"
    elif verdict == "PASS":
        rejection_reason = (
            reason
            or "judge did not establish both a concrete development and useful reader value"
        )
    else:
        rejection_reason = f"invalid substantive value verdict: {verdict or 'empty'}"

    print(
        f"[SUBSTANTIVE STORY VALUE] REJECT | "
        f"confidence={confidence} | concrete_development={str(concrete_development).lower()} | "
        f"reader_value={str(reader_value).lower()} | {rejection_reason}"
    )

    raise ValueError(
        f"Substantive Story Value Gate rejected candidate "
        f"(confidence={confidence}): {rejection_reason}"
    )


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
            "excerpt": excerpt,
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

    # Deterministically separate the strongest facts from optional supporting
    # details. Extraction order is intentionally preserved because the prompt
    # asks the model to return core facts first. No facts are discarded.
    clean = clean[:EVIDENCE_MAX_FACTS]
    for index, fact_item in enumerate(clean):
        fact_item["role"] = "core" if index < CORE_FACTS_MAX else "supporting"

    return {
        "primary_group": "G1",
        "facts": clean,
        "core_fact_ids": [f["id"] for f in clean if f.get("role") == "core"],
        "supporting_fact_ids": [f["id"] for f in clean if f.get("role") == "supporting"],
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

    # Keep all provenance-verified facts, then collapse only facts that represent
    # the same underlying information unit. This prevents syndicated/repeated
    # reporting from inflating evidence count while preserving genuinely new details.
    locked, lineage_stats = _deduplicate_evidence_facts(facts)
    redundancy = (
        1.0 - (lineage_stats["unique_information_units"] / max(1, lineage_stats["raw_facts"]))
    )

    print(
        f"[FACT LINEAGE] raw_facts={lineage_stats['raw_facts']} "
        f"| unique_information_units={lineage_stats['unique_information_units']} "
        f"| merged_facts={lineage_stats['merged_facts']} "
        f"| redundancy={redundancy:.3f} "
        f"| candidate_pairs={lineage_stats.get('candidate_pairs', 0)}"
    )

    group_counts = {}
    for fact in locked:
        group_counts[fact["group"]] = group_counts.get(fact["group"], 0) + 1

    primary_group = max(
        group_counts,
        key=group_counts.get,
        default="C1-G1",
    )

    # Re-assign CORE/SUPPORTING only AFTER lineage deduplication.
    # The final locked fact list is the authoritative evidence universe, so
    # core IDs must be derived from that final list (not from pre-lineage facts).
    # This also guarantees that downstream article coverage cannot silently
    # fall back to all facts when lineage renumbers F1..Fn.
    locked = locked[:EVIDENCE_MAX_FACTS]
    for index, fact_item in enumerate(locked):
        fact_item["role"] = "core" if index < CORE_FACTS_MAX else "supporting"

    core_fact_ids = [
        f["id"] for f in locked if f.get("role") == "core" and f.get("id")
    ]
    supporting_fact_ids = [
        f["id"] for f in locked if f.get("role") == "supporting" and f.get("id")
    ]

    evidence = {
        "primary_group": primary_group,
        "facts": locked,
        "core_fact_ids": core_fact_ids,
        "supporting_fact_ids": supporting_fact_ids,
        "fact_lineage": lineage_stats,
    }

    print(
        f"[PERF] Evidence ready | facts={len(evidence['facts'])} "
        f"| core={len(core_fact_ids)} | supporting={len(supporting_fact_ids)} "
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


def _body_word_count(article):
    """Count only actual article-body text for monitoring; never used as a floor."""
    values = list(article.get("paragraphs", []))
    return len(" ".join(str(value) for value in values if value).split())


def _required_paragraphs(fact_count):
    """Return the minimum structural paragraph count without imposing a fact-based floor."""
    try:
        count = int(fact_count)
    except (TypeError, ValueError):
        count = 0
    return 1 if count <= 0 else 1


def _article_structure_check(article, evidence):
    """
    Deterministic schema/structure gate.

    Paragraph count is intentionally NOT derived from fact count. Coverage is
    validated separately through the internal fact_ids lock, while factual
    correctness is owned by the production Fact Guard in generate.py.
    """
    facts = evidence.get("facts", []) if isinstance(evidence, dict) else []
    fact_count = len(facts) if isinstance(facts, list) else 0
    paragraphs = article.get("paragraphs", []) if isinstance(article, dict) else []

    if not isinstance(paragraphs, list) or not paragraphs:
        return {
            "passed": False,
            "reason": "article has no paragraphs",
            "required_paragraphs": 1,
            "actual_paragraphs": 0,
            "fact_count": fact_count,
        }

    actual = len([p for p in paragraphs if isinstance(p, str) and p.strip()])
    if actual < 1:
        return {
            "passed": False,
            "reason": "article has no substantive paragraphs",
            "required_paragraphs": 1,
            "actual_paragraphs": actual,
            "fact_count": fact_count,
        }

    return {
        "passed": True,
        "reason": "article structure satisfied without a fact-count paragraph floor",
        "required_paragraphs": 1,
        "actual_paragraphs": actual,
        "fact_count": fact_count,
    }


# ============================================================
# Article generation

# The model writes an internal fact coverage map together with the prose.
# The map is never published; Python validates it before the factual audit.
_ARTICLE_FORMAT = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "h1": {"type": "string"},
        "paragraphs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "fact_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["text", "fact_ids"],
            },
        },
    },
    "required": [
        "title",
        "description",
        "h1",
        "paragraphs",
    ],
}


def _article_prompt(evidence, source_context=None):
    context_block = ""
    if WRITER_SOURCE_CONTEXT and source_context:
        bounded_context = str(source_context)[:WRITER_SOURCE_CONTEXT_CHARS]
        context_block = f"""

SOURCE CONTEXT — NON-AUTHORITATIVE:
{bounded_context}

IMPORTANT: This source context is provided only to help understand wording and
relationships between the locked facts. It is NOT an evidence source.
You MUST NOT extract, add, strengthen, infer or introduce any fact from it
unless that fact is explicitly present in LOCKED EVIDENCE. If SOURCE CONTEXT
and LOCKED EVIDENCE differ, LOCKED EVIDENCE always wins.
"""

    facts = evidence.get("facts", []) if isinstance(evidence, dict) else []
    core_ids = set(evidence.get("core_fact_ids", [])) if isinstance(evidence, dict) else set()
    fact_inventory = "\n".join(
        f"- {str(f.get('id','')).strip()} [{('CORE' if str(f.get('id','')).strip() in core_ids else 'SUPPORTING')}]: {str(f.get('fact','')).strip()}"
        for f in facts
        if isinstance(f, dict) and str(f.get('id','')).strip()
    )

    return f"""
Write a clear TrendCurrent news article in {LANGUAGE}.

LANGUAGE LOCK:
- TITLE, DESCRIPTION, H1 and EVERY paragraph text MUST be written in {LANGUAGE}.
- Translate/paraphrase only supported facts.
- Proper names and official names may remain in their original form.

SOURCE-LOCKED FACTUAL RULES:
- LOCKED EVIDENCE is the complete and closed factual universe.
- Use ONLY LOCKED EVIDENCE for factual content.
- Never invent, infer, strengthen, embellish or add outside knowledge.
- Every material sentence must be directly supported by LOCKED EVIDENCE.

ENTITY / ATTRIBUTION / TIME / NUMBER LOCKS:
- Never transfer roles, actions, responsibility, employers or relationships between entities.
- Preserve event status exactly: scheduled != completed; announced != implemented; proposed != completed.
- Never infer dates, recency, weekday or timing from publication metadata.
- Preserve names, numbers, prices, dates, scores, percentages and certainty exactly.
- Never calculate or derive new factual numbers.

COVERAGE-FIRST CONTRACT — MANDATORY FOR CORE FACTS ONLY:
- CORE facts are the mandatory factual spine of the article. Every CORE fact listed below MUST be explicitly represented in the article body.
- SUPPORTING facts are verified optional details. Use them when they improve clarity or information density, but do NOT force them into the article.
- Before writing, assign each CORE fact ID to one or more paragraphs.
- The returned paragraphs must include a fact_ids array naming the exact locked facts explicitly represented in that paragraph.
- Every CORE fact ID must appear in at least one paragraph's fact_ids.
- A SUPPORTING fact ID may appear only when the paragraph explicitly communicates that fact or a faithful paraphrase.
- Do NOT place a fact ID in fact_ids unless the paragraph text explicitly communicates that fact or a faithful paraphrase.
- Closely related facts may share a paragraph, but each remains explicit.
- Do not use a vague summary as coverage for multiple distinct facts.
- Every paragraph must add new verified information; never repeat facts merely for length.
- The fact_ids field is INTERNAL metadata and will never be published.
- This is a coverage requirement, NOT a word-count rule.
- Do not pad, repeat or invent facts to satisfy coverage.

STRUCTURE:
- Use as many substantive paragraphs as needed to present all locked facts clearly and coherently.
- Do NOT force a paragraph count based on the number of locked facts.
- Closely related facts may share a paragraph when each fact remains explicit.
- Do not compress distinct facts into vague summaries merely to reduce paragraph count.
- Do not pad or split paragraphs artificially just to satisfy a structural target.

STYLE:
- Natural, fluent {LANGUAGE}; professional, clear, objective and precise.
- No clickbait, speculation, filler or unsupported conclusions.
- Write ONE coherent article about ONE concrete story.

HEADLINE:
- Maximum 10 words AND 65 characters.
- TITLE and H1 must be identical.
- Use only the core verified entity and core verified development.

FINAL SELF-CHECK BEFORE RETURNING:
1. Enumerate every CORE fact ID in LOCKED EVIDENCE.
2. Verify every CORE fact ID appears in at least one paragraph fact_ids array.
3. Verify each paragraph's fact_ids are actually expressed in that paragraph text.
4. Verify every material sentence against LOCKED EVIDENCE.
5. Verify no fact used in the article was omitted from the locked evidence or transferred to the wrong entity.
6. Verify SUPPORTING facts are used only when they add real information and are not forced into the article.
7. Verify the result remains one coherent story.

Return ONLY this JSON shape:
{{
  "title": "...",
  "description": "...",
  "h1": "...",
  "paragraphs": [
    {{"text": "...", "fact_ids": ["F1", "F2"]}}
  ]
}}

LOCKED FACT INVENTORY:
{fact_inventory}

LOCKED EVIDENCE:
{_compact(evidence)}
{context_block}
"""


def _normalize_generated_article(article, evidence):
    if not isinstance(article, dict):
        raise ValueError("Article generator returned invalid object.")

    for key in ("title", "description", "h1"):
        if not isinstance(article.get(key), str):
            raise ValueError(f"Article generator returned invalid {key}.")

    raw_paragraphs = article.get("paragraphs")
    if not isinstance(raw_paragraphs, list) or not raw_paragraphs:
        raise ValueError("Article generator returned no paragraphs.")

    # Every locked fact ID is valid paragraph metadata. CORE facts are the
    # mandatory coverage universe; SUPPORTING facts remain optional but may be
    # explicitly cited by a paragraph when the paragraph actually communicates them.
    all_fact_ids = {
        str(f.get("id", "")).strip()
        for f in (evidence.get("facts", []) if isinstance(evidence, dict) else [])
        if isinstance(f, dict) and str(f.get("id", "")).strip()
    }
    core_fact_ids = (
        set(evidence.get("core_fact_ids", []))
        if isinstance(evidence, dict)
        else set()
    )
    valid_fact_ids = all_fact_ids
    required_core_ids = core_fact_ids & all_fact_ids
    covered = set()
    paragraphs = []

    for index, item in enumerate(raw_paragraphs, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Paragraph {index} is not an object.")
        text = str(item.get("text", "")).strip()
        ids = item.get("fact_ids", [])
        if not text or not isinstance(ids, list):
            raise ValueError(f"Paragraph {index} has invalid text/fact_ids.")
        clean_ids = []
        for fid in ids:
            fid = str(fid).strip()
            if not fid:
                continue
            if fid not in valid_fact_ids:
                raise ValueError(f"Generator returned unknown fact ID: {fid}")
            clean_ids.append(fid)
            covered.add(fid)
        if not clean_ids:
            raise ValueError(f"Paragraph {index} has no fact coverage metadata.")
        paragraphs.append(text)

    # Only CORE facts are mandatory for coverage. SUPPORTING facts are optional.
    missing = sorted(
        required_core_ids - covered,
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 999999,
    )
    if missing:
        raise ValueError(
            "Coverage-first generation failed before audit; missing fact IDs: "
            + ", ".join(missing)
        )

    clean = {
        "title": article["title"].strip(),
        "description": article["description"].strip(),
        "h1": article["h1"].strip(),
        "paragraphs": paragraphs,
    }
    return clean


def _generate_article(evidence, source_context=None):
    fact_count = len(evidence.get("core_fact_ids", [])) if isinstance(evidence, dict) else 0
    dynamic_tokens = max(ARTICLE_TOKENS, min(1500, 650 + fact_count * 100))
    raw_article = _call(
        _article_prompt(evidence, source_context=source_context),
        temperature=0.0,
        num_predict=dynamic_tokens,
        num_thread=NUM_THREADS,
        response_format=_ARTICLE_FORMAT,
        stage="article_generation",
    )
    return _normalize_generated_article(raw_article, evidence)


# ============================================================
# Public compatibility API / Fact Guard delegation
# ============================================================

# Public compatibility API used by generate.py.
# These wrappers intentionally contain no LLM factual audit or repair path.
def extract_evidence(source):
    return _extract_evidence(source)


def validate_article_structure(article, evidence, label="Article structure"):
    result = _article_structure_check(article, evidence)
    if not result.get("passed"):
        raise ValueError(
            f"{label} failed: {result.get('reason', 'invalid article structure')}"
        )
    print(
        f"[STRUCTURE] PASS | label={label} | "
        f"paragraphs={result.get('actual_paragraphs', 0)} | "
        f"required={result.get('required_paragraphs', 0)} | "
        f"facts={result.get('fact_count', 0)}"
    )
    return result


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


def generate(prompt, retries=0, evidence=None):
    """
    Coverage-first, fail-closed publication pipeline.

    Flow:
        evidence -> coverage-first article -> deterministic coverage gate ->
        structure gate -> return to caller for the single production Fact Guard

    There is deliberately NO repair, recursive regeneration, second article
    generation, or duplicate factual audit in this module. Fact Guard is the
    single production factual-validation authority in generate.py.
    """
    pipeline_start = time.perf_counter()

    print(
        f"[TIMER] PIPELINE START | prompt_chars={len(prompt or '')}"
    )

    if evidence is not None:
        print(
            f"[PIPELINE] Reusing preflight evidence lock | "
            f"facts={len(evidence.get('facts', []))}"
        )
    else:
        print("[PIPELINE] Building balanced evidence lock...")
        evidence = _extract_evidence(prompt)

    print("[PIPELINE] Generating coverage-first evidence-locked article...")
    article = _generate_article(
        evidence,
        source_context=prompt if WRITER_SOURCE_CONTEXT else None,
    )
    article = _sanitize_article(article)

    print(
        f"[PIPELINE] Article generated | "
        f"words={_body_word_count(article)}"
    )

    structure = validate_article_structure(
        article, evidence, label="Coverage-first structure"
    )

    # Factual validation is intentionally NOT performed here.
    # generate.py owns the single production factual-validation authority
    # (Fact Guard). Keeping a second LLM factual audit here would duplicate
    # expensive inference and recreate the latency/failure path we removed.
    print("[PIPELINE] Article generation complete | factual validation delegated to Fact Guard")
    print(
        f"[TIMER] PIPELINE TOTAL | "
        f"elapsed={time.perf_counter() - pipeline_start:.2f}s"
    )
    return article

