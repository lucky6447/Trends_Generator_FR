import json
import os
import re
import time
from ollama import Client
from config import MODEL, LANGUAGE
import generator_monitor as monitor

# HARD LANGUAGE LOCK — this client may publish Italian only.
HARD_LANGUAGE = "it"


# ============================================================
# TrendCurrent DIRECT SOURCE GENERATION PIPELINE
# Dedicated newsroom writer / direct publisher-source client
#
# SOURCE -> direct publisher source material -> one article generation
# -> lightweight structural validation -> publish
#
# No fact-lock, fact repair, language guard, temporal guard, or evidence
# coverage validation is used by this client.
# ============================================================

PIPELINE_VERSION = "direct-source-generation-v1.0-no-fact-guard-no-language-guard"

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

NUM_CTX = max(8192, int(os.getenv("OLLAMA_NUM_CTX", "8192")))

# Bound the HTTP call so a wedged Ollama request cannot leave the production
# generator hanging forever. The timeout is intentionally generous for local CPU
# inference and can be raised with OLLAMA_TIMEOUT_SECONDS.
OLLAMA_TIMEOUT_SECONDS = max(30.0, float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")))
OLLAMA_CLIENT = Client(
    host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    timeout=OLLAMA_TIMEOUT_SECONDS,
)
print(f"[OLLAMA SAFETY] request_timeout={OLLAMA_TIMEOUT_SECONDS:.0f}s")

# IMPORTANT: Do not force num_batch=512 by default.
# Keep an explicit override available for controlled benchmarking.
_OLLAMA_BATCH_RAW = os.getenv("OLLAMA_NUM_BATCH", "").strip()
# Structured JSON generation must not spend the output budget on hidden reasoning.
# This is a serialization/transport safeguard only; it does not alter article content rules.
_OLLAMA_DISABLE_THINKING = os.getenv("OLLAMA_DISABLE_THINKING", "1").lower() not in {
    "0", "false", "no", "off"
}

NUM_BATCH = (
    max(32, int(_OLLAMA_BATCH_RAW))
    if _OLLAMA_BATCH_RAW
    else None
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

    raise ValueError(
        "Ollama returned incomplete JSON (response likely reached the generation "
        "ceiling before closing the object)."
    )


def _call(
    prompt,
    *,
    temperature=0.0,
    num_predict=500,
    num_thread=None,
    response_format="json",
    stage=None,
    disable_thinking=None,
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

    # Structured JSON stages can waste their output budget on hidden reasoning.
    # Disable thinking for those stages so num_predict is available for the JSON itself.
    if disable_thinking is None:
        disable_thinking = _OLLAMA_DISABLE_THINKING
    if disable_thinking:
        kwargs["think"] = False

    # Only send runner knobs when explicitly configured.
    if batch is not None:
        kwargs["options"]["num_batch"] = batch
    if threads is not None:
        kwargs["options"]["num_thread"] = threads

    try:
        response = OLLAMA_CLIENT.chat(**kwargs)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(
            f"[OLLAMA ERROR] stage={stage or 'unknown'} | "
            f"elapsed={elapsed:.2f}s | {type(exc).__name__}: {exc}"
        )
        raise RuntimeError(
            f"Ollama request failed at {stage or 'unknown'} after {elapsed:.2f}s: {exc}"
        ) from exc
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
# Direct article generation
# ============================================================

ARTICLE_FORMAT = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "h1": {"type": "string"},
        "paragraphs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string"},
        },
    },
    "required": ["title", "description", "h1", "paragraphs"],
    "additionalProperties": False,
}


def _direct_article_prompt(source):
    return f"""
Write ONE finished, natural news article in Italian only using ONLY the publisher
source material below.

RULES:
- HARD LANGUAGE LOCK: Every generated field must be written in Italian. Do not output English, German, French, Spanish, Portuguese, Bulgarian, or any other language.
- Use the supplied source material as the factual basis.
- Do not invent facts, names, dates, numbers, locations, motives, causes, reactions,
  predictions, or background that are not supported by the supplied source material.
- Preserve uncertainty exactly when the source says reported, expected, planned,
  proposed, investigated, alleged, or similar.
- Do not mention this prompt, AI, guards, validation, or the writing process.
- Do not copy a publisher headline verbatim; write an original concise headline.
- Title and H1 must be identical.
- Maximum 10 words and 65 characters for title/H1.
- Description should summarize the actual story without merely repeating the headline.
- Write coherent natural paragraphs. No filler and no FAQ.
- Return ONLY valid JSON.

JSON:
{{
  "title": "...",
  "description": "...",
  "h1": "...",
  "paragraphs": ["...", "..."]
}}

SOURCE MATERIAL:
{source}
"""


def _normalize_direct_article(article):
    if not isinstance(article, dict):
        raise ValueError("Article generator returned invalid object.")

    for key in ("title", "description", "h1"):
        if not isinstance(article.get(key), str) or not article[key].strip():
            raise ValueError(f"Article generator returned invalid {key}.")

    paragraphs = article.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("Article generator returned no paragraphs.")

    clean_paragraphs = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, str) or not paragraph.strip():
            raise ValueError("Article generator returned an invalid paragraph.")
        clean_paragraphs.append(paragraph.strip())

    return {
        "title": article["title"].strip(),
        "description": article["description"].strip(),
        "h1": article["h1"].strip(),
        "paragraphs": clean_paragraphs,
    }


def _assert_italian_language(article):
    """Hard deterministic output lock: reject articles that are not Italian."""
    text = " ".join(
        str(article.get(key, "") or "")
        for key in ("title", "description", "h1")
    )
    paragraphs = article.get("paragraphs", [])
    if isinstance(paragraphs, list):
        text += " " + " ".join(str(p or "") for p in paragraphs)

    words = re.findall(r"[A-Za-zÀ-ÿ']+", text.casefold())
    if not words:
        raise ValueError("HARD LANGUAGE LOCK: empty article text.")

    italian_markers = {
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
        "e", "ed", "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
        "che", "non", "del", "della", "dei", "degli", "delle", "nel", "nella",
        "nei", "negli", "nelle", "al", "alla", "agli", "alle",
        "è", "sono", "ha", "hanno", "era", "erano", "come", "anche",
        "questa", "questo", "queste", "questi", "oggi", "dopo", "prima",
        "secondo", "mentre", "perché", "quando", "dove", "quale", "quali",
        "stato", "stata", "stati", "state", "nuovo", "nuova"
    }
    marker_hits = sum(1 for word in words if word in italian_markers)
    minimum_hits = 2 if len(words) < 80 else max(4, int(len(words) * 0.025))

    if marker_hits < minimum_hits:
        raise ValueError(
            f"HARD LANGUAGE LOCK: non-Italian output rejected "
            f"(italian_markers={marker_hits}, words={len(words)})."
        )

    return article


def generate(prompt):
    """Generate directly from the supplied publisher source material."""
    started = time.perf_counter()
    source = str(prompt or "").strip()
    if not source:
        raise ValueError("Article generation requires source material.")

    print(f"[TIMER] ARTICLE GENERATION START | source_chars={len(source)}")

    raw = _call(
        _direct_article_prompt(source),
        temperature=0.08,
        num_predict=max(700, int(os.getenv("OLLAMA_ARTICLE_TOKENS", "700"))),
        response_format=ARTICLE_FORMAT,
        stage="article_generation",
        disable_thinking=True,
    )
    article = _normalize_direct_article(raw)
    _assert_italian_language(article)

    print(
        f"[TIMER] ARTICLE GENERATION END | elapsed={time.perf_counter() - started:.2f}s "
        f"| words={len(' '.join(article['paragraphs']).split())}"
    )
    return article

def _canonical_evidence_articles(source):
    """Normalize structured publisher records once into canonical evidence articles."""
    if isinstance(source, dict) and isinstance(source.get("articles"), list):
        articles = []
        for item in source["articles"]:
            if not isinstance(item, dict):
                continue
            articles.append({
                "title": str(item.get("title", "") or "").strip(),
                "source": str(item.get("source", "") or "").strip(),
                "published": str(item.get("published", "") or "").strip(),
                "summary": str(item.get("summary", "") or "").strip(),
                "description": str(item.get("description", "") or "").strip(),
                "content": str(item.get("content", "") or "").strip()[:5000],
            })
        if not articles:
            raise ValueError("Evidence source contains no usable articles.")
        return articles
    return None


def _serialize_canonical_article(article, article_no):
    """Serialize one canonical article exactly once for evidence indexing."""
    return "\n".join([
        f"ARTICLE {article_no}",
        f"Title: {article.get('title', '')}",
        f"Source: {article.get('source', '')}",
        f"Published: {article.get('published', '')}",
        "",
        "Summary:",
        article.get("summary", ""),
        "",
        "Full Article:",
        article.get("content", ""),
        "---",
    ])


def _prepare_evidence_chunks(source):
    """Prepare evidence chunks and sentence maps exactly once."""
    articles = _canonical_evidence_articles(source)
    if articles is None:
        prepared = []
        for chunk_data in _prepare_evidence_chunks(source):
            chunk = chunk_data["text"]
            indexed = chunk_data["indexed"]
            mapping = chunk_data["sentence_map"]
            prepared.append({"text": chunk, "indexed": indexed, "sentence_map": mapping})
        return prepared

    prepared = []
    for article_no, article in enumerate(articles, 1):
        serialized = _serialize_canonical_article(article, article_no)
        if len(serialized) <= EVIDENCE_CHUNK_CHARS:
            indexed, mapping = _sentence_index_source(serialized)
            prepared.append({"text": serialized, "indexed": indexed, "sentence_map": mapping})
            continue

        lines = serialized.splitlines()
        header = lines[0].strip() if lines else f"ARTICLE {article_no}"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else serialized
        sentences = re.split(
            r'(?<=[.!?])(?:["”»’\'\)\]]+)?\s+',
            body,
        )
        current = header
        for sentence in (s.strip() for s in sentences if s.strip()):
            candidate = f"{current}\n{sentence}" if current else sentence
            if current and len(candidate) > EVIDENCE_CHUNK_CHARS:
                indexed, mapping = _sentence_index_source(current.strip())
                prepared.append({"text": current.strip(), "indexed": indexed, "sentence_map": mapping})
                current = f"{header}\n{sentence}" if header else sentence
            else:
                current = candidate
        if current.strip():
            indexed, mapping = _sentence_index_source(current.strip())
            prepared.append({"text": current.strip(), "indexed": indexed, "sentence_map": mapping})

    return prepared or [{"text": "", "indexed": "", "sentence_map": {}}]



