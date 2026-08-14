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

PIPELINE_VERSION = "universal-fact-lock-v2.1-balanced"

NUM_THREADS = max(1, int(os.getenv("OLLAMA_NUM_THREADS", "16")))
NUM_CTX = max(4096, int(os.getenv("OLLAMA_NUM_CTX", "8192")))
NUM_BATCH = max(64, int(os.getenv("OLLAMA_NUM_BATCH", "512")))

# The old extractor asked the model for facts + quotes + groups at once.
# That made a 500-token ceiling very easy to hit.  The balanced extractor
# keeps one compact fact record and a small number of records.
EVIDENCE_CHUNK_CHARS = max(
    7000, int(os.getenv("OLLAMA_EVIDENCE_CHUNK_CHARS", "14000"))
)
EVIDENCE_TOKENS = max(
    240, int(os.getenv("OLLAMA_EVIDENCE_TOKENS", "360"))
)
EVIDENCE_MAX_FACTS = max(
    4, int(os.getenv("OLLAMA_EVIDENCE_MAX_FACTS", "8"))
)

ARTICLE_TOKENS = max(
    500, int(os.getenv("OLLAMA_ARTICLE_TOKENS", "760"))
)
AUDIT_TOKENS = max(
    180, int(os.getenv("OLLAMA_AUDIT_TOKENS", "300"))
)
REPAIR_TOKENS = max(
    300, int(os.getenv("OLLAMA_REPAIR_TOKENS", "520"))
)

MIN_ARTICLE_WORDS = max(
    0, int(os.getenv("OLLAMA_MIN_ARTICLE_WORDS", "0"))
)

# Evidence-aware minimum. This prevents tiny articles when the ledger is rich,
# while allowing genuinely short stories to remain short instead of padding.
ADAPTIVE_MIN_1_2 = max(0, int(os.getenv("OLLAMA_MIN_1_2_FACTS", "45")))
ADAPTIVE_MIN_3_4 = max(0, int(os.getenv("OLLAMA_MIN_3_4_FACTS", "65")))
ADAPTIVE_MIN_5_PLUS = max(0, int(os.getenv("OLLAMA_MIN_5_PLUS_FACTS", "85")))
MAX_SECTIONS = 4

# Evidence is deliberately sequential on CPU.
EVIDENCE_PARALLEL = os.getenv(
    "OLLAMA_EVIDENCE_PARALLEL", "0"
).lower() not in {"0", "false", "no", "off"}

# One retry is allowed only for malformed/truncated evidence JSON.
# The retry uses a smaller output contract, not another huge prompt.
EVIDENCE_RETRY_TOKENS = max(
    180, int(os.getenv("OLLAMA_EVIDENCE_RETRY_TOKENS", "220"))
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
    threads = max(1, int(num_thread or NUM_THREADS))

    print(
        f"[TIMER] Ollama START | predict={num_predict} | temp={temperature} "
        f"| prompt_chars={len(prompt)} | threads={threads}"
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
            "num_batch": NUM_BATCH,
            "num_thread": threads,
        },
        "format": response_format,
    }

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

    print(
        f"[TIMER] Ollama END   | elapsed={elapsed:.2f}s "
        f"| response_chars={len(raw)}"
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
                    "g": {"type": "string"},
                    "f": {"type": "string"},
                    "e": {"type": "string"},
                    "s": {"type": "string"},
                    "d": {"type": "string"},
                    "t": {"type": "string"},
                },
                "required": ["g", "f", "e", "s", "d", "t"],
            },
        },
    },
    "required": ["facts"],
}


def _evidence_prompt(source, max_facts=None):
    limit = max_facts or EVIDENCE_MAX_FACTS

    return f"""
You are TrendCurrent's source-evidence extractor.

Read ONLY the SOURCE MATERIAL. Build evidence for ONE publishable news story.

PRIMARY-EVENT RULE:
- Identify the main story/event in the supplied material.
- Keep unrelated games, injuries, statements, people, dates or incidents out.
- Facts that belong to the SAME main event must share the same group value.
- Do NOT stop after one fact if the source contains more directly useful facts.
- Extract up to {limit} useful facts from the main event. Prefer 4-8 when available.
- A useful fact can be the event itself, date, location, result/status, named people,
  numbers, direct developments or other material details explicitly supported by the source.
- Do not manufacture supporting facts just to reach a count.

FACT RULES:
- Every fact must be directly supported by its excerpt.
- Preserve names, roles, dates, numbers and status exactly.
- Never turn reported/expected/talks into confirmed facts.
- Do not infer motives, causes, consequences or significance.
- Do not use outside knowledge.
- Keep excerpts short.
- Return ONLY JSON. No article and no commentary.

Use this compact JSON shape:
{{"facts":[{{"g":"G1","f":"fact","e":"short source excerpt","s":"source","d":"date or empty","t":"status or empty"}}]}}

SOURCE MATERIAL:
{source}
"""


def _evidence_retry_prompt(source):
    return f"""
Extract the MAIN EVENT from this source. Return ONLY compact JSON.

Use 2-6 facts if the source supports them; do not invent facts.
All returned facts must belong to the same main event. Exclude unrelated events.
Every fact needs a short supporting excerpt copied from the source.

{{"facts":[{{"g":"G1","f":"fact","e":"excerpt","s":"source","d":"date or empty","t":"status or empty"}}]}}

SOURCE:
{source}
"""


def _normalize_evidence(data):
    if not isinstance(data, dict):
        raise ValueError("Evidence response is not an object.")

    raw_facts = data.get("facts", [])
    if not isinstance(raw_facts, list):
        raw_facts = []

    clean = []
    seen = set()

    for item in raw_facts:
        if not isinstance(item, dict):
            continue

        group = str(item.get("g", "")).strip()
        fact = str(item.get("f", "")).strip()
        excerpt = str(item.get("e", "")).strip()
        source = str(item.get("s", "")).strip()
        date = str(item.get("d", "")).strip()
        status = str(item.get("t", "")).strip()

        if not group or not fact or not excerpt:
            continue

        key = (group.lower(), fact.lower(), excerpt.lower())
        if key in seen:
            continue
        seen.add(key)

        clean.append({
            "id": f"F{len(clean) + 1}",
            "group": group,
            "fact": fact,
            "excerpt": excerpt[:220],
            "source": source,
            "date": date,
            "status": status,
        })

        if len(clean) >= EVIDENCE_MAX_FACTS:
            break

    if not clean:
        raise ValueError("Evidence extraction produced no usable facts.")

    # Select the group with the most facts. Tie-break by first appearance.
    counts = {}
    first_seen = {}
    for index, fact in enumerate(clean):
        group = fact["group"]
        counts[group] = counts.get(group, 0) + 1
        first_seen.setdefault(group, index)

    primary = max(
        counts,
        key=lambda g: (counts[g], -first_seen[g]),
    )

    locked = [fact for fact in clean if fact["group"] == primary]
    if not locked:
        raise ValueError("Primary event group contains no usable facts.")

    return {
        "primary_group": primary,
        "facts": locked[:EVIDENCE_MAX_FACTS],
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
            maps.append(_normalize_evidence(data))
        except ValueError as exc:
            # Only malformed/truncated evidence is retried.  The retry is
            # compact, deterministic and deliberately smaller.
            if "Invalid Ollama JSON" not in str(exc):
                raise

            print(
                f"[PIPELINE] Evidence JSON retry | chunk={index}"
            )
            data = _call(
                _evidence_retry_prompt(chunk),
                temperature=0.0,
                num_predict=EVIDENCE_RETRY_TOKENS,
                num_thread=NUM_THREADS,
                response_format=_EVIDENCE_FORMAT,
            )
            maps.append(_normalize_evidence(data))

    # Merge only within the already selected primary group of each chunk.
    # Different chunks remain different groups, so they cannot accidentally
    # become one event.
    facts = []
    seen = set()

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

    # In normal one-chunk inputs this is the selected event.
    # For multi-chunk inputs, choose the group with the most evidence.
    group_counts = {}
    for fact in facts:
        group_counts[fact["group"]] = group_counts.get(fact["group"], 0) + 1

    primary_group = max(group_counts, key=group_counts.get)
    locked = [
        fact for fact in facts
        if fact["group"] == primary_group
    ]

    evidence = {
        "primary_group": primary_group,
        "facts": locked[:EVIDENCE_MAX_FACTS],
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
- You may naturally paraphrase supported facts.
- You may combine facts when they clearly describe the same event.
- Do not add outside facts.
- Do not invent dates, numbers, roles, locations, causes or motives.
- Preserve uncertainty and status.
- Do not turn a report into a confirmed fact.
- Do not use unsupported quotes.
- Do not add generic filler or speculation.
- Titles, descriptions and headings must also stay factual.
- Aim for roughly 90-140 words when the evidence contains enough material.
- Use the available supporting facts naturally; do not collapse a multi-fact story into a single sentence.
- If the evidence genuinely contains only one or two facts, a shorter article is acceptable.
- Never add filler just to reach a word target.

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
- Preserve supported material.
- Do not add new facts.
- Do not change dates, numbers, roles, status or certainty.
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

def _adaptive_min_words(evidence):
    count = len(evidence.get("facts", [])) if isinstance(evidence, dict) else 0
    if MIN_ARTICLE_WORDS > 0:
        return MIN_ARTICLE_WORDS
    if count <= 2:
        return ADAPTIVE_MIN_1_2
    if count <= 4:
        return ADAPTIVE_MIN_3_4
    return ADAPTIVE_MIN_5_PLUS


def generate(prompt, retries=0):
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
