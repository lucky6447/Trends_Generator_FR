print("[TrendCurrent PIPELINE] universal-fact-lock-v2.4-evidence-gate")
import re
import os
import subprocess
from datetime import date

from config import MAX_ARTICLES_PER_RUN, LANGUAGE
from rss import fetch_trends
from news import fetch_news, extract_article
from prompt import build_prompt
from ollama_client import generate, extract_evidence
from ollama import chat
from config import MODEL
from fact_guard import validate as fact_guard_validate
from fact_guard_repair import repair as fact_guard_repair
import json
import unicodedata
from html_generator import render_article, save_article
from processed import load_processed, add_processed
from index_generator import update_all
from topic_scorer import rank_trends, score_with_news, select_final_candidates, filter_relevant_news

REQUIRED_FIELDS = ["title", "description", "h1", "paragraphs"]

# Article length is determined by the amount of usable verified evidence.
# There is no artificial word-count target or evidence-count-based minimum.

SKIP_PATTERNS = [
    " vs ",
    " v ",
    " live",
    " score",
    " result",
    " calendario",
    " alineación",
    " pronóstico",
    " stream",
    " streaming",
]


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _build_targeted_news_queries(title):
    """Build a small deterministic query ladder without changing the topic itself."""
    text = " ".join(str(title or "").split()).strip()
    if not text:
        return []
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").casefold()
    norm = re.sub(r"[-–—]+", " ", norm)
    states = (
        "baden wuerttemberg", "bayern", "berlin", "brandenburg", "bremen", "hamburg",
        "hessen", "mecklenburg vorpommern", "niedersachsen", "nordrhein westfalen",
        "rheinland pfalz", "saarland", "sachsen anhalt", "sachsen", "schleswig holstein", "thueringen",
    )
    state = next((x for x in states if re.search(rf"(?<!\w){re.escape(x)}(?!\w)", norm)), "")
    queries=[]
    if state:
        state_display = {
            "baden wuerttemberg": "Baden-Württemberg",
            "mecklenburg vorpommern": "Mecklenburg-Vorpommern",
            "nordrhein westfalen": "Nordrhein-Westfalen",
            "rheinland pfalz": "Rheinland-Pfalz",
            "sachsen anhalt": "Sachsen-Anhalt",
            "schleswig holstein": "Schleswig-Holstein",
            "thueringen": "Thüringen",
        }.get(state, state.title())
        queries.extend([
            f'"{state_display}" Wahl',
            f'"{state_display}" Landtagswahl',
            f'"{state_display}" Wahlprognose',
            f'"{state_display}" Umfrage',
        ])
    queries.append(text)
    return list(dict.fromkeys(queries))


def _build_related_story_query(title):
    """Build a compact entity/event query for Google News corroboration.

    Discovery headlines can be long publisher headlines. Re-searching the
    entire headline is too restrictive and often returns no corroborating
    results. Keep the core named entities/event words instead.
    """
    text = " ".join(str(title or "").split()).strip()
    if not text:
        return ""

    # Remove common publisher suffixes when the discovery headline exposes one.
    text = re.sub(
        r"\s*[-|]\s*(?:suara\.com|wolipop|detikcom|detik\.com|"
        r"antaranews|antara|jpnn|liputan6|fimela|inews|idn times|"
        r"kapanlagi(?:\.com)?|haibunda|katadata(?:\.co\.id)?|"
        r"rri(?:\.co\.id)?|sindonews(?:\.com)?|merdeka(?:\.com)?|"
        r"kompas(?:\.com)?|grid\.id)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove obvious source-headline scaffolding that does not identify the
    # underlying story. Keep the entity/event itself.
    text = re.sub(
        r"^(?:sinopsis|daftar|rekomendasi|cara menonton|cara nonton|"
        r"jadwal tayang)\s*[:!-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    words = re.findall(r"[A-Za-z0-9À-ÿ']+", text)
    return " ".join(words[:10])


def _evidence_words(text):
    """Return conservative significant tokens for deterministic story clustering."""
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", text.casefold())
    stop = {
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "yang", "telah",
        "akan", "jadi", "sebuah", "para", "pada", "dalam", "atas", "oleh", "sebagai", "lebih",
        "the", "and", "for", "with", "from", "this", "that", "has", "have", "was", "were", "are",
        "its", "into", "after", "before", "over", "under", "about", "news", "today", "world"
    }
    return {w for w in words if len(w) >= 3 and w not in stop}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _story_pool_profile(news, topic):
    """Cheap lexical profile used only by the single story-source decision."""
    items = list(news or [])
    if not items:
        return {
            "status": "REJECT", "reason": "empty source pool", "count": 0,
            "dominant": 0, "ratio": 0.0, "cohesion": 0.0,
            "member_cohesion": 0.0, "cluster": [], "components": [],
        }

    profiles = []
    for idx, item in enumerate(items):
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        title_words = _evidence_words(title)
        body_words = _evidence_words(summary[:700])
        profiles.append({
            "idx": idx,
            "title_words": title_words,
            "words": title_words | body_words,
        })

    n = len(profiles)
    if n < 3:
        return {
            "status": "PASS", "reason": "small source pool", "count": n,
            "dominant": n, "ratio": 1.0, "cohesion": 1.0,
            "member_cohesion": 1.0, "cluster": [p["idx"] for p in profiles],
            "components": [[p["idx"] for p in profiles]],
        }

    topic_words = _evidence_words(topic)
    frequency = {}
    for p in profiles:
        for word in p["title_words"]:
            frequency[word] = frequency.get(word, 0) + 1

    common_words = {
        word for word, count in frequency.items()
        if count >= max(3, int(n * 0.60 + 0.999))
    }
    ignored = topic_words | common_words

    residual = []
    for p in profiles:
        core = p["title_words"] - ignored
        if len(core) < 2:
            core = p["title_words"] - topic_words
        if not core:
            core = p["title_words"]
        p["core"] = core
        residual.append(core)

    similarities = {}
    for i in range(n):
        for j in range(i + 1, n):
            sim_title = _jaccard(residual[i], residual[j])
            sim_full = _jaccard(
                profiles[i]["words"] - topic_words,
                profiles[j]["words"] - topic_words,
            )
            similarities[(i, j)] = max(sim_title, sim_full * 0.85)

    edge_threshold = 0.16
    adjacency = {i: set() for i in range(n)}
    for (i, j), sim in similarities.items():
        if sim >= edge_threshold:
            adjacency[i].add(j)
            adjacency[j].add(i)

    components = []
    unseen = set(range(n))
    while unseen:
        start_idx = min(unseen)
        stack = [start_idx]
        unseen.remove(start_idx)
        component = []
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nxt in adjacency[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    stack.append(nxt)
        components.append(sorted(component))
    components.sort(key=lambda c: (-len(c), c[0]))

    dominant = components[0]
    dominant_size = len(dominant)
    ratio = dominant_size / n

    internal = []
    for pos, i in enumerate(dominant):
        for j in dominant[pos + 1:]:
            internal.append(similarities.get((min(i, j), max(i, j)), 0.0))
    cohesion = sum(internal) / len(internal) if internal else 1.0

    if dominant_size >= 2:
        member_strength = []
        for i in dominant:
            sims = [
                similarities.get((min(i, j), max(i, j)), 0.0)
                for j in dominant if j != i
            ]
            member_strength.append(sum(sims) / len(sims) if sims else 0.0)
        member_cohesion = sum(member_strength) / len(member_strength)
    else:
        member_cohesion = 0.0

    if dominant_size >= 4 and ratio >= 0.50 and cohesion >= 0.16 and member_cohesion >= 0.14:
        status = "PASS"
        reason = "dominant story cluster"
    elif dominant_size >= 3 and ratio >= 0.50 and cohesion >= 0.24 and member_cohesion >= 0.20:
        status = "PASS"
        reason = "strong dominant story core"
    else:
        status = "REJECT"
        reason = "mixed source pool"

    return {
        "status": status,
        "reason": reason,
        "count": n,
        "dominant": dominant_size,
        "ratio": ratio,
        "cohesion": cohesion,
        "member_cohesion": member_cohesion,
        "cluster": dominant,
        "components": components,
    }


def _semantic_story_concentration_judge(news, topic):
    """One compact semantic judgment for borderline source pools."""
    items = list(news or [])
    lines = []
    for idx, item in enumerate(items[:12], 1):
        title = str(item.get("title", "")).strip()
        summary = re.sub(r"\s+", " ", str(item.get("summary", "")).strip())[:260]
        if title:
            lines.append(f"{idx}. TITLE: {title}\n   SUMMARY: {summary}")
    if not lines:
        return {"status": "MIXED", "confidence": 0, "reason": "no semantic input", "source_numbers": []}

    prompt = f"""
You are TrendCurrent's pre-evidence story selector.

Classify the retrieved source pool into exactly one:
1) ONE_STORY - most sources corroborate one concrete news story/event.
2) DOMINANT_STORY - a clear majority corroborates one concrete story and the
   remaining sources are unrelated/outliers.
3) MIXED - there is no safely isolatable majority story.

TOPIC: {str(topic or '').strip()}

Rules:
- Same broad topic is NOT the same story.
- Different wording or languages for the SAME event counts as the same story.
- DOMINANT_STORY is valid only when there is a real majority that can be isolated safely.
- Separate local/regional stories, separate people/events, programmes, lists,
  roundups, or unrelated developments are outliers and must not be selected.
- Be conservative. Never invent a connection.

Return ONLY JSON:
{{"verdict":"DOMINANT_STORY","confidence":95,"source_numbers":[1,2,3],"reason":"brief reason"}}

SOURCE HEADLINES:
{chr(10).join(lines)}
"""

    try:
        started = __import__("time").perf_counter()
        raw = chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "top_p": 0.85,
                "top_k": 40,
                "num_ctx": max(4096, int(os.getenv("OLLAMA_NUM_CTX", "6144"))),
                "num_predict": 160,
            },
            format="json",
        )
        elapsed = __import__("time").perf_counter() - started
        content = getattr(getattr(raw, "message", None), "content", "") or ""
        start_json = content.find("{")
        end_json = content.rfind("}")
        if start_json < 0 or end_json <= start_json:
            raise ValueError("semantic selector returned no JSON object")
        result = json.loads(content[start_json:end_json + 1])
        verdict = str(result.get("verdict", "")).strip().upper()
        confidence = max(0, min(100, int(result.get("confidence", 0))))
        reason = str(result.get("reason", "")).strip()[:300]
        raw_numbers = result.get("source_numbers", [])
        source_numbers = []
        if isinstance(raw_numbers, list):
            for value in raw_numbers:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= number <= len(items) and number not in source_numbers:
                    source_numbers.append(number)
        if verdict not in {"PASS", "ONE_STORY", "DOMINANT_STORY", "MIXED"}:
            raise ValueError(f"invalid semantic verdict: {verdict}")

        print(
            f"[TOPIC FILTER] STORY SOURCE JUDGE | {verdict} | "
            f"confidence={confidence} | elapsed={elapsed:.2f}s | {reason}"
        )
        return {
            "status": verdict,
            "confidence": confidence,
            "reason": reason or "semantic story selection judgment",
            "source_numbers": source_numbers,
        }
    except Exception as exc:
        # This judge is part of the source-safety decision. If it was required
        # to resolve a borderline/rejected pool, an unavailable result cannot
        # safely become PASS.
        print(f"[TOPIC FILTER] STORY SOURCE JUDGE | ERROR | fail-closed | {exc}")
        return {
            "status": "ERROR",
            "confidence": 0,
            "reason": f"semantic selector unavailable: {exc}",
            "source_numbers": [],
        }


def _decide_story_sources(news, topic):
    """Single authority for choosing the exact sources used by evidence extraction."""
    items = list(news or [])
    profile = _story_pool_profile(items, topic)
    count = profile["count"]

    print(
        f"[TOPIC FILTER] STORY SOURCE DECISION | lexical={profile['status']} "
        f"| sources={count} | dominant={profile['dominant']}/{count} "
        f"| ratio={profile['ratio']:.2f} | cohesion={profile['cohesion']:.2f}"
    )
    if profile.get("components"):
        sizes = ", ".join(str(len(c)) for c in profile["components"][:5])
        print(f"[TOPIC FILTER] Story pool components | sizes={sizes}")

    # A pool with fewer than three sources has no meaningful concentration
    # signal; use all available sources. Three or more sources must obey the
    # same deterministic/semantic decision policy as every larger pool.
    if count < 3:
        selected_indices = list(range(count))
        reason = "small source pool"
        status = "PASS"
        semantic_status = "SKIPPED"
    else:
        # IMPORTANT: a lexical REJECT is not a final story decision.
        # The lexical clusterer is intentionally conservative and can split
        # legitimate corroborating headlines when publishers use different
        # wording. Every 3+ source REJECT therefore gets one semantic
        # arbitration pass. The semantic judge may still reject the pool.
        #
        # This is the critical recovery path for cases such as 1/8 lexical
        # concentration where the sources can still describe one real event.
        semantic_needed = (
            profile["status"] == "REJECT"
        ) or (
            profile["status"] == "PASS"
            and (
                profile.get("ratio", 0.0) < 0.75
                or profile.get("cohesion", 0.0) < 0.28
            )
        )

        semantic = (
            _semantic_story_concentration_judge(items, topic)
            if semantic_needed
            else {"status": "SKIPPED", "confidence": 100, "reason": "strong deterministic concentration", "source_numbers": []}
        )
        semantic_status = semantic["status"]

        if semantic["status"] == "ERROR":
            raise Exception(
                "Story source decision rejected because semantic judgment failed "
                f"(reason={semantic.get('reason', '')})"
            )

        if semantic["status"] == "MIXED":
            raise Exception(
                "Story source decision rejected mixed source pool "
                f"(confidence={semantic.get('confidence', 0)}; reason={semantic.get('reason', '')})"
            )

        if semantic["status"] == "ONE_STORY":
            selected_indices = list(range(count))
            reason = "semantic one story"
            status = "PASS"
            print(
                f"[TOPIC FILTER] STORY SOURCE DECISION CONFIRM | "
                f"kept={len(selected_indices)}/{count} | one story"
            )

        elif semantic["status"] == "DOMINANT_STORY":
            numbers = semantic.get("source_numbers", [])
            minimum_majority = max(3, int(count * 0.50 + 0.999))
            if len(numbers) < minimum_majority:
                raise Exception("Story source judge returned an insufficient dominant majority")
            selected_indices = [n - 1 for n in numbers]
            reason = "semantic dominant story"
            status = "PASS"
            print(
                f"[TOPIC FILTER] STORY SOURCE DECISION PRUNE | "
                f"kept={len(selected_indices)}/{count} | outliers={count-len(selected_indices)}"
            )
        else:
            # This branch is reachable only for a deterministic PASS with
            # semantic verification skipped. No semantic decision is overwritten.
            selected_indices = list(profile.get("cluster") or [])
            if not selected_indices:
                raise Exception("Story source decision produced no usable source cluster")
            reason = profile["reason"]
            status = "PASS"

    selected_indices = sorted(dict.fromkeys(i for i in selected_indices if 0 <= i < count))
    if not selected_indices:
        raise Exception("Story source decision produced an empty evidence source set")

    print(
        f"[TOPIC FILTER] STORY SOURCE DECISION PASS | "
        f"kept={len(selected_indices)}/{count} | reason={reason} | semantic={semantic_status}"
    )
    for idx in selected_indices:
        print(f"[TOPIC FILTER] Evidence source -> {str(items[idx].get('title', '')).strip()}")

    return {
        "status": status,
        "reason": reason,
        "count": count,
        "selected_indices": selected_indices,
        "selected_count": len(selected_indices),
        "semantic_status": semantic_status,
    }


def _build_evidence_source(news, story_selection):
    """Build evidence input from the exact source selection already made upstream."""
    items = list(news or [])
    selection = story_selection or {}
    indices = list(selection.get("selected_indices") or [])
    selected_news = [items[i] for i in indices if 0 <= i < len(items)]
    if not selected_news:
        raise Exception("Evidence source received no sources from story selection")

    compact_sources = []
    for item in selected_news:
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        source = str(item.get("source", "")).strip()
        published = str(item.get("published", "")).strip()
        content = str(item.get("content", "")).strip()
        evidence_text = content if len(content) >= 300 else summary
        compact_sources.append({
            "title": title,
            "source": source,
            "published": published,
            "summary": summary,
            "content": evidence_text[:5000],
        })

    return "\n".join([
        "SOURCE MATERIAL:",
        *[
            "\n".join([
                f"SOURCE S{i}",
                f"ARTICLE {i}",
                f"Title: {item['title']}",
                f"Source: {item['source']}",
                f"Published: {item['published']}",
                "",
                "Summary:",
                item["summary"],
                "",
                "Full Article:",
                item["content"],
                "---",
            ])
            for i, item in enumerate(compact_sources, 1)
        ],
    ])


def _build_fact_guard_source(news):
    """
    Build a deterministic, factual-complete representation for the external
    Fact Guard.

    This intentionally does NOT summarize, truncate, reorder, or deduplicate
    news items. It only removes transport/formatting overhead and omits the
    URL field, which is not factual article content for the semantic audit.

    Summary is retained unless it is substantially redundant with the article
    content. This is conservative: if there is meaningful information in the
    summary that is not present in content, it stays.
    """
    compact_sources = []

    for item in news or []:
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        source = str(item.get("source", "")).strip()
        published = str(item.get("published", "")).strip()
        content = str(item.get("content", "")).strip()

        record = {
            "title": title,
            "source": source,
            "published": published,
            "content": content,
        }

        # Conservative redundancy check. Normalize whitespace and compare
        # whether the complete summary is already contained in article text.
        summary_norm = " ".join(summary.split()).casefold()
        content_norm = " ".join(content.split()).casefold()

        if summary and (
            not content_norm
            or not summary_norm
            or summary_norm not in content_norm
        ):
            record["summary"] = summary

        compact_sources.append(record)

    # No pretty-print indentation and no link field. The factual fields above
    # remain unchanged; only serialization overhead is reduced.
    return json.dumps(
        compact_sources,
        ensure_ascii=False,
        separators=(",", ":"),
    )




HEADLINE_MAX_WORDS = 10
HEADLINE_MAX_CHARS = 65

# Headline-only repair must never invoke the full universal fact-lock pipeline.
HEADLINE_REPAIR_THREADS = max(1, int(os.getenv("HEADLINE_REPAIR_THREADS", "16")))
HEADLINE_REPAIR_CTX = max(2048, int(os.getenv("HEADLINE_REPAIR_CTX", "4096")))
HEADLINE_REPAIR_BATCH = max(64, int(os.getenv("HEADLINE_REPAIR_BATCH", "256")))
HEADLINE_REPAIR_TOKENS = max(48, int(os.getenv("HEADLINE_REPAIR_TOKENS", "96")))


def _normalise_headline(text):
    text = "" if text is None else str(text)
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("|", "-")
    return " ".join(text.split()).strip().casefold()


def _headline_source_suffix(title, news):
    """Return a source suffix if the headline visibly leaks a publisher name."""
    normalized = _normalise_headline(title)
    if not normalized:
        return None

    parts = re.split(r"\s-\s", normalized)
    if len(parts) < 2:
        return None

    tail = parts[-1].strip(" .,:;\u2013\u2014")
    if not tail:
        return None

    for item in news or []:
        source = _normalise_headline(item.get("source", ""))
        if not source:
            continue
        source = source.strip(" .,:;\u2013\u2014")
        if tail == source or tail.endswith(source) or source.endswith(tail):
            return tail

    if re.search(r"\.(?:co|com|id|net|org)(?:\.[a-z]{2,3})?$", tail):
        return tail

    return None


def _headline_violations(article, trend):
    title = " ".join(str(article.get("title", "")).split()).strip()
    h1 = " ".join(str(article.get("h1", "")).split()).strip()
    source_titles = [
        str(x.get("title", "")).strip()
        for x in trend.get("news", [])
    ]

    violations = []

    if not title:
        violations.append("empty title")
    if len(title.split()) > HEADLINE_MAX_WORDS:
        violations.append(f"title exceeds {HEADLINE_MAX_WORDS} words")
    if len(title) > HEADLINE_MAX_CHARS:
        violations.append(f"title exceeds {HEADLINE_MAX_CHARS} characters")

    if not h1:
        violations.append("empty h1")
    if len(h1.split()) > HEADLINE_MAX_WORDS:
        violations.append(f"h1 exceeds {HEADLINE_MAX_WORDS} words")
    if len(h1) > HEADLINE_MAX_CHARS:
        violations.append(f"h1 exceeds {HEADLINE_MAX_CHARS} characters")

    if _headline_source_suffix(title, trend.get("news", [])):
        violations.append("publisher/source suffix in title")
    if _headline_source_suffix(h1, trend.get("news", [])):
        violations.append("publisher/source suffix in h1")

    title_norm = _normalise_headline(title)
    if title_norm and any(
        title_norm == _normalise_headline(x)
        for x in source_titles if x
    ):
        violations.append("title copies source headline")

    h1_norm = _normalise_headline(h1)
    if h1_norm and any(
        h1_norm == _normalise_headline(x)
        for x in source_titles if x
    ):
        violations.append("h1 copies source headline")

    if title_norm and h1_norm and title_norm != h1_norm:
        violations.append("title and h1 differ")

    return violations


def _repair_headline(article, trend):
    """Repair only title/H1 when the model violates the editorial headline policy."""
    current_title = str(article.get("title", "")).strip()
    current_h1 = str(article.get("h1", "")).strip()
    topic = str(trend.get("title", "")).strip()

    source_titles = [
        str(x.get("title", "")).strip()
        for x in trend.get("news", [])
        if str(x.get("title", "")).strip()
    ][:8]

    source_block = "\n".join(f"- {x}" for x in source_titles)

    repair_prompt = f"""
You are a professional {LANGUAGE} news headline editor.

You are NOT rewriting the article. You are repairing ONLY its public headline.
The article has already been generated from source-locked evidence.

MAIN TOPIC:
{topic}

CURRENT TITLE:
{current_title}

CURRENT H1:
{current_h1}

SOURCE HEADLINES FOR CONTEXT:
{source_block}

STRICT HEADLINE RULES:
- Return exactly ONE clean editorial headline in {LANGUAGE}.
- Maximum 10 words.
- Maximum 65 characters.
- Prefer 7-10 words when natural.
- Keep only the core verified entity + core verified event/development.
- Do NOT copy any source headline verbatim.
- Do NOT include a publisher, website, domain, author or source name.
- Do NOT use SEO listicle wording, keyword stuffing or filler.
- Do NOT add any fact that is not already present in the supplied topic/headline context.
- Preserve the factual status of the existing headline; shorten it rather than changing the claim.
- The result must be suitable for a professional news card.

Return ONLY valid JSON:
{{"title":""}}
"""

    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": repair_prompt}],
        options={
            "temperature": 0.0,
            "top_p": 0.85,
            "top_k": 40,
            "num_ctx": HEADLINE_REPAIR_CTX,
            "num_predict": HEADLINE_REPAIR_TOKENS,
            "num_batch": HEADLINE_REPAIR_BATCH,
            "num_thread": HEADLINE_REPAIR_THREADS,
        },
        format={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    )
    raw = response.message.content or ""
    try:
        repaired = json.loads(raw)
    except Exception as exc:
        raise Exception(f"Headline repair returned invalid JSON: {exc}") from exc
    if not isinstance(repaired, dict):
        raise Exception("Headline repair did not return a JSON object")

    new_title = " ".join(str(repaired.get("title", "")).split()).strip()
    if not new_title:
        raise Exception("Headline repair returned an empty title")

    # Deterministic local fallback: never re-call Ollama for a headline
    # that is still over the editorial limits after repair.
    if len(new_title.split()) > HEADLINE_MAX_WORDS or len(new_title) > HEADLINE_MAX_CHARS:
        words = new_title.split()
        new_title = " ".join(words[:HEADLINE_MAX_WORDS]).strip()

        if len(new_title) > HEADLINE_MAX_CHARS:
            new_title = new_title[:HEADLINE_MAX_CHARS].rsplit(" ", 1)[0].strip(" -,:;")

        if not new_title:
            raise Exception("Headline repair produced no usable headline")

    article["title"] = new_title
    article["h1"] = new_title

    violations = _headline_violations(article, trend)
    if violations:
        raise Exception("Headline repair failed: " + "; ".join(violations))

    print(f"[HEADLINE GUARD] PASS -> {new_title}")
    return article


def enforce_headline_policy(article, trend):
    violations = _headline_violations(article, trend)
    if not violations:
        print(f"[HEADLINE GUARD] PASS -> {article.get('title', '')}")
        return article

    print("[HEADLINE GUARD] REPAIR REQUIRED | " + "; ".join(violations))
    return _repair_headline(article, trend)

def validate_article(article):
    if not isinstance(article, dict):
        raise Exception("Article is not JSON object")

    for f in REQUIRED_FIELDS:
        if f not in article:
            raise Exception(f"Missing field: {f}")

    if not isinstance(article["paragraphs"], list):
        raise Exception("paragraphs must be a list")

    if not article["paragraphs"]:
        raise Exception("Article must contain at least one paragraph.")

    for paragraph in article["paragraphs"]:
        if not isinstance(paragraph, str) or not paragraph.strip():
            raise Exception("Invalid paragraph")

    return True


# ============================================================
# FINAL ARTICLE STORY QUALITY GATE
# ============================================================
# This is an editorial gate, not a factuality gate.
# Fact Guard answers: "Are the claims supported?"
# Story Quality Gate answers: "Is this ONE coherent news story?"
#
# IMPORTANT:
# - Short but clean articles PASS.
# - Relevant context around the main event is allowed.
# - Roundups, mixed stories, generic topic digests and unrelated
#   second stories are REJECTED.
# - BORDERLINE is treated as a production rejection. We never spend
#   downstream resources (including image generation) on an uncertain article.
# - PASS also requires minimum editorial confidence; a low-confidence PASS
#   is converted to BORDERLINE.
#
# This gate deliberately runs AFTER generate_valid_article() and BEFORE
# save_article(), so an article cannot become a production success until
# it passes both factual and editorial validation.
STORY_QUALITY_THREADS = max(1, int(os.getenv("STORY_QUALITY_THREADS", "16")))
STORY_QUALITY_CTX = max(4096, int(os.getenv("STORY_QUALITY_CTX", "8192")))
STORY_QUALITY_BATCH = max(64, int(os.getenv("STORY_QUALITY_BATCH", "256")))
STORY_QUALITY_TOKENS = max(128, int(os.getenv("STORY_QUALITY_TOKENS", "256")))
STORY_QUALITY_MIN_PASS_CONFIDENCE = max(
    0,
    min(100, int(os.getenv("STORY_QUALITY_MIN_PASS_CONFIDENCE", "70"))),
)


def _story_quality_text(article):
    paragraphs = article.get("paragraphs", [])
    if not isinstance(paragraphs, list):
        return ""
    return "\n\n".join(
        str(p).strip() for p in paragraphs if str(p).strip()
    )


def _story_quality_evidence_summary(evidence):
    """Keep the quality-gate context compact while preserving story identity."""
    if not isinstance(evidence, dict):
        return ""

    facts = evidence.get("facts", [])
    if not isinstance(facts, list):
        facts = []

    lines = []
    for idx, fact in enumerate(facts[:12], 1):
        if isinstance(fact, dict):
            # Evidence schemas can evolve; keep only useful semantic fields.
            parts = []
            for key in ("fact", "claim", "statement", "text", "event", "entity"):
                value = str(fact.get(key, "")).strip()
                if value:
                    parts.append(value)
            if parts:
                lines.append(f"FACT {idx}: " + " | ".join(dict.fromkeys(parts)))
        elif str(fact).strip():
            lines.append(f"FACT {idx}: {str(fact).strip()}")

    if not lines:
        return "No structured locked facts were available."

    return "\n".join(lines)


def _story_quality_source_titles(trend):
    titles = []
    for item in trend.get("news", []) or []:
        title = str(item.get("title", "")).strip()
        if title:
            titles.append(title)
        if len(titles) >= 8:
            break
    return "\n".join(f"- {title}" for title in titles) or "- No source headlines available."


def _deterministic_story_quality_flags(article):
    """
    Conservative local red flags for obvious roundup/list behavior.

    These are NOT used as automatic rejection rules because normal news
    language can contain similar words. They are passed to the editorial
    model as warning signals so the model can make the final decision.
    """
    text = _story_quality_text(article).casefold()
    flags = []

    roundup_patterns = [
        # English
        r"\bother (?:major|key|notable|important) (?:news|stories|developments)\b",
        r"\bother stories\b",
        r"\bhere are (?:the|some) (?:latest|top|key|major)\b",
        r"\bmeanwhile\b",
        r"\bin other news\b",
        r"\balso in (?:sports|business|technology|entertainment|news)\b",
        r"\bseveral (?:other|major|key) (?:events|developments|stories)\b",
        r"\btop \d+\b",
        r"\b\d+ things to know\b",
        r"\bwhat you need to know\b",

        # Indonesian
        r"\bberita (?:lain|terkini|utama)\b",
        r"\bberita lainnya\b",
        r"\bsementara itu\b",
        r"\bdi sisi lain\b",
        r"\bselain itu\b",
        r"\bbeberapa (?:berita|peristiwa|perkembangan)\b",
        r"\bberikut (?:berita|hal|informasi)\b",
        r"\b\d+ hal yang perlu diketahui\b",
        r"\bapa yang perlu diketahui\b",

        # Spanish
        r"\botras (?:noticias|historias|novedades)\b",
        r"\bmientras tanto\b",
        r"\bpor otro lado\b",
        r"\ben otras noticias\b",
        r"\bvarias (?:noticias|historias|novedades)\b",
        r"\b\d+ cosas que debes saber\b",

        # French
        r"\bd'autres (?:actualités|nouvelles|informations)\b",
        r"\bpendant ce temps\b",
        r"\bdans d'autres actualités\b",
        r"\bplusieurs (?:actualités|nouvelles|événements)\b",
        r"\bà savoir\b",

        # German
        r"\bweitere (?:nachrichten|meldungen|entwicklungen)\b",
        r"\bindessen\b",
        r"\bin anderen nachrichten\b",
        r"\bmehrere (?:nachrichten|ereignisse|entwicklungen)\b",
        r"\bwas sie wissen müssen\b",

        # Italian
        r"\baltre (?:notizie|storie|novità)\b",
        r"\bnel frattempo\b",
        r"\bin altre notizie\b",
        r"\bdiverse (?:notizie|storie|novità)\b",
        r"\bcose da sapere\b",

        # Portuguese
        r"\boutros (?:notícias|casos|acontecimentos)\b",
        r"\benquanto isso\b",
        r"\bem outras notícias\b",
        r"\bvárias (?:notícias|histórias|atualizações)\b",
        r"\bo que você precisa saber\b",
    ]

    for pattern in roundup_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            flags.append(pattern)

    # Repeated topic shifts are a useful warning, but not a hard rule.
    transition_hits = len(re.findall(
        r"\b(?:"
        r"meanwhile|separately|in a separate development|elsewhere|on another front|"
        r"sementara itu|di sisi lain|selain itu|"
        r"mientras tanto|por otro lado|"
        r"pendant ce temps|d'un autre côté|"
        r"indessen|andererseits|"
        r"nel frattempo|d'altra parte|"
        r"enquanto isso|por outro lado"
        r")\b",
        text,
        flags=re.IGNORECASE,
    ))
    if transition_hits >= 2:
        flags.append(f"multiple_story_transition_markers:{transition_hits}")

    return flags


def story_quality_gate(article, trend):
    """
    Final editorial quality decision for the generated article.

    Returns the article only on PASS. REMOVE and BORDERLINE raise an
    exception, preventing save_article() and therefore preventing any
    downstream image generation for the rejected candidate.
    """
    validate_article(article)

    topic = str(trend.get("title", "")).strip()
    article_title = str(article.get("title", "")).strip()
    article_description = str(article.get("description", "")).strip()
    article_h1 = str(article.get("h1", "")).strip()
    article_body = _story_quality_text(article)
    evidence = trend.get("_evidence_lock", {})

    local_flags = _deterministic_story_quality_flags(article)
    local_flag_text = (
        "\n".join(f"- {flag}" for flag in local_flags)
        if local_flags
        else "- None detected"
    )

    prompt = f"""
You are the FINAL EDITORIAL STORY QUALITY GATE for a professional {LANGUAGE} news site.

Your task is NOT to fact-check the article and NOT to improve or rewrite it.
Fact Guard has already handled factual validation.

Your ONLY job is to determine whether the generated article is ONE clearly
recognizable, coherent news story suitable for publication as a single article.

MAIN SELECTED TOPIC:
{topic}

ARTICLE TITLE:
{article_title}

ARTICLE H1:
{article_h1}

ARTICLE DESCRIPTION:
{article_description}

GENERATED ARTICLE BODY:
{article_body}

LOCKED EVIDENCE / STORY FACTS:
{_story_quality_evidence_summary(evidence)}

SOURCE HEADLINES FOR CONTEXT ONLY:
{_story_quality_source_titles(trend)}

DETERMINISTIC WARNING SIGNALS:
{local_flag_text}

EDITORIAL STANDARD:

PASS when:
- The article clearly centers on ONE identifiable event, development, person,
  decision, announcement, incident, match, transfer, release, or other single
  news story.
- Every paragraph materially belongs to that same story.
- Relevant background/context about the same story is allowed.
- A short article is completely acceptable if it cleanly covers one story.
- A small amount of closely related context does NOT make it a roundup.
- A second paragraph can explain consequences, reactions, history or context
  when those details are directly connected to the same main story.

REMOVE when:
- It is a roundup or digest of multiple independent news stories.
- It combines several unrelated people, events, topics, announcements or
  developments under one article.
- It is a finance/investment roundup rather than one specific financial story.
- It is a sports roundup rather than one specific sports story.
- It is a generic "latest news" / "AI news" / topic roundup containing
  unrelated developments.
- It begins with one story but then changes into a different independent story.
- It contains a list of separate stories disguised as one article.
- The article's identity is broad enough that there is no single central event
  or development.

BORDERLINE when:
- It is genuinely unclear whether the article is one story or multiple stories.
- The article has a central story but a substantial portion shifts into
  independent developments that cannot reasonably be treated as context.
- The editorial decision itself is low-confidence.

IMPORTANT:
- DO NOT reject an article merely because it is short.
- DO NOT reject an article merely because it has multiple paragraphs.
- DO NOT reject relevant consequences, reactions, background or context.
- DO NOT use article length, source count or number of facts as the decision.
- Judge STORY IDENTITY and COHERENCE.
- Do not infer missing facts.
- Do not reward an article simply because every sentence is individually factual.
- A factual roundup is still REMOVE.
- A concise single-story article is PASS.

DECISION PRIORITY:
1. Is there one unmistakable central story?
2. Do the paragraphs remain about that story?
3. Are additional details genuinely related context rather than independent news?
4. Only then consider whether there is a clear reason for REMOVE/BORDERLINE.

Return ONLY valid JSON:
{{
  "verdict": "PASS",
  "confidence": 0,
  "reason": ""
}}

Allowed verdict values: PASS, REMOVE, BORDERLINE.
confidence must be an integer from 0 to 100.
reason must be a concise editorial explanation, maximum 35 words.
"""

    print("[STORY QUALITY] Checking final article...")

    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0,
            "top_p": 0.85,
            "top_k": 40,
            "num_ctx": STORY_QUALITY_CTX,
            "num_predict": STORY_QUALITY_TOKENS,
            "num_batch": STORY_QUALITY_BATCH,
            "num_thread": STORY_QUALITY_THREADS,
        },
        format={
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["PASS", "REMOVE", "BORDERLINE"],
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                },
                "reason": {"type": "string"},
            },
            "required": ["verdict", "confidence", "reason"],
        },
    )

    raw = response.message.content or ""
    try:
        result = json.loads(raw)
    except Exception as exc:
        raise Exception(f"Story Quality Gate returned invalid JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise Exception("Story Quality Gate did not return a JSON object")

    verdict = str(result.get("verdict", "")).strip().upper()
    try:
        confidence = int(result.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0
    reason = " ".join(str(result.get("reason", "")).split()).strip()

    if verdict not in {"PASS", "REMOVE", "BORDERLINE"}:
        raise Exception(f"Story Quality Gate returned invalid verdict: {verdict!r}")

    confidence = max(0, min(100, confidence))

    # A PASS with low confidence is not safe enough for production.
    # Treat it as BORDERLINE so uncertain articles never reach persistence
    # or any downstream image-generation pipeline.
    if verdict == "PASS" and confidence < STORY_QUALITY_MIN_PASS_CONFIDENCE:
        verdict = "BORDERLINE"
        reason = (
            f"Low editorial confidence ({confidence} < "
            f"{STORY_QUALITY_MIN_PASS_CONFIDENCE})"
            + (f": {reason}" if reason else "")
        )

    print(
        f"[STORY QUALITY] {verdict} | confidence={confidence} "
        f"| {reason or 'No reason supplied'}"
    )

    if verdict != "PASS":
        raise Exception(
            f"Story Quality Gate {verdict.lower()} "
            f"(confidence={confidence}): {reason or 'article is not one coherent story'}"
        )

    return article



def _enrich_evidence_for_generation(evidence, trend):
    """
    Add deterministic topic context to the already locked evidence.

    The article generator receives the evidence object, not the original trend prompt.
    Keeping the exact selected topic inside that object reduces cross-story contamination
    when Google News returns multiple related pages.
    """
    if not isinstance(evidence, dict):
        raise Exception("Evidence lock is not an object.")

    enriched = dict(evidence)
    topic = str(trend.get("title", "")).strip()

    if topic:
        enriched["selected_topic"] = topic

    # IMPORTANT: Source/publisher attribution is not part of the article narrative.
    # The model must synthesize the locked facts into a continuous news story.
    # Source names/headlines may be used internally for provenance, but must not
    # become "X reported..." / "Y said..." prose unless the attribution itself
    # is an essential verified fact (for example, an official statement).
    enriched["editorial_generation_policy"] = (
        "NARRATIVE SYNTHESIS: Write the article as an original, continuous news "
        "story from the locked verified facts. Do not organize paragraphs by source. "
        "Do not mention publisher names, websites, source headlines, or phrases such "
        "as 'reported by', 'according to [publisher]', 'X has reported', 'Y published', "
        "or similar source-digest wording. State the verified information directly. "
        "Only retain attribution when who made the statement or which official body "
        "confirmed it is itself an essential part of the verified fact. Never use a "
        "source headline to add specificity that the locked evidence does not support."
    )

    # IMPORTANT: Do not pass publisher/source headlines into article generation.
    # Headlines are discovery metadata and can contain claims that are stronger,
    # newer, or more specific than the source body. The locked evidence facts are
    # the sole factual authority for generation.
    # Evidence-density article length policy. When the locked evidence contains
    # only one verified fact, the article must stay naturally concise rather
    # than expanding a single fact into repetitive 150-200 word coverage.
    # This is a generation instruction only; it never changes or weakens the
    # underlying evidence lock or Fact Guard source.
    facts = enriched.get("facts", [])
    if isinstance(facts, list) and len(facts) == 1:
        enriched["article_length_policy"] = (
            "SINGLE-FACT EVIDENCE: Write a naturally short article. "
            "Use only the single verified fact available. Do not pad, repeat, "
            "generalize, speculate, or manufacture context to reach a word count. "
            "The article may be "
            "substantially shorter than a typical article. "
            "Completeness and factual precision take priority over length."
        )

    return enriched


def generate_valid_article(prompt, fact_guard_source, reference_date, trend, max_attempts=1, prelocked_evidence=None):
    last = None

    for i in range(max_attempts):
        try:
            generation_evidence = _enrich_evidence_for_generation(
                prelocked_evidence,
                trend,
            )
            article = generate(prompt, evidence=generation_evidence)
            validate_article(article)

            locked_facts = generation_evidence.get("facts", [])
            paragraph_text = " ".join(
                str(p) for p in article.get("paragraphs", [])
            ).strip()
            if isinstance(locked_facts, list) and len(locked_facts) >= 3:
                print(
                    f"[EVIDENCE COVERAGE] locked_facts={len(locked_facts)} "
                    f"| article_words={len(paragraph_text.split())}"
                )
            article = enforce_headline_policy(article, trend)
            validate_article(article)

            print("[FACT GUARD] Checking generated article...")
            guard = fact_guard_validate(
                fact_guard_source,
                article,
                reference_date=reference_date,
            )

            if guard["status"] != "PASS":
                print("[FACT GUARD] FLAG - article requires repair.")
                print(json.dumps(guard, ensure_ascii=False, indent=2))

                try:
                    print("[FACT GUARD REPAIR] Attempting targeted repair v1.0...")
                    repaired = fact_guard_repair(
                        article,
                        fact_guard_source,
                        guard,
                    )
                    validate_article(repaired)
                    repaired = enforce_headline_policy(repaired, trend)
                    validate_article(repaired)

                    print("[FACT GUARD REPAIR] Re-checking repaired article...")
                    repaired_guard = fact_guard_validate(
                        fact_guard_source,
                        repaired,
                        reference_date=reference_date,
                    )

                    if repaired_guard["status"] != "PASS":
                        print("[FACT GUARD REPAIR] FAIL - repaired article blocked.")
                        print(
                            json.dumps(
                                repaired_guard,
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                        raise Exception(
                            "Fact Guard repair failed re-validation "
                            f"({repaired_guard['blocking_issues']} blocking issue(s))"
                        )

                    print("[FACT GUARD REPAIR] PASS - repaired article accepted.")

                    if repaired_guard.get("review_items", 0):
                        print(
                            f"[FACT GUARD] PASS with "
                            f"{repaired_guard['review_items']} review item(s)."
                        )
                    else:
                        print("[FACT GUARD] PASS")

                    return repaired

                except Exception as repair_error:
                    raise Exception(
                        f"Fact Guard blocked article; repair failed: {repair_error}"
                    ) from repair_error

            if guard.get("review_items", 0):
                print(
                    f"[FACT GUARD] PASS with "
                    f"{guard['review_items']} review item(s)."
                )
            else:
                print("[FACT GUARD] PASS")

            return article

        except Exception as e:
            last = e
            print(f"Validation failed ({i+1}/{max_attempts}): {e}")

            # A failed Fact Guard repair is terminal for this candidate.
            # Re-generating from the same evidence only repeats the expensive
            # audit/repair cycle instead of improving the underlying condition.
            if "Fact Guard blocked article; repair failed:" in str(e):
                break

    raise Exception(last)


def run_git(cmd):
    print("\n" + "=" * 60)
    print("Running:", " ".join(cmd))
    print("=" * 60)

    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.stdout:
        print(r.stdout)

    if r.stderr:
        print(r.stderr)

    return r.returncode == 0


def git_push():
    if not run_git(["git", "status"]):
        return

    if not run_git(["git", "add", "."]):
        return

    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print("No changes to commit.")
        return

    if not run_git(["git", "commit", "-m", "Auto update"]):
        return

    if not run_git(["git", "push", "origin", "main"]):
        return

    print("SUCCESS: GitHub updated.")


def main():
    processed = load_processed()
    trends = fetch_trends()
    print(f"[TOPIC FILTER] Raw trends received: {len(trends)}")

    # One explicit reference date for the entire production run.
    # This is the date against which event state is evaluated.
    reference_date = date.today()
    print(f"[FACT GUARD] Validation reference date: {reference_date.isoformat()}")

    generated = 0
    new_keywords = []

    # ============================================================
    # TOPIC PRIORITY FILTER
    # ============================================================
    # IMPORTANT: article generation must NEVER run directly over the raw
    # trend feed. We first rank the entire trend set, then retrieve news only
    # for the small candidate pool, then generate only the final winners.
    #
    # This is deliberately outside Fact Guard. Fact Guard protects factual
    # correctness AFTER a topic has been selected; it does not decide which
    # topics deserve production capacity.
    # ============================================================
    # Keep a wider cheap candidate pool than the final article capacity.
    # This allows the evidence-usability gate to reject a top-ranked topic
    # and fall through to the next-ranked candidate instead of ending the
    # production run with an unused article slot.
    candidate_pool_size = max(
        MAX_ARTICLES_PER_RUN * 4,
        MAX_ARTICLES_PER_RUN + 3,
    )

    candidate_trends = rank_trends(
        trends,
        processed,
        limit=candidate_pool_size,
    )

    source_scored_candidates = []

    for trend in candidate_trends:
        keyword = trend["title"]

        try:
            print(
                f"\n[TOPIC FILTER] Retrieving sources for candidate: "
                f"{keyword}"
            )
            # Discovery candidates already come from fresh Google News RSS.
            # Re-query using the category discovery query rather than the full
            # publisher headline; the latter is often too specific and can
            # return zero results. The discovered item is retained as seed
            # evidence so the exact fresh story is not lost.
            discovery_query = trend.get("discovery_query")
            seed = None
            if discovery_query:
                # The discovery headline is already a fresh story. Use it as
                # seed evidence and make a second, entity-focused search for
                # corroboration. Never discard the seed just because Google
                # News cannot find the publisher headline again.
                seed = dict(trend)
                if not seed.get("content") and seed.get("link"):
                    try:
                        seed["content"] = extract_article(seed["link"])
                    except Exception:
                        seed["content"] = ""
                entity_query = _build_related_story_query(keyword)
                if not entity_query:
                    entity_query = keyword.split(" - ")[0].strip()

                # The discovery category is only a discovery hint. Never append it
                # to the corroboration query: doing so can misclassify a local
                # Indonesian artist as K-pop simply because the same headline was
                # returned by a K-pop discovery query. Entity/event terms are safer.
                search_query = entity_query

                print(f"[TOPIC FILTER] Related story search: {search_query}")
                news = fetch_news(search_query)
                # If the seed page extracted very little text, borrow the RSS
                # summary/content from the closest matching search result. This
                # prevents a short publisher extraction from becoming the only
                # evidence source when Google News already has the same story.
                seed_title_norm = _normalise_headline(seed.get("title", ""))
                if len(str(seed.get("content", "")).strip()) < 900 and news:
                    best = None
                    best_overlap = 0
                    seed_words = {w for w in re.findall(r"[a-z0-9]+", seed_title_norm.casefold()) if len(w) >= 4}
                    for candidate in news:
                        cand_norm = _normalise_headline(candidate.get("title", ""))
                        cand_words = {w for w in re.findall(r"[a-z0-9]+", cand_norm.casefold()) if len(w) >= 4}
                        overlap = len(seed_words & cand_words)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best = candidate
                    if best is not None and best_overlap >= 2:
                        if not seed.get("content") and best.get("content"):
                            seed["content"] = best.get("content")
                        if best.get("summary"):
                            seed["summary"] = best.get("summary")

                if seed.get("title") and not any(
                    str(x.get("title", "")).strip().casefold() == str(seed.get("title", "")).strip().casefold()
                    for x in news
                ):
                    news.insert(0, seed)
            else:
                news = []
                for search_query in _build_targeted_news_queries(keyword):
                    print(f"[TOPIC FILTER] Related story search: {search_query}")
                    found = fetch_news(search_query)
                    if found:
                        news.extend(found)
                    if filter_relevant_news(trend, news):
                        break

            if not news:
                print(f"[TOPIC FILTER] DROP AFTER NEWS | {keyword} | no usable news result(s)")
                continue

            # --------------------------------------------------------
            # TOPIC / NEWS RELEVANCE GATE
            # --------------------------------------------------------
            # Do not let corroboration for a different entity/region enter
            # Story Concentration or source scoring.
            relevant_news = filter_relevant_news(trend, news)
            if not relevant_news:
                print(
                    f"[TOPIC FILTER] DROP AFTER NEWS | {keyword} | "
                    f"no topic-relevant news result(s)"
                )
                continue
            news = relevant_news

            # --------------------------------------------------------
            # STORY SOURCE DECISION
            # --------------------------------------------------------
            # This runs before Ollama evidence extraction.
            # Its only job is to detect a source pool that contains multiple
            # independent stories under a broad/generic retrieval query.
            try:
                story_selection = _decide_story_sources(news, keyword)
                trend["_story_selection"] = story_selection
            except Exception as concentration_error:
                trend["_production_status"] = "REJECT"
                trend["_production_reject_reason"] = str(concentration_error)
                print(
                    f"[TOPIC FILTER] REJECT BEFORE EVIDENCE | article_slot=0 | "
                    f"{keyword} | {concentration_error}"
                )
                continue

            trend["news"] = news
            topic_final = score_with_news(trend, news)
            trend["_topic_final"] = topic_final
            source_scored_candidates.append(trend)

            print(
                f"[TOPIC FILTER] SOURCE SCORE | "
                f"{topic_final['final_score']:.1f} | {keyword}"
            )

        except Exception as e:
            print(f"[TOPIC FILTER] ERROR: {keyword}: {e}")

    # Evidence usability is part of production candidate selection.
    # Keep the full source-scored ranking available so a top topic that has
    # unusable evidence does not consume the article slot.
    evidence_candidate_pool = select_final_candidates(
        source_scored_candidates,
        limit=len(source_scored_candidates),
    )

    for trend in evidence_candidate_pool:
        if generated >= MAX_ARTICLES_PER_RUN:
            break
        keyword = trend["title"]
        news = trend["news"]
        story_selection = trend.get("_story_selection")

        print(
            f"\n[GENERATION] Selected topic | "
            f"score={trend['_topic_final']['final_score']:.1f} | {keyword}"
        )

        try:
            generation_prompt = build_prompt(trend)
            fact_guard_source = _build_fact_guard_source(news)

            print(
                f"[FACT GUARD] Source prepared | "
                f"news_items={len(news)} | "
                f"source_chars={len(fact_guard_source)}"
            )

            # --------------------------------------------------------
            # EVIDENCE USABILITY GATE
            # --------------------------------------------------------
            # Use the exact same source-grounded extractor as the universal
            # fact-lock pipeline. If this candidate cannot produce at least
            # one provenance-verified fact, skip it and try the next-ranked
            # candidate instead of consuming the article slot.
            print(
                f"[TOPIC FILTER] EVIDENCE USABILITY CHECK | {keyword}"
            )

            try:
                evidence_source = _build_evidence_source(news, story_selection)
                print(
                    f"[TOPIC FILTER] Evidence source prepared | "
                    f"source_chars={len(evidence_source)}"
                )
                evidence_lock = extract_evidence(evidence_source)
                evidence_lock = _enrich_evidence_for_generation(
                    evidence_lock,
                    trend,
                )
                trend["_evidence_lock"] = evidence_lock
                print(
                    f"[TOPIC FILTER] EVIDENCE USABILITY PASS | "
                    f"facts={len(evidence_lock.get('facts', []))} | {keyword}"
                )
            except Exception as evidence_error:
                # Terminal candidate rejection: this topic has no usable
                # source-locked evidence. It consumes ZERO article slots and
                # the pipeline immediately falls through to the next ranked
                # candidate. Never retry an already-rejected evidence
                # candidate or add another repair layer here.
                trend["_production_status"] = "REJECT"
                trend["_production_reject_reason"] = str(evidence_error)
                print(
                    f"[TOPIC FILTER] REJECT | article_slot=0 | "
                    f"{keyword} | evidence={evidence_error}"
                )
                continue

            article = generate_valid_article(
                generation_prompt,
                fact_guard_source,
                reference_date,
                trend,
                max_attempts=1,
                prelocked_evidence=evidence_lock,
            )

            # --------------------------------------------------------
            # FINAL ARTICLE STORY QUALITY GATE
            # --------------------------------------------------------
            # This is intentionally the last gate before persistence.
            # If the article is a roundup, mixed story, generic topic digest,
            # or otherwise editorially incoherent, it consumes ZERO production
            # article slots and NEVER reaches save_article() / image generation.
            article = story_quality_gate(article, trend)

            slug = slugify(keyword)
            article["slug"] = slug

            save_article(slug, render_article(article, news=news))

            new_keywords.append(keyword)
            generated += 1

            print(f"OK -> {slug}.html")

        except Exception as e:
            # Any candidate that fails after evidence lock is terminal for this
            # run. Do not retry the same expensive candidate; move immediately
            # to the next ranked topic. The article counter remains unchanged.
            # This also covers Story Quality Gate REMOVE/BORDERLINE decisions.
            trend["_production_status"] = "REJECT"
            trend["_production_reject_reason"] = str(e)
            print(
                f"[GENERATION] REJECT | article_slot=0 | "
                f"{keyword} | {e}"
            )

    try:
        update_all()

        for k in new_keywords:
            add_processed(k, processed)

    except Exception as e:
        print("UPDATE ERROR:", e)

    print(f"Finished. Generated {generated} article(s).")

    if generated:
        git_push()


if __name__ == "__main__":
    main()
