import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import re
import os
import difflib
import subprocess
from datetime import date

from config import MAX_ARTICLES_PER_RUN, LANGUAGE, SOURCE_FIRST, TREND_DIR
from rss import fetch_trends
from rss_source_discovery import fetch_source_stories
from news import fetch_news, fetch_news_discovery, hydrate_story_sources, hydrate_news_items
from prompt import build_prompt
from ollama_client import generate, extract_evidence, validate_article_structure, substantive_story_value_gate
from ollama import chat
from config import MODEL

# Minimum confidence required before rescuing a safe semantic sub-cluster.
# Keep this conservative: semantic rescue must still have >=2 explicit sources.
SEMANTIC_RESCUE_MIN_CONFIDENCE = 85
import json
import unicodedata
from urllib.parse import urljoin, urlparse, parse_qs
from html_generator import render_article, save_article
from processed import load_processed, add_processed
from index_generator import update_all
from topic_scorer import filter_relevant_news, _is_sports_match_topic, _skip_reason, _norm, _clean
import generator_monitor as monitor

REQUIRED_FIELDS = ["title", "description", "h1", "paragraphs"]


def _strip_markdown_formatting(value):
    """Remove Markdown emphasis markers from model output before HTML rendering.

    The article renderer expects plain text/HTML, not Markdown. A model may still
    return emphasis such as *text* or **text**, which would otherwise be exposed
    literally in the published page as stray asterisks.
    """
    text = str(value or "")
    # Handle strongest emphasis first so the inner passes cannot leave markers.
    text = re.sub(r"\*\*\*(.*?)\*\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*(?!\*)", r"\1", text, flags=re.DOTALL)
    return text


def _sanitize_article_markdown(article):
    """Normalize model output that is intended to be rendered as HTML."""
    if not isinstance(article, dict):
        return article

    cleaned = dict(article)
    for field in ("title", "description", "h1"):
        if field in cleaned:
            cleaned[field] = _strip_markdown_formatting(cleaned[field])

    paragraphs = cleaned.get("paragraphs")
    if isinstance(paragraphs, list):
        cleaned["paragraphs"] = [
            _strip_markdown_formatting(paragraph)
            for paragraph in paragraphs
        ]

    return cleaned


def _article_text(article):
    if not isinstance(article, dict):
        return ""
    parts = [
        article.get("title", ""),
        article.get("description", ""),
        article.get("h1", ""),
    ]
    paragraphs = article.get("paragraphs", [])
    if isinstance(paragraphs, list):
        parts.extend(paragraphs)
    return " ".join(str(value) for value in parts if value).strip()


def validate_language_integrity(article):
    """Fail closed on obvious language/script leakage or prompt/instruction output."""
    text = _article_text(article)
    if not text:
        raise ValueError("Language integrity check failed: empty article text.")

    # Current TrendCurrent production languages are Latin-script except Bulgarian.
    # Detecting foreign scripts is deterministic and directly catches the Ceuta-type
    # Chinese leakage without asking another model to judge its own output.
    language = str(LANGUAGE or "").strip().casefold()
    cyrillic_allowed = language in {"bulgarian", "bg", "български"}
    latin_languages = {
        "english", "en", "german", "de", "deutsch", "french", "fr", "français",
        "italian", "it", "italiano", "spanish", "es", "español",
        "indonesian", "id", "bahasa indonesia",
    }

    forbidden_scripts = []
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "CJK UNIFIED IDEOGRAPH" in name or "HIRAGANA" in name or "KATAKANA" in name or "HANGUL" in name:
            forbidden_scripts.append("CJK/East Asian")
        elif "ARABIC" in name:
            forbidden_scripts.append("Arabic")
        elif "HEBREW" in name:
            forbidden_scripts.append("Hebrew")
        elif "DEVANAGARI" in name:
            forbidden_scripts.append("Devanagari")
        elif "THAI" in name:
            forbidden_scripts.append("Thai")
        elif "CYRILLIC" in name and language in latin_languages:
            forbidden_scripts.append("Cyrillic")

    # Obvious generator/instruction leakage is never valid article prose.
    lower = text.casefold()
    leakage_markers = (
        "return only the required json",
        "locked evidence:",
        "final entitlement check",
        "you are a professional",
        "write a clear trendcurrent news article",
        "article generator",
        "do not pad, speculate, manufacture context",
    )
    matched = [marker for marker in leakage_markers if marker in lower]

    if forbidden_scripts:
        scripts = ", ".join(sorted(set(forbidden_scripts)))
        raise ValueError(f"Language integrity check failed: forbidden script detected ({scripts}).")
    if matched:
        raise ValueError("Language integrity check failed: generator/instruction leakage detected.")

    # For known non-Bulgarian production languages, a tiny Latin share is expected
    # for names/official terms, so this gate intentionally does not require a
    # particular percentage of Latin letters. It only blocks clearly foreign scripts.
    if not cyrillic_allowed and language not in latin_languages:
        # Unknown future language: keep the deterministic leakage markers above,
        # but do not guess its valid writing system.
        return True

    return True

# Article length is determined by the amount of usable verified evidence.
# There is no artificial word-count target or evidence-count-based minimum.
# Evidence with 1-2 facts is allowed to reach the substantive story-value gate;
# that gate remains responsible for deciding whether the evidence can support
# a meaningful standalone article.

# Evidence sufficiency is a hard pre-generation eligibility gate.
# Count only provenance-verified, deduplicated information units; syndicated
# duplicates must not inflate the minimum. Candidates with fewer than 3
# genuinely distinct usable facts are rejected before substantive-value judging
# and before the expensive article-generation pipeline.
EVIDENCE_MIN_FACTS_FOR_GENERATION = max(
    3,
    int(os.getenv("EVIDENCE_MIN_FACTS_FOR_GENERATION", "3")),
)

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

# Sports exclusion: this generator is not a sports-news publisher.
# Match/fixture filtering alone is insufficient because athlete, tournament,
# league and motorsport trends can still enter the production reservoir.
SPORTS_TOPIC_TERMS = (
    "tennis", "football", "soccer", "basketball", "baseball", "hockey",
    "golf", "cricket", "rugby", "boxing", "ufc", "mma", "wrestling",
    "motogp", "nascar", "formula 1", "formula one", "f1", "grand prix",
    "premier league", "champions league", "europa league", "bundesliga",
    "la liga", "ligue 1", "serie a", "mlb", "nfl", "nba", "nhl",
    "fifa", "uefa", "atp", "wta", "us open", "wimbledon", "olympics",
    "olympic", "world cup", "tournament", "championship", "playoff",
    "matchday", "fixture", "kickoff", "kick-off", "marathon", "cycling",
    "cyclist", "swimming", "gymnastics", "volleyball", "weightlifting",
    "track and field", "athletics", "esports",
)
SPORTS_SOURCE_TERMS = (
    "espn", "sky sports", "sports illustrated", "sportskeeda", "sporting news",
    "the athletic", "bbc sport", "cbssports", "yahoo sports", "eurosport",
    "formula1.com", "formula1", "motorsport", "tennis.com", "atptour",
    "wtatennis", "mlb.com", "nfl.com", "nba.com", "nhl.com", "fifa.com",
    "uefa.com", "golf.com", "golf monthly", "cricbuzz",
)

def _sports_term_in_text(text):
    """Match sports vocabulary as whole words/phrases, not arbitrary substrings."""
    text = " ".join(str(text or "").split()).casefold()
    if not text:
        return False
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text)
        for term in SPORTS_TOPIC_TERMS
    )


def _sports_source_in_item(item):
    """Match a known sports publisher/domain only in source/link fields."""
    if not isinstance(item, dict):
        return False
    source_text = " ".join(
        str(item.get(key, "") or "")
        for key in ("source", "url", "link")
    ).casefold()
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", source_text)
        for term in SPORTS_SOURCE_TERMS
    )


def _is_sports_topic(title, news=None):
    """
    Return True only for a strong sports signal.

    The title itself is authoritative for explicit sports topics. When news is
    supplied, it must already be topic-relevant; a single unrelated sports
    result must never veto a legitimate non-sports topic.
    """
    if _sports_term_in_text(title):
        return True

    items = [item for item in (news or ()) if isinstance(item, dict)]
    if not items:
        return False

    sports_signals = 0
    strong_source_signals = 0

    for item in items:
        title_sports = _sports_term_in_text(item.get("title", ""))
        summary_sports = _sports_term_in_text(item.get("summary", ""))
        source_sports = _sports_source_in_item(item)

        if source_sports:
            strong_source_signals += 1
        if title_sports:
            sports_signals += 2
        elif summary_sports:
            sports_signals += 1

    if len(items) == 1:
        return strong_source_signals >= 1 or sports_signals >= 2

    return (
        strong_source_signals >= 2
        or sports_signals >= 3
        or (strong_source_signals >= 1 and sports_signals >= 2)
    )


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _normalize_existing_story_title(text):
    """Normalize a headline for the pre-generation existing-story check."""
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    # Publisher suffixes are not part of the underlying story identity.
    text = re.sub(
        r"\s*[-|–—]\s*(?:bbc(?: news)?|reuters|associated press|ap|cnn|"
        r"the guardian|nytimes|new york times|sky news|abc news|cbs news|"
        r"nbc news|fox news)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_EXISTING_STORY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "amid", "after", "before", "during",
    "with", "without", "from", "into", "over", "under", "for", "of", "to", "in",
    "on", "at", "as", "by", "is", "are", "was", "were", "has", "have", "had",
    "new", "latest", "news", "report", "reports", "update", "updates",
}


def _story_identity_tokens(text):
    normalized = _normalize_existing_story_title(text)
    return {
        token for token in normalized.split()
        if len(token) >= 3 and token not in _EXISTING_STORY_STOPWORDS
    }


def _existing_story_title_match(candidate_title, existing_title):
    """Return conservative deterministic evidence that two titles cover one story.

    This is intentionally stricter than topical similarity. It is only a
    pre-generation guard: false positives are more harmful than allowing a
    borderline story through to later quality gates.
    """
    a = _normalize_existing_story_title(candidate_title)
    b = _normalize_existing_story_title(existing_title)
    if not a or not b:
        return False, 0.0, 0, 0.0

    if a == b:
        return True, 1.0, len(_story_identity_tokens(a)), 1.0

    sequence_ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta = _story_identity_tokens(a)
    tb = _story_identity_tokens(b)
    common = ta & tb
    min_coverage = (
        len(common) / min(len(ta), len(tb))
        if ta and tb else 0.0
    )

    # Exact/near-identical headline.
    if sequence_ratio >= 0.82:
        return True, sequence_ratio, len(common), min_coverage

    # Strong shared story vocabulary. Requiring >=4 non-generic tokens and
    # >=50% coverage avoids rejecting merely related stories.
    if len(common) >= 4 and min_coverage >= 0.50:
        return True, sequence_ratio, len(common), min_coverage

    # Three shared tokens can still be enough when the headlines are clearly
    # paraphrases rather than merely sharing a broad topic.
    if len(common) >= 3 and sequence_ratio >= 0.68 and min_coverage >= 0.50:
        return True, sequence_ratio, len(common), min_coverage

    return False, sequence_ratio, len(common), min_coverage


def _extract_existing_article_titles():
    """Read already-published article titles from this language's trends directory.

    Only local published HTML is inspected. No network call and no LLM call are
    used here. Malformed/unreadable files are ignored so one bad article cannot
    stop the production run.
    """
    titles = []
    try:
        paths = sorted(TREND_DIR.glob("*.html"))
    except Exception as exc:
        print(f"[EXISTING STORY] inventory unavailable: {exc}")
        return titles

    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        found = []
        for pattern in (
            r"<title[^>]*>(.*?)</title>",
            r"<h1[^>]*>(.*?)</h1>",
            r'"headline"\s*:\s*"([^"]+)"',
        ):
            for match in re.findall(pattern, raw, flags=re.IGNORECASE | re.DOTALL):
                value = re.sub(r"<[^>]+>", " ", match)
                value = re.sub(r"\s+", " ", value).strip()
                if value:
                    found.append(value)

        # One title is enough per article. Prefer <title>, then h1/JSON-LD.
        title = next((value for value in found if value), "")
        if title:
            titles.append({
                "path": str(path),
                "title": title,
            })

    return titles


def _existing_story_check(topic_title, news, story_selection):
    """Check whether the concrete story is already represented by a published article.

    The check runs before publisher hydration, evidence extraction and article
    generation. It uses the canonical trend title plus the selected source
    headlines, because a publisher headline often names the same event more
    precisely than the discovery seed.
    """
    existing = _extract_existing_article_titles()
    if not existing:
        return None

    selected_indices = story_selection.get("selected_indices", []) if isinstance(story_selection, dict) else []
    candidate_titles = [str(topic_title or "").strip()]

    for index in selected_indices:
        try:
            item = news[int(index)]
        except (IndexError, TypeError, ValueError):
            continue
        if not isinstance(item, dict):
            continue
        for key in ("title",):
            value = str(item.get(key, "") or "").strip()
            if value:
                candidate_titles.append(value)

    # De-duplicate candidate headlines before comparing.
    candidate_titles = list(dict.fromkeys(candidate_titles))

    best = None
    for candidate_title in candidate_titles:
        for item in existing:
            matched, ratio, common_count, coverage = _existing_story_title_match(
                candidate_title,
                item["title"],
            )
            if not matched:
                continue

            score = (
                1.0 if _normalize_existing_story_title(candidate_title)
                == _normalize_existing_story_title(item["title"])
                else (ratio + min(coverage, 1.0) * 0.20 + min(common_count, 6) * 0.03)
            )
            if best is None or score > best["score"]:
                best = {
                    "candidate_title": candidate_title,
                    "existing_title": item["title"],
                    "path": item["path"],
                    "ratio": ratio,
                    "common_tokens": common_count,
                    "coverage": coverage,
                    "score": score,
                }

    if best:
        print(
            f"[EXISTING STORY] REJECT | already_covered=true | "
            f"existing={best['existing_title']} | "
            f"candidate={best['candidate_title']} | "
            f"similarity={best['ratio']:.3f} | "
            f"common_tokens={best['common_tokens']} | "
            f"coverage={best['coverage']:.3f} | "
            f"file={best['path']}"
        )
        return best

    print(
        f"[EXISTING STORY] PASS | published_articles={len(existing)} | "
        f"candidate_headlines={len(candidate_titles)}"
    )
    return None


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


class StorySourceUnavailable(Exception):
    """Semantic Story Source Judge could not complete after bounded retry."""
    pass


def _semantic_story_concentration_judge(news, topic):
    """Fallback semantic splitter for genuinely ambiguous source pools.

    Deterministic clustering is attempted first. This LLM call is deliberately
    single-shot and compact; it must never be the normal path for an obvious
    cluster.
    """
    items = list(news or [])
    lines = []
    for idx, item in enumerate(items[:12], 1):
        title = str(item.get("title", "")).strip()
        summary = re.sub(r"\s+", " ", str(item.get("summary", "")).strip())[:180]
        if title:
            lines.append(f"{idx}. {title} | {summary}")
    if not lines:
        return {"status": "MIXED", "confidence": 0, "reason": "no input", "source_numbers": []}

    prompt = f"""
Identify one safe concrete story cluster in this source pool.

Return ONLY JSON:
{{"verdict":"ONE_STORY|DOMINANT_STORY|MIXED","confidence":0-100,"source_numbers":[1,2],"reason":"brief"}}

Rules:
- Same concrete event counts as one story, even with different wording.
- Shared person/company/topic alone is not enough.
- DOMINANT_STORY requires at least 2 sources clearly describing the same event.
- MIXED means no safe cluster can be isolated.
- Never invent a connection.

TOPIC: {str(topic or "").strip()}
SOURCES:
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
                "num_ctx": max(4096, int(os.getenv("OLLAMA_NUM_CTX", "4096"))),
                "num_predict": 96,
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
        confidence = max(0, min(100, int(result.get("confidence", 0) or 0)))
        reason = str(result.get("reason", "")).strip()[:300]
        raw_numbers = result.get("source_numbers", [])
        source_numbers = []
        if isinstance(raw_numbers, list):
            for value in raw_numbers:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= number <= len(items):
                    source_numbers.append(number)
        source_numbers = list(dict.fromkeys(source_numbers))
        status = verdict if verdict in {"ONE_STORY", "DOMINANT_STORY", "MIXED"} else "MIXED"
        print(
            f"[TOPIC FILTER] STORY SOURCE JUDGE | {status} | "
            f"confidence={confidence} | elapsed={elapsed:.2f}s"
        )
        return {
            "status": status,
            "confidence": confidence,
            "source_numbers": source_numbers,
            "reason": reason,
        }
    except Exception as exc:
        raise StorySourceUnavailable(str(exc)) from exc
def _source_independence_text(item):
    """Return publisher text used only for conservative syndication detection."""
    if not isinstance(item, dict):
        return ""
    title = str(item.get("title", "") or "").strip()
    summary = str(item.get("summary", "") or "").strip()
    content = str(item.get("content", "") or "").strip()
    # Keep the publisher body as the primary signal. RSS summary is useful when
    # extraction is short, but should never dominate a long article body.
    body = content if len(content) >= 500 else summary
    text = " ".join([title, summary, body]).strip()
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text)
    return text


def _source_shingles(text, size=5):
    """Conservative word shingles; useful for detecting copied/syndicated text."""
    words = re.findall(r"[a-z0-9À-ÿ']+", unicodedata.normalize("NFKD", text))
    if len(words) < size:
        return set()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def _source_pair_similarity(a, b):
    """
    Return deterministic similarity signals for source independence.

    This is intentionally much stricter than Story Concentration:
    two sources reporting the same event are NOT duplicates merely because they
    share entities, dates, names or a few facts. We require substantial textual
    overlap before treating one as a syndicated/reprinted copy.
    """
    ta = _source_independence_text(a)
    tb = _source_independence_text(b)
    if not ta or not tb:
        return 0.0, 0.0, 0.0

    title_a = " ".join(str(a.get("title", "") or "").split()).casefold()
    title_b = " ".join(str(b.get("title", "") or "").split()).casefold()
    title_ratio = difflib.SequenceMatcher(None, title_a, title_b).ratio()

    sa = _source_shingles(ta)
    sb = _source_shingles(tb)
    shingle_jaccard = (
        len(sa & sb) / len(sa | sb)
        if sa and sb else 0.0
    )

    # Character-level similarity is a secondary safety signal. It is useful for
    # small rewrites but cannot independently trigger deduplication.
    char_ratio = difflib.SequenceMatcher(
        None,
        ta[:6000],
        tb[:6000],
    ).ratio()

    return title_ratio, shingle_jaccard, char_ratio


def _deduplicate_syndicated_sources(news):
    """
    Collapse only high-confidence syndicated/reprinted copies.

    Critical distinction:
      6 sources about one event != 6 independent sources.
      6 copies of one report = 1 independent source family.

    Independent reporting is preserved unless there is strong textual evidence
    that two publisher articles are substantially the same underlying copy.
    This layer is language-agnostic at the pipeline level and deliberately does
    not use source-name allow/deny lists.
    """
    items = list(news or [])
    n = len(items)
    if n < 2:
        return {
            "items": items,
            "kept_indices": list(range(n)),
            "duplicate_indices": [],
            "families": [[i] for i in range(n)],
        }

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_debug = []
    for i in range(n):
        for j in range(i + 1, n):
            title_ratio, shingle_jaccard, char_ratio = _source_pair_similarity(items[i], items[j])

            # Very high title + body overlap is strong evidence of a copied story.
            # Body shingle overlap is the principal signal because independent
            # reporting of the same event normally uses materially different prose.
            high_copy = (
                shingle_jaccard >= 0.62
                and char_ratio >= 0.72
            )
            near_verbatim = (
                shingle_jaccard >= 0.48
                and char_ratio >= 0.82
                and title_ratio >= 0.72
            )
            short_copy = (
                title_ratio >= 0.88
                and shingle_jaccard >= 0.38
                and char_ratio >= 0.78
            )

            if high_copy or near_verbatim or short_copy:
                union(i, j)
                pair_debug.append({
                    "pair": [i, j],
                    "title": round(title_ratio, 3),
                    "shingle": round(shingle_jaccard, 3),
                    "char": round(char_ratio, 3),
                })

    families = {}
    for idx in range(n):
        families.setdefault(find(idx), []).append(idx)
    families = [sorted(v) for v in families.values()]
    families.sort(key=lambda family: family[0])

    duplicate_indices = []
    kept_indices = []
    for family in families:
        # Preserve the first source. The discovery seed is intentionally inserted
        # first, so the concrete fresh story is never displaced by a later copy.
        kept_indices.append(family[0])
        duplicate_indices.extend(family[1:])

    print(
        f"[SOURCE INDEPENDENCE] input={n} | "
        f"independent_families={len(families)} | "
        f"duplicates_removed={len(duplicate_indices)}"
    )
    if duplicate_indices:
        print(
            f"[SOURCE INDEPENDENCE] duplicate_indices="
            f"{','.join(str(i + 1) for i in duplicate_indices)}"
        )
        for pair in pair_debug[:12]:
            print(
                f"[SOURCE INDEPENDENCE] pair={pair['pair'][0] + 1},{pair['pair'][1] + 1} "
                f"title={pair['title']:.3f} shingle={pair['shingle']:.3f} "
                f"char={pair['char']:.3f}"
            )

    return {
        "items": [items[i] for i in kept_indices],
        "kept_indices": kept_indices,
        "duplicate_indices": duplicate_indices,
        "families": families,
        "pair_debug": pair_debug,
    }


def _discover_concrete_story_candidates(news, topic):
    """Identify concrete stories using deterministic clustering first.

    Semantic selection is a fallback only when lexical clustering cannot safely
    isolate a corroborated story. This keeps LLM inference off the common path.
    """
    original_items = list(news or [])
    if not original_items:
        return []

    independence = _deduplicate_syndicated_sources(original_items)
    items = list(independence.get("items") or [])
    if not items:
        return []

    kept_original = list(independence.get("kept_indices", []))
    remaining = list(range(len(items)))
    candidates = []
    seen = set()

    def add_candidate(selected, semantic_status="NOT_REQUIRED"):
        selected = sorted(dict.fromkeys(i for i in selected if 0 <= i < len(items)))
        if not selected:
            return
        original_indices = sorted(dict.fromkeys(
            kept_original[i] for i in selected if 0 <= i < len(kept_original)
        ))
        if not original_indices or tuple(original_indices) in seen:
            return
        seen.add(tuple(original_indices))
        candidates.append({
            "status": "PASS",
            "reason": "concrete story cluster",
            "count": len(original_items),
            "selected_indices": original_indices,
            "selected_count": len(original_indices),
            "semantic_status": semantic_status,
            "original_source_count": len(original_items),
            "independent_source_count": len(items),
            "duplicates_removed": len(independence.get("duplicate_indices", [])),
            "independence_families": independence.get("families", []),
            "component_index": len(candidates),
        })

    if len(items) < 3:
        add_candidate(remaining)
    else:
        profile = _story_pool_profile([items[i] for i in remaining], topic)
        dominant = list(profile.get("cluster") or [])
        if profile.get("status") == "PASS" and len(dominant) >= 2:
            add_candidate(dominant, "DETERMINISTIC_DOMINANT_STORY")
            remaining = [i for i in remaining if i not in set(dominant)]
        else:
            try:
                judge = _semantic_story_concentration_judge(
                    [items[i] for i in remaining], topic
                )
                status = str(judge.get("status", "MIXED")).upper()
                nums = judge.get("source_numbers") or []
                selected = [
                    remaining[n - 1] for n in nums
                    if 1 <= n <= len(remaining)
                ]
                confidence = int(judge.get("confidence", 0) or 0)

                if status in {"ONE_STORY", "PASS"} and confidence >= 70:
                    add_candidate(remaining, status)
                    remaining = []
                elif status == "DOMINANT_STORY" and confidence >= 70 and len(selected) >= 2:
                    add_candidate(selected, status)
                    remaining = [i for i in remaining if i not in set(selected)]
                elif (
                    status == "MIXED"
                    and confidence >= SEMANTIC_RESCUE_MIN_CONFIDENCE
                    and len(selected) >= 2
                ):
                    print(
                        f"[STORY DISCOVERY] SEMANTIC SUB-CLUSTER | "
                        f"sources={len(selected)} | confidence={confidence}"
                    )
                    add_candidate(selected, "MIXED_SEMANTIC_SUBCLUSTER")
                    remaining = [i for i in remaining if i not in set(selected)]
            except Exception as exc:
                print(f"[STORY DISCOVERY] semantic splitter unavailable | {exc}")

    if remaining:
        residual = [items[i] for i in remaining]
        profile = _story_pool_profile(residual, topic)
        components = list(profile.get("components") or [])
        if len(residual) < 3:
            components = [list(range(len(residual)))]
        for component in components:
            selected = [remaining[i] for i in component if 0 <= i < len(remaining)]
            if len(selected) < 2:
                print(
                    f"[STORY DISCOVERY] DROP residual singleton | "
                    f"sources={len(selected)} | topic={topic}"
                )
                continue
            add_candidate(selected, "MIXED_RESIDUAL")

    print(
        f"[STORY DISCOVERY] topic={topic} | sources={len(original_items)} | "
        f"independent={len(items)} | story_candidates={len(candidates)}"
    )
    for idx, candidate in enumerate(candidates, 1):
        print(
            f"[STORY DISCOVERY] candidate={idx}/{len(candidates)} | "
            f"sources={candidate['selected_count']} | "
            f"semantic={candidate['semantic_status']} | topic={topic}"
        )
    return candidates

def _deterministic_production_reservoir(trends, processed):
    """Return every deterministic-eligible trend without ranking or scoring."""
    processed_norm = {_norm(x) for x in processed}
    seen_titles = set()
    eligible = []

    for trend in trends or []:
        if not isinstance(trend, dict):
            continue

        title = _clean(trend.get("title"))
        if not title:
            continue

        norm_title = _norm(title)
        if norm_title in processed_norm or norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)

        reason = _skip_reason(title)
        if reason:
            print(f"[TOPIC FILTER] DROP | {title} | {reason}")
            continue

        item = dict(trend)
        item["title"] = title
        eligible.append(item)

    print(
        f"[TOPIC FILTER] PRODUCTION RESERVOIR | "
        f"raw={len(list(trends or []))} | deterministic_eligible={len(eligible)} | "
        f"selection=scoring_disabled"
    )
    return eligible


def _filter_trend_discovery_news(trend, news):
    """Keep discovery results relevant to either the canonical trend or a
    Google-Trends related-news headline, without weakening the existing
    deterministic relevance rules.
    """
    canonical = filter_relevant_news(trend, news)
    if canonical:
        return canonical

    recovered = []
    seen = set()
    for related in (trend.get("news") or []):
        if not isinstance(related, dict):
            continue
        headline = str(related.get("title", "") or "").strip()
        if not headline:
            continue
        probe = {"title": headline}
        for item in filter_relevant_news(probe, news):
            key = (str(item.get("url", "") or "").strip().casefold(),
                   str(item.get("title", "") or "").strip().casefold())
            if key not in seen:
                seen.add(key)
                recovered.append(item)

    print(
        f"[TOPIC FILTER] Related-news relevance recovery | "
        f"canonical={len(canonical)} | recovered={len(recovered)}"
    )
    return recovered


def _build_trend_discovery_queries(trend, keyword, max_queries=3):
    """Build concrete discovery queries from Google Trends related-news context.

    The original Trends title remains the canonical topic. Related-news
    headlines are used only as concrete search signals for the existing news
    discovery mechanism. No new quality gate is introduced here.
    """
    queries = []
    seen = set()

    def add(value):
        value = str(value or "").strip()
        if not value:
            return
        key = _norm(value)
        if not key or key in seen:
            return
        seen.add(key)
        queries.append(value)

    # Prefer concrete publisher headlines supplied by Google Trends.
    for item in (trend.get("news") or []):
        if not isinstance(item, dict):
            continue
        headline = str(item.get("title", "") or "").strip()
        if not headline:
            continue
        query = _build_related_story_query(headline)
        if query:
            add(query)
        if len(queries) >= max_queries:
            break

    # Always retain the original trend as a fallback/anchor.
    if not queries:
        add(keyword)

    return queries[:max_queries]


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
        description = str(item.get("description", "")).strip()
        source = str(item.get("source", "")).strip()
        published = str(item.get("published", "")).strip()
        content = str(item.get("content", "")).strip()
        fallback_text = " ".join(x for x in (summary, description) if x).strip()
        evidence_text = content if len(content) >= 300 else fallback_text
        compact_sources.append({
            "title": title,
            "source": source,
            "published": published,
            "summary": summary,
            "description": description,
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


def _selected_story_news(news, story_selection):
    """Return exactly the sources selected by Story Source Decision."""
    items = list(news or [])
    selection = story_selection or {}
    indices = list(selection.get("selected_indices") or [])
    selected = [items[i] for i in indices if 0 <= i < len(items)]
    if not selected:
        raise Exception("Evidence source received no sources from story selection")
    return selected



def _is_valid_publisher_image_url(url):
    """Accept only plausible publisher image URLs; reject media/placeholders/thumbnails."""
    value = str(url or "").strip()
    if not value or not re.match(r"^https?://", value, re.I):
        return False
    lower = value.casefold()
    blocked = (
        "placeholder", "place-holder", "default-image", "default_image",
        "no-image", "no_image", "spacer.gif", "transparent.gif",
        "video", ".mp4", ".webm", ".m3u8", ".mp3", ".wav", ".aac",
        "favicon", "/favicon", "sprite", "tracking", "pixel",
        "googlelogo", "googleusercontent", "gstatic.com/images/branding",
    )
    if any(token in lower for token in blocked):
        return False

    # Reject obvious thumbnail-sized variants such as ?w=96 / ?width=120.
    try:
        query = parse_qs(urlparse(value).query)
        for key in ("w", "width", "h", "height", "size"):
            for raw in query.get(key, []):
                m = re.search(r"\d+", str(raw))
                if m and int(m.group()) <= 160:
                    return False
    except Exception:
        pass

    path = lower.split("?", 1)[0].split("#", 1)[0]
    return bool(re.search(r"\.(?:jpg|jpeg|png|webp|avif)(?:$|/)", path)) or any(
        token in lower for token in ("/image/", "/images/", "/photo/", "/photos/", "/media/")
    )


def _extract_jsonld_image_url(html, base_url=""):
    """Return the best article image from JSON-LD image.url only.

    Article/NewsArticle/BlogPosting/etc. images are preferred. Person,
    Organization, Brand, ImageObject and Logo nodes are never treated as
    article images. If JSON-LD has no usable image, fall back to standard
    publisher social metadata (og:image/twitter:image), then a sufficiently
    large content <img>.
    """
    article_types = {"article", "newsarticle", "blogposting", "techarticle", "report"}
    excluded_types = {"person", "organization", "brand", "imageobject", "logo", "website"}
    primary = []
    generic = []

    def node_types(node):
        raw = node.get("@type") if isinstance(node, dict) else None
        values = raw if isinstance(raw, list) else [raw]
        return {str(v).split("/")[-1].casefold() for v in values if v}

    def add_image_from_node(node, bucket):
        if not isinstance(node, dict):
            return
        image = node.get("image")
        image_items = image if isinstance(image, list) else [image]
        for item in image_items:
            if not isinstance(item, dict):
                continue
            # Deliberately ONLY image.url. Never contentUrl/thumbnailUrl.
            image_url = item.get("url")
            if isinstance(image_url, str):
                absolute = urljoin(base_url, image_url.strip()) if base_url else image_url.strip()
                if _is_valid_publisher_image_url(absolute):
                    bucket.append(absolute)

    for block in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html or "", flags=re.I | re.S,
    ):
        raw = re.sub(r"^\s*<!--|-->\s*$", "", block.strip()).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                types = node_types(node)
                if types & excluded_types:
                    pass
                elif types & article_types:
                    add_image_from_node(node, primary)
                else:
                    # WebPage and unknown containers are only generic fallback.
                    add_image_from_node(node, generic)
                for key, value in node.items():
                    if key == "image":
                        continue
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)

    seen = set()
    for url in primary + generic:
        if url not in seen:
            seen.add(url)
            return url

    # Publisher-standard fallback when JSON-LD does not expose an image.url.
    meta_patterns = (
        r'<meta[^>]+(?:property|name)=["\']og:image["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+(?:property|name)=["\']og:image:url["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+(?:property|name)=["\']og:image:secure_url["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+(?:property|name)=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)',
    )
    for pattern in meta_patterns:
        for match in re.finditer(pattern, html or "", flags=re.I):
            candidate = urljoin(base_url, match.group(1).strip()) if base_url else match.group(1).strip()
            if _is_valid_publisher_image_url(candidate):
                return candidate

    # Last fallback: real content <img> with explicit reasonable dimensions.
    for tag in re.findall(r'<img\b[^>]*>', html or "", flags=re.I):
        src_match = re.search(r'\b(?:src|data-src|data-lazy-src|data-original)=["\']([^"\']+)', tag, flags=re.I)
        if not src_match:
            continue
        candidate = urljoin(base_url, src_match.group(1).strip()) if base_url else src_match.group(1).strip()
        if not _is_valid_publisher_image_url(candidate):
            continue
        dims = []
        for attr in ("width", "height"):
            m = re.search(rf'\b{attr}=["\'](\d+)', tag, flags=re.I)
            dims.append(int(m.group(1)) if m else None)
        if dims[0] and dims[1] and (dims[0] < 500 or dims[1] < 250):
            continue
        return candidate

    return None


def _ensure_publisher_images(selected_news):
    """Second-pass publisher image extraction after article quality gates."""
    import urllib.request

    items = list(selected_news or [])
    found = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        existing = str(item.get("image") or item.get("image_url") or "").strip()
        if existing and _is_valid_publisher_image_url(existing):
            item["image"] = existing
            found += 1
            continue

        url = str(item.get("url") or item.get("link") or "").strip()
        if not url:
            continue
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (TrendCurrent publisher image resolver)"},
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                html = response.read(5000000).decode("utf-8", errors="replace")
            image_url = _extract_jsonld_image_url(html, base_url=url)
            if image_url:
                item["image"] = image_url
                found += 1
                print(f"[SOURCE IMAGE] FOUND: {image_url} | source=publisher-image-resolver")
            else:
                print(f"[SOURCE IMAGE] NONE: {url} | jsonld.image.url not found")
        except Exception as exc:
            print(f"[SOURCE IMAGE] ERROR: {url} | {exc}")

    print(f"[SOURCE IMAGE] FINAL | publisher_images={found} | sources={len(items)}")
    return items



def _ensure_rendered_publisher_image(rendered_html, article, selected_news):
    """Guarantee the verified publisher image reaches final HTML presentation and stays responsive."""
    html = str(rendered_html or "")
    if not html:
        return html

    # The renderer's image markup is intentionally presentation-only.
    # Keep the image inside the article width on desktop and mobile; otherwise
    # a native publisher image (often 1200-1920px wide) can overflow the page.
    image_css = (
        '<style id="tc-responsive-article-image">'
        '.tc-article-image{width:100%;max-width:100%;margin:0 0 32px;overflow:hidden;}'
        '.tc-article-image__frame{width:100%;max-width:100%;overflow:hidden;}'
        '.tc-article-image img{display:block;width:100%;max-width:100%;height:auto;object-fit:cover;}'
        '@media(max-width:768px){.tc-article-image{margin-bottom:24px;}'
        '.tc-article-image img{width:100%;max-width:100%;height:auto;}}'
        '</style>'
    )
    if 'id="tc-responsive-article-image"' not in html:
        head = re.search(r'</head\s*>', html, flags=re.IGNORECASE)
        if head:
            html = html[:head.start()] + image_css + html[head.start():]
        else:
            html = image_css + html

    # If the renderer already supplied an image, keep it and only repair its
    # responsive presentation.
    if re.search(r"<img\b[^>]+src\s*=", html, flags=re.IGNORECASE):
        return html

    image_url = ""
    for item in selected_news or []:
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("image") or item.get("image_url") or "").strip()
        if _is_valid_publisher_image_url(candidate):
            image_url = candidate
            break
    if not image_url:
        return html

    from html import escape
    title = str((article or {}).get("title") or (article or {}).get("h1") or "").strip()
    figure = (
        '<figure class="tc-article-image"><div class="tc-article-image__frame">'
        f'<img src="{escape(image_url, quote=True)}" alt="{escape(title, quote=True)}" '
        'loading="eager" decoding="async" fetchpriority="high">'
        '</div></figure>'
    )
    for pattern in (r'(<article\b[^>]*>)', r'(<main\b[^>]*>)', r'(<body\b[^>]*>)'):
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            pos = match.end()
            return html[:pos] + figure + html[pos:]
    return html

def _shorten_headline(title):
    title = " ".join(str(title or "").split()).strip()
    if not title:
        return title
    if len(title) <= 65 and len(title.split()) <= 10:
        return title

    candidates = []
    for sep in (" — ", " – ", " - ", ": ", "; ", ", "):
        if sep in title:
            candidates.extend(part.strip() for part in title.split(sep) if part.strip())
    valid = [c for c in candidates if len(c) <= 65 and len(c.split()) <= 10]
    if valid:
        return max(valid, key=lambda x: (len(x), len(x.split())))

    words = title.split()
    kept = []
    for word in words:
        candidate = " ".join(kept + [word])
        if len(candidate) > 65 or len(kept) + 1 > 10:
            break
        kept.append(word)
    return " ".join(kept).rstrip(" ,:;–—-")

HEADLINE_MAX_WORDS = 10
HEADLINE_MAX_CHARS = 65


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
        if source:
            source = source.strip(" .,:;\u2013\u2014")
            if tail == source or tail.endswith(source) or source.endswith(tail):
                return tail
    if re.search(r"\.(?:co|com|id|net|org)(?:\.[a-z]{2,3})?$", tail):
        return tail
    return None


def _headline_violations(article, trend):
    title = " ".join(str(article.get("title", "")).split()).strip()
    h1 = " ".join(str(article.get("h1", "")).split()).strip()
    source_titles = [str(x.get("title", "")).strip() for x in trend.get("news", [])]
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
    if title_norm and any(title_norm == _normalise_headline(x) for x in source_titles if x):
        violations.append("title copies source headline")
    h1_norm = _normalise_headline(h1)
    if h1_norm and any(h1_norm == _normalise_headline(x) for x in source_titles if x):
        violations.append("h1 copies source headline")
    if title_norm and h1_norm and title_norm != h1_norm:
        violations.append("title and h1 differ")
    return violations


def enforce_headline_policy(article, trend):
    """Deterministic headline policy only; never call the LLM for repair."""
    title = str(article.get("title", "")).strip()
    h1 = str(article.get("h1", "")).strip()
    if not title or title != h1:
        raise ValueError("Headline policy failed: title and H1 must be identical and non-empty.")
    words = title.split()
    if len(words) > 10 or len(title) > 65:
        raise ValueError("Headline policy failed: title exceeds 10 words or 65 characters.")
    return article

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
    #
    return enriched

def _run_repetition_guard(article, generation_evidence, event_name="repetition_guard"):
    """Cheap deterministic post-generation repetition gate.

    This gate is deterministic and blocks only obvious paragraph duplication.
    It never calls Ollama and never repairs or regenerates the article.
    """
    import re

    paragraphs = article.get("paragraphs", []) if isinstance(article, dict) else []
    paragraphs = [str(p).strip() for p in paragraphs if str(p).strip()]
    paragraph_count = len(paragraphs)

    def words(text):
        return [w for w in re.findall(r"[\w’'-]+", text.casefold()) if len(w) > 2]

    def shingles(tokens, n=5):
        return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

    repeated_pairs = []
    redundant = 0
    for i in range(paragraph_count):
        a = words(paragraphs[i])
        sa = shingles(a)
        for j in range(i + 1, paragraph_count):
            b = words(paragraphs[j])
            if not a or not b:
                continue
            sb = shingles(b)
            if sa and sb:
                inter = len(sa & sb)
                union = len(sa | sb)
                similarity = inter / union if union else 0.0
            else:
                aset, bset = set(a), set(b)
                similarity = len(aset & bset) / max(1, min(len(aset), len(bset)))

            # Very high phrase overlap means the later paragraph is almost certainly
            # restating the same prose. This intentionally does not try to judge
            # semantic paraphrase; false positives would be worse than missed nuances.
            if similarity >= 0.72:
                repeated_pairs.append([i + 1, j + 1])

    redundant = len({pair[1] for pair in repeated_pairs})
    status = "REJECT" if repeated_pairs else "PASS"
    reason = (
        "obvious paragraph duplication detected" if repeated_pairs
        else "no obvious paragraph duplication"
    )

    repetition = {
        "status": status,
        "reason": reason,
        "paragraph_count": paragraph_count,
        "paragraphs_with_new_information": paragraph_count - redundant,
        "redundant_paragraphs": redundant,
        "repeated_information_units": len(repeated_pairs),
        "repeated_pairs": repeated_pairs,
        "unique_information_units": max(0, paragraph_count - redundant),
        "information_density": round((paragraph_count - redundant) / paragraph_count, 2) if paragraph_count else 0.0,
    }

    print(
        f"[REPETITION GUARD] {status} | deterministic | "
        f"paragraphs={paragraph_count} | repeated_pairs={len(repeated_pairs)}"
    )
    monitor.candidate_event(event_name, **repetition)
    return repetition



# ============================================================
# Deterministic Fact Consistency Guard
# ============================================================

_FACT_GUARD_STOPWORDS = {
    "the","and","for","with","from","that","this","was","were","has","have","had",
    "are","is","its","into","after","before","over","under","about","than","then",
    "they","their","them","there","which","while","also","been","being","will",
    "would","could","should","said","says","according","official","officials",
    "new","latest","news","report","reports","story","article","podcast",
}

def _fc_tokens(value):
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return [
        token for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 3 and token not in _FACT_GUARD_STOPWORDS
    ]

def _fc_numeric_tokens(value):
    return set(re.findall(
        r"\b\d+(?:[.,]\d+)?%?\b|\b(?:19|20)\d{2}\b",
        str(value or ""),
    ))

def generate_valid_article(prompt, reference_date, trend, prelocked_evidence=None):
    """Generate once and publish after deterministic/language/repetition validation.

    No post-generation LLM factual validation, repair, or regeneration occurs.
    """
    try:
        generation_evidence = _enrich_evidence_for_generation(prelocked_evidence, trend)
        article = generate(prompt, evidence=generation_evidence)
        # The HTML renderer does not parse Markdown. Strip model-emitted
        # emphasis markers before any validation/rendering so literal "*" and
        # "**" cannot leak into titles, metadata, or article paragraphs.
        article = _sanitize_article_markdown(article)
        validate_article(article)
        # Paragraph count is intentionally unrestricted; this gate checks only usable structure.
        validate_article_structure(article, generation_evidence, label="Initial newsroom article")
        locked_facts = generation_evidence.get("facts", [])
        core_fact_ids = generation_evidence.get("core_fact_ids", [])
        supporting_fact_ids = generation_evidence.get("supporting_fact_ids", [])
        paragraph_text = " ".join(str(p) for p in article.get("paragraphs", [])).strip()
        if isinstance(locked_facts, list) and len(locked_facts) >= 1:
            print(f"[EVIDENCE COVERAGE] locked_facts={len(locked_facts)} | core_facts={len(core_fact_ids)} | supporting_facts={len(supporting_fact_ids)} | article_words={len(paragraph_text.split())}")
        print("[FACT CONSISTENCY GUARD] DISABLED — publication not blocked by deterministic fact-expression guard")
        normalized_headline = _shorten_headline(article.get("title", ""))
        article["title"] = normalized_headline; article["h1"] = normalized_headline
        article = enforce_headline_policy(article, trend)
        validate_article(article); validate_language_integrity(article)
        print("[LANGUAGE GUARD] PASS")
        repetition = _run_repetition_guard(article, generation_evidence)
        if repetition.get("status") != "PASS":
            print("[REPETITION GUARD] FAIL — article discarded; NO REPAIR")
            print(json.dumps(repetition, ensure_ascii=False, indent=2))
            raise Exception("Repetition Guard blocked article; publication blocked.")
        validate_language_integrity(article)
        print("[LANGUAGE GUARD] FINAL ARTICLE PASS")
        return article
    except Exception as e:
        print(f"Validation failed: {e}")
        raise
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
    monitor.start_run(language=LANGUAGE, model=MODEL, pipeline="universal-fact-lock-v2.9.2-ministral-newsroom-no-word-floor", max_articles=MAX_ARTICLES_PER_RUN)
    processed = load_processed()

    # Direct publisher RSS is the permanent discovery root.
    source_first = SOURCE_FIRST

    if source_first:
        trends = fetch_source_stories(language=LANGUAGE)
        print(
            f"[SOURCE FIRST] Direct publisher RSS enabled | "
            f"language={LANGUAGE} | seeds={len(trends)}"
        )

        # Direct publisher RSS is preferred, but it must never be a single
        # point of failure for discovery. If the configured publisher has no
        # fresh item, immediately fall back to the existing Google News ->
        # Bing discovery mechanism. Freshness, deduplication and all downstream
        if not trends:
            fallback_queries = {
                "en": ["latest news", "breaking news", "top news"],
                "en-us": ["latest news", "breaking news", "top news"],
                "de": ["aktuelle nachrichten", "eilmeldungen", "top nachrichten"],
                "es": ["últimas noticias", "última hora", "principales noticias"],
                "it": ["ultime notizie", "ultim'ora", "principali notizie"],
                "fr": ["dernières nouvelles", "dernière minute", "actualités principales"],
                "pt": ["últimas notícias", "última hora", "principais notícias"],
                "pt-br": ["últimas notícias", "última hora", "principais notícias"],
                "id": ["berita terbaru", "berita terkini", "berita utama"],
            }.get(str(LANGUAGE or "").strip().casefold(), ["latest news", "breaking news", "top news"])

            fallback = fetch_news_discovery(
                fallback_queries,
                per_query_limit=8,
                max_results=12,
            )
            trends = []
            for item in fallback:
                seed = dict(item)
                seed["discovery_provider"] = "google_news_bing_fallback"
                seed["discovery_source"] = str(item.get("source") or "").strip()
                seed["discovery_context"] = "broad_news_fallback"
                trends.append(seed)

            print(
                f"[SOURCE FIRST] Direct RSS empty -> broad news fallback | "
                f"queries={len(fallback_queries)} | seeds={len(trends)}"
            )
    else:
        trends = fetch_trends()
        print(f"[TOPIC FILTER] Raw trends received: {len(trends)}")

    # One explicit reference date for the entire production run.
    # This is the date against which event state is evaluated.
    reference_date = date.today()
    print(f"[RUN] Reference date: {reference_date.isoformat()}")

    generated = 0
    new_keywords = []

    # ============================================================
    # DISCOVERY -> CONCRETE STORIES
    # ============================================================
    # Trends are discovery signals only. They do not represent the production
    # article candidate by themselves. There is no score/ranking admission
    # path and no target-relative reservoir. Every deterministic-eligible trend
    # can be explored until a real story passes the unchanged quality gates.
    candidate_trends = _deterministic_production_reservoir(
        trends,
        processed,
    )

    print(
        f"[STORY DISCOVERY] seeds={len(candidate_trends)} | "
        f"article_target={MAX_ARTICLES_PER_RUN}"
    )

    for trend in candidate_trends:
        if generated >= MAX_ARTICLES_PER_RUN:
            break

        keyword = trend["title"]

        # Cheap deterministic sports exclusion MUST happen before any news
        # retrieval. Sports-only trends must not consume RSS/network capacity.
        if _is_sports_topic(keyword):
            print(f"[TrendCurrent] SKIP sports topic: {keyword}")
            continue

        try:
            print(
                f"\n[STORY DISCOVERY] Retrieving sources for discovery seed: "
                f"{keyword}"
            )

            discovery_context = str(trend.get("discovery_context", "") or "").strip()
            seed = None

            if source_first:
                # Direct RSS items remain preferred discovery seeds. Broad-news
                # fallback items are also valid concrete seeds and are expanded
                # through the same corroboration path.
                seed = dict(trend)
                news = [seed]

                related_query = _build_related_story_query(keyword) or keyword
                if trend.get("discovery_context") == "broad_news_fallback":
                    print(f"[SOURCE FIRST] Fallback corroboration search: {related_query}")
                else:
                    print(f"[SOURCE FIRST] Corroboration search: {related_query}")
                found = fetch_news(related_query)

                # A broad-news fallback seed is only a discovery lead. If the
                # corroboration lookup returns nothing (including a network/API
                # failure), do not let the singleton seed bypass the concrete-story
                # evidence standard. Direct publisher seeds retain their existing
                # behavior.
                if (
                    trend.get("discovery_context") == "broad_news_fallback"
                    and not (found or [])
                ):
                    print(
                        f"[STORY DISCOVERY] REJECT singleton fallback seed | "
                        f"{keyword} | corroboration=0"
                    )
                    continue

                seen = {
                    (
                        str(item.get("url", "") or item.get("link", "")).strip().casefold(),
                        str(item.get("title", "") or "").strip().casefold(),
                    )
                    for item in news
                }
                for item in found or []:
                    key = (
                        str(item.get("url", "") or item.get("link", "")).strip().casefold(),
                        str(item.get("title", "") or "").strip().casefold(),
                    )
                    if key not in seen:
                        seen.add(key)
                        news.append(item)

                # RSS discovery is already language/source-specific. Do not run the
                # Trends relevance gate against a direct publisher seed.
            elif discovery_context:
                # Google Trends related-news headlines are concrete story signals.
                # Query them through the EXISTING discovery mechanism, while the
                # original Trends title remains the canonical topic for relevance,
                # evidence and generation.
                discovery_queries = _build_trend_discovery_queries(
                    trend,
                    keyword,
                    max_queries=3,
                )
                news = []

                for entity_query in discovery_queries:
                    print(f"[TOPIC FILTER] Related story search: {entity_query}")
                    found = fetch_news(entity_query)
                    if found:
                        news.extend(found)

                # Discovery remains metadata-only. Full article extraction is deferred
                # until a concrete story candidate has been selected.
            else:
                news = []
                for search_query in _build_targeted_news_queries(keyword):
                    print(f"[TOPIC FILTER] Related story search: {search_query}")
                    found = fetch_news(search_query)
                    if found:
                        news.extend(found)
                    # Keep the targeted-query ladder cheap: stop once a query
                    # produces a relevant pool, but carry that already-filtered
                    # pool forward so the relevance gate is not executed twice.
                    if found:
                        candidate_relevant = filter_relevant_news(trend, news)
                        if candidate_relevant:
                            news = candidate_relevant
                            break

            if not news:
                print(
                    f"[TOPIC FILTER] DROP AFTER NEWS | {keyword} | "
                    f"no usable news result(s)"
                )
                continue

            # Apply the existing Trends relevance gate only to Trends discovery.
            # Direct publisher RSS seeds are already the canonical discovery source.
            if discovery_context and not source_first:
                news = _filter_trend_discovery_news(trend, news)

            if not news:
                print(
                    f"[TOPIC FILTER] DROP AFTER NEWS | {keyword} | "
                    f"no topic-relevant news result(s)"
                )
                continue

            if _is_sports_topic(keyword, news):
                print(f"[TrendCurrent] SKIP sports topic after relevance: {keyword}")
                continue

            story_candidates = _discover_concrete_story_candidates(news, keyword)
            if not story_candidates:
                print(
                    f"[STORY DISCOVERY] REJECT | {keyword} | "
                    f"no concrete story candidate"
                )
                continue

            # Each concrete story is evaluated immediately through the existing
            # story never rejects the remaining stories from this topic.
            for story_number, story_selection in enumerate(story_candidates, 1):
                if generated >= MAX_ARTICLES_PER_RUN:
                    break

                trend["_story_selection"] = story_selection
                trend["news"] = news
                trend["_story_number"] = story_number
                trend["_story_candidate_count"] = len(story_candidates)

                monitor.start_candidate(
                    keyword,
                    trend=trend,
                )

                print(
                    f"\n[GENERATION] Concrete story candidate "
                    f"{story_number}/{len(story_candidates)} | {keyword}"
                )

                try:
                    existing_story = _existing_story_check(
                        keyword,
                        news,
                        story_selection,
                    )
                    if existing_story:
                        trend["_production_status"] = "REJECT"
                        trend["_production_reject_reason"] = "existing story already covered"
                        monitor.candidate_event(
                            "existing_story_check",
                            status="REJECT",
                            reason="already_covered",
                            existing_title=existing_story.get("existing_title"),
                            existing_file=existing_story.get("path"),
                            candidate_title=existing_story.get("candidate_title"),
                            similarity=existing_story.get("ratio"),
                            common_tokens=existing_story.get("common_tokens"),
                            coverage=existing_story.get("coverage"),
                        )
                        monitor.finish_candidate(
                            "REJECT",
                            reason="existing story already covered",
                        )
                        continue

                    monitor.candidate_event(
                        "existing_story_check",
                        status="PASS",
                    )

                    generation_prompt = build_prompt(trend)

                    # STAGED SOURCE ACQUISITION:
                    # story discovery uses RSS metadata only; full publisher extraction
                    # is performed only for the already selected concrete story sources.
                    #
                    # A MIXED_RESIDUAL candidate must contain at least two sources.
                    # Singleton residuals are filtered during discovery; this second
                    # invariant protects the downstream boundary if candidate data
                    # changes or is malformed.
                    if (
                        str(story_selection.get("semantic_status", "")).upper()
                        == "MIXED_RESIDUAL"
                        and int(story_selection.get("selected_count", 0) or 0) < 2
                    ):
                        raise ValueError(
                            "Mixed residual story requires at least two independent sources."
                        )

                    selected_news = hydrate_story_sources(
                        news,
                        story_selection,
                        include_images=False,
                    )
                    if not selected_news:
                        raise ValueError("Selected story has no usable hydrated sources.")

                    # A selected source with no body and no usable summary is not usable
                    # evidence and must never reach Ollama generation.
                    usable_selected_news = [
                        item for item in selected_news
                        if len(str(item.get("content", "")).strip()) >= 120
                        or len(str(item.get("summary", "")).strip()) >= 120
                    ]
                    if not usable_selected_news:
                        raise ValueError("Selected story sources have no usable factual text.")
                    selected_news = usable_selected_news

                    print(
                        f"[TOPIC FILTER] EVIDENCE USABILITY CHECK | {keyword} | "
                        f"story={story_number}/{len(story_candidates)}"
                    )

                    evidence_source = _build_evidence_source(news, story_selection)
                    print(
                        f"[TOPIC FILTER] Evidence source prepared | "
                        f"source_chars={len(evidence_source)}"
                    )
                    evidence_lock = extract_evidence(evidence_source)

                    locked_facts = (
                        evidence_lock.get("facts", [])
                        if isinstance(evidence_lock, dict)
                        else []
                    )
                    evidence_fact_count = (
                        len(locked_facts)
                        if isinstance(locked_facts, list)
                        else 0
                    )
                    lineage = (
                        evidence_lock.get("fact_lineage", {})
                        if isinstance(evidence_lock, dict)
                        else {}
                    )
                    unique_information_units = (
                        int(lineage.get("unique_information_units", evidence_fact_count))
                        if isinstance(lineage, dict)
                        else evidence_fact_count
                    )

                    # Hard article-eligibility gate: a standalone TrendCurrent
                    # article must have at least three distinct verified
                    # information units. This is NOT a word floor, NOT a retry,
                    # and NOT a ranking rule. It prevents thin 1-2 fact evidence
                    # from reaching generation and producing non-articles.
                    if unique_information_units < EVIDENCE_MIN_FACTS_FOR_GENERATION:
                        trend["_production_status"] = "REJECT"
                        trend["_production_reject_reason"] = (
                            f"insufficient unique evidence facts "
                            f"({unique_information_units} < "
                            f"{EVIDENCE_MIN_FACTS_FOR_GENERATION})"
                        )
                        print(
                            f"[TOPIC FILTER] EVIDENCE SUFFICIENCY REJECT | "
                            f"unique_information_units={unique_information_units} | "
                            f"minimum={EVIDENCE_MIN_FACTS_FOR_GENERATION} | "
                            f"{keyword} | story={story_number} | "
                            f"reason=insufficient distinct evidence for meaningful article"
                        )
                        monitor.candidate_event(
                            "evidence_sufficiency",
                            status="REJECT",
                            news_count=len(news),
                            selected_source_indices=story_selection.get("selected_indices", []),
                            selected_source_count=story_selection.get("selected_count"),
                            evidence_source_chars=len(evidence_source),
                            evidence_fact_count=evidence_fact_count,
                            unique_information_units=unique_information_units,
                            evidence_facts=locked_facts,
                            minimum_facts=EVIDENCE_MIN_FACTS_FOR_GENERATION,
                        )
                        monitor.finish_candidate(
                            "REJECT",
                            reason=trend["_production_reject_reason"],
                        )
                        continue

                    print(
                        f"[TOPIC FILTER] SUBSTANTIVE STORY VALUE CHECK | "
                        f"{keyword} | story={story_number}"
                    )
                    try:
                        substantive_story_value_gate(evidence_lock)
                    except Exception as substantive_error:
                        trend["_production_status"] = "REJECT"
                        trend["_production_reject_reason"] = str(substantive_error)
                        monitor.candidate_event(
                            "substantive_story_value",
                            status="REJECT",
                            reason=str(substantive_error),
                            news_count=len(news),
                            selected_source_indices=story_selection.get("selected_indices", []),
                            selected_source_count=story_selection.get("selected_count"),
                            evidence_source_chars=len(evidence_source),
                            evidence_facts=locked_facts,
                            evidence_fact_count=evidence_fact_count,
                        )
                        monitor.finish_candidate(
                            "REJECT",
                            reason=f"substantive_value={substantive_error}",
                        )
                        continue

                    monitor.candidate_event(
                        "substantive_story_value",
                        status="PASS",
                        news_count=len(news),
                        selected_source_indices=story_selection.get("selected_indices", []),
                        selected_source_count=story_selection.get("selected_count"),
                        evidence_source_chars=len(evidence_source),
                        evidence_facts=locked_facts,
                        evidence_fact_count=evidence_fact_count,
                        unique_information_units=unique_information_units,
                    )

                    evidence_lock = _enrich_evidence_for_generation(
                        evidence_lock,
                        trend,
                    )
                    trend["_evidence_lock"] = evidence_lock
                    print(
                        f"[TOPIC FILTER] EVIDENCE USABILITY PASS | "
                        f"facts={evidence_fact_count} | {keyword} | story={story_number}"
                    )
                    monitor.candidate_event(
                        "evidence_locked",
                        news_count=len(news),
                        selected_source_indices=story_selection.get("selected_indices", []),
                        selected_source_count=story_selection.get("selected_count"),
                        evidence_source_chars=len(evidence_source),
                        evidence_facts=locked_facts,
                        evidence_fact_count=evidence_fact_count,
                        evidence_lock=evidence_lock,
                    )

                    article = generate_valid_article(
                        generation_prompt,
                        reference_date,
                        trend,
                        prelocked_evidence=evidence_lock,
                    )

                    _paragraph_text = " ".join(
                        str(p) for p in article.get("paragraphs", [])
                    ).strip()
                    _locked_facts = (
                        evidence_lock.get("facts", [])
                        if isinstance(evidence_lock, dict)
                        else []
                    )
                    monitor.candidate_event(
                        "article_generated",
                        article=article,
                        article_word_count=len(_paragraph_text.split()),
                        evidence_fact_count=(
                            len(_locked_facts)
                            if isinstance(_locked_facts, list)
                            else None
                        ),
                        evidence_facts=_locked_facts,
                        evidence_coverage=None,
                        information_density=None,
                    )

                    slug = slugify(keyword)
                    article["slug"] = slug

                    # Images are presentation metadata, not discovery/evidence data.
                    # Extract them only after the article has passed every quality gate.
                    selected_news = hydrate_news_items(selected_news, include_images=True)
                    selected_news = _ensure_publisher_images(selected_news)
                    rendered_html = render_article(article, news=selected_news)
                    rendered_html = _ensure_rendered_publisher_image(
                        rendered_html, article, selected_news
                    )
                    save_article(slug, rendered_html)

                    new_keywords.append(keyword)
                    generated += 1

                    print(f"OK -> {slug}.html")
                    monitor.finish_candidate("PASS", slug=slug)

                except Exception as e:
                    trend["_production_status"] = "REJECT"
                    trend["_production_reject_reason"] = str(e)
                    print(
                        f"[GENERATION] REJECT | article_slot=0 | "
                        f"{keyword} | story={story_number} | {e}"
                    )
                    monitor.finish_candidate("REJECT", reason=str(e))
                    continue

        except Exception as e:
            print(f"[TOPIC FILTER] ERROR: {keyword}: {e}")

    try:
        update_all()

        for k in new_keywords:
            add_processed(k, processed)

    except Exception as e:
        print("UPDATE ERROR:", e)

    print(f"Finished. Generated {generated} article(s).")
    monitor.end_run(generated=generated, status="FINISHED")

    if generated:
        git_push()


if __name__ == "__main__":
    main()
