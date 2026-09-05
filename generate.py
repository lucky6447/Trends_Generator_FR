import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print("[TrendCurrent PIPELINE] universal-fact-lock-v2.7.0-source-independence-fact-lineage-substantive-value")

import re
import os
import difflib
import subprocess
from datetime import date

from config import MAX_ARTICLES_PER_RUN, LANGUAGE
from rss import fetch_trends
from news import fetch_news, extract_article
from prompt import build_prompt
from ollama_client import generate, extract_evidence, validate_article_structure, substantive_story_value_gate
from ollama import chat
from config import MODEL
from fact_guard import validate as fact_guard_validate
from fact_guard_repair import repair as fact_guard_repair
import json
import unicodedata
from html_generator import render_article, save_article
from processed import load_processed, add_processed
from index_generator import update_all
from topic_scorer import filter_relevant_news, _is_sports_match_topic, _skip_reason, _norm, _clean
import generator_monitor as monitor

REQUIRED_FIELDS = ["title", "description", "h1", "paragraphs"]


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
#
# Evidence sufficiency is a separate production-capacity gate: if the locked
# evidence contains only 0-2 facts, the candidate is rejected before article
# generation. This avoids spending the expensive generation/audit pipeline on
# articles that cannot carry enough verified information to be meaningful.
EVIDENCE_MIN_FACTS_FOR_GENERATION = max(
    1,
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
1) ONE_STORY - the source pool broadly corroborates one concrete news story/event.
2) DOMINANT_STORY - a clearly identifiable, strongly corroborated story cluster
   can be isolated from a noisy pool, while the remaining sources are unrelated
   outliers. The isolated cluster does NOT have to be 50% or more of the pool.
3) MIXED - no single concrete story can be isolated safely.

TOPIC: {str(topic or '').strip()}

Rules:
- Same broad topic is NOT the same story.
- Different wording or languages for the SAME event counts as the same story.
- DOMINANT_STORY does NOT require a numerical majority of the full source pool.
- A smaller cluster may qualify when at least 2 sources clearly describe the
  same concrete event/development and the cluster can be isolated safely.
- Do not select a smaller cluster merely because the sources share a person,
  team, tournament, programme, region, or broad topic.
- Separate local/regional stories, separate people/events, programmes, lists,
  roundups, or unrelated developments are outliers and must not be selected.
- Be conservative. Never invent a connection.

Return ONLY JSON:
{{"verdict":"DOMINANT_STORY","confidence":95,"source_numbers":[1,2,3],"reason":"brief reason"}}

SOURCE HEADLINES:
{chr(10).join(lines)}
"""

    # Bounded infrastructure recovery ONLY for the semantic judge.
    # This is not an article/evidence/generation retry.
    # Attempt 1 is normal; attempt 2 is allowed only when the first attempt
    # fails before producing a valid semantic verdict.
    max_attempts = 2
    last_error = None

    for attempt in range(1, max_attempts + 1):
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
                f"confidence={confidence} | attempt={attempt}/{max_attempts} | "
                f"elapsed={elapsed:.2f}s | {reason}"
            )
            return {
                "status": verdict,
                "confidence": confidence,
                "reason": reason or "semantic story selection judgment",
                "source_numbers": source_numbers,
            }

        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                print(
                    f"[TOPIC FILTER] STORY SOURCE JUDGE | RETRY | "
                    f"attempt={attempt}/{max_attempts} | {exc}"
                )
                continue

            print(
                f"[TOPIC FILTER] STORY SOURCE JUDGE | UNAVAILABLE | "
                f"attempts={max_attempts} | {exc}"
            )
            return {
                "status": "UNAVAILABLE",
                "confidence": 0,
                "reason": f"semantic selector unavailable after {max_attempts} attempts: {last_error}",
                "source_numbers": [],
            }


SEMANTIC_RESCUE_MIN_CONFIDENCE = 90


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
    """Identify concrete stories inside a topic without topic-level rejection.

    Lexical clustering is used as the deterministic baseline. When the pool is
    ambiguous, the existing semantic story judge is used to isolate a concrete
    story cluster. MIXED never means "reject topic": the remaining sources are
    still split/evaluated as independent candidates.
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
            kept_original[i] for i in selected
            if 0 <= i < len(kept_original)
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

    # Very small pools do not justify an additional semantic call.
    if len(items) < 3:
        add_candidate(remaining)
    else:
        # First pass: semantic identification of a concrete story. This is a
        # splitter, never a topic-level gate.
        try:
            judge = _semantic_story_concentration_judge(
                [items[i] for i in remaining], topic
            )
            status = str(judge.get("status", "MIXED")).upper()
            nums = judge.get("source_numbers") or []
            selected = [remaining[n - 1] for n in nums if 1 <= n <= len(remaining)]
            confidence = int(judge.get("confidence", 0) or 0)

            if status in {"ONE_STORY", "PASS"} and confidence >= 70:
                add_candidate(remaining, status)
                remaining = []
            elif status == "DOMINANT_STORY" and confidence >= 70 and len(selected) >= 2:
                add_candidate(selected, status)
                remaining = [i for i in remaining if i not in set(selected)]
        except Exception as exc:
            # Semantic discovery is auxiliary; deterministic fallback remains.
            print(f"[STORY DISCOVERY] semantic splitter unavailable | {exc}")

    # Split whatever remains deterministically. This is intentionally applied
    # after semantic extraction so separate stories survive a mixed topic pool.
    if remaining:
        residual = [items[i] for i in remaining]
        profile = _story_pool_profile(residual, topic)
        components = list(profile.get("components") or [])
        if len(residual) < 3:
            components = [list(range(len(residual)))]
        for component in components:
            selected = [remaining[i] for i in component if 0 <= i < len(remaining)]
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


def _selected_story_news(news, story_selection):
    """Return exactly the sources selected by Story Source Decision."""
    items = list(news or [])
    selection = story_selection or {}
    indices = list(selection.get("selected_indices") or [])
    selected = [items[i] for i in indices if 0 <= i < len(items)]
    if not selected:
        raise Exception("Fact Guard source received no sources from story selection")
    return selected


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
    # Article length is now owned entirely by ollama_client._article_length_policy()
    # and its deterministic body-word floor. Do not add a second fact-count-specific
    # override here: a conflicting single-fact instruction previously encouraged
    # extreme compression even when the density gate required a minimum body size.

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
            validate_article_structure(article, generation_evidence, label="Initial generated article")

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
            validate_language_integrity(article)
            print("[LANGUAGE GUARD] PASS")

            print("[FACT GUARD] Checking generated article...")
            guard = fact_guard_validate(
                fact_guard_source,
                article,
                reference_date=reference_date,
            )

            monitor.candidate_event(
                "fact_guard",
                status=guard.get("status"),
                review_items=guard.get("review_items"),
                blocking_issues=guard.get("blocking_issues"),
                guard=guard,
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
                    validate_article_structure(repaired, generation_evidence, label="Fact Guard repaired article")
                    repaired = enforce_headline_policy(repaired, trend)
                    validate_article(repaired)
                    validate_article_structure(repaired, generation_evidence, label="Fact Guard repaired article final")
                    validate_language_integrity(repaired)
                    print("[LANGUAGE GUARD] REPAIRED ARTICLE PASS")

                    print("[FACT GUARD REPAIR] Re-checking repaired article...")
                    repaired_guard = fact_guard_validate(
                        fact_guard_source,
                        repaired,
                        reference_date=reference_date,
                    )

                    monitor.candidate_event(
                        "fact_guard_repair_check",
                        status=repaired_guard.get("status"),
                        review_items=repaired_guard.get("review_items"),
                        blocking_issues=repaired_guard.get("blocking_issues"),
                        guard=repaired_guard,
                        repaired_article=repaired,
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

                    validate_language_integrity(repaired)
                    print("[LANGUAGE GUARD] FINAL REPAIRED ARTICLE PASS")
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

            validate_language_integrity(article)
            print("[LANGUAGE GUARD] FINAL ARTICLE PASS")
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
    monitor.start_run(language=LANGUAGE, model=MODEL, pipeline="universal-fact-lock-v2.7.0-source-independence-fact-lineage-substantive-value", max_articles=MAX_ARTICLES_PER_RUN)
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

        if _is_sports_topic(keyword):
            print(f"[TrendCurrent] SKIP sports topic: {keyword}")
            continue

        try:
            print(
                f"\n[STORY DISCOVERY] Retrieving sources for discovery seed: "
                f"{keyword}"
            )

            discovery_query = trend.get("discovery_query")
            seed = None
            if discovery_query:
                seed = dict(trend)
                if not seed.get("content") and seed.get("link"):
                    try:
                        seed["content"] = extract_article(seed["link"])
                    except Exception:
                        seed["content"] = ""

                seed_anchor = str(seed.get("title", "")).strip() or keyword
                entity_query = _build_related_story_query(seed_anchor)
                if not entity_query:
                    entity_query = seed_anchor.split(" - ")[0].strip()

                print(f"[TOPIC FILTER] Related story search: {entity_query}")
                news = fetch_news(entity_query)

                seed_title_norm = _normalise_headline(seed.get("title", ""))
                if len(str(seed.get("content", "")).strip()) < 900 and news:
                    best = None
                    best_overlap = 0
                    seed_words = {
                        w for w in re.findall(r"[a-z0-9]+", seed_title_norm.casefold())
                        if len(w) >= 4
                    }
                    for candidate in news:
                        cand_norm = _normalise_headline(candidate.get("title", ""))
                        cand_words = {
                            w for w in re.findall(r"[a-z0-9]+", cand_norm.casefold())
                            if len(w) >= 4
                        }
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
                    str(x.get("title", "")).strip().casefold() ==
                    str(seed.get("title", "")).strip().casefold()
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
                print(
                    f"[TOPIC FILTER] DROP AFTER NEWS | {keyword} | "
                    f"no usable news result(s)"
                )
                continue

            relevant_news = filter_relevant_news(trend, news)
            if not relevant_news:
                print(
                    f"[TOPIC FILTER] DROP AFTER NEWS | {keyword} | "
                    f"no topic-relevant news result(s)"
                )
                continue
            news = relevant_news

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
            # evidence/substantive/generation/Fact Guard pipeline. A failed
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
                    generation_prompt = build_prompt(trend)
                    selected_news = _selected_story_news(news, story_selection)
                    fact_guard_source = _build_fact_guard_source(selected_news)

                    print(
                        f"[FACT GUARD] Source prepared | "
                        f"news_items={len(selected_news)} | "
                        f"source_chars={len(fact_guard_source)}"
                    )

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

                    if evidence_fact_count < EVIDENCE_MIN_FACTS_FOR_GENERATION:
                        trend["_production_status"] = "REJECT"
                        trend["_production_reject_reason"] = (
                            f"insufficient evidence facts "
                            f"({evidence_fact_count} < "
                            f"{EVIDENCE_MIN_FACTS_FOR_GENERATION})"
                        )
                        print(
                            f"[TOPIC FILTER] EVIDENCE SUFFICIENCY REJECT | "
                            f"facts={evidence_fact_count} | "
                            f"minimum={EVIDENCE_MIN_FACTS_FOR_GENERATION} | "
                            f"{keyword} | story={story_number} | "
                            f"reason=insufficient evidence for meaningful article"
                        )
                        monitor.candidate_event(
                            "evidence_sufficiency",
                            status="REJECT",
                            reason="insufficient evidence for meaningful article",
                            news_count=len(news),
                            selected_source_indices=story_selection.get("selected_indices", []),
                            selected_source_count=story_selection.get("selected_count"),
                            evidence_source_chars=len(evidence_source),
                            evidence_fact_count=evidence_fact_count,
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
                        fact_guard_source,
                        reference_date,
                        trend,
                        max_attempts=1,
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
                    save_article(slug, render_article(article, news=selected_news))

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
