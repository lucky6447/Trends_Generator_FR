import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import re
import os
from pathlib import Path
import difflib
import hashlib
import subprocess
import time
import html as _html
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from datetime import date

from config import MAX_ARTICLES_PER_RUN, LANGUAGE, SOURCE_FIRST, TREND_DIR, RUN_TIME_BUDGET_SECONDS, MAX_CONCRETE_STORY_CANDIDATES_PER_RUN
from rss import fetch_trends
from rss_source_discovery import fetch_source_stories
from news import fetch_news, fetch_news_discovery, hydrate_story_sources, hydrate_news_items
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
    """Fail closed on obvious script leakage, instruction leakage, or clear wrong-language output."""
    text = _article_text(article)
    if not text:
        raise ValueError("Language integrity check failed: empty article text.")

    language = str(LANGUAGE or "").strip().casefold()
    cyrillic_allowed = language in {"bulgarian", "bg", "български"}
    language_aliases = {
        "english": "en", "en": "en", "en-us": "en", "en-gb": "en",
        "german": "de", "de": "de", "deutsch": "de",
        "french": "fr", "fr": "fr", "français": "fr",
        "italian": "it", "it": "it", "italiano": "it",
        "spanish": "es", "es": "es", "español": "es",
        "indonesian": "id", "id": "id", "bahasa indonesia": "id",
        "bulgarian": "bg", "bg": "bg", "български": "bg",
    }
    target_lang = language_aliases.get(language, language)

    latin_languages = {"en", "de", "fr", "it", "es", "id"}

    # Deterministic script leakage detection remains unchanged in purpose.
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
        elif "CYRILLIC" in name and target_lang in latin_languages:
            forbidden_scripts.append("Cyrillic")

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

    if target_lang in latin_languages:
        # The previous guard only checked Unicode script. That cannot distinguish
        # English from Italian/German/French/Spanish/Indonesian because all use Latin.
        # This is a deterministic target-language check: common function/content
        # words and language-specific morphology are scored against the full article.
        #
        # Proper names, publisher names and internationally used terms are not enough
        # to fail the article. We reject only a clear competing-language signal.
        tokens = re.findall(r"[a-zà-ÿ]+", unicodedata.normalize("NFKC", text).casefold())
        if len(tokens) >= 18:
            signatures = {
                "en": {
                    "the","and","of","to","in","for","on","with","from","that","this",
                    "was","were","has","have","had","are","is","as","by","at","after",
                    "before","will","said","about","into","over","their","they","which",
                    "also","more","than","its","who","what","when","how","new","news",
                },
                "de": {
                    "der","die","das","und","von","zu","den","dem","des","ein","eine",
                    "einer","einem","einen","ist","sind","war","wurde","wurden","mit",
                    "auf","für","im","in","aus","nach","über","auch","nicht","sich",
                    "als","bei","hat","haben","wie","dass","durch","werden","wird",
                },
                "fr": {
                    "le","la","les","des","du","de","un","une","et","en","dans","pour",
                    "sur","avec","par","est","sont","était","ont","a","au","aux","ce",
                    "cette","ces","qui","que","pas","plus","mais","comme","après",
                    "avant","leur","leurs","dans","entre","vers","selon",
                },
                "it": {
                    "il","lo","la","i","gli","le","di","del","della","dei","degli",
                    "delle","un","uno","una","e","che","in","con","per","su","da",
                    "al","alla","agli","alle","è","sono","era","sono","ha","hanno",
                    "non","anche","come","dopo","prima","nel","nella","nelle","degli",
                    "questa","questo","quello","secondo",
                },
                "es": {
                    "el","la","los","las","del","de","un","una","unos","unas","y",
                    "que","en","con","por","para","sobre","desde","entre","al","es",
                    "son","era","fue","han","ha","no","también","como","más","menos",
                    "después","antes","esta","este","estos","estas","según",
                },
                "id": {
                    "yang","dan","di","ke","dari","untuk","dengan","pada","dalam",
                    "ini","itu","akan","telah","adalah","sebagai","oleh","lebih",
                    "juga","tidak","dapat","bisa","setelah","sebelum","tentang",
                    "dengan","menjadi","sudah","masih","mereka","para","karena",
                    "hingga","tersebut","bahwa","atau","serta","terhadap",
                },
            }

            # Extra morphology/orthography signals help distinguish closely related
            # Latin languages when function-word evidence is sparse.
            morphology = {
                "en": (r"\b\w+(?:ing|ed|ly)\b",),
                "de": (r"\b\w+(?:ung|keit|heit|lich|ischen|isch)\b",),
                "fr": (r"\b\w+(?:ment|tion|ique|eur|euse|aient|ées|és)\b",),
                "it": (r"\b\w+(?:zione|zioni|mente|ità|ismo|are|ere|ire)\b",),
                "es": (r"\b\w+(?:ción|ciones|mente|ando|iendo|ado|ido|ación)\b",),
                "id": (r"\b\w+(?:kan|nya|lah|kah|pun|per|ber|ter|meng|mem|men)\b",),
            }

            scores = {lang: 0.0 for lang in signatures}
            token_set = set(tokens)
            for lang, words in signatures.items():
                # Function/content words are strong signals. Cap contribution per
                # repeated word so names or repeated phrasing cannot dominate.
                hits = sum(1 for word in words if word in token_set)
                freq_hits = sum(1 for token in tokens if token in words)
                scores[lang] += hits * 1.0 + min(freq_hits, 10) * 0.35

                for pattern in morphology.get(lang, ()):
                    scores[lang] += min(len(re.findall(pattern, lower)), 6) * 0.45

            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            best_lang, best_score = ranked[0]
            target_score = scores.get(target_lang, 0.0)
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0

            # Only fail on a strong and materially separated competing-language
            # signal. This deliberately favors false negatives over false positives.
            if (
                best_lang != target_lang
                and best_score >= 4.5
                and best_score >= target_score + 2.5
                and best_score >= second_score + 0.75
            ):
                raise ValueError(
                    f"Language integrity check failed: target={target_lang} "
                    f"detected={best_lang} score={best_score:.2f} target_score={target_score:.2f}."
                )

    # Unknown future languages keep the existing conservative script/leakage behavior.
    return True

# Article length is determined by the amount of usable verified evidence.
# There is no artificial word-count target, word-count retry, or minimum
# evidence-fact gate before substantive story-value judging. Small but concrete
# stories are allowed to reach the unchanged substantive-value gate.

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


LOCAL_NEWS_TOPIC_PATTERNS = (
    r"\blocal news\b", r"\blocal news update\b", r"\blocal headlines?\b",
    r"\blocal stories\b", r"\blocal report\b", r"\blocal reports\b",
    r"\blocal police\b", r"\blocal politics\b", r"\blocal government\b",
    r"\blocal council\b", r"\bcity council\b", r"\btown council\b",
    r"\bmunicipal news\b", r"\bmunicipal government\b",
    r"\bmunicipal council\b", r"\bdistrict council\b",
    r"\bregional news\b", r"\bregional headlines?\b", r"\bregional update\b",
    r"\bcommunity news\b", r"\bcommunity update\b",
    r"\bactualidad local\b", r"\bnoticias locales?\b",
    r"\bnachrichten aus der region\b", r"\blokalnachrichten\b",
    r"\bnotizie locali\b", r"\bactualités locales?\b",
)

def _is_local_news_topic(title, news=None):
    """Return True for explicit local/regional-news topics."""
    text = " ".join(str(title or "").split()).casefold()
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in LOCAL_NEWS_TOPIC_PATTERNS):
        return True

    # Only use an explicit local/regional signal from source metadata/headlines.
    # A normal national/international story mentioning a city must not be rejected.
    for item in (news or ()):
        if not isinstance(item, dict):
            continue
        item_text = " ".join(
            str(item.get(key, "") or "")
            for key in ("title", "summary", "source")
        ).casefold()
        if any(re.search(pattern, item_text, flags=re.IGNORECASE)
               for pattern in LOCAL_NEWS_TOPIC_PATTERNS):
            return True

    return False


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
    # English
    "the", "a", "an", "and", "or", "but", "amid", "after", "before", "during",
    "with", "without", "from", "into", "over", "under", "for", "of", "to", "in",
    "on", "at", "as", "by", "is", "are", "was", "were", "has", "have", "had",
    "new", "latest", "news", "report", "reports", "update", "updates",
    # French -- essential for FR cross-run story identity.
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "au", "aux",
    "et", "ou", "mais", "avec", "sans", "pour", "par", "dans", "sur", "sous",
    "entre", "vers", "chez", "apres", "avant", "pendant", "selon", "est", "sont",
    "a", "ont", "avait", "avaient", "etre", "ete", "ce", "cet", "cette", "ces",
    "qui", "que", "quoi", "dont", "plus", "moins", "tres", "comme", "aussi",
    "nouveau", "nouvelle", "nouvelles", "actualite", "actualites", "rapport", "rapports",
    "mise", "jour",
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

    # Three shared concrete tokens can be enough for a clear paraphrase.
    if len(common) >= 3 and sequence_ratio >= 0.68 and min_coverage >= 0.50:
        return True, sequence_ratio, len(common), min_coverage

    # French headlines are often compact paraphrases where sequence similarity
    # drops after removing articles/prepositions. If the two titles still share
    # most of their concrete vocabulary, treat them as the same story.
    if len(common) >= 3 and min_coverage >= 0.67 and sequence_ratio >= 0.58:
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

def _is_non_story_discovery_seed(trend):
    """Reject obvious roundup/bulletin/headline-format seeds before corroboration.

    This is a narrow discovery-quality guard. It does NOT judge whether a story is
    important, popular, or factual; it only removes publisher feed entries that are
    clearly containers for multiple stories rather than one concrete event. Such
    seeds should never consume corroboration, hydration, evidence extraction, or
    Ollama time.
    """
    if not isinstance(trend, dict):
        return False, "invalid seed"

    title = re.sub(r"\s+", " ", str(trend.get("title", "") or "")).strip().casefold()
    if not title:
        return True, "empty title"

    # Deliberately narrow: only unmistakable multi-story/bulletin formats.
    non_story_patterns = (
        r"\blatest news headlines?\b",
        r"\btoday(?:'s|s)? headlines?\b",
        r"\btop headlines?\b",
        r"\bheadlines? from .*\bat \d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b",
        r"\b(?:morning|midday|afternoon|evening|night) headlines?\b",
        r"\bnews roundup\b",
        r"\broundup of (?:the )?(?:latest )?news\b",
        r"\b(?:daily|morning|evening|night) news briefing\b",
        r"\bnews briefing\b",
        r"\btop stories (?:today|tonight)\b",
        r"\bnews(?:cast| bulletin)\b.*\bat \d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b",
    )
    for pattern in non_story_patterns:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return True, "non-story roundup/bulletin format"

    # A very explicit headline-list construction is also safe to reject when it
    # names a broadcast time. This catches publisher variants not covered above.
    if "headlines" in title and re.search(
        r"\b(?:at|@)\s*\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b",
        title,
        flags=re.IGNORECASE,
    ):
        return True, "timed headline bulletin"

    return False, ""


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

        non_story, reject_reason = _is_non_story_discovery_seed(trend)
        if non_story:
            print(
                f"[STORY DISCOVERY] SKIP non-story seed | reason={reject_reason} | title={title}"
            )
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
    """Keep discovery results relevant to the canonical trend only.

    Google-Trends related-news headlines are used upstream only as concrete
    discovery query signals. They must never become an alternative relevance
    authority, because that can replace the canonical trend with an unrelated
    story before story selection and evidence locking.
    """
    canonical = filter_relevant_news(trend, news)
    print(
        f"[TOPIC FILTER] Canonical trend relevance | "
        f"canonical={len(canonical)} | input={len(news or [])}"
    )
    return canonical


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
    """Build the canonical structured evidence source from the exact selected sources."""
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

    # Keep the source structured until ollama_client builds the single canonical
    # sentence index. This removes dict -> large string -> chunks -> re-index churn.
    return {"articles": compact_sources}

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
    """Validate a URL already obtained from the canonical publisher-image resolver."""
    value = str(url or "").strip()
    if not value or not re.match(r"^https?://", value, re.I):
        return False

    lower = value.casefold()
    blocked = (
        "placeholder", "place-holder", "default-image", "default_image",
        "no-image", "no_image", "spacer.gif", "transparent.gif",
        "favicon", "/favicon", "sprite", "tracking", "pixel",
        "googlelogo", "googleusercontent", "gstatic.com/images/branding",
    )
    if any(token in lower for token in blocked):
        return False

    # Never allow media resources to be rendered as article images.
    path = lower.split("?", 1)[0].split("#", 1)[0]
    if path.endswith((
        ".mp4", ".m4v", ".webm", ".mov", ".m3u8", ".mpd",
        ".avi", ".mkv", ".flv", ".wmv", ".mp3", ".m4a", ".wav", ".ogg",
        ".svg",
    )):
        return False

    # Reject obvious tiny thumbnail variants such as ?w=96 / ?width=120.
    try:
        query = parse_qs(urlparse(value).query)
        for key in ("w", "width", "h", "height", "size"):
            for raw in query.get(key, []):
                m = re.search(r"\d+", str(raw))
                if m and int(m.group()) <= 160:
                    return False
    except Exception:
        pass

    # Do not require a file extension: many publisher CDNs expose image URLs
    # without .jpg/.webp in the path. The canonical news.py resolver has already
    # established that this URL came from JSON-LD image.url.
    return True


class _JSONLDImageParser(HTMLParser):
    """Collect application/ld+json blocks without treating HTML images as evidence."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.blocks = []
        self._capture = False
        self._buffer = []

    def handle_starttag(self, tag, attrs):
        if tag.casefold() != "script":
            return
        attrs_map = {str(k).casefold(): str(v or "") for k, v in attrs}
        if attrs_map.get("type", "").casefold().split(";")[0].strip() == "application/ld+json":
            self._capture = True
            self._buffer = []

    def handle_endtag(self, tag):
        if tag.casefold() == "script" and self._capture:
            self.blocks.append("".join(self._buffer))
            self._capture = False
            self._buffer = []

    def handle_data(self, data):
        if self._capture:
            self._buffer.append(data)

    def handle_entityref(self, name):
        if self._capture:
            self._buffer.append(f"&{name};")

    def handle_charref(self, name):
        if self._capture:
            self._buffer.append(f"&#{name};")


def _jsonld_image_urls(payload):
    """Yield only values from JSON-LD ``image`` objects that expose ``url``."""
    found = []
    seen = set()

    def visit(node):
        if isinstance(node, dict):
            image_value = node.get("image")
            if isinstance(image_value, dict):
                candidate = str(image_value.get("url") or "").strip()
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    found.append(candidate)
            elif isinstance(image_value, list):
                for image_obj in image_value:
                    if isinstance(image_obj, dict):
                        candidate = str(image_obj.get("url") or "").strip()
                        if candidate and candidate not in seen:
                            seen.add(candidate)
                            found.append(candidate)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(payload)
    return found


def _extract_jsonld_publisher_image(publisher_url):
    """Fetch the publisher page and accept only JSON-LD image.url.

    This is a narrow recovery path for publishers whose page is reachable but
    whose image metadata was not surfaced by news.extract_source_image(). It
    deliberately does NOT inspect og:image, twitter:image, <img>, CSS, or
    arbitrary page URLs.
    """
    url = str(publisher_url or "").strip()
    if not url or not re.match(r"^https?://", url, re.I):
        return ""

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        },
    )
    with urlopen(request, timeout=12) as response:
        raw = response.read(2_500_000)
        charset = response.headers.get_content_charset() or "utf-8"
        try:
            text = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = raw.decode("utf-8", errors="replace")

    parser = _JSONLDImageParser()
    parser.feed(text)
    parser.close()

    for block in parser.blocks:
        block = _html.unescape(block).strip()
        if not block:
            continue
        try:
            payload = json.loads(block)
        except Exception:
            # Some publishers put more than one JSON object in a JSON-LD block.
            # Do not guess at non-JSON content; skip the malformed block.
            continue
        for candidate in _jsonld_image_urls(payload):
            if _is_valid_publisher_image_url(candidate):
                return candidate
    return ""


def _publisher_image_candidate_urls(item, image_data=None):
    """Return publisher-page URLs, prioritizing resolver-confirmed source URLs."""
    candidates = []
    for value in (
        (image_data or {}).get("source_url", ""),
        item.get("publisher_url", "") if isinstance(item, dict) else "",
        item.get("publisher", "") if isinstance(item, dict) else "",
        item.get("source_url", "") if isinstance(item, dict) else "",
    ):
        value = str(value or "").strip()
        if value and re.match(r"^https?://", value, re.I) and value not in candidates:
            candidates.append(value)
    return candidates


def _ensure_publisher_images(selected_news):
    """Finalize publisher images without erasing a verified cross-publisher fallback.

    Canonical policy:
      - direct publisher JSON-LD image.url is accepted;
      - if direct fails, news.py may return a verified independent publisher
        fallback JSON-LD image.url;
      - once FOUND, that verified image is authoritative for the selected source;
      - this finalizer must never call the resolver without the story title,
        because title is required to activate cross-publisher fallback.
    """
    from news import extract_source_image

    items = list(selected_news or [])
    found = 0
    allowed_reasons = {
        "jsonld_image_url",
        "fallback_jsonld_image_url",
        "article_img_tag",
    }

    for item in items:
        if not isinstance(item, dict):
            continue

        existing = str(item.get("image") or item.get("image_url") or "").strip()
        existing_status = str(item.get("image_status") or "").strip().upper()
        existing_reason = str(item.get("image_reason") or "").strip().casefold()

        # IMPORTANT: preserve BOTH direct and verified fallback images.
        if (
            existing
            and _is_valid_publisher_image_url(existing)
            and existing_status == "FOUND"
            and existing_reason in allowed_reasons
        ):
            item["image"] = existing
            found += 1
            print(
                f"[SOURCE IMAGE] PRESERVE | image={existing} | "
                f"reason={existing_reason}"
            )
            continue

        url = str(item.get("url") or item.get("link") or "").strip()
        title = str(item.get("title") or "").strip()
        if not url:
            continue

        try:
            # Pass the title. Without it, extract_source_image() cannot perform
            # the cross-publisher fallback query.
            image_data = extract_source_image(url, title=title) or {}
        except Exception as exc:
            print(f"[SOURCE IMAGE] CANONICAL ERROR: {url} | {exc}")
            image_data = {}

        image_url = str(image_data.get("image") or "").strip()
        image_reason = str(
            image_data.get("image_reason")
            or "jsonld_image_url_not_found_or_invalid"
        ).strip().casefold()

        if (
            image_url
            and _is_valid_publisher_image_url(image_url)
            and image_data.get("image_status") == "FOUND"
            and image_reason in allowed_reasons
        ):
            item["image"] = image_url
            item["image_source_url"] = image_data.get("source_url", "")
            item["image_source"] = image_data.get("source", "")
            item["image_status"] = "FOUND"
            item["image_reason"] = image_reason
            found += 1
            print(
                f"[SOURCE IMAGE] FOUND: {image_url} | "
                f"source=canonical-news-resolver | reason={image_reason}"
            )
        else:
            item["image"] = ""
            item["image_status"] = "NONE"
            item["image_reason"] = image_reason
            print(
                f"[SOURCE IMAGE] NONE: {url} | reason={image_reason}"
            )

    print(f"[SOURCE IMAGE] FINAL | publisher_images={found} | sources={len(items)}")
    return items

def _ensure_rendered_publisher_image(rendered_html, article, selected_news):
    """Guarantee the verified publisher image reaches final HTML presentation and stays responsive."""
    html = str(rendered_html or "")
    if not html:
        return html

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

    # Only a verified JSON-LD publisher image satisfies this gate. Both direct
    # and cross-publisher fallback JSON-LD image.url results are valid.
    image_url = ""
    verified_image_urls = []
    for item in selected_news or []:
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("image") or item.get("image_url") or "").strip()
        if (
            _is_valid_publisher_image_url(candidate)
            and str(item.get("image_status") or "").strip().upper() == "FOUND"
            and str(item.get("image_reason") or "").strip().casefold()
                in {"jsonld_image_url", "fallback_jsonld_image_url"}
        ):
            if not image_url:
                image_url = candidate
            verified_image_urls.append(candidate)

    if not verified_image_urls:
        print("[SOURCE IMAGE] OPTIONAL | no verified publisher image; publishing without image.")
        return html

    # If the exact verified publisher image is already rendered, keep it.
    if any(candidate in html or candidate.replace("&", "&amp;") in html
           for candidate in verified_image_urls):
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
    raise ValueError("Publisher image gate failed: could not inject verified image into HTML.")


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
    """Deterministic headline gate; never call the LLM for repair."""
    violations = _headline_violations(article, trend)
    if violations:
        raise ValueError(
            "Headline policy failed: " + "; ".join(violations)
        )
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
def _deterministic_temporal_event_guard(evidence, trend=None, reference_date=None):
    """Reject only a narrow, high-confidence temporal/event mismatch in locked CORE facts.

    This is not a general factual audit. It rejects materially stale CORE events:
    an explicit topic-year conflict, a current-year CORE event mixed with a materially
    older CORE event, or a CORE event whose explicit years are all materially older
    than the current run year without a verified current-year development.
    SUPPORTING facts are ignored because they may legitimately provide historical
    context. No LLM call, inference, repair, or regeneration is performed.
    """
    if not isinstance(evidence, dict):
        raise ValueError("Temporal/event guard requires an evidence object.")

    facts = evidence.get("facts", [])
    if not isinstance(facts, list) or not facts:
        return {"status": "PASS", "checked": False, "reason": "no locked facts"}

    year_re = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
    core_facts = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).strip().casefold() != "core":
            continue
        fact_text = str(item.get("fact", "") or "").strip()
        years = sorted({int(y) for y in year_re.findall(fact_text)})
        if years:
            core_facts.append({"id": str(item.get("id", "")), "fact": fact_text, "years": years})

    if not core_facts:
        return {"status": "PASS", "checked": False, "reason": "no explicit years in CORE facts"}

    topic = str((trend or {}).get("title", "") or "").strip()
    topic_years = sorted({int(y) for y in year_re.findall(topic)})
    core_years = sorted({year for item in core_facts for year in item["years"]})
    current_year = getattr(reference_date, "year", None)

    if topic_years:
        conflicts = sorted({
            (topic_year, fact_year)
            for topic_year in topic_years
            for fact_year in core_years
            if abs(topic_year - fact_year) >= 2
        })
        if conflicts:
            print(
                "[TEMPORAL/EVENT GUARD] REJECT | "
                f"topic_years={topic_years} | core_years={core_years} | conflicts={conflicts}"
            )
            return {
                "status": "REJECT",
                "reason": "explicit topic year conflicts with CORE event year",
                "topic_years": topic_years,
                "core_years": core_years,
                "conflicts": conflicts,
            }

    if current_year is not None and current_year in core_years:
        older_core_years = [year for year in core_years if year <= current_year - 2]
        if older_core_years:
            print(
                "[TEMPORAL/EVENT GUARD] REJECT | "
                f"current_year={current_year} | core_years={core_years} | "
                f"older_core_years={older_core_years}"
            )
            return {
                "status": "REJECT",
                "reason": "current-year CORE event mixed with materially older CORE event",
                "current_year": current_year,
                "core_years": core_years,
                "older_core_years": older_core_years,
            }

    # A fresh source is not enough to make an old event a current story.
    # If every explicit CORE year is materially older than the run year, there
    # is no verified current-year development in the locked CORE evidence.
    # Supporting facts may still contain historical context; only CORE facts
    # determine whether the concrete story itself is stale.
    if current_year is not None and core_years:
        materially_old_core_years = [
            year for year in core_years if year <= current_year - 2
        ]
        if materially_old_core_years and current_year not in core_years:
            print(
                "[TEMPORAL/EVENT GUARD] REJECT | "
                f"current_year={current_year} | core_years={core_years} | "
                f"materially_old_core_years={materially_old_core_years} | "
                "reason=historical CORE event without current-year development"
            )
            return {
                "status": "REJECT",
                "reason": "historical CORE event without current-year development",
                "current_year": current_year,
                "core_years": core_years,
                "materially_old_core_years": materially_old_core_years,
            }

    print(
        "[TEMPORAL/EVENT GUARD] PASS | "
        f"topic_years={topic_years or []} | core_years={core_years or []}"
    )
    return {"status": "PASS", "checked": True, "topic_years": topic_years, "core_years": core_years}


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

    # CORE/SUPPORTING roles are preserved from the locked evidence. Supporting
    # facts are optional context and must not be promoted to mandatory coverage.
    #
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
# Cross-run Story Identity Guard
#
# Existing-story title matching is useful but cannot close the race where two
# production runs check before either one has published. It also cannot reliably
# catch paraphrased headlines. This guard persists a small story-identity claim
# built from the already locked, semantically deduplicated evidence.
#
# The claim is reserved atomically before expensive generation. A failed
# generation releases its reservation; a successfully saved article keeps it.
# A filesystem lock makes the check+reserve operation atomic across concurrent
# generator processes on the same filesystem.
# ============================================================

_CROSS_RUN_STORY_CLAIMS_DIRNAME = ".story_claims"
_CROSS_RUN_STORY_LOCK_NAME = ".reserve.lock"
# Conservative crash-recovery windows. Reservations are normally much shorter.
_CROSS_RUN_STALE_RESERVATION_SECONDS = max(60, int(os.getenv("CROSS_RUN_STALE_RESERVATION_SECONDS", "1800")))
_CROSS_RUN_STALE_LOCK_SECONDS = max(30, int(os.getenv("CROSS_RUN_STALE_LOCK_SECONDS", "120")))


def _cross_run_fact_tokens(value):
    """Return conservative factual tokens for cross-run story identity."""
    text = unicodedata.normalize(
        "NFKD", str(value or "")
    ).encode("ascii", "ignore").decode("ascii").casefold()
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "was", "were",
        "has", "have", "had", "are", "is", "its", "into", "after", "before",
        "over", "under", "about", "than", "then", "they", "their", "them",
        "there", "which", "while", "also", "been", "being", "will", "would",
        "could", "should", "said", "says", "according", "official", "officials",
        "new", "latest", "news", "report", "reports", "story", "article",
        "according", "publisher", "source",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) >= 3 and token not in stop
    }


def _cross_run_numeric_tokens(value):
    return set(re.findall(
        r"\b\d+(?:[.,]\d+)?%?\b|\b(?:19|20)\d{2}\b",
        str(value or ""),
    ))


def _cross_run_fact_similarity(a, b):
    """Conservative paraphrase similarity for two factual claims."""
    ta = _cross_run_fact_tokens(a)
    tb = _cross_run_fact_tokens(b)
    if len(ta) < 3 or len(tb) < 3:
        return 0.0

    overlap = len(ta & tb) / min(len(ta), len(tb))

    na = _cross_run_numeric_tokens(a)
    nb = _cross_run_numeric_tokens(b)
    if na and nb and not (na & nb):
        return 0.0

    return overlap


def _cross_run_evidence_duplicate(current_evidence, prior_evidence):
    """Return whether two evidence locks describe the same concrete story."""
    if not isinstance(current_evidence, dict) or not isinstance(prior_evidence, dict):
        return False

    current_facts = current_evidence.get("facts", [])
    prior_facts = prior_evidence.get("facts", [])
    if not isinstance(current_facts, list) or not isinstance(prior_facts, list):
        return False

    current_facts = [str(x).strip() for x in current_facts if str(x).strip()]
    prior_facts = [str(x).strip() for x in prior_facts if str(x).strip()]
    if not current_facts or not prior_facts:
        return False

    matched = 0
    matched_scores = []
    for current in current_facts:
        best = max(
            (_cross_run_fact_similarity(current, prior) for prior in prior_facts),
            default=0.0,
        )
        if best >= 0.50:
            matched += 1
            matched_scores.append(best)

    # Two independently locked factual claims matching strongly is enough.
    # One matching fact is deliberately insufficient: related stories often
    # share a person, place, or single event detail.
    return matched >= 2 and (
        sum(matched_scores) / len(matched_scores) >= 0.52
        if matched_scores else False
    )


def _cross_run_claim_lock():
    """Acquire the process-safe cross-run reservation lock.

    Recover a lock left behind by a hard crash only after a conservative age
    threshold; the critical section itself normally lasts only milliseconds.
    """
    claims_dir = TREND_DIR / _CROSS_RUN_STORY_CLAIMS_DIRNAME
    claims_dir.mkdir(parents=True, exist_ok=True)
    lock_path = claims_dir / _CROSS_RUN_STORY_LOCK_NAME

    for _ in range(100):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={os.getpid()}\ntime={time.time()}\n".encode("ascii"))
            finally:
                os.close(fd)
            return lock_path
        except FileExistsError:
            try:
                age = max(0.0, time.time() - lock_path.stat().st_mtime)
                if age >= _CROSS_RUN_STALE_LOCK_SECONDS:
                    lock_pid = None
                    try:
                        lock_text = lock_path.read_text(encoding="ascii", errors="replace")
                        match = re.search(r"(?m)^pid=(\d+)\s*$", lock_text)
                        if match:
                            lock_pid = int(match.group(1))
                    except OSError:
                        lock_pid = None

                    # Never steal an old lock from a still-running generator.
                    # The lock is expected to be held only for the tiny
                    # check+reserve critical section, so an old lock whose
                    # owner PID is gone is safe to recover after the TTL.
                    if lock_pid is not None and _cross_run_pid_alive(lock_pid):
                        time.sleep(0.05)
                        continue

                    # If the lock is old but its owner PID cannot be read,
                    # treat it as stale only when it is clearly orphaned.
                    # This protects against permanent blockage after a hard
                    # crash while remaining conservative about live locks.
                    if lock_pid is not None or age >= (_CROSS_RUN_STALE_LOCK_SECONDS * 2):
                        lock_path.unlink()
                        print(
                            f"[CROSS-RUN STORY] stale reservation lock recovered | "
                            f"age={age:.1f}s | pid={lock_pid}"
                        )
                        continue
            except FileNotFoundError:
                continue
            except OSError:
                pass
            time.sleep(0.05)

    raise RuntimeError("Cross-run story reservation lock is busy.")

def _cross_run_pid_alive(pid):
    """Return whether a local process PID currently exists."""
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (ValueError, TypeError, ProcessLookupError, PermissionError, OSError):
        return False


def _cross_run_claims(claims_dir):
    """Load active claims and recover only genuinely stale reservations.

    A claim is stale only when it is older than the generous TTL AND its
    recorded process is no longer alive. This keeps healthy slow generations
    protected while allowing restart/crash recovery.
    """
    claims = []
    now = time.time()
    try:
        paths = sorted(claims_dir.glob("*.json"))
    except Exception:
        return claims

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        try:
            age = now - float(data.get("created_at"))
        except (TypeError, ValueError):
            age = 0.0
        pid = data.get("pid")
        stale = age >= _CROSS_RUN_STALE_RESERVATION_SECONDS and not _cross_run_pid_alive(pid)

        if stale:
            try:
                path.unlink()
                print(f"[CROSS-RUN STORY] stale reservation recovered | age={age:.1f}s | pid={pid} | file={path}")
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[CROSS-RUN STORY] stale reservation cleanup failed | file={path} | {exc}")
            continue

        claims.append((path, data))

    return claims

def _reserve_cross_run_story(topic_title, evidence_lock):
    """
    Atomically reserve a concrete story across production runs.

    Returns:
      {"reserved": True, "path": "..."} -> reservation acquired.
      dict without "reserved" -> an earlier/concurrent story is already covered.
    """
    if not isinstance(evidence_lock, dict):
        return None

    claims_dir = TREND_DIR / _CROSS_RUN_STORY_CLAIMS_DIRNAME
    lock_path = None
    try:
        lock_path = _cross_run_claim_lock()
        existing_titles = _extract_existing_article_titles()

        # First use the same conservative title identity logic already used for
        # published HTML. This catches identical/near-identical Chris Rokos titles.
        for item in existing_titles:
            matched, ratio, common_count, coverage = _existing_story_title_match(
                topic_title, item["title"]
            )
            if matched:
                return {
                    "reason": "published_story_title_match",
                    "existing_title": item["title"],
                    "path": item["path"],
                    "ratio": ratio,
                    "common_tokens": common_count,
                    "coverage": coverage,
                }

        # Then inspect persistent evidence identities. This catches paraphrased
        # headlines and concurrent runs whose titles differ.
        for path, claim in _cross_run_claims(claims_dir):
            prior_title = str(claim.get("topic_title", "") or "").strip()
            if prior_title:
                matched, ratio, common_count, coverage = _existing_story_title_match(
                    topic_title, prior_title
                )
                if matched:
                    return {
                        "reason": "reserved_story_title_match",
                        "existing_title": prior_title,
                        "path": str(path),
                        "ratio": ratio,
                        "common_tokens": common_count,
                        "coverage": coverage,
                    }

            if _cross_run_evidence_duplicate(
                evidence_lock,
                claim.get("evidence_lock"),
            ):
                return {
                    "reason": "reserved_story_evidence_match",
                    "existing_title": prior_title,
                    "path": str(path),
                }

        # Unique filename avoids collisions between different stories while the
        # lock guarantees that the duplicate scan + reservation is atomic.
        fingerprint = hashlib.sha256(
            (
                str(topic_title or "").strip().casefold()
                + "\n"
                + "\n".join(
                    sorted(
                        str(f).strip().casefold()
                        for f in evidence_lock.get("facts", [])
                        if str(f).strip()
                    )
                )
                + f"\n{os.getpid()}\n{time.time_ns()}"
            ).encode("utf-8")
        ).hexdigest()

        claim_path = claims_dir / f"{fingerprint}.json"
        claim_path.write_text(
            json.dumps(
                {
                    "topic_title": str(topic_title or "").strip(),
                    "evidence_lock": evidence_lock,
                    "created_at": time.time(),
                    "pid": os.getpid(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"reserved": True, "path": str(claim_path)}
    finally:
        if lock_path is not None:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _release_cross_run_story_reservation(reservation):
    """Release only reservations that never reached successful publication."""
    if not isinstance(reservation, dict) or not reservation.get("reserved"):
        return
    path = reservation.get("path")
    if not path:
        return
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass




# ============================================================
# Evidence -> Generation Coverage
# ============================================================

_EVIDENCE_COVERAGE_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this",
    "was", "were", "has", "have", "had", "are", "is",
    "its", "into", "after", "before", "over", "under",
    "about", "than", "then", "they", "their", "them",
    "there", "which", "while", "also", "been", "being",
    "will", "would", "could", "should", "said", "says",
    "according", "official", "officials", "new", "latest",
    "news", "report", "reports", "story", "article",
}


def _evidence_coverage_tokens(text):
    """Return conservative meaningful tokens for evidence coverage."""
    normalized = unicodedata.normalize(
        "NFKD",
        str(text or ""),
    ).encode("ascii", "ignore").decode("ascii").casefold()

    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 3
        and token not in _EVIDENCE_COVERAGE_STOPWORDS
    }


def _evidence_coverage_numbers(text):
    """Return factual numeric tokens used as a strong coverage signal."""
    return set(
        re.findall(
            r"\b\d+(?:[.,:/-]\d+)*%?\b",
            str(text or ""),
        )
    )


def _check_evidence_generation_coverage(article, generation_evidence):
    """
    Lightweight deterministic post-generation evidence coverage check.

    Purpose:
      Verify that locked evidence facts are actually expressed by the
      generated article.

    This is NOT:
      - a factuality checker
      - an LLM judge
      - a repair mechanism
      - a regeneration mechanism

    Policy:
      - every locked evidence fact is mandatory;
      - 100% coverage is required;
      - missing facts cause rejection;
      - no repair and no regeneration are attempted.
    """
    facts = (
        generation_evidence.get("facts", [])
        if isinstance(generation_evidence, dict)
        else []
    )

    paragraphs = (
        article.get("paragraphs", [])
        if isinstance(article, dict)
        else []
    )

    facts = [
        fact
        for fact in facts
        if isinstance(fact, dict)
        and str(fact.get("id", "")).strip()
        and str(fact.get("fact", "")).strip()
    ]

    paragraphs = [
        str(paragraph).strip()
        for paragraph in paragraphs
        if str(paragraph).strip()
    ]

    if not facts:
        result = {
            "status": "PASS",
            "locked_facts": 0,
            "covered_facts": 0,
            "coverage": 1.0,
            "missing_fact_ids": [],
            "covered_fact_ids": [],
            "support_debug": {},
        }

        monitor.candidate_event(
            "evidence_generation_coverage",
            **result,
        )

        print(
            "[EVIDENCE -> GENERATION COVERAGE] PASS | "
            "locked=0 | covered=0 | coverage=1.000"
        )

        return result

    if not paragraphs:
        missing_ids = [
            str(fact["id"]).strip()
            for fact in facts
        ]

        result = {
            "status": "REJECT",
            "locked_facts": len(facts),
            "covered_facts": 0,
            "coverage": 0.0,
            "missing_fact_ids": missing_ids,
            "covered_fact_ids": [],
            "support_debug": {},
        }

        monitor.candidate_event(
            "evidence_generation_coverage",
            **result,
        )

        print(
            "[EVIDENCE -> GENERATION COVERAGE] REJECT | "
            f"locked={len(facts)} | covered=0 | "
            f"coverage=0.000 | missing={missing_ids}"
        )

        return result

    article_text = " ".join(paragraphs)
    article_tokens = _evidence_coverage_tokens(article_text)
    article_numbers = _evidence_coverage_numbers(article_text)

    covered_ids = []
    missing_ids = []
    support_debug = {}

    for fact in facts:
        fact_id = str(fact["id"]).strip()
        fact_text = str(fact.get("fact", "")).strip()
        excerpt = str(fact.get("excerpt", "")).strip()

        reference_tokens = (
            _evidence_coverage_tokens(fact_text)
            | _evidence_coverage_tokens(excerpt)
        )

        fact_numbers = (
            _evidence_coverage_numbers(fact_text)
            | _evidence_coverage_numbers(excerpt)
        )

        overlap = len(reference_tokens & article_tokens)
        token_coverage = (
            overlap / len(reference_tokens)
            if reference_tokens
            else 0.0
        )
        number_match = bool(fact_numbers & article_numbers)

        # Conservative coverage rule:
        #   - numbered facts require the number + at least two meaningful tokens;
        #   - short facts require two meaningful tokens;
        #   - longer facts require at least three meaningful tokens and 25% coverage.
        if fact_numbers:
            covered = number_match and overlap >= 2
        elif len(reference_tokens) <= 4:
            covered = overlap >= 2
        else:
            covered = overlap >= 3 and token_coverage >= 0.25

        support_debug[fact_id] = {
            "overlap": overlap,
            "reference_tokens": len(reference_tokens),
            "token_coverage": round(token_coverage, 3),
            "number_match": number_match,
            "covered": covered,
        }

        if covered:
            covered_ids.append(fact_id)
        else:
            missing_ids.append(fact_id)

    coverage = len(covered_ids) / len(facts) if facts else 1.0
    status = "PASS" if not missing_ids else "REJECT"

    result = {
        "status": status,
        "locked_facts": len(facts),
        "covered_facts": len(covered_ids),
        "coverage": coverage,
        "missing_fact_ids": missing_ids,
        "covered_fact_ids": covered_ids,
        "support_debug": support_debug,
    }

    print(
        f"[EVIDENCE -> GENERATION COVERAGE] {status} | "
        f"locked={len(facts)} | covered={len(covered_ids)} | "
        f"coverage={coverage:.3f} | missing={missing_ids}"
    )

    monitor.candidate_event(
        "evidence_generation_coverage",
        status=status,
        locked_facts=len(facts),
        covered_facts=len(covered_ids),
        coverage=coverage,
        missing_fact_ids=missing_ids,
    )

    return result


def generate_valid_article(prompt=None, reference_date=None, trend=None, prelocked_evidence=None):
    """Generate once and validate without fact repair or regeneration.

    Locked evidence is authoritative at extraction/lock time. After the single
    generation pass, a lightweight deterministic coverage check verifies that
    every locked evidence fact is expressed by the generated article.
    """
    try:
        generation_evidence = _enrich_evidence_for_generation(
            prelocked_evidence,
            trend,
        )

        article = generate(
            "",
            evidence=generation_evidence,
        )

        article = _sanitize_article_markdown(article)
        validate_article(article)

        validate_article_structure(
            article,
            generation_evidence,
            label="Initial newsroom article",
        )

        locked_facts = generation_evidence.get("facts", [])
        core_fact_ids = generation_evidence.get("core_fact_ids", [])
        supporting_fact_ids = generation_evidence.get("supporting_fact_ids", [])
        paragraph_text = " ".join(
            str(p) for p in article.get("paragraphs", [])
        ).strip()

        if isinstance(locked_facts, list) and len(locked_facts) >= 1:
            print(
                f"[EVIDENCE COVERAGE] locked_facts={len(locked_facts)} | "
                f"core_facts={len(core_fact_ids)} | "
                f"supporting_facts={len(supporting_fact_ids)} | "
                f"article_words={len(paragraph_text.split())}"
            )

        evidence_coverage = _check_evidence_generation_coverage(
            article,
            generation_evidence,
        )

        article.pop("_declared_fact_ids", None)

        if evidence_coverage.get("status") != "PASS":
            print(
                "[EVIDENCE -> GENERATION COVERAGE] FAIL | "
                "article discarded; NO REPAIR; NO REGENERATION"
            )
            raise Exception(
                "Generated article omitted one or more locked evidence facts."
            )

        normalized_headline = _shorten_headline(
            article.get("title", "")
        )
        article["title"] = normalized_headline
        article["h1"] = normalized_headline

        article = enforce_headline_policy(article, trend)
        validate_article(article)
        validate_language_integrity(article)
        print("[LANGUAGE GUARD] PASS")

        repetition = _run_repetition_guard(
            article,
            generation_evidence,
        )

        if repetition.get("status") != "PASS":
            print(
                "[REPETITION GUARD] FAIL — "
                "article discarded; NO REPAIR"
            )
            print(
                json.dumps(
                    repetition,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise Exception(
                "Repetition Guard blocked article; publication blocked."
            )

        validate_language_integrity(article)
        print("[LANGUAGE GUARD] FINAL ARTICLE PASS")

        article["_evidence_generation_coverage"] = evidence_coverage
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


def _fetch_broad_news_fallback_seeds():
    """Fetch broad FR news leads when direct-source discovery is insufficient.

    Direct publisher RSS remains first choice. This fallback is supplementary
    only: every returned lead still passes the normal deterministic eligibility,
    story discovery, evidence, substantive-value, and generation gates.
    """
    fallback_queries = {
        "en": ["latest news", "breaking news", "top news"],
        "en-us": ["latest news", "breaking news", "top news"],
        "de": ["aktuelle nachrichten", "eilmeldungen", "top nachrichten"],
        "es": ["últimas noticias", "última hora", "principales noticias"],
        "it": ["ultime notizie", "ultim'ora", "principali notizie"],
        "fr": ["dernières actualités", "actualité dernière minute", "actualités France"],
        "pt": ["últimas notícias", "última hora", "principais notícias"],
        "pt-br": ["últimas notícias", "última hora", "principais notícias"],
        "id": ["berita terbaru", "berita terkini", "berita utama"],
    }.get(str(LANGUAGE or "").strip().casefold(), ["latest news", "breaking news", "top news"])

    fallback = fetch_news_discovery(
        fallback_queries,
        per_query_limit=8,
        max_results=12,
    )
    seeds = []
    for item in fallback:
        seed = dict(item)
        seed["discovery_provider"] = "google_news_bing_fallback"
        seed["discovery_source"] = str(item.get("source") or "").strip()
        seed["discovery_context"] = "broad_news_fallback"
        seeds.append(seed)

    print(
        f"[SOURCE FIRST] Broad news fallback | "
        f"queries={len(fallback_queries)} | seeds={len(seeds)}"
    )
    return seeds


def main():
    monitor.start_run(language=LANGUAGE, model=MODEL, pipeline="universal-evidence-lock-v3.0-ministral-all-facts-no-word-floor", max_articles=MAX_ARTICLES_PER_RUN)
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
        # point of failure. A weak direct feed is not considered healthy merely
        # because it returned one or two fresh rows; those rows may all be
        # processed, irrelevant, or fail downstream story quality. The broad
        # Google News -> Bing mechanism is therefore available as a supplementary
        # discovery pool when the direct-source reservoir is insufficient.
        if not trends:
            trends = _fetch_broad_news_fallback_seeds()
            print(
                f"[SOURCE FIRST] Direct RSS empty -> broad news fallback | "
                f"seeds={len(trends)}"
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
    run_started_monotonic = time.monotonic()
    concrete_candidates_processed = 0
    budget_exhausted = False

    def _run_budget_exhausted(stage=""):
        nonlocal budget_exhausted
        elapsed = time.monotonic() - run_started_monotonic
        if elapsed >= RUN_TIME_BUDGET_SECONDS:
            if not budget_exhausted:
                budget_exhausted = True
                print(f"[RUN BUDGET] HARD STOP | elapsed={elapsed:.1f}s | budget={RUN_TIME_BUDGET_SECONDS}s | stage={stage}")
            return True
        return False

    print(f"[RUN BUDGET] limit={RUN_TIME_BUDGET_SECONDS}s | max_concrete_candidates={MAX_CONCRETE_STORY_CANDIDATES_PER_RUN}")

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

    # A direct RSS feed can be technically non-empty while yielding no usable
    # production candidates after deterministic filtering. Treat that as a weak
    # discovery result and supplement it with the existing Google News -> Bing
    # fallback. Direct-source candidates remain first in the queue.
    if source_first and len(candidate_trends) < max(4, MAX_ARTICLES_PER_RUN * 4):
        fallback_seeds = _fetch_broad_news_fallback_seeds()
        if fallback_seeds:
            existing_keys = {
                (
                    str(item.get("url") or item.get("link") or "").strip().casefold(),
                    _norm(item.get("title")),
                )
                for item in trends
                if isinstance(item, dict)
            }
            added = 0
            for seed in fallback_seeds:
                key = (
                    str(seed.get("url") or seed.get("link") or "").strip().casefold(),
                    _norm(seed.get("title")),
                )
                if key in existing_keys:
                    continue
                trends.append(seed)
                existing_keys.add(key)
                added += 1

            if added:
                candidate_trends = _deterministic_production_reservoir(
                    trends,
                    processed,
                )
                print(
                    f"[SOURCE FIRST] Weak direct reservoir -> fallback supplemented | "
                    f"added={added} | production_candidates={len(candidate_trends)}"
                )

    print(
        f"[STORY DISCOVERY] seeds={len(candidate_trends)} | "
        f"article_target={MAX_ARTICLES_PER_RUN}"
    )

    for trend in candidate_trends:
        if generated >= MAX_ARTICLES_PER_RUN:
            break
        if _run_budget_exhausted("before_seed"):
            break

        keyword = trend["title"]

        # Cheap deterministic sports/local-news exclusion MUST happen before any news
        # retrieval. Excluded topic types must not consume RSS/network capacity.
        if _is_sports_topic(keyword):
            print(f"[TrendCurrent] SKIP sports topic: {keyword}")
            continue
        if _is_local_news_topic(keyword):
            print(f"[TrendCurrent] SKIP local-news topic: {keyword}")
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
            if _is_local_news_topic(keyword, news):
                print(f"[TrendCurrent] SKIP local-news topic after relevance: {keyword}")
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
                if concrete_candidates_processed >= MAX_CONCRETE_STORY_CANDIDATES_PER_RUN:
                    budget_exhausted = True
                    print(f"[RUN BUDGET] HARD CANDIDATE STOP | processed={concrete_candidates_processed} | limit={MAX_CONCRETE_STORY_CANDIDATES_PER_RUN}")
                    break
                if _run_budget_exhausted("before_candidate"):
                    break
                concrete_candidates_processed += 1

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

                cross_run_reservation = None
                cross_run_published = False

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

                    if _run_budget_exhausted("before_evidence_extraction"):
                        raise RuntimeError("run time budget exhausted before evidence extraction")
                    evidence_source = _build_evidence_source(selected_news, {"selected_indices": list(range(len(selected_news)))})
                    evidence_source_chars = sum(
                        len(str(value or ""))
                        for item in evidence_source.get("articles", [])
                        for value in (
                            item.get("title", ""),
                            item.get("source", ""),
                            item.get("published", ""),
                            item.get("summary", ""),
                            item.get("description", ""),
                            item.get("content", ""),
                        )
                    )
                    print(
                        f"[TOPIC FILTER] Evidence source prepared | "
                        f"source_chars={evidence_source_chars}"
                    )
                    evidence_lock = extract_evidence(evidence_source)

                    temporal_event_check = _deterministic_temporal_event_guard(
                        evidence_lock,
                        trend=trend,
                        reference_date=reference_date,
                    )
                    if temporal_event_check.get("status") != "PASS":
                        trend["_production_status"] = "REJECT"
                        trend["_production_reject_reason"] = temporal_event_check.get(
                            "reason", "temporal/event guard rejected evidence"
                        )
                        monitor.candidate_event(
                            "temporal_event_guard",
                            **temporal_event_check,
                        )
                        monitor.finish_candidate(
                            "REJECT",
                            reason=f"temporal_event_guard={temporal_event_check.get('reason')}"
                        )
                        continue

                    monitor.candidate_event(
                        "temporal_event_guard",
                        **temporal_event_check,
                    )

                    # FAST PATH:
                    # extract_evidence() now returns the authoritative deduplicated
                    # evidence + lineage state. Do not run a second generation-side
                    # semantic dedup pass over the same locked facts.
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
                            evidence_source_chars=evidence_source_chars,
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
                        evidence_source_chars=evidence_source_chars,
                        evidence_facts=locked_facts,
                        evidence_fact_count=evidence_fact_count,
                        unique_information_units=unique_information_units,
                    )

                    # CROSS-RUN STORY PROTECTION:
                    # Reserve the concrete story only after evidence is locked and
                    # substantive value passes, but before expensive generation.
                    # This closes both sequential and concurrent duplicate runs.
                    cross_run_reservation = _reserve_cross_run_story(
                        keyword,
                        evidence_lock,
                    )
                    if not cross_run_reservation.get("reserved"):
                        trend["_production_status"] = "REJECT"
                        trend["_production_reject_reason"] = "story already covered across production runs"
                        monitor.candidate_event(
                            "cross_run_story_check",
                            status="REJECT",
                            reason=cross_run_reservation.get("reason"),
                            existing_title=cross_run_reservation.get("existing_title"),
                            existing_file=cross_run_reservation.get("path"),
                            similarity=cross_run_reservation.get("ratio"),
                            common_tokens=cross_run_reservation.get("common_tokens"),
                            coverage=cross_run_reservation.get("coverage"),
                        )
                        monitor.finish_candidate(
                            "REJECT",
                            reason="story already covered across production runs",
                        )
                        continue

                    monitor.candidate_event(
                        "cross_run_story_check",
                        status="PASS",
                        reason="new story reservation acquired",
                        reservation_file=cross_run_reservation.get("path"),
                    )

                    print(
                        f"[TOPIC FILTER] EVIDENCE USABILITY PASS | "
                        f"facts={evidence_fact_count} | {keyword} | story={story_number}"
                    )
                    monitor.candidate_event(
                        "evidence_locked",
                        news_count=len(news),
                        selected_source_indices=story_selection.get("selected_indices", []),
                        selected_source_count=story_selection.get("selected_count"),
                        evidence_source_chars=evidence_source_chars,
                        evidence_facts=locked_facts,
                        evidence_fact_count=evidence_fact_count,
                        evidence_lock=evidence_lock,
                    )

                    if _run_budget_exhausted("before_article_generation"):
                        raise RuntimeError("run time budget exhausted before article generation")
                    article = generate_valid_article(
                        reference_date=reference_date,
                        trend=trend,
                        prelocked_evidence=evidence_lock,
                    )

                    evidence_coverage = article.pop(
                        "_evidence_generation_coverage",
                        {},
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
                        evidence_coverage=evidence_coverage.get("coverage"),
                        evidence_covered_facts=evidence_coverage.get("covered_facts"),
                        evidence_missing_fact_ids=evidence_coverage.get("missing_fact_ids"),
                    )

                    base_slug = slugify(keyword)
                    if not base_slug:
                        raise ValueError("Article slug is empty after normalization.")

                    # One trend may legitimately yield multiple concrete stories.
                    # Never let a later story overwrite the earlier story's HTML.
                    # Keep the first story's normal SEO slug and suffix subsequent
                    # candidates deterministically. If a suffixed slug already exists,
                    # advance until the filesystem path is unused.
                    story_number = int(trend.get("_story_number", 1) or 1)
                    story_count = int(trend.get("_story_candidate_count", 1) or 1)
                    slug = base_slug if story_count <= 1 and story_number <= 1 else f"{base_slug}-story-{story_number}"
                    slug_path = TREND_DIR / f"{slug}.html"
                    suffix = 2
                    while slug_path.exists():
                        slug = f"{base_slug}-story-{story_number}-{suffix}"
                        slug_path = TREND_DIR / f"{slug}.html"
                        suffix += 1
                        if suffix > 1000:
                            raise ValueError("Unable to allocate a unique article slug.")
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
                    cross_run_published = True

                    new_keywords.append(keyword)
                    generated += 1

                    print(f"OK -> {slug}.html")
                    monitor.finish_candidate("PASS", slug=slug)

                except Exception as e:
                    if cross_run_reservation and not cross_run_published:
                        _release_cross_run_story_reservation(cross_run_reservation)
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

    elapsed_total = time.monotonic() - run_started_monotonic
    final_status = "TIME_BUDGET" if budget_exhausted else "FINISHED"
    print(f"Finished. Generated {generated} article(s). | elapsed={elapsed_total:.1f}s | candidates={concrete_candidates_processed} | status={final_status}")
    monitor.end_run(generated=generated, status=final_status)

    if generated:
        git_push()


if __name__ == "__main__":
    main()
