"""
TrendCurrent Topic Priority Filter v1.0

Purpose:
    Select the small number of trends that deserve expensive news retrieval
    and article-generation capacity.

This module is intentionally independent from:
    - universal-fact-lock / ollama_client
    - Fact Guard
    - article generation
    - index generation

It is a deterministic ranking layer. It does NOT generate articles.

Pipeline:
    ALL TRENDS
        -> cheap pre-score
        -> top candidate pool
        -> fetch_news only for candidates
        -> source-quality re-score
        -> article generation only for final winners
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

try:
    from config import COUNTRY
except Exception:
    COUNTRY = os.getenv("COUNTRY", "")

# These defaults are deliberately conservative. They can be changed with
# environment variables without touching the stable factual pipeline.
TOPIC_CANDIDATE_MULTIPLIER = max(
    2, int(os.getenv("TOPIC_CANDIDATE_MULTIPLIER", "4"))
)
TOPIC_MAX_CANDIDATES = max(
    8, int(os.getenv("TOPIC_MAX_CANDIDATES", "24"))
)
TOPIC_MIN_PRE_SCORE = float(os.getenv("TOPIC_MIN_PRE_SCORE", "42"))
TOPIC_MIN_FINAL_SCORE = float(os.getenv("TOPIC_MIN_FINAL_SCORE", "55"))

# Production GEO priority. During the current UK/US experiment this gives
# those markets a meaningful advantage without hard-rejecting other stories.
DEFAULT_PRIORITY_GEOS = ("US", "UK")
_priority_geos_env = os.getenv("TOPIC_PRIORITY_GEOS", "")
PRIORITY_GEOS = tuple(
    x.strip().upper()
    for x in (_priority_geos_env.split(",") if _priority_geos_env else DEFAULT_PRIORITY_GEOS)
    if x.strip()
)

SKIP_PATTERNS = (
    " vs ",
    " v ",
    " live",
    " score",
    " result",
    " calendario",
    " alineación",
    " prognóstico",
    " stream",
    " streaming",
)

NOISE_PATTERNS = (
    r"\bhoroscope\b",
    r"\bhoroscopes\b",
    r"\blottery results?\b",
    r"\bweather forecast\b",
    r"\bexchange rate\b",
    r"\bprice forecast\b",
    r"\bstock forecast\b",
    r"\bflight status\b",
)

NEWS_PATTERNS = (
    r"\bannounc(?:e|ed|es|ement)\b",
    r"\bconfirm(?:ed|s|ation)?\b",
    r"\bappoint(?:ed|ment|s)?\b",
    r"\bsign(?:ed|s|ing)?\b",
    r"\bjoin(?:ed|s|ing)?\b",
    r"\bwin(?:s|ner|ning)?\b",
    r"\bwon\b",
    r"\bdefeat(?:ed|s)?\b",
    r"\bkill(?:ed|s)?\b",
    r"\bdie(?:d|s)?\b",
    r"\bdeath\b",
    r"\barrest(?:ed|s)?\b",
    r"\bresign(?:ed|s|ation)?\b",
    r"\blaunch(?:ed|es)?\b",
    r"\brelease(?:d|s)?\b",
    r"\bcrash(?:ed|es)?\b",
    r"\bearthquake\b",
    r"\bhurricane\b",
    r"\bstorm\b",
    r"\bfire\b",
    r"\bshooting\b",
    r"\battack\b",
    r"\belection\b",
    r"\bdecision\b",
    r"\bcourt\b",
    r"\btrial\b",
    r"\bpolice\b",
    r"\bgovernment\b",
    r"\bminister\b",
    r"\bpresident\b",
)

GEO_TERMS = {
    "US": ("united states", "u.s.", "u.s", "america", "american", "usa"),
    "UK": ("united kingdom", "uk", "britain", "british", "england", "scotland", "wales"),
    "CA": ("canada", "canadian"),
    "AU": ("australia", "australian"),
}

def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def _norm(text: Any) -> str:
    return _clean(text).casefold()

def _tokens(text: Any) -> Set[str]:
    return {
        t for t in re.findall(r"[a-z0-9']+", _norm(text))
        if len(t) >= 3
    }

def _traffic_value(trend: Dict[str, Any]) -> float:
    keys = (
        "traffic", "approx_traffic", "ht_approx_traffic",
        "approxTraffic", "traffic_value", "search_volume",
    )
    for key in keys:
        value = trend.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        m = re.search(r"([\d,.]+)\s*([KMB])?", str(value), re.I)
        if not m:
            continue
        try:
            n = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        suffix = (m.group(2) or "").upper()
        return n * {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix, 1)
    return 0.0

def _published_age_hours(trend: Dict[str, Any]) -> float | None:
    value = (
        trend.get("published")
        or trend.get("pubDate")
        or trend.get("published_at")
        or trend.get("updated")
    )
    if not value:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        # RSS / ISO formats used by feedparser and common trend feeds.
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(text)
        except Exception:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except Exception:
                return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600)

def _geo_score(title: str) -> Tuple[float, str]:
    t = _norm(title)
    country = _norm(COUNTRY)

    # Strong explicit current production GEO preference.
    for geo in PRIORITY_GEOS:
        if any(term in t for term in GEO_TERMS.get(geo, ())):
            return 15.0, geo

    # If config itself names a priority country, give a smaller local signal.
    for geo, terms in GEO_TERMS.items():
        if any(term in country for term in terms) and any(term in t for term in terms):
            return 12.0, geo

    return 5.0, "GLOBAL"

def _noise_penalty(title: str) -> float:
    t = _norm(title)
    if any(re.search(p, t) for p in NOISE_PATTERNS):
        return 30.0
    return 0.0

def _skip_reason(title: str) -> str | None:
    t = _norm(title)
    for pattern in SKIP_PATTERNS:
        if pattern in t:
            return f"existing skip pattern: {pattern.strip()}"
    if _noise_penalty(title) >= 30:
        return "low-value utility/noise topic"
    return None

def _pre_score(trend: Dict[str, Any]) -> Dict[str, Any]:
    title = _clean(trend.get("title"))
    traffic = _traffic_value(trend)
    age = _published_age_hours(trend)

    # 0-30: trend momentum. If the feed exposes traffic, use it. Otherwise
    # use a neutral baseline plus a small news-language signal.
    if traffic > 0:
        momentum = min(30.0, 8.0 + 8.0 * math.log10(max(1.0, traffic)))
    else:
        momentum = 12.0

    # 0-20: freshness.
    if age is None:
        freshness = 12.0
    elif age <= 3:
        freshness = 20.0
    elif age <= 6:
        freshness = 18.0
    elif age <= 12:
        freshness = 16.0
    elif age <= 24:
        freshness = 13.0
    elif age <= 48:
        freshness = 8.0
    else:
        freshness = 3.0

    geo, geo_name = _geo_score(title)

    newsworthiness = min(
        15.0,
        7.0 + 2.0 * sum(bool(re.search(p, _norm(title))) for p in NEWS_PATTERNS),
    )

    # Clear, entity-rich topics tend to be more usable than vague one-word
    # trends. This is a quality signal, not an article-fact claim.
    token_count = len(_tokens(title))
    clarity = min(10.0, 3.0 + token_count * 1.5)

    penalty = _noise_penalty(title)

    score = max(0.0, min(100.0, momentum + freshness + geo + newsworthiness + clarity - penalty))

    return {
        "title": title,
        "pre_score": round(score, 2),
        "momentum": round(momentum, 2),
        "freshness": round(freshness, 2),
        "geo": round(geo, 2),
        "geo_name": geo_name,
        "newsworthiness": round(newsworthiness, 2),
        "clarity": round(clarity, 2),
        "penalty": round(penalty, 2),
        "traffic": traffic,
        "age_hours": round(age, 2) if age is not None else None,
    }

def rank_trends(trends: Sequence[Dict[str, Any]], processed: Iterable[str], limit: int) -> List[Dict[str, Any]]:
    """
    Cheap first-pass ranking.

    This function does NOT fetch news and does NOT invoke Ollama.
    """
    processed_norm = {_norm(x) for x in processed}
    scored: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    for trend in trends:
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

        data = _pre_score(trend)
        if data["pre_score"] < TOPIC_MIN_PRE_SCORE:
            print(
                f"[TOPIC FILTER] DROP | {title} | "
                f"pre={data['pre_score']:.1f} < {TOPIC_MIN_PRE_SCORE:.1f}"
            )
            continue

        item = dict(trend)
        item["_topic"] = data
        scored.append(item)

    scored.sort(key=lambda x: x["_topic"]["pre_score"], reverse=True)

    pool_size = min(
        len(scored),
        max(limit * TOPIC_CANDIDATE_MULTIPLIER, 8),
        TOPIC_MAX_CANDIDATES,
    )
    selected = scored[:pool_size]

    print(
        f"[TOPIC FILTER] {len(trends)} trends -> "
        f"{len(scored)} viable -> {len(selected)} candidates before news retrieval"
    )

    for rank, item in enumerate(selected, 1):
        d = item["_topic"]
        print(
            f"[TOPIC FILTER] #{rank:02d} "
            f"{d['pre_score']:5.1f} | {d['geo_name']:6s} | {d['title']}"
        )

    return selected

def score_with_news(trend: Dict[str, Any], news: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Second-stage sourceability score.

    Uses only the news already fetched for this candidate.
    No additional network calls and no LLM.
    """
    base = dict(trend.get("_topic") or _pre_score(trend))
    news = [x for x in news if isinstance(x, dict)]

    count = len(news)
    source_names = {
        _norm(x.get("source"))
        for x in news
        if _clean(x.get("source"))
    }
    content_count = sum(1 for x in news if len(_clean(x.get("content"))) >= 700)
    title_count = sum(1 for x in news if _clean(x.get("title")))

    # 0-20 sourceability: independent source count + usable article bodies.
    sourceability = min(20.0, count * 2.0 + min(5.0, len(source_names)) + min(5.0, content_count))

    # Strong penalty for a topic that has only one weak/duplicate source.
    evidence_penalty = 0.0
    if count < 2:
        evidence_penalty = 30.0
    elif len(source_names) <= 1:
        evidence_penalty = 6.0
    elif content_count == 0:
        evidence_penalty = 5.0

    final_score = max(
        0.0,
        min(
            100.0,
            base["pre_score"] * 0.78
            + sourceability
            - evidence_penalty,
        ),
    )

    result = {
        **base,
        "news_count": count,
        "unique_sources": len(source_names),
        "usable_content": content_count,
        "sourceability": round(sourceability, 2),
        "evidence_penalty": round(evidence_penalty, 2),
        "final_score": round(final_score, 2),
    }
    return result

def select_final_candidates(candidates: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    viable = []
    for item in candidates:
        d = item.get("_topic_final") or {}
        if d.get("final_score", 0) < TOPIC_MIN_FINAL_SCORE:
            print(
                f"[TOPIC FILTER] DROP AFTER NEWS | {item.get('title')} | "
                f"final={d.get('final_score', 0):.1f} < {TOPIC_MIN_FINAL_SCORE:.1f}"
            )
            continue
        viable.append(item)

    viable.sort(
        key=lambda x: (
            x["_topic_final"]["final_score"],
            x["_topic_final"].get("sourceability", 0),
            x["_topic_final"].get("freshness", 0),
        ),
        reverse=True,
    )

    selected = viable[:max(0, limit)]

    print(
        f"[TOPIC FILTER] {len(candidates)} candidates -> "
        f"{len(viable)} viable after source scoring -> "
        f"{len(selected)} article slots"
    )

    for rank, item in enumerate(selected, 1):
        d = item["_topic_final"]
        print(
            f"[TOPIC FILTER] FINAL #{rank:02d} "
            f"{d['final_score']:5.1f} | {d['geo_name']:6s} | "
            f"sources={d['unique_sources']} | {d['title']}"
        )

    return selected
