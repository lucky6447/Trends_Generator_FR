import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re

from config import RSS_URL

# Production hard freshness gate: only trends <= 24 hours old are eligible.
MAX_TREND_AGE_HOURS = 24.0


SKIP_KEYWORDS = {
    # Live / резултати
    "live score",
    "livescore",
    "results",
    "fixtures",
    "fixture",
    "standings",
    "table",

    # Залагания
    "odds",
    "betting",
    "bookmaker",
    "bet365",
    "1xbet",
    "tips",
    "prediction",
    "predictions",

    # Потоци
    "live stream",
    "stream",
    "streaming",
    "watch live",

    # Промоции
    "promo code",
    "coupon",
    "bonus code",
}


def clean(value):
    if not value:
        return ""
    return " ".join(str(value).split())


def should_skip(title):
    title = clean(title).casefold()
    return any(keyword in title for keyword in SKIP_KEYWORDS)


def _published_value(item):
    """Use the first valid RSS publication/update field available."""
    for field in ("published", "pubDate", "updated", "date", "dc_date"):
        value = item.get(field)
        if value:
            return clean(value)
    return ""


def _title_key(title):
    value = clean(title).casefold()
    value = re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _age_hours(published):
    if not published:
        return None
    try:
        dt = parsedate_to_datetime(str(published))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0)
    except Exception:
        return None


NEWS_ITEM_KEYS = ("ht_news_item", "news_item")


def _as_mapping(value):
    return value if hasattr(value, "get") else None


def _iter_news_items(item):
    """Yield Google Trends related-news objects in a parser-tolerant way."""
    candidates = []
    for key in NEWS_ITEM_KEYS:
        value = item.get(key)
        if value:
            candidates.append(value)

    flattened = {
        "title": clean(item.get("ht_news_item_title")),
        "url": clean(item.get("ht_news_item_url")),
        "source": clean(item.get("ht_news_item_source")),
        "snippet": clean(item.get("ht_news_item_snippet")),
    }
    if any(flattened.values()):
        candidates.append(flattened)

    for value in candidates:
        values = value if isinstance(value, (list, tuple)) else [value]
        for raw in values:
            mapping = _as_mapping(raw)
            if mapping is None:
                continue
            title = clean(mapping.get("ht_news_item_title") or mapping.get("title"))
            url = clean(mapping.get("ht_news_item_url") or mapping.get("url"))
            source = clean(mapping.get("ht_news_item_source") or mapping.get("source"))
            snippet = clean(mapping.get("ht_news_item_snippet") or mapping.get("snippet"))
            if title or url or source or snippet:
                yield {"title": title, "url": url, "source": source, "snippet": snippet}


def _related_news(item):
    """Return clean, unique Google Trends related-news metadata."""
    out = []
    seen = set()
    for news in _iter_news_items(item):
        title = clean(news.get("title"))
        if not title or should_skip(title):
            continue
        key = _title_key(title)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({
            "title": title,
            "url": clean(news.get("url")),
            "source": clean(news.get("source")),
            "snippet": clean(news.get("snippet")),
        })
    return out


def _build_discovery_context(related_news):
    """Build bounded metadata context without changing the canonical trend title."""
    parts = []
    for item in related_news[:4]:
        headline = clean(item.get("title"))
        source = clean(item.get("source"))
        if headline:
            parts.append(f"{headline} | {source}" if source else headline)
    return " || ".join(parts)


def fetch_trends():
    """Return fresh, non-duplicate Trends signals with related-news metadata."""
    try:
        feed = feedparser.parse(RSS_URL)
    except Exception as exc:
        print(f"[TREND RSS] parse failed: {exc}")
        return []

    trends = []
    dropped_old = 0
    dropped_unknown_age = 0
    dropped_duplicates = 0
    seen_titles = set()
    seen_links = set()
    feed_link_key = clean(RSS_URL).casefold()
    enriched_from_news = 0

    for item in getattr(feed, "entries", []) or []:
        title = clean(item.get("title"))
        published = _published_value(item)
        link = clean(item.get("link"))

        if not title or should_skip(title):
            continue

        age = _age_hours(published)
        if age is None:
            dropped_unknown_age += 1
            continue

        if age > MAX_TREND_AGE_HOURS:
            dropped_old += 1
            continue

        related_news = _related_news(item)
        discovery_context = _build_discovery_context(related_news)
        if related_news:
            enriched_from_news += 1

        title_key = _title_key(title)
        link_key = link.casefold()

        # Google Trends RSS can repeat the feed URL in every item's <link>.
        # That URL identifies the feed, not the individual trend.
        meaningful_link = bool(link_key and link_key != feed_link_key)

        # Title is the primary item identity. Use the link only when it is
        # genuinely item-specific, preventing 10 entries -> 1 kept when
        # every RSS item shares the same feed URL.
        if title_key in seen_titles or (meaningful_link and link_key in seen_links):
            dropped_duplicates += 1
            continue

        seen_titles.add(title_key)
        if meaningful_link:
            seen_links.add(link_key)

        trends.append({
            # Canonical identity ALWAYS remains the original Google Trends title.
            "title": title,
            "trend_title": title,
            "news": related_news,
            "discovery_context": discovery_context,
            "link": link,
            "published": published,
            "traffic": clean(item.get("ht_approx_traffic")),
            "age_hours": round(age, 2),
        })

    print(
        f"[TREND FRESHNESS] RSS entries={len(getattr(feed, 'entries', []) or [])} | "
        f"kept={len(trends)} | old>{MAX_TREND_AGE_HOURS:g}h={dropped_old} | "
        f"unknown_age={dropped_unknown_age} | duplicates={dropped_duplicates} | "
        f"related_news_enriched={enriched_from_news}"
    )

    return trends
