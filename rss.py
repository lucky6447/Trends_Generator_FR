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


def fetch_trends():
    """Return only fresh, non-duplicate trend signals with normalized metadata."""
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

        title_key = _title_key(title)
        link_key = link.casefold()
        if title_key in seen_titles or (link_key and link_key in seen_links):
            dropped_duplicates += 1
            continue

        seen_titles.add(title_key)
        if link_key:
            seen_links.add(link_key)

        trends.append({
            "title": title,
            "link": link,
            "published": published,
            "traffic": clean(item.get("ht_approx_traffic")),
            "age_hours": round(age, 2),
        })

    print(
        f"[TREND FRESHNESS] RSS entries={len(getattr(feed, 'entries', []) or [])} | "
        f"kept={len(trends)} | old>{MAX_TREND_AGE_HOURS:g}h={dropped_old} | "
        f"unknown_age={dropped_unknown_age} | duplicates={dropped_duplicates}"
    )

    return trends
