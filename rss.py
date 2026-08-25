import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from config import RSS_URL

# Production hard freshness gate: only trends <= 6 hours old are eligible.
MAX_TREND_AGE_HOURS = 6.0


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
    title = title.lower()
    return any(keyword in title for keyword in SKIP_KEYWORDS)


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
    feed = feedparser.parse(RSS_URL)

    trends = []
    dropped_old = 0
    dropped_unknown_age = 0

    for item in feed.entries:
        title = clean(item.get("title"))
        published = clean(item.get("published"))

        if not title:
            continue

        if should_skip(title):
            continue

        age = _age_hours(published)
        if age is None:
            # Strict experimental mode: unknown age is not considered fresh.
            dropped_unknown_age += 1
            continue

        if age > MAX_TREND_AGE_HOURS:
            dropped_old += 1
            continue

        trends.append({
            "title": title,
            "link": clean(item.get("link")),
            "published": published,
            "traffic": clean(item.get("ht_approx_traffic")),
            "age_hours": round(age, 2),
        })

    print(
        f"[TREND FRESHNESS] RSS entries={len(feed.entries)} | "
        f"kept={len(trends)} | old>{MAX_TREND_AGE_HOURS:g}h={dropped_old} | "
        f"unknown_age={dropped_unknown_age}"
    )

    return trends