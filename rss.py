import feedparser

from config import RSS_URL


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


def fetch_trends():
    feed = feedparser.parse(RSS_URL)

    trends = []

    for item in feed.entries:
        title = clean(item.get("title"))

        if not title:
            continue

        if should_skip(title):
            continue

        trends.append({
            "title": title,
            "link": clean(item.get("link")),
            "published": clean(item.get("published")),
            "traffic": clean(item.get("ht_approx_traffic")),
        })

    return trends