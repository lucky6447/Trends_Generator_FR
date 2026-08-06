import re
import feedparser
import requests
import trafilatura

from collections import Counter
from urllib.parse import quote_plus


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def clean(value):
    if not value:
        return ""

    value = re.sub(r"<[^>]+>", " ", str(value))
    value = value.replace("&nbsp;", " ")
    value = value.replace("&amp;", "&")

    return " ".join(value.split())


def extract_article(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
            allow_redirects=True
        )

        final_url = response.url

        downloaded = trafilatura.fetch_url(final_url)

        if downloaded:

            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                include_links=False
            )

            if text:
                return clean(text)[:5000]

    except Exception:
        pass

    return ""


def filter_similar_articles(articles):

    if len(articles) < 5:
        return []

    words = []

    for article in articles:

        tokens = re.findall(
            r"[a-zA-Z0-9]+",
            article["title"].lower()
        )

        tokens = [
            t for t in tokens
            if len(t) > 3
        ]

        words.extend(tokens)

    common = Counter(words)

    keywords = {
        word
        for word, count in common.items()
        if count >= 3
    }

    filtered = []

    for article in articles:

        title = article["title"].lower()

        if any(word in title for word in keywords):
            filtered.append(article)

    if len(filtered) < 5:
        return []

    return filtered


def fetch_news(query, limit=10):

    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query + ' when:7d')}"
        "&hl=fr"
        "&gl=FR"
        "&ceid=FR:fr"
    )

    feed = feedparser.parse(url)

    articles = []

    for item in feed.entries[:limit]:

        source = ""

        if hasattr(item, "source"):
            source = clean(item.source.get("title"))

        link = clean(item.get("link"))

        content = extract_article(link)

        articles.append({
            "title": clean(item.get("title")),
            "summary": clean(item.get("summary")),
            "content": content,
            "source": source,
            "link": link,
            "published": clean(item.get("published")),
        })

    return filter_similar_articles(articles)