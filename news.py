import re
import feedparser
import requests
import trafilatura
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

HEADERS = {"User-Agent": "Mozilla/5.0"}

def clean(value):
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", str(value))
    value = value.replace("&nbsp;", " ").replace("&amp;", "&")
    return " ".join(value.split())

def extract_article(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=False, include_links=False
            )
            if text:
                return clean(text)[:5000]
    except Exception:
        pass
    return ""

def filter_similar_articles(articles):
    """Deduplicate only near-identical results; never cluster independent stories."""
    seen = set()
    result = []
    for article in articles:
        title = clean(article.get("title"))
        key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(article)
        if len(result) >= 12:
            break
    return result

def fetch_news(query, limit=20):
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query + ' when:7d')}"
        "&hl=en-GB&gl=GB&ceid=GB:en"
    )
    feed = feedparser.parse(url)
    items = feed.entries[:limit]

    prepared = []
    for item in items:
        source = ""
        if hasattr(item, "source"):
            source = clean(item.source.get("title"))
        source_href = ""
        if hasattr(item, "source"):
            source_href = clean(item.source.get("href"))

        prepared.append({
            "title": clean(item.get("title")),
            "summary": clean(item.get("summary")),
            "source": source,
            "source_href": source_href,
            "link": clean(item.get("link")),
            "published": clean(item.get("published")),
        })

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(prepared)))) as executor:
        contents = list(executor.map(lambda a: extract_article(a["link"]), prepared))

    articles = []
    for article, content in zip(prepared, contents):
        article["content"] = content
        articles.append(article)

    return filter_similar_articles(articles)
