import re
import feedparser
import requests
import trafilatura
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

HEADERS = {"User-Agent": "Mozilla/5.0"}

MAX_NEWS_AGE_HOURS = 6.0

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

def _is_fresh_news(published):
    age = _age_hours(published)
    return age is not None and age <= MAX_NEWS_AGE_HOURS


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

def filter_similar_articles(articles, max_results=12):
    """Deduplicate near-identical headlines while preserving independent reports."""
    seen = set()
    result = []
    for article in articles:
        title = clean(article.get("title"))
        key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(article)
        if len(result) >= max(1, int(max_results)):
            break
    return result


def fetch_news_multi(queries, per_query_limit=8, max_results=18):
    """Fetch and merge several tightly scoped Google News searches.

    RSS entries are deduplicated before article extraction so query expansion
    increases source coverage without multiplying page downloads for duplicates.
    """
    clean_queries = []
    seen_queries = set()
    for query in queries or []:
        q = " ".join(str(query or "").split()).strip()
        key = q.casefold()
        if not q or key in seen_queries:
            continue
        seen_queries.add(key)
        clean_queries.append(q)

    if not clean_queries:
        return []

    candidates = []
    seen_links = set()
    seen_titles = set()

    for query in clean_queries:
        url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(query + ' when:6h')}"
            "&hl=en-GB&gl=GB&ceid=GB:en"
        )
        feed = feedparser.parse(url)
        for item in feed.entries[:max(1, int(per_query_limit))]:
            title = clean(item.get("title"))
            link = clean(item.get("link"))
            title_key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
            link_key = link.casefold()
            if not title_key or title_key in seen_titles or (link_key and link_key in seen_links):
                continue
            seen_titles.add(title_key)
            if link_key:
                seen_links.add(link_key)

            source = ""
            if hasattr(item, "source"):
                source = clean(item.source.get("title"))
            source_href = ""
            if hasattr(item, "source"):
                source_href = clean(item.source.get("href"))

            candidates.append({
                "title": title,
                "summary": clean(item.get("summary")),
                "source": source,
                "source_href": source_href,
                "link": link,
                "published": clean(item.get("published")),
            })

            if len(candidates) >= max(1, int(max_results)):
                break
        if len(candidates) >= max(1, int(max_results)):
            break

    if not candidates:
        return []

    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
        contents = list(executor.map(lambda a: extract_article(a["link"]), candidates))

    for article, content in zip(candidates, contents):
        article["content"] = content

    return filter_similar_articles(candidates, max_results=max_results)


def fetch_news(query, limit=20):
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query + ' when:6h')}"
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
