import re
import feedparser
import requests
import trafilatura
from concurrent.futures import ThreadPoolExecutor

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
    """
    Group Google News results by shared keywords and keep only the
    dominant cluster. Falls back to the original list if no clear
    dominant cluster exists.
    """
    if len(articles) < 5:
        return []

    stopwords = {
        "the","and","for","with","from","this","that","into","about",
        "der","die","das","und","mit","von","den","des","ein","eine",
        "il","lo","la","gli","le","dei","della","delle","del","con",
        "el","los","las","del","para","una","uno","por","como",
        "le","les","des","une","dans","avec","pour","sur","aux","est"
    }

    token_sets = []

    for article in articles:
        tokens = {
            t for t in re.findall(r"[a-zA-Z0-9]+", article["title"].lower())
            if len(t) > 3 and t not in stopwords
        }
        token_sets.append(tokens)

    groups = []

    for idx, tokens in enumerate(token_sets):
        placed = False
        for group in groups:
            common = len(tokens & group["tokens"])
            if common >= 2:
                group["items"].append(idx)
                group["tokens"] |= tokens
                placed = True
                break
        if not placed:
            groups.append({"items":[idx], "tokens":set(tokens)})

    largest = max(groups, key=lambda g: len(g["items"]))

    if len(largest["items"]) < 3:
        return articles

    filtered = [articles[i] for i in largest["items"]]

    if len(filtered) < 5:
        return articles[:5]

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

    items = feed.entries[:limit]

    prepared = []

    for item in items:
        source = ""
        if hasattr(item, "source"):
            source = clean(item.source.get("title"))

        prepared.append({
            "title": clean(item.get("title")),
            "summary": clean(item.get("summary")),
            "source": source,
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