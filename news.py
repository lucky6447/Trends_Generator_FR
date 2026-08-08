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
    """
    Group Google News results by meaningful shared keywords and keep only
    the dominant cluster. Generic words are ignored so unrelated stories
    are not merged into the same news topic.
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

    generic = {
        "france","français","française","francais","francaise",
        "actualité","actualités","news","nouvelle","nouvelles",
        "politique","candidat","candidate",
        "candidats","candidates","candidature","candidatures",
        "président","présidente","présidentielle","élection",
        "élections","monde","sport","sports","football","rugby",
        "joueur","joueuse","équipe","club","match","transfert",
        "transferts","nouveau","nouvelle","nouveaux","nouvelles",
        "annonce","annonces","rapport","selon","après","avant",
        "contre","face","parti","partis","gouvernement","ministre"
    }

    token_sets = []
    for article in articles:
        tokens = {
            t for t in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", article["title"].lower())
            if len(t) > 3 and t not in stopwords and t not in generic
        }
        token_sets.append(tokens)

    frequency = Counter(token for tokens in token_sets for token in tokens)

    def token_weight(token):
        return 1.0 / frequency[token]

    groups = []

    for idx, tokens in enumerate(token_sets):
        best_group = None
        best_score = 0.0

        for group in groups:
            common = tokens & group["tokens"]
            score = sum(token_weight(t) for t in common)

            if common and score > best_score:
                best_score = score
                best_group = group

        if best_group is not None and best_score >= 0.5:
            best_group["items"].append(idx)
            best_group["tokens"] |= tokens
        else:
            groups.append({"items": [idx], "tokens": set(tokens)})

    if not groups:
        return []

    largest = max(
        groups,
        key=lambda g: (
            len(g["items"]),
            sum(token_weight(t) for t in g["tokens"])
        )
    )

    if len(largest["items"]) < 3:
        return []

    filtered = [articles[i] for i in largest["items"]]

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