import re
import feedparser
import requests
import trafilatura
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urljoin
from html.parser import HTMLParser

try:
    from googlenewsdecoder import gnewsdecoder
except Exception:
    gnewsdecoder = None

HEADERS = {"User-Agent": "Mozilla/5.0"}

MAX_NEWS_AGE_HOURS = 24.0

def _age_hours(published):
    if not published:
        return None
    value = str(published).strip()
    try:
        dt = parsedate_to_datetime(value)
    except Exception:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0)

def _published_value(item):
    for field in ("published", "pubDate", "updated", "date", "dc_date"):
        value = clean(item.get(field))
        if value:
            return value
    for field in ("published_parsed", "updated_parsed"):
        value = item.get(field)
        if value:
            try:
                from email.utils import format_datetime
                dt = datetime(*value[:6], tzinfo=timezone.utc)
                return format_datetime(dt, usegmt=True)
            except Exception:
                pass
    source = item.get("source")
    if isinstance(source, dict):
        for field in ("published", "pubDate", "updated", "date"):
            value = clean(source.get(field))
            if value:
                return value
    return ""

def _is_fresh_news(published):
    age = _age_hours(published)
    return age is not None and age <= MAX_NEWS_AGE_HOURS


def clean(value):
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", str(value))
    value = value.replace("&nbsp;", " ").replace("&amp;", "&")
    return " ".join(value.split())


_GOOGLE_NEWS_DECODE_CACHE = {}
_GOOGLE_NEWS_DECODE_LOCK = __import__("threading").Lock()


class _SourceImageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.candidates = []
        self._in_jsonld = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag.lower() == "meta":
            key = (attrs.get("property") or attrs.get("name") or "").strip().lower()
            value = (attrs.get("content") or "").strip()
            if value and key in {
                "og:image",
                "og:image:url",
                "og:image:secure_url",
                "twitter:image",
                "twitter:image:src",
            }:
                self.candidates.append(value)

        elif tag.lower() == "script":
            self._in_jsonld = (
                (attrs.get("type") or "").strip().lower()
                == "application/ld+json"
            )

    def handle_endtag(self, tag):
        if tag.lower() == "script":
            self._in_jsonld = False

    def handle_data(self, data):
        if not self._in_jsonld:
            return

        # JSON-LD is handled separately so that only image.url is accepted.
        return


def _extract_jsonld_images(html):
    """Extract only JSON-LD image.url values; never use contentUrl."""
    import json

    images = []
    pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'

    for match in re.finditer(pattern, html, flags=re.I | re.S):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
        except Exception:
            continue

        def walk(obj):
            if isinstance(obj, dict):
                image = obj.get("image")

                if isinstance(image, dict):
                    image_url = image.get("url")
                    if isinstance(image_url, str):
                        images.append(image_url)

                elif isinstance(image, list):
                    for item in image:
                        if isinstance(item, dict):
                            image_url = item.get("url")
                            if isinstance(image_url, str):
                                images.append(image_url)

                for value in obj.values():
                    walk(value)

            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(data)

    return images


def resolve_google_news_url(url):
    """Return the real publisher URL for a Google News article URL."""
    url = str(url or "").strip()
    if not url:
        return ""

    if "news.google.com" not in url:
        return url

    with _GOOGLE_NEWS_DECODE_LOCK:
        if url in _GOOGLE_NEWS_DECODE_CACHE:
            return _GOOGLE_NEWS_DECODE_CACHE[url]

    if gnewsdecoder is None:
        return ""

    try:
        result = gnewsdecoder(url, interval=0.5)
        decoded = ""

        if isinstance(result, dict):
            if result.get("status"):
                decoded = str(result.get("decoded_url") or "").strip()
        else:
            if getattr(result, "status", False):
                decoded = str(
                    getattr(result, "decoded_url", "") or ""
                ).strip()

        if decoded and "news.google.com" not in decoded:
            with _GOOGLE_NEWS_DECODE_LOCK:
                _GOOGLE_NEWS_DECODE_CACHE[url] = decoded
            return decoded

    except Exception as exc:
        print(f"[SOURCE IMAGE] Google News decode failed: {exc}")

    with _GOOGLE_NEWS_DECODE_LOCK:
        _GOOGLE_NEWS_DECODE_CACHE[url] = ""

    return ""


def _fetch_publisher_html(url):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15,
            allow_redirects=True,
        )
        if response.ok and response.text:
            return response.text, response.url
    except Exception:
        pass

    return "", ""


def _is_usable_source_image(url):
    value = str(url or "").strip().replace("\\/\\/", "//").replace("\\/", "/").strip()
    if not value:
        return False

    low = value.casefold()

    if low.startswith(("data:", "javascript:")):
        return False

    blocked = (
        "favicon",
        "sprite",
        "placeholder",
        "default-image",
        "default_image",
        "logo",
        "site-logo",
        "apple-touch-icon",
        "avatar",
        "icon",
        "1x1",
        "pixel.gif",
        "spacer.gif",
    )

    if any(token in low for token in blocked):
        return False

    path = low.split("?", 1)[0].split("#", 1)[0]
    if path.endswith((".svg", ".mp4", ".m4v", ".webm", ".mov", ".m3u8", ".mpd",
                      ".avi", ".mkv", ".flv", ".wmv", ".mp3", ".m4a", ".wav", ".ogg")):
        return False

    if not (low.startswith("http://") or low.startswith("https://")):
        return False

    return True


def extract_source_image(url):
    """
    Resolve a Google News URL to the publisher page and extract the publisher's
    declared article image. Only page HTML is fetched; the image itself is never
    downloaded or stored locally.
    """
    publisher_url = resolve_google_news_url(url)
    if not publisher_url:
        return {"image": "", "source_url": "", "source": ""}

    html, final_url = _fetch_publisher_html(publisher_url)
    if not html:
        return {
            "image": "",
            "source_url": final_url or publisher_url,
            "source": "",
        }

    parser = _SourceImageParser()

    try:
        parser.feed(html)
    except Exception:
        pass

    candidates = list(parser.candidates)
    candidates.extend(_extract_jsonld_images(html))

    seen = set()

    for candidate in candidates:
        image_url = urljoin(
            final_url or publisher_url,
            str(candidate).strip(),
        )

        if not image_url or image_url in seen:
            continue

        seen.add(image_url)

        if _is_usable_source_image(image_url):
            host = re.sub(
                r"^www\.",
                "",
                requests.utils.urlparse(image_url).hostname or "",
            )

            return {
                "image": image_url,
                "source_url": final_url or publisher_url,
                "source": host,
            }

    return {
        "image": "",
        "source_url": final_url or publisher_url,
        "source": "",
    }


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


def _parse_news_rss(url):
    """Fetch an RSS URL explicitly so empty/error responses are observable."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
        feed = feedparser.parse(response.content)
        return response, feed
    except Exception as exc:
        print(f"[NEWS RSS] request failed: {exc}")
        return None, feedparser.FeedParserDict(entries=[])


def _fetch_topic_feed(query):
    """Google News topic search first; Bing News RSS is a fallback only."""
    google_url = ("https://news.google.com/rss/search?"
                  f"q={quote_plus(query + ' when:24h')}&hl=fr-FR&gl=FR&ceid=FR:fr")
    response, feed = _parse_news_rss(google_url)
    entries = list(getattr(feed, "entries", []) or [])
    if response is not None:
        print(f"[NEWS RSS] Google -> HTTP {response.status_code} | bytes={len(response.content)} | entries={len(entries)}")
    if entries:
        return feed

    bing_url = ("https://www.bing.com/news/search?"
                f"q={quote_plus(query)}&format=rss&setlang=fr-FR&cc=FR")
    response, feed = _parse_news_rss(bing_url)
    entries = list(getattr(feed, "entries", []) or [])
    print(f"[NEWS RSS] Google empty -> Bing fallback | HTTP {response.status_code if response is not None else 'ERR'} | bytes={len(response.content) if response is not None else 0} | entries={len(entries)}")
    return feed


def _entry_source(item):
    source = ""
    source_href = ""
    if hasattr(item, "source"):
        source = clean(item.source.get("title"))
        source_href = clean(item.source.get("href"))
    if not source:
        source = clean(item.get("source"))
    return source, source_href


def _prepare_entry(item):
    title = clean(item.get("title"))
    link = clean(item.get("link"))
    published = _published_value(item)
    if not _is_fresh_news(published):
        return None
    source, source_href = _entry_source(item)
    summary = clean(item.get("summary") or item.get("description"))
    return {
        "title": title,
        "summary": summary,
        "source": source,
        "source_href": source_href,
        "link": link,
        "published": published,
    }


def fetch_news_multi(queries, per_query_limit=8, max_results=18):
    """Fetch and merge several tightly scoped Google News searches."""
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
        feed = _fetch_topic_feed(query)
        for item in feed.entries[:max(1, int(per_query_limit))]:
            article = _prepare_entry(item)
            if not article:
                continue
            title_key = re.sub(r"[^a-z0-9]+", " ", article["title"].lower()).strip()
            link_key = article["link"].casefold()
            if not title_key or title_key in seen_titles or (link_key and link_key in seen_links):
                continue
            seen_titles.add(title_key)
            if link_key:
                seen_links.add(link_key)
            candidates.append(article)
            if len(candidates) >= max(1, int(max_results)):
                break
        if len(candidates) >= max(1, int(max_results)):
            break

    if not candidates:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
        contents = list(executor.map(lambda a: extract_article(a["link"]), candidates))
    for article, content in zip(candidates, contents):
        article["content"] = content or article.get("summary", "")

    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
        images = list(executor.map(lambda a: extract_source_image(a["link"]), candidates))

    for article, image_data in zip(candidates, images):
        image_data = image_data or {}
        article["image"] = image_data.get("image", "")
        article["image_source_url"] = image_data.get("source_url", "")
        article["image_source"] = image_data.get("source", "")

    return filter_similar_articles(candidates, max_results=max_results)


def fetch_news(query, limit=20):
    feed = _fetch_topic_feed(query)
    prepared = []
    for item in feed.entries[:limit]:
        article = _prepare_entry(item)
        if article:
            prepared.append(article)
    if not prepared:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(prepared))) as executor:
        contents = list(executor.map(lambda a: extract_article(a["link"]), prepared))
    for article, content in zip(prepared, contents):
        article["content"] = content or article.get("summary", "")

    with ThreadPoolExecutor(max_workers=min(8, len(prepared))) as executor:
        images = list(executor.map(lambda a: extract_source_image(a["link"]), prepared))

    for article, image_data in zip(prepared, images):
        image_data = image_data or {}
        article["image"] = image_data.get("image", "")
        article["image_source_url"] = image_data.get("source_url", "")
        article["image_source"] = image_data.get("source", "")

    return filter_similar_articles(prepared)
