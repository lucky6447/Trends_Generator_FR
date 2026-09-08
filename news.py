import re
import json
import os
import feedparser
import requests
import trafilatura
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urljoin
from html.parser import HTMLParser

try:
    from config import LANGUAGE
except Exception:
    LANGUAGE = "english"

try:
    from googlenewsdecoder import gnewsdecoder
except Exception:
    gnewsdecoder = None

HEADERS = {"User-Agent": "Mozilla/5.0"}

MAX_NEWS_AGE_HOURS = 24.0
MIN_DISCOVERY_RESULTS = 4
MAX_DISCOVERY_RESULTS = 12
DEFAULT_FULL_CONTENT_MAX_CHARS = 5000

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


def _published_value(item):
    """Return a normalized publication timestamp from common RSS field variants."""
    for field in ("published", "pubDate", "updated", "date", "dc_date"):
        value = item.get(field)
        if value:
            return clean(value)
    return ""


_GOOGLE_NEWS_DECODE_CACHE = {}
_GOOGLE_NEWS_DECODE_LOCK = __import__("threading").Lock()


class _SourceImageParser(HTMLParser):
    """Extract JSON-LD article images with type-aware priority.

    Many publishers expose several image.url values in the same JSON-LD payload:
    the article image, author/employee portraits, organization logos, thumbnails,
    etc.  The old recursive collector treated all of them equally and returned
    whichever appeared first.  Keep article-type images separate so the article
    image wins deterministically.
    """

    ARTICLE_TYPES = {
        "article",
        "newsarticle",
        "blogposting",
        "techarticle",
        "report",
    }

    NON_ARTICLE_TYPES = {
        "person",
        "organization",
        "brand",
        "website",
        "webpage",
        "imageobject",
        "logo",
    }

    def __init__(self):
        super().__init__()
        self.primary_candidates = []
        self.generic_candidates = []
        self._jsonld_buffer = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() != "script":
            return

        script_type = (attrs.get("type") or "").strip().lower()
        # Accept normal JSON-LD content types, including harmless parameters such
        # as charset. Do not broaden this to meta/OG/Twitter image sources.
        if script_type.split(";", 1)[0].strip() == "application/ld+json":
            self._jsonld_buffer = []

    def handle_endtag(self, tag):
        if tag.lower() != "script" or self._jsonld_buffer is None:
            return

        raw = "".join(self._jsonld_buffer).strip()
        self._jsonld_buffer = None
        if not raw:
            return

        try:
            payload = json.loads(raw)
        except Exception:
            return

        def type_names(value):
            raw_types = value.get("@type") if isinstance(value, dict) else None
            if isinstance(raw_types, str):
                raw_types = [raw_types]
            if not isinstance(raw_types, list):
                return set()
            return {
                str(item).strip().casefold().split("/")[-1]
                for item in raw_types
                if str(item).strip()
            }

        def add_image_value(image, target):
            values = image if isinstance(image, list) else [image]
            for item in values:
                if isinstance(item, dict):
                    image_url = item.get("url")
                    if isinstance(image_url, str) and image_url.strip():
                        target.append(image_url.strip())

        def walk(value, article_context=False, excluded_context=False):
            if isinstance(value, dict):
                types = type_names(value)
                is_article = bool(types & self.ARTICLE_TYPES)
                is_excluded = bool(types & self.NON_ARTICLE_TYPES)
                current_article_context = article_context or is_article
                current_excluded_context = excluded_context or (is_excluded and not is_article)

                if "image" in value:
                    if current_article_context and not current_excluded_context:
                        add_image_value(value.get("image"), self.primary_candidates)
                    elif not current_excluded_context:
                        add_image_value(value.get("image"), self.generic_candidates)

                for child in value.values():
                    if isinstance(child, (dict, list)):
                        walk(child, current_article_context, current_excluded_context)

            elif isinstance(value, list):
                for child in value:
                    walk(child, article_context, excluded_context)

        walk(payload)

    def handle_data(self, data):
        if self._jsonld_buffer is not None:
            self._jsonld_buffer.append(data)


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
    value = str(url or "").strip()
    if not value:
        return False

    # JSON-LD URLs may contain escaped forward slashes (\\/).
    value = value.replace("\\/\\/", "//").replace("\\/", "/").strip()
    low = value.casefold()

    # Reject obvious tiny thumbnail variants. Article pages frequently expose
    # author/profile images with URLs such as ?w=96; these are technically valid
    # images but are not suitable as the article hero image.
    query = low.split("?", 1)[1].split("#", 1)[0] if "?" in low else ""
    if query:
        for match in re.finditer(r"(?:^|&)(?:w|width|h|height|size)=([0-9]{1,4})(?:&|$)", query):
            if int(match.group(1)) <= 160:
                return False

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

    if low.endswith(".svg"):
        return False

    # Never feed video/audio media URLs into an <img> element.
    path = low.split("?", 1)[0].split("#", 1)[0]
    if path.endswith((
        ".mp4", ".m4v", ".webm", ".mov", ".m3u8", ".mpd",
        ".avi", ".mkv", ".flv", ".wmv", ".mp3", ".m4a", ".wav", ".ogg",
    )):
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

    seen = set()
    candidates = list(parser.primary_candidates) + list(parser.generic_candidates)

    for candidate in candidates:
        candidate = str(candidate).strip().replace("\\/\\/", "//").replace("\\/", "/")
        image_url = urljoin(
            final_url or publisher_url,
            candidate,
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

    print(
        f"[SOURCE IMAGE] NONE | publisher={final_url or publisher_url} "
        f"| article_jsonld_candidates={len(parser.primary_candidates)} "
        f"| generic_jsonld_candidates={len(parser.generic_candidates)}"
    )
    return {
        "image": "",
        "source_url": final_url or publisher_url,
        "source": "",
    }


def extract_article(url):
    """Extract publisher article text, resolving Google News URLs first."""
    try:
        publisher_url = resolve_google_news_url(url) or str(url or "").strip()
        if not publisher_url:
            return ""
        downloaded = trafilatura.fetch_url(publisher_url)
        if downloaded:
            text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=False, include_links=False
            )
            if text:
                return clean(text)[:DEFAULT_FULL_CONTENT_MAX_CHARS]
    except Exception as exc:
        print(f"[NEWS EXTRACT] failed | url={str(url or '')[:180]} | {exc}")
    return ""


def _title_key(title):
    value = clean(title).casefold()
    return re.sub(r"[^a-z0-9\s]+", " ", value).strip()


def _url_key(url):
    value = str(url or "").strip().casefold()
    value = value.split("#", 1)[0]
    return value.rstrip("/")


def _source_key(item):
    source = clean(item.get("source"))
    if source:
        return source.casefold()
    return re.sub(r"^www\.", "", (requests.utils.urlparse(str(item.get("link") or "")).hostname or "").casefold())


def _discovery_record(item):
    published = _published_value(item)
    if not _is_fresh_news(published):
        return None
    title = clean(item.get("title"))
    link = clean(item.get("link"))
    if not title or not link:
        return None
    source = ""
    if hasattr(item, "source"):
        source = clean(item.source.get("title"))
    if not source:
        source = clean(item.get("source"))
    source_href = ""
    if hasattr(item, "source"):
        source_href = clean(item.source.get("href"))
    return {
        "title": title,
        "summary": clean(item.get("summary")),
        "description": clean(item.get("description")),
        "source": source,
        "source_href": source_href,
        "link": link,
        "published": published,
    }


def _dedupe_discovery(items, max_results=MAX_DISCOVERY_RESULTS):
    result = []
    seen_titles = set()
    seen_urls = set()
    for item in items or []:
        title_key = _title_key(item.get("title"))
        url_key = _url_key(item.get("link"))
        if not title_key or (title_key in seen_titles) or (url_key and url_key in seen_urls):
            continue
        seen_titles.add(title_key)
        if url_key:
            seen_urls.add(url_key)
        result.append(item)
        if len(result) >= max(1, int(max_results)):
            break
    return result

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
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
            allow_redirects=True,
        )
        feed = feedparser.parse(response.content)
        return response, feed
    except Exception as exc:
        print(f"[NEWS RSS] request failed: {exc}")
        return None, feedparser.FeedParserDict(entries=[])


def _feed_candidates(feed):
    candidates = []
    for item in list(getattr(feed, "entries", []) or []):
        record = _discovery_record(item)
        if record:
            candidates.append(record)
    return candidates


def _news_locale():
    """Return Google/Bing locale settings for the active TrendCurrent language."""
    language = str(LANGUAGE or "").strip().casefold()
    mapping = {
        "english": ("en-GB", "GB", "en-GB"),
        "en": ("en-GB", "GB", "en-GB"),
        "english (us)": ("en-US", "US", "en-US"),
        "en-us": ("en-US", "US", "en-US"),
        "en_us": ("en-US", "US", "en-US"),
        "german": ("de-DE", "DE", "de-DE"),
        "de": ("de-DE", "DE", "de-DE"),
        "italian": ("it-IT", "IT", "it-IT"),
        "it": ("it-IT", "IT", "it-IT"),
        "french": ("fr-FR", "FR", "fr-FR"),
        "fr": ("fr-FR", "FR", "fr-FR"),
        "spanish": ("es-ES", "ES", "es-ES"),
        "es": ("es-ES", "ES", "es-ES"),
        "portuguese": ("pt-PT", "PT", "pt-PT"),
        "pt": ("pt-PT", "PT", "pt-PT"),
        "brazilian portuguese": ("pt-BR", "BR", "pt-BR"),
        "pt-br": ("pt-BR", "BR", "pt-BR"),
        "indonesian": ("id-ID", "ID", "id-ID"),
        "id": ("id-ID", "ID", "id-ID"),
        "bulgarian": ("bg-BG", "BG", "bg-BG"),
        "bg": ("bg-BG", "BG", "bg-BG"),
    }
    return mapping.get(language, ("en-US", "US", "en-US"))


def _fetch_topic_feed(query):
    """Google first, then supplement/fallback with Bing when Google is weak."""
    hl, gl, bing_lang = _news_locale()
    google_url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query + ' when:24h')}"
        f"&hl={quote_plus(hl)}&gl={quote_plus(gl)}&ceid={quote_plus(f'{gl}:{bing_lang.split("-")[0]}')}"
    )
    response, google_feed = _parse_news_rss(google_url)
    google_candidates = _feed_candidates(google_feed)
    if response is not None:
        print(
            f"[NEWS RSS] Google -> HTTP {response.status_code} | "
            f"bytes={len(response.content)} | fresh_usable={len(google_candidates)}"
        )

    if len(google_candidates) >= MIN_DISCOVERY_RESULTS:
        return google_feed

    bing_url = (
        "https://www.bing.com/news/search?"
        f"q={quote_plus(query)}&format=rss&setlang={quote_plus(bing_lang)}&cc={quote_plus(gl)}"
    )
    response, bing_feed = _parse_news_rss(bing_url)
    bing_candidates = _feed_candidates(bing_feed)
    print(
        f"[NEWS RSS] Google weak ({len(google_candidates)} fresh usable) -> Bing supplement/fallback | "
        f"HTTP {response.status_code if response is not None else 'ERR'} | "
        f"bytes={len(response.content) if response is not None else 0} | "
        f"fresh_usable={len(bing_candidates)}"
    )

    merged = _dedupe_discovery(google_candidates + bing_candidates, MAX_DISCOVERY_RESULTS)
    return feedparser.FeedParserDict(entries=merged)


def fetch_news_discovery(queries, per_query_limit=8, max_results=MAX_DISCOVERY_RESULTS):
    """Return fresh discovery metadata only; do not download publisher pages yet."""
    clean_queries = []
    seen_queries = set()
    for query in queries or []:
        q = " ".join(str(query or "").split()).strip()
        key = q.casefold()
        if not q or key in seen_queries:
            continue
        seen_queries.add(key)
        clean_queries.append(q)

    candidates = []
    for query in clean_queries:
        feed = _fetch_topic_feed(query)
        batch = _feed_candidates(feed)
        for item in batch[:max(1, int(per_query_limit))]:
            candidates.append(item)
        candidates = _dedupe_discovery(candidates, max_results=max_results)
        if len(candidates) >= max(1, int(max_results)):
            break

    result = _dedupe_discovery(candidates, max_results=max_results)
    print(f"[NEWS DISCOVERY] queries={len(clean_queries)} | candidates={len(result)} | full_extract=deferred")
    return result


def hydrate_news_items(items, include_images=False, max_content_chars=DEFAULT_FULL_CONTENT_MAX_CHARS):
    """Download full article text only for the already selected source set."""
    prepared = [dict(item) for item in (items or []) if isinstance(item, dict)]
    if not prepared:
        return []

    with ThreadPoolExecutor(max_workers=min(6, len(prepared))) as executor:
        contents = list(executor.map(lambda a: extract_article(a.get("link", "")), prepared))

    hydrated = []
    for article, content in zip(prepared, contents):
        if content:
            article["content"] = clean(content)[:max_content_chars]
        else:
            fallback = " ".join(
                x for x in (
                    article.get("summary", ""),
                    article.get("description", ""),
                )
                if clean(x)
            ).strip()
            article["content"] = clean(fallback)
        hydrated.append(article)

    if include_images:
        with ThreadPoolExecutor(max_workers=min(6, len(hydrated))) as executor:
            images = list(executor.map(lambda a: extract_source_image(a.get("link", "")), hydrated))
        for article, image_data in zip(hydrated, images):
            article["image"] = image_data.get("image", "")
            article["image_source_url"] = image_data.get("source_url", "")
            article["image_source"] = ""

    print(f"[NEWS HYDRATE] sources={len(hydrated)} | images={include_images}")
    return hydrated


def hydrate_story_sources(news, story_selection, include_images=False):
    items = list(news or [])
    indices = list((story_selection or {}).get("selected_indices") or [])
    selected = [items[i] for i in indices if 0 <= i < len(items)]
    if not selected:
        raise ValueError("No sources selected for story hydration")
    return hydrate_news_items(selected, include_images=include_images)


def fetch_news_multi(queries, per_query_limit=8, max_results=MAX_DISCOVERY_RESULTS):
    """Compatibility wrapper: discovery only; caller hydrates selected sources later."""
    return fetch_news_discovery(queries, per_query_limit=per_query_limit, max_results=max_results)


def fetch_news(query, limit=MAX_DISCOVERY_RESULTS):
    """Compatibility wrapper: discovery only. Use hydrate_news_items() for full content."""
    return fetch_news_discovery([query], per_query_limit=limit, max_results=limit)
