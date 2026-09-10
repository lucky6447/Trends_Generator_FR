import re
import json
import os
import feedparser
import requests
import trafilatura
import html as _html
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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Cache-Control": "no-cache",
}

MAX_NEWS_AGE_HOURS = 24.0
MIN_DISCOVERY_RESULTS = 4
MAX_DISCOVERY_RESULTS = 12
DEFAULT_FULL_CONTENT_MAX_CHARS = 5000
IMAGE_PIPELINE_VERSION = "2026-09-10-universal-v3"
# Image fallback: if the selected publisher has no verified JSON-LD image.url,
# search fresh independent news coverage for a usable publisher image.
MAX_IMAGE_FALLBACK_SOURCES = 6

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
    """Extract publisher article images from JSON-LD only.

    Hard policy:
      * accept ONLY an image object's ``url`` field;
      * never accept ``contentUrl``, ``thumbnailUrl``, OG/Twitter tags or <img>;
      * prefer Article/NewsArticle/BlogPosting/Report nodes;
      * support @graph, arrays, @id references and tolerant JSON-LD parsing.
    """

    ARTICLE_TYPES = {
        "article", "newsarticle", "blogposting", "techarticle", "report",
    }
    EXCLUDED_TYPES = {
        "person", "organization", "brand", "website", "webpage",
        "imageobject", "logo",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.primary_candidates = []
        self.generic_candidates = []
        self.jsonld_blocks = 0
        self.jsonld_parse_failures = 0
        self._jsonld_buffer = None
        self._script_buffer = None
        self._script_is_json_candidate = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "script":
            return
        attrs = dict(attrs)
        script_type = (attrs.get("type") or "").strip().casefold()
        base_type = script_type.split(";", 1)[0].strip()
        if base_type == "application/ld+json":
            self._jsonld_buffer = []
            self._script_buffer = None
            self._script_is_json_candidate = False
        else:
            # Some publishers emit valid JSON-LD without the standard MIME type
            # (or as application/json). Keep this fallback narrow: it is accepted
            # only when the payload parses as JSON and explicitly contains @context.
            self._jsonld_buffer = None
            self._script_buffer = []
            self._script_is_json_candidate = base_type in ("", "application/json")

    def handle_startendtag(self, tag, attrs):
        # JSON-LD is not normally self-closing, but explicitly terminate any
        # accidental parser state rather than leaking it into the next script.
        if tag.lower() == "script":
            self._jsonld_buffer = None
            self._script_buffer = None
            self._script_is_json_candidate = False

    @staticmethod
    def _type_names(node):
        raw = node.get("@type") if isinstance(node, dict) else None
        raw = raw if isinstance(raw, list) else [raw]
        result = set()
        for value in raw:
            text = str(value or "").strip().casefold()
            if text:
                result.add(text.rsplit("/", 1)[-1].rsplit("#", 1)[-1])
        return result

    @staticmethod
    def _node_ids(node):
        if not isinstance(node, dict):
            return []
        values = node.get("@id")
        values = values if isinstance(values, list) else [values]
        return [
            str(v).strip()
            for v in values
            if isinstance(v, str) and v.strip()
        ]

    @staticmethod
    def _decode_jsonld(raw):
        """Parse common real-world JSON-LD variants without changing semantics."""
        text = str(raw or "").lstrip("\ufeff\u200b").strip()
        if not text:
            return None

        candidates = [text]
        # HTML entities occasionally appear in CMS-produced JSON-LD.
        unescaped = _html.unescape(text)
        if unescaped != text:
            candidates.append(unescaped)

        for candidate in candidates:
            candidate = candidate.strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass

            # A few publishers append a harmless semicolon after JSON.
            if candidate.endswith(";"):
                try:
                    return json.loads(candidate[:-1].rstrip())
                except Exception:
                    pass

            # raw_decode tolerates leading/trailing non-JSON whitespace while
            # still requiring the actual payload to be valid JSON.
            try:
                decoder = json.JSONDecoder()
                first = next(
                    (i for i, ch in enumerate(candidate) if not ch.isspace()),
                    None,
                )
                if first is not None:
                    payload, _ = decoder.raw_decode(candidate[first:])
                    return payload
            except Exception:
                pass

        return None

    def handle_endtag(self, tag):
        if tag.lower() != "script":
            return

        if self._jsonld_buffer is not None:
            raw = "".join(self._jsonld_buffer)
            self._jsonld_buffer = None
            self._script_buffer = None
            self._script_is_json_candidate = False
            if not raw.strip():
                return
            self._consume_jsonld_payload(raw, count_block=True)
            return

        if self._script_buffer is not None and self._script_is_json_candidate:
            raw = "".join(self._script_buffer)
            self._script_buffer = None
            self._script_is_json_candidate = False
            if not raw.strip():
                return
            payload = self._decode_jsonld(raw)
            # Only treat an untyped/application-json script as JSON-LD when it
            # explicitly declares @context. This prevents arbitrary JavaScript
            # objects from entering the image pipeline.
            if isinstance(payload, dict) and "@context" in payload:
                self._consume_jsonld_payload(raw, count_block=False)
            elif isinstance(payload, list) and any(
                isinstance(node, dict) and "@context" in node for node in payload
            ):
                self._consume_jsonld_payload(raw, count_block=False)
            return

        self._script_buffer = None
        self._script_is_json_candidate = False

    def _consume_jsonld_payload(self, raw, count_block=True):
        if count_block:
            self.jsonld_blocks += 1

        payload = self._decode_jsonld(raw)
        if payload is None:
            if count_block:
                self.jsonld_parse_failures += 1
            return

        nodes = []

        def collect(value):
            if isinstance(value, dict):
                nodes.append(value)
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(payload)

        by_id = {}
        for node in nodes:
            # Do not let a reference-only {"@id": "..."} stub overwrite the
            # full node collected earlier from the same JSON-LD payload.
            if isinstance(node, dict) and not (set(node) - {"@id"}):
                continue
            for node_id in self._node_ids(node):
                by_id[node_id] = node

        def add_image_value(image, target, visited_ids=None):
            """Collect ONLY JSON-LD ImageObject.url values, including @id refs."""
            visited_ids = set(visited_ids or ())
            values = image if isinstance(image, list) else [image]

            for item in values:
                if not isinstance(item, dict):
                    continue

                image_url = item.get("url")
                if isinstance(image_url, str) and image_url.strip():
                    target.append(image_url.strip())
                elif isinstance(image_url, list):
                    for value in image_url:
                        if isinstance(value, str) and value.strip():
                            target.append(value.strip())

                ref = item.get("@id")
                if isinstance(ref, str) and ref.strip() and ref not in visited_ids:
                    referenced = by_id.get(ref.strip())
                    if isinstance(referenced, dict):
                        add_image_value(
                            referenced,
                            target,
                            visited_ids | {ref.strip()},
                        )

        for node in nodes:
            if self._type_names(node) & self.ARTICLE_TYPES:
                add_image_value(node.get("image"), self.primary_candidates)

        for node in nodes:
            types = self._type_names(node)
            if not types & self.ARTICLE_TYPES and not types & self.EXCLUDED_TYPES:
                add_image_value(node.get("image"), self.generic_candidates)

    def handle_data(self, data):
        if self._jsonld_buffer is not None:
            self._jsonld_buffer.append(data)
        elif self._script_buffer is not None and self._script_is_json_candidate:
            self._script_buffer.append(data)

    def handle_comment(self, data):
        # Do not treat commented JSON-LD as executable metadata.
        return

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
    """Fetch publisher HTML with bounded browser-like retries.

    Image discovery is a metadata operation, but publishers/CDNs sometimes
    return transient 403/429/5xx responses to a generic client. Retry with a
    second browser profile and a fresh connection before declaring failure.
    """
    target = str(url or "").strip()
    if not target:
        return "", "", "empty_url"

    profiles = [
        HEADERS,
        {
            **HEADERS,
            "Referer": "https://www.google.com/",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.6 Safari/605.1.15"
            ),
            "Accept-Language": "en-US,en;q=0.8",
        },
    ]

    last_reason = "request_failed"
    for attempt, headers in enumerate(profiles, 1):
        try:
            response = requests.get(
                target,
                headers=headers,
                timeout=(8, 18),
                allow_redirects=True,
            )
            if response.ok and response.content:
                # Do not reject pages solely because a publisher omitted
                # Content-Type or returned a generic text type.
                # requests decodes publisher HTML using the server charset;
                # fall back to UTF-8 only when that detection is unusable.
                try:
                    text = response.text
                except Exception:
                    text = response.content.decode("utf-8", errors="replace")
                return text, response.url, f"http_{response.status_code}"

            last_reason = f"http_{response.status_code}"
            print(
                f"[SOURCE IMAGE] FETCH RETRY | attempt={attempt} "
                f"| status={response.status_code} | url={target}"
            )
        except requests.RequestException as exc:
            last_reason = f"{type(exc).__name__}"
            print(
                f"[SOURCE IMAGE] FETCH RETRY | attempt={attempt} "
                f"| error={exc} | url={target}"
            )

    return "", "", last_reason


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


def _extract_article_img_tag(html, base_url=""):
    """Return a validated image from an article's normal HTML <img src>.

    JSON-LD image.url remains preferred. This fallback exists for publishers
    that expose the real article image only as a normal image tag, while still
    using the existing strict URL validation and never using og:image,
    twitter:image, CSS, or arbitrary metadata.
    """
    html = str(html or "")
    if not html:
        return ""

    article_match = re.search(
        r"<article\b[^>]*>.*?</article>",
        html,
        flags=re.I | re.S,
    )
    scope = article_match.group(0) if article_match else html

    seen = set()
    for match in re.finditer(
        r"<img\b[^>]*\bsrc\s*=\s*([\"'])(.*?)\1",
        scope,
        flags=re.I | re.S,
    ):
        candidate = str(match.group(2) or "").strip()
        if not candidate:
            continue
        candidate = candidate.replace("\\/", "/").replace("&amp;", "&")
        image_url = urljoin(base_url, candidate) if base_url else candidate
        image_url = str(image_url or "").strip()
        if not image_url or image_url in seen:
            continue
        seen.add(image_url)

        if _is_usable_source_image(image_url):
            return image_url

    return ""

def _extract_source_image_direct(url):
    """Try the supplied publisher only. Never performs cross-publisher fallback."""
    original_url = str(url or "").strip()
    publisher_url = resolve_google_news_url(original_url)

    if not publisher_url:
        return {
            "image": "",
            "source_url": "",
            "source": "",
            "image_status": "NONE",
            "image_reason": (
                "google_news_decode_failed"
                if "news.google.com" in original_url
                else "empty_url"
            ),
        }

    html, final_url, fetch_reason = _fetch_publisher_html(publisher_url)
    effective_url = final_url or publisher_url

    if not html:
        print(
            f"[SOURCE IMAGE] NONE | original={original_url} "
            f"| publisher={effective_url} | reason={fetch_reason}"
        )
        return {
            "image": "",
            "source_url": effective_url,
            "source": "",
            "image_status": "NONE",
            "image_reason": f"publisher_fetch_{fetch_reason}",
        }

    parser = _SourceImageParser()
    parse_failed = False
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        parse_failed = True
        print(
            f"[SOURCE IMAGE] PARSE ERROR | publisher={effective_url} "
            f"| error={exc}"
        )

    # Fail closed on parser errors: never trust partially parsed JSON-LD.
    if parse_failed:
        return {
            "image": "",
            "source_url": effective_url,
            "source": "",
            "image_status": "NONE",
            "image_reason": "publisher_html_parse_failed",
        }

    seen = set()
    candidates = list(parser.primary_candidates) + list(parser.generic_candidates)

    for candidate in candidates:
        candidate = (
            str(candidate).strip()
            .replace("\\/\\/", "//")
            .replace("\\/", "/")
        )
        image_url = urljoin(effective_url, candidate)
        if not image_url or image_url in seen:
            continue
        seen.add(image_url)

        if _is_usable_source_image(image_url):
            host = re.sub(
                r"^www\\.",
                "",
                requests.utils.urlparse(image_url).hostname or "",
            )
            print(
                f"[SOURCE IMAGE] FOUND | original={original_url} "
                f"| publisher={effective_url} | image={image_url}"
            )
            return {
                "image": image_url,
                "source_url": effective_url,
                "source": host,
                "image_status": "FOUND",
                "image_reason": "jsonld_image_url",
            }


    # Some legitimate publishers expose the article image only as a normal
    # <img src> inside the article. JSON-LD remains preferred; this is a
    # narrowly scoped fallback using the same strict image URL validation.
    html_img = _extract_article_img_tag(
        html,
        base_url=effective_url,
    )
    if html_img:
        host = re.sub(
            r"^www\\.",
            "",
            requests.utils.urlparse(html_img).hostname or "",
        )
        print(
            f"[SOURCE IMAGE] FOUND | original={original_url} "
            f"| publisher={effective_url} | source=article_img_tag "
            f"| image={html_img}"
        )
        return {
            "image": html_img,
            "source_url": effective_url,
            "source": host,
            "image_status": "FOUND",
            "image_reason": "article_img_tag",
        }

    print(
        f"[SOURCE IMAGE] NONE | original={original_url} "
        f"| publisher={effective_url} "
        f"| jsonld_blocks={parser.jsonld_blocks} "
        f"| jsonld_parse_failures={parser.jsonld_parse_failures} "
        f"| jsonld_primary={len(parser.primary_candidates)} "
        f"| jsonld_generic={len(parser.generic_candidates)}"
    )
    return {
        "image": "",
        "source_url": effective_url,
        "source": "",
        "image_status": "NONE",
        "image_reason": "jsonld_image_url_not_found_or_invalid",
    }


def _image_fallback_queries(title):
    """Create a small set of precise news-image discovery queries."""
    clean_title = clean(title)
    if not clean_title:
        return []

    queries = [clean_title]

    # Keep the fallback precise: title + news is preferable to broad keyword
    # searches that can return an unrelated stock/illustration image.
    if len(clean_title) > 140:
        queries.append(clean_title[:140])

    return list(dict.fromkeys(q for q in queries if q))


def _find_fallback_publisher_image(title, original_url=""):
    """Find a usable image on an independent fresh news publisher.

    This fallback is image-only. It does NOT replace the evidence/source set
    used to validate the story. Only a publisher's verified JSON-LD image.url
    can be returned.
    """
    queries = _image_fallback_queries(title)
    if not queries:
        return None

    original_publisher = resolve_google_news_url(original_url)
    original_host = re.sub(
        r"^www\.",
        "",
        requests.utils.urlparse(original_publisher).hostname or "",
    ).casefold()

    candidates = []
    seen = set()

    for query in queries:
        feed = _fetch_topic_feed(query)
        for item in _feed_candidates(feed):
            link = str(item.get("link") or "").strip()
            if not link:
                continue

            publisher_url = resolve_google_news_url(link)
            host = re.sub(
                r"^www\.",
                "",
                requests.utils.urlparse(publisher_url).hostname or "",
            ).casefold()

            # Never use the same publisher that already failed, and never
            # revisit the exact same URL.
            key = _url_key(publisher_url or link)
            if not key or key in seen:
                continue
            if original_host and host == original_host:
                continue

            seen.add(key)
            candidates.append(item)

            if len(candidates) >= MAX_IMAGE_FALLBACK_SOURCES:
                break

        if len(candidates) >= MAX_IMAGE_FALLBACK_SOURCES:
            break

    if not candidates:
        print(
            f"[SOURCE IMAGE FALLBACK] NONE | title={clean(title)} "
            f"| reason=no_independent_fresh_candidates"
        )
        return None

    print(
        f"[SOURCE IMAGE FALLBACK] candidates={len(candidates)} "
        f"| title={clean(title)}"
    )

    # Fetch candidate publisher pages concurrently, but keep the first
    # verified publisher image in deterministic discovery order.
    with ThreadPoolExecutor(max_workers=min(6, len(candidates))) as executor:
        results = list(
            executor.map(
                lambda item: _extract_source_image_direct(item.get("link", "")),
                candidates,
            )
        )

    for item, result in zip(candidates, results):
        if result.get("image_status") != "FOUND":
            continue

        print(
            f"[SOURCE IMAGE FALLBACK] FOUND | story={clean(title)} "
            f"| fallback_publisher={result.get('source_url', '')} "
            f"| image={result.get('image', '')}"
        )
        result["image_reason"] = "fallback_jsonld_image_url"
        return result

    print(
        f"[SOURCE IMAGE FALLBACK] NONE | title={clean(title)} "
        f"| checked={len(candidates)} | reason=no_verified_jsonld_image"
    )
    return None


def extract_source_image(url, title=""):
    """Resolve a source image with reliable cross-publisher fallback.

    Stage 1: selected publisher, JSON-LD image.url only.
    Stage 2: if missing, search fresh independent news coverage and accept
             only another publisher's JSON-LD image.url.
    No contentUrl, thumbnailUrl, OG/Twitter tags, or arbitrary <img> is used.
    """
    direct = _extract_source_image_direct(url)
    if direct.get("image_status") == "FOUND":
        return direct

    fallback = _find_fallback_publisher_image(
        title=title,
        original_url=url,
    )
    if fallback:
        return fallback

    return direct


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
            images = list(executor.map(lambda a: extract_source_image(a.get("link", ""), a.get("title", "")), hydrated))
        for article, image_data in zip(hydrated, images):
            article["image"] = image_data.get("image", "")
            article["image_source_url"] = image_data.get("source_url", "")
            article["image_source"] = image_data.get("source", "")
            article["image_status"] = image_data.get("image_status", "NONE")
            article["image_reason"] = image_data.get("image_reason", "")

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
