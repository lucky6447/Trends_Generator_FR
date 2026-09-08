"""
TrendCurrent universal direct-publisher RSS discovery.

This module is language-agnostic. Feed configuration is supplied through
TREND_CURRENT_SOURCE_FEEDS as JSON and can be either:
  1) a list of feed objects, used for every language; or
  2) a dict keyed by language code/name, with a list of feed objects per language.

Each feed object:
  {"name": "...", "url": "...", "enabled": true}

The module ONLY discovers fresh publisher stories. It does not generate articles,
judge evidence, or bypass any downstream quality gate.
"""

import json
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

try:
    from dateutil.parser import isoparse
except Exception:
    isoparse = None
from urllib.parse import urlparse

import feedparser
import requests

HEADERS = {"User-Agent": "TrendCurrent/1.0 (+news discovery)"}
DEFAULT_MAX_AGE_HOURS = 24.0
DEFAULT_PER_FEED_LIMIT = 50
DEFAULT_MAX_SEEDS = 100
LOCAL_FEEDS_FILE = os.path.join(os.path.dirname(__file__), "source_feeds.json")


def _clean(value):
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", str(value))
    value = value.replace("&nbsp;", " ").replace("&amp;", "&")
    return " ".join(value.split()).strip()


def _published_value(entry):
    for field in ("published", "pubDate", "updated", "date", "dc_date"):
        value = entry.get(field)
        if value:
            return _clean(value)
    return ""


def _age_hours(value, parsed_struct=None):
    # Prefer feedparser's normalized time tuple when available.
    if parsed_struct:
        try:
            import calendar
            ts = calendar.timegm(parsed_struct)
            return max(0.0, (datetime.now(timezone.utc).timestamp() - ts) / 3600.0)
        except Exception:
            pass

    if not value:
        return None

    raw = str(value).strip()
    candidates = [raw, raw.replace("Z", "+00:00")]

    # RFC 2822 / RSS dates.
    for candidate in candidates:
        try:
            dt = parsedate_to_datetime(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0,
            )
        except Exception:
            pass

    # ISO-8601 / Atom dates, including fractional seconds and explicit offsets.
    for candidate in candidates:
        try:
            dt = isoparse(candidate) if isoparse else datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0,
            )
        except Exception:
            pass

    return None


def _entry_fresh_age(entry):
    """Return the freshest usable age among common RSS/Atom date fields.

    Some publishers keep an old ``published`` value while updating the item
    via ``updated``. Freshness must not incorrectly reject such an item.
    """
    ages = []
    for field in ("published", "pubDate", "updated", "date", "dc_date"):
        raw = entry.get(field)
        if not raw:
            continue
        parsed_struct = entry.get(f"{field}_parsed")
        age = _age_hours(_clean(raw), parsed_struct)
        if age is not None:
            ages.append(age)
    return min(ages) if ages else None


def _entry_fresh(entry, max_age_hours):
    age = _entry_fresh_age(entry)
    return age is not None and age <= max_age_hours


def _fresh(value, max_age_hours, parsed_struct=None):
    age = _age_hours(value, parsed_struct)
    return age is not None and age <= max_age_hours


def _url_key(url):
    value = str(url or "").strip().casefold()
    return value.split("#", 1)[0].rstrip("/")


def _title_key(title):
    value = _clean(title).casefold()
    return re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE).strip()


def _load_config(language):
    """Load only the local per-project source_feeds.json.

    Environment variables are intentionally ignored so one language instance
    can never inherit another instance's feed configuration.
    """
    if not os.path.exists(LOCAL_FEEDS_FILE):
        return []

    try:
        with open(LOCAL_FEEDS_FILE, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exc:
        print(f"[SOURCE FIRST] Could not read {LOCAL_FEEDS_FILE}: {exc}")
        return []

    if isinstance(config, list):
        feeds = config
    elif isinstance(config, dict):
        key = str(language or "").strip().casefold()
        aliases = {
            "english": ["en"],
            "english (us)": ["en-us"],
            "german": ["de"], "deutsch": ["de"],
            "italian": ["it"], "italiano": ["it"],
            "french": ["fr"], "français": ["fr"],
            "spanish": ["es"], "español": ["es"],
            "indonesian": ["id"], "bahasa indonesia": ["id"],
        }
        keys=[key] + aliases.get(key, [])
        feeds=[]
        for candidate in keys:
            if candidate in config and isinstance(config[candidate], list):
                feeds=config[candidate]
                break
    else:
        feeds=[]

    return [
        item for item in feeds
        if isinstance(item, dict)
        and item.get("enabled", True)
        and str(item.get("url", "")).strip()
    ]


def _fetch_feed(feed_url):
    try:
        response = requests.get(
            feed_url,
            headers=HEADERS,
            timeout=20,
            allow_redirects=True,
        )
        if not response.ok:
            print(f"[SOURCE FIRST] RSS HTTP {response.status_code} | {feed_url}")
            return None
        return feedparser.parse(response.content)
    except Exception as exc:
        print(f"[SOURCE FIRST] RSS fetch failed | {feed_url} | {exc}")
        return None


def _record(entry, feed):
    published = _published_value(entry)
    title = _clean(entry.get("title"))
    link = _clean(entry.get("link"))
    if not title or not link:
        return None

    return {
        "title": title,
        "summary": _clean(entry.get("summary")),
        "description": _clean(entry.get("description")),
        "source": _clean(feed.get("name")) or _clean(
            urlparse(link).hostname or ""
        ),
        "source_href": _clean(feed.get("url")),
        "link": link,
        "url": link,
        "published": published,
        "_published_value": published,
        "guid": _clean(entry.get("id") or entry.get("guid")),
        "discovery_provider": "direct_rss",
        "discovery_source": _clean(feed.get("name")),
        "discovery_feed": _clean(feed.get("url")),
        "discovery_context": "direct_publisher_rss",
    }


def fetch_source_stories(language=None):
    max_age = float(os.getenv(
        "RSS_DISCOVERY_MAX_AGE_HOURS",
        str(DEFAULT_MAX_AGE_HOURS),
    ))
    per_feed = max(1, int(os.getenv(
        "RSS_DISCOVERY_PER_FEED_LIMIT",
        str(DEFAULT_PER_FEED_LIMIT),
    )))
    max_seeds = max(1, int(os.getenv(
        "RSS_DISCOVERY_MAX_SEEDS",
        str(DEFAULT_MAX_SEEDS),
    )))

    feeds = _load_config(language)
    if not feeds:
        print(
            f"[SOURCE FIRST] No configured direct publisher feeds | "
            f"language={language}"
        )
        return []

    result = []
    seen_urls = set()
    seen_titles = set()

    for feed in feeds:
        feed_url = str(feed.get("url", "")).strip()
        parsed = _fetch_feed(feed_url)
        if parsed is None:
            continue

        accepted = 0
        inspected = 0
        for entry in list(getattr(parsed, "entries", []) or [])[:per_feed]:
            inspected += 1
            published = _published_value(entry)
            if not _entry_fresh(entry, max_age):
                continue

            item = _record(entry, feed)
            if not item:
                continue

            uk = _url_key(item["link"])
            tk = _title_key(item["title"])
            if not uk or uk in seen_urls or not tk or tk in seen_titles:
                continue

            seen_urls.add(uk)
            seen_titles.add(tk)
            result.append(item)
            accepted += 1

            if len(result) >= max_seeds:
                break

        print(
            f"[SOURCE FIRST] feed={feed.get('name') or feed_url} | "
            f"entries_inspected={inspected} | fresh_accepted={accepted}"
        )

        if len(result) >= max_seeds:
            break

    print(
        f"[SOURCE FIRST] language={language} | feeds={len(feeds)} | "
        f"fresh_unique_seeds={len(result)} | max_age_hours={max_age}"
    )
    return result
