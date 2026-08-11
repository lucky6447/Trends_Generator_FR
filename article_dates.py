import json
from datetime import datetime
from pathlib import Path

from config import ROOT, TREND_DIR

DATES_FILE = ROOT / "article_dates.json"


def load_dates():
    if not DATES_FILE.exists():
        return {}
    try:
        data = json.loads(DATES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_dates(data):
    DATES_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def register_article(slug):
    """Register a new article once and keep its timestamp stable forever."""
    data = load_dates()
    if slug in data:
        return data[slug]

    now = datetime.now().astimezone().replace(microsecond=0)
    existing_orders = [
        item.get("order", 0)
        for item in data.values()
        if isinstance(item, dict) and isinstance(item.get("order"), int)
    ]
    order = min(existing_orders, default=0) - 1

    data[slug] = {
        "published_at": now.strftime("%Y-%m-%d %H:%M"),
        "order": order,
    }
    save_dates(data)
    return data[slug]


def ordered_slugs():
    """Return existing article slugs in newest-first order."""
    data = load_dates()

    # Any article not yet registered gets a stable timestamp now.
    changed = False
    for f in TREND_DIR.glob("*.html"):
        if f.stem not in data:
            now = datetime.fromtimestamp(f.stat().st_mtime).astimezone()
            orders = [
                item.get("order", 0)
                for item in data.values()
                if isinstance(item, dict) and isinstance(item.get("order"), int)
            ]
            data[f.stem] = {
                "published_at": now.strftime("%Y-%m-%d %H:%M"),
                "order": min(orders, default=0) - 1,
            }
            changed = True

    if changed:
        save_dates(data)

    def key(slug):
        item = data.get(slug, {})
        published = item.get("published_at", "")
        order = item.get("order", 0)
        return published, -order

    return [
        f.stem
        for f in sorted(
            TREND_DIR.glob("*.html"),
            key=lambda f: key(f.stem),
            reverse=True,
        )
    ]
