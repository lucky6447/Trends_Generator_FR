import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

from config import ROOT, TREND_DIR, SITE_NAME, SITE_URL, ARTICLES_PER_PAGE

env = Environment(
    loader=FileSystemLoader(ROOT / "templates"),
    autoescape=True
)

template = env.get_template("index.html")


def _load_saved_article_times():
    """
    Keep the original published date/time from index.json for existing articles.
    This prevents a regeneration from changing old article timestamps.
    """
    path = Path("index.json")

    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return {}

        return {
            item.get("url"): item
            for item in data
            if isinstance(item, dict) and item.get("url")
        }
    except Exception:
        return {}


def get_articles():
    saved = _load_saved_article_times()
    articles = []

    for f in TREND_DIR.glob("*.html"):
        url = f"/trends/{f.name}"
        old = saved.get(url)

        if old and old.get("updated"):
            updated = old["updated"]
            lastmod = old.get("lastmod") or updated[:10]
        else:
            file_time = datetime.fromtimestamp(f.stat().st_mtime)
            updated = file_time.strftime("%Y-%m-%d %H:%M")
            lastmod = file_time.strftime("%Y-%m-%d")

        articles.append(
            {
                "title": f.stem.replace("-", " ").title(),
                "url": url,
                "updated": updated,
                "lastmod": lastmod,
            }
        )

    def sort_key(article):
        try:
            return datetime.strptime(article["updated"], "%Y-%m-%d %H:%M")
        except Exception:
            return datetime.min

    articles.sort(key=sort_key, reverse=True)
    return articles



def update_search_json():
    """
    Create search index used by frontend search.
    Generated from the same article list as index.json.
    """
    Path("search.json").write_text(
        json.dumps(
            get_articles(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def update_index_json():
    Path("index.json").write_text(
        json.dumps(
            get_articles(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def update_index_html():
    all_articles = get_articles()
    articles = all_articles[:ARTICLES_PER_PAGE]
    total_pages = max(
        1,
        (len(all_articles) + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE
    )

    Path("index.html").write_text(
        template.render(
            site_name=SITE_NAME,
            articles=articles,
            site_url=SITE_URL,
            page_number=1,
            total_pages=total_pages,
        ),
        encoding="utf-8",
    )


def update_pagination():
    articles = get_articles()

    pages = [
        articles[i:i + ARTICLES_PER_PAGE]
        for i in range(0, len(articles), ARTICLES_PER_PAGE)
    ]

    total_pages = len(pages)

    for number, page_articles in enumerate(pages[1:], start=2):
        page_dir = Path(f"page/{number}")
        page_dir.mkdir(parents=True, exist_ok=True)

        (page_dir / "index.html").write_text(
            template.render(
                site_name=SITE_NAME,
                articles=page_articles,
                site_url=SITE_URL,
                page_number=number,
                total_pages=total_pages,
            ),
            encoding="utf-8",
        )

    # Remove pagination directories that no longer correspond to real pages.
    page_root = Path("page")
    if page_root.exists():
        for directory in page_root.iterdir():
            if directory.is_dir() and directory.name.isdigit():
                if int(directory.name) > total_pages:
                    for child in directory.iterdir():
                        if child.is_file() or child.is_symlink():
                            child.unlink()
                    try:
                        directory.rmdir()
                    except OSError:
                        pass


def update_sitemap():
    articles = get_articles()

    urlset = ET.Element(
        "urlset",
        xmlns="http://www.sitemaps.org/schemas/sitemap/0.9",
    )

    # Homepage
    home = ET.SubElement(urlset, "url")
    ET.SubElement(home, "loc").text = SITE_URL.rstrip("/") + "/"
    ET.SubElement(home, "lastmod").text = datetime.utcnow().strftime("%Y-%m-%d")
    ET.SubElement(home, "changefreq").text = "hourly"
    ET.SubElement(home, "priority").text = "1.0"

    # Articles
    for article in articles:
        url = ET.SubElement(urlset, "url")

        ET.SubElement(
            url,
            "loc",
        ).text = SITE_URL.rstrip("/") + article["url"]

        ET.SubElement(
            url,
            "lastmod",
        ).text = article["lastmod"]

        ET.SubElement(
            url,
            "changefreq",
        ).text = "monthly"

        ET.SubElement(
            url,
            "priority",
        ).text = "0.8"

    tree = ET.ElementTree(urlset)

    ET.indent(tree, space="  ")

    xml_content = ET.tostring(
        urlset,
        encoding="unicode"
    )

    with open("sitemap.xml", "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(xml_content)


def update_all():
    update_index_json()
    update_search_json()
    update_index_html()
    update_pagination()
    update_sitemap()