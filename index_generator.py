import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

from config import ROOT, TREND_DIR, SITE_NAME, SITE_URL, ARTICLES_PER_PAGE
from article_dates import load_dates, ordered_slugs

env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=True)
template = env.get_template("index.html")


def get_articles():
    """Build the article list using the permanent publication timestamps."""
    ordered = ordered_slugs()
    dates = load_dates()
    articles = []

    for slug in ordered:
        f = TREND_DIR / f"{slug}.html"
        if not f.exists():
            continue

        item = dates.get(slug, {})
        updated = item.get("published_at")

        if not updated:
            file_time = datetime.fromtimestamp(f.stat().st_mtime).astimezone()
            updated = file_time.strftime("%Y-%m-%d %H:%M")

        articles.append({
            "title": f.stem.replace("-", " ").title(),
            "url": f"/trends/{f.name}",
            "updated": updated,
            "lastmod": updated[:10],
        })

    return articles


def update_search_json():
    Path("search.json").write_text(
        json.dumps(get_articles(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def update_index_json():
    Path("index.json").write_text(
        json.dumps(get_articles(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def update_index_html():
    articles = get_articles()
    total_pages = max(
        1,
        (len(articles) + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE
    )

    Path("index.html").write_text(
        template.render(
            site_name=SITE_NAME,
            articles=articles[:ARTICLES_PER_PAGE],
            site_url=SITE_URL,
            page_number=1,
            total_pages=total_pages
        ),
        encoding="utf-8"
    )


def update_pagination():
    articles = get_articles()
    pages = [
        articles[i:i + ARTICLES_PER_PAGE]
        for i in range(0, len(articles), ARTICLES_PER_PAGE)
    ]

    # Remove old generated pagination pages so stale pages never remain.
    page_root = ROOT / "page"
    if page_root.exists():
        for old_dir in page_root.iterdir():
            if old_dir.is_dir():
                for old_file in old_dir.rglob("*"):
                    if old_file.is_file():
                        old_file.unlink()
                for old_subdir in sorted(
                    [p for p in old_dir.rglob("*") if p.is_dir()],
                    reverse=True
                ):
                    old_subdir.rmdir()
                old_dir.rmdir()
    else:
        page_root.mkdir(parents=True, exist_ok=True)

    total_pages = len(pages)

    for number, page_articles in enumerate(pages[1:], start=2):
        page_dir = page_root / str(number)
        page_dir.mkdir(parents=True, exist_ok=True)

        (page_dir / "index.html").write_text(
            template.render(
                site_name=SITE_NAME,
                articles=page_articles,
                site_url=SITE_URL,
                page_number=number,
                total_pages=total_pages
            ),
            encoding="utf-8"
        )


def update_sitemap():
    articles = get_articles()

    urlset = ET.Element(
        "urlset",
        xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
    )

    home = ET.SubElement(urlset, "url")
    ET.SubElement(home, "loc").text = SITE_URL.rstrip("/") + "/"
    ET.SubElement(home, "lastmod").text = datetime.utcnow().strftime("%Y-%m-%d")
    ET.SubElement(home, "changefreq").text = "hourly"
    ET.SubElement(home, "priority").text = "1.0"

    for article in articles:
        url = ET.SubElement(urlset, "url")
        sitemap_url = article["url"].removesuffix(".html")
        ET.SubElement(url, "loc").text = SITE_URL.rstrip("/") + sitemap_url
        ET.SubElement(url, "lastmod").text = article["lastmod"]
        ET.SubElement(url, "changefreq").text = "monthly"
        ET.SubElement(url, "priority").text = "0.8"

    ET.indent(ET.ElementTree(urlset), space="  ")
    xml_content = ET.tostring(urlset, encoding="unicode")

    with open("sitemap.xml", "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(xml_content)


def update_all():
    update_index_json()
    update_search_json()
    update_index_html()
    update_pagination()
    update_sitemap()
