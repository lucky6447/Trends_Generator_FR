import html
import json
import re
from html.parser import HTMLParser
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader

from config import ROOT, TREND_DIR, SITE_NAME, SITE_URL, ARTICLES_PER_PAGE, LANGUAGE
from article_dates import load_dates, ordered_slugs

env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=True)
template = env.get_template("index.html")


# Date-navigation copy is derived from the site's existing LANGUAGE setting.
# No language-specific template changes are required.
_DATE_I18N = {
    "en": {"title": "Articles by date", "today": "Today", "yesterday": "Yesterday", "more": "More dates", "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]},
    "de": {"title": "Artikel nach Datum", "today": "Heute", "yesterday": "Gestern", "more": "Weitere Daten", "months": ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]},
    "fr": {"title": "Articles par date", "today": "Aujourd’hui", "yesterday": "Hier", "more": "Plus de dates", "months": ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]},
    "it": {"title": "Articoli per data", "today": "Oggi", "yesterday": "Ieri", "more": "Altre date", "months": ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]},
    "es": {"title": "Artículos por fecha", "today": "Hoy", "yesterday": "Ayer", "more": "Más fechas", "months": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]},
    "id": {"title": "Artikel berdasarkan tanggal", "today": "Hari ini", "yesterday": "Kemarin", "more": "Tanggal lainnya", "months": ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]},
}


def _language_key():
    value = (LANGUAGE or "").strip().lower()
    aliases = {
        "en": "en",
        "english": "en",
        "english (us)": "en",
        "en-us": "en",
        "en_us": "en",
        "id": "id",
        "bahasa indonesia": "id",
        "indonesian": "id",
        "id-id": "id",
        "de": "de",
        "deutsch": "de",
        "german": "de",
        "de-de": "de",
        "fr": "fr",
        "français": "fr",
        "french": "fr",
        "fr-fr": "fr",
        "it": "it",
        "italiano": "it",
        "italian": "it",
        "it-it": "it",
        "es": "es",
        "español": "es",
        "spanish": "es",
        "es-es": "es",
    }
    if value in aliases:
        return aliases[value]

    # Also accept common config values such as "French (France)" or
    # locale strings such as "fr-FR". This keeps the generator universal
    # without requiring any changes to language-specific config.py files.
    normalized = re.sub(r"[_-]+", "-", value)
    if normalized.startswith(("fr-", "fr ")):
        return "fr"
    if normalized.startswith(("de-", "de ")):
        return "de"
    if normalized.startswith(("it-", "it ")):
        return "it"
    if normalized.startswith(("es-", "es ")):
        return "es"
    if normalized.startswith(("id-", "id ")):
        return "id"
    if normalized.startswith(("en-", "en ")):
        return "en"
    if "français" in value or "french" in value:
        return "fr"
    if "deutsch" in value or "german" in value:
        return "de"
    if "italiano" in value or "italian" in value:
        return "it"
    if "español" in value or "spanish" in value:
        return "es"
    if "indonesian" in value or "bahasa indonesia" in value:
        return "id"
    if "english" in value:
        return "en"
    return "en"


def _date_copy(date_obj, today):
    copy = _DATE_I18N[_language_key()]
    if date_obj == today:
        return copy["today"]
    if date_obj == today - timedelta(days=1):
        return copy["yesterday"]
    return f"{date_obj.day} {copy['months'][date_obj.month - 1]} {date_obj.year}"




class _TitleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_title = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.parts.append(data)


def _extract_public_title(html_text, fallback):
    """Read the generated editorial <title>; never derive card text from slug."""
    parser = _TitleParser()
    try:
        parser.feed(html_text)
        title = " ".join("".join(parser.parts).split()).strip()
        return title or fallback
    except Exception:
        return fallback

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

        try:
            html_text = f.read_text(encoding="utf-8")
        except Exception:
            html_text = ""

        fallback_title = f.stem.replace("-", " ").title()
        public_title = _extract_public_title(html_text, fallback_title)

        articles.append({
            "title": public_title,
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


def get_date_groups(articles):
    groups = {}
    for article in articles:
        date = article["lastmod"]
        groups.setdefault(date, []).append(article)

    today = datetime.now().astimezone().date()
    navigation = []
    copy = _DATE_I18N[_language_key()]

    for date_string, date_articles in sorted(groups.items(), reverse=True):
        date_obj = datetime.strptime(date_string, "%Y-%m-%d").date()
        navigation.append({
            "date": date_string,
            "label": _date_copy(date_obj, today),
            "count": len(date_articles),
            "url": f"/date/{date_string}/",
        })

    return groups, navigation, copy["title"]


_DATE_NAV_CSS = r"""
<style id="tc-date-navigation-css">
.tc-date-navigation{margin:0 0 28px;padding:14px 16px;background:#0f1b2d;border:1px solid rgba(75,130,255,.2);border-radius:16px;box-shadow:0 12px 30px rgba(0,0,0,.22)}
.tc-date-navigation-title{margin:0 0 10px;color:#f3f6ff;font-size:14px;font-weight:700}
.tc-date-navigation-row{display:flex;align-items:center;gap:8px;min-width:0;padding:2px 2px 5px}
.tc-date-navigation-scroll{display:flex;align-items:center;gap:8px;min-width:0;flex:1 1 auto;overflow-x:auto;overflow-y:hidden;scrollbar-width:thin;-webkit-overflow-scrolling:touch}
.tc-date-link{flex:0 0 auto;display:inline-flex;align-items:center;gap:7px;min-height:42px;padding:0 12px;border:1px solid rgba(59,130,246,.25);border-radius:10px;background:#101b2d;color:#c9d6ee;text-decoration:none;font-size:14px;font-weight:600;white-space:nowrap;transition:.2s}
.tc-date-link:hover{background:#1c3152;border-color:#4f8cff;color:#fff}
.tc-date-link.active{background:linear-gradient(135deg,#3b82f6,#2563eb);border-color:#3b82f6;color:#fff;box-shadow:0 7px 18px rgba(59,130,246,.22)}
.tc-date-count{min-width:22px;height:22px;padding:0 6px;display:inline-flex;align-items:center;justify-content:center;border-radius:7px;background:rgba(255,255,255,.09);font-size:12px;line-height:1}
.tc-date-link.active .tc-date-count{background:rgba(255,255,255,.18)}
.tc-date-more{margin-top:0;flex:0 0 auto}
.tc-date-more summary{display:inline-flex;align-items:center;gap:7px;min-height:40px;padding:0 13px;border:1px solid rgba(59,130,246,.25);border-radius:10px;background:#101b2d;color:#c9d6ee;cursor:pointer;font-size:13px;font-weight:600;list-style:none}
.tc-date-more summary::-webkit-details-marker{display:none}
.tc-date-more summary::after{content:"▾";font-size:12px;opacity:.8}
.tc-date-more[open] summary::after{content:"▴"}
.tc-date-more summary:hover{background:#1c3152;border-color:#4f8cff;color:#fff}
.tc-date-more-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-top:9px;padding:10px;background:#0b1627;border:1px solid rgba(75,130,255,.16);border-radius:12px;max-height:260px;overflow-y:auto}
.tc-date-more-list .tc-date-link{width:100%;box-sizing:border-box}
@media(max-width:768px){
  .tc-date-navigation{margin:0 0 22px;padding:13px;border-radius:14px}
  .tc-date-navigation-title{font-size:13px}
  .tc-date-navigation-row{flex-wrap:nowrap;min-width:0}
  .tc-date-navigation-scroll{flex:1 1 auto;min-width:0}
  .tc-date-link{min-height:44px;padding:0 12px;font-size:13px}
  .tc-date-more-list{grid-template-columns:1fr;max-height:300px}
}
</style>
"""



def _strip_old_date_navigation(rendered):
    # Older versions of the template may already contain the date block.
    # Remove it so replacing this generator never creates duplicate navigation.
    return re.sub(
        r'\s*<div class="date-navigation"[^>]*>.*?</div>\s*(?=<div class="grid">)',
        "\n",
        rendered,
        flags=re.DOTALL,
    )


def _build_date_navigation(date_navigation, title, active_date, more_label):
    # Keep the primary navigation compact forever: only the seven newest
    # publication dates are shown directly. Older dates remain accessible
    # through a native <details> panel, with no JavaScript required.
    if active_date and active_date not in {item["date"] for item in date_navigation[:7]}:
        # Keep six newest dates plus the currently selected older date visible.
        # This prevents an old date page from hiding its active state in "More dates".
        active_item = next((item for item in date_navigation if item["date"] == active_date), None)
        visible_items = date_navigation[:6] + ([active_item] if active_item else date_navigation[6:7])
        visible_dates = {item["date"] for item in visible_items}
        older_items = [item for item in date_navigation if item["date"] not in visible_dates]
    else:
        visible_items = date_navigation[:7]
        visible_dates = {item["date"] for item in visible_items}
        older_items = [item for item in date_navigation if item["date"] not in visible_dates]

    def build_link(item):
        active = " active" if item["date"] == active_date else ""
        return (
            f'<a class="tc-date-link{active}" href="{html.escape(item["url"], quote=True)}">'
            f'<span>{html.escape(item["label"])}</span>'
            f'<span class="tc-date-count">{item["count"]}</span></a>'
        )

    visible_links = "".join(build_link(item) for item in visible_items)

    more_block = ""
    if older_items:
        older_links = "".join(build_link(item) for item in older_items)
        more_block = (
            '<details class="tc-date-more">'
            f'<summary>{html.escape(more_label)}</summary>'
            f'<div class="tc-date-more-list">{older_links}</div>'
            '</details>'
        )

    return (
        '<div class="tc-date-navigation" aria-label="'
        + html.escape(title, quote=True)
        + '">'
        f'<div class="tc-date-navigation-title">{html.escape(title)}</div>'
        '<div class="tc-date-navigation-row">'
        '<div class="tc-date-navigation-scroll">'
        + visible_links
        + '</div>'
        + more_block
        + '</div>'
        + '</div>'
    )


def _inject_date_navigation(rendered, date_navigation, title, active_date, more_label):
    rendered = _strip_old_date_navigation(rendered)
    navigation = _build_date_navigation(date_navigation, title, active_date, more_label)

    if "</head>" in rendered and "tc-date-navigation-css" not in rendered:
        rendered = rendered.replace("</head>", _DATE_NAV_CSS + "\n</head>", 1)

    # Insert immediately before the article grid. This works across language
    # templates as long as the existing article grid remains the same.
    marker = '<div class="grid">'
    if marker not in rendered:
        raise RuntimeError("Could not find article grid marker in templates/index.html")
    return rendered.replace(marker, navigation + "\n\n" + marker, 1)


def render_index(articles, page_number, total_pages, *, canonical_url=None, page_title=None, active_date=None, pagination_base="/"):
    all_articles = get_articles()
    _, date_navigation, date_title = get_date_groups(all_articles)
    rendered = template.render(
        site_name=SITE_NAME,
        articles=articles,
        site_url=SITE_URL,
        page_number=page_number,
        total_pages=total_pages,
        canonical_url=canonical_url or SITE_URL.rstrip("/") + "/",
        page_title=page_title or f"{SITE_NAME} | Latest Trends",
        date_navigation=[],
        active_date=active_date,
        pagination_base=pagination_base,
    )
    return _inject_date_navigation(rendered, date_navigation, date_title, active_date, _DATE_I18N[_language_key()]["more"])


def update_index_html():
    articles = get_articles()
    total_pages = max(
        1,
        (len(articles) + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE
    )

    Path("index.html").write_text(
        render_index(articles[:ARTICLES_PER_PAGE], 1, total_pages),
        encoding="utf-8"
    )


def update_pagination():
    articles = get_articles()
    pages = [articles[i:i + ARTICLES_PER_PAGE] for i in range(0, len(articles), ARTICLES_PER_PAGE)]

    page_root = ROOT / "page"
    if page_root.exists():
        for old_dir in page_root.iterdir():
            if old_dir.is_dir():
                for old_file in old_dir.rglob("*"):
                    if old_file.is_file():
                        old_file.unlink()
                for old_subdir in sorted([p for p in old_dir.rglob("*") if p.is_dir()], reverse=True):
                    old_subdir.rmdir()
                old_dir.rmdir()
    else:
        page_root.mkdir(parents=True, exist_ok=True)

    total_pages = len(pages)
    for number, page_articles in enumerate(pages[1:], start=2):
        page_dir = page_root / str(number)
        page_dir.mkdir(parents=True, exist_ok=True)
        (page_dir / "index.html").write_text(
            render_index(page_articles, number, total_pages),
            encoding="utf-8"
        )


def update_date_pages():
    articles = get_articles()
    groups, date_navigation, _ = get_date_groups(articles)
    date_root = ROOT / "date"

    if date_root.exists():
        for old_dir in date_root.iterdir():
            if old_dir.is_dir():
                for old_file in old_dir.rglob("*"):
                    if old_file.is_file():
                        old_file.unlink()
                for old_subdir in sorted([p for p in old_dir.rglob("*") if p.is_dir()], reverse=True):
                    old_subdir.rmdir()
                old_dir.rmdir()
    else:
        date_root.mkdir(parents=True, exist_ok=True)

    for date_string, date_articles in groups.items():
        nav_item = next(item for item in date_navigation if item["date"] == date_string)
        title = f"{nav_item['label']} ({len(date_articles)}) | {SITE_NAME}"
        canonical = SITE_URL.rstrip("/") + f"/date/{date_string}/"
        base_path = f"/date/{date_string}/"
        pages = [date_articles[i:i + ARTICLES_PER_PAGE] for i in range(0, len(date_articles), ARTICLES_PER_PAGE)]
        total_pages = max(1, len(pages))

        for number, page_articles in enumerate(pages, start=1):
            page_dir = date_root / date_string if number == 1 else date_root / date_string / "page" / str(number)
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / "index.html").write_text(
                render_index(
                    page_articles,
                    number,
                    total_pages,
                    canonical_url=canonical,
                    page_title=title,
                    active_date=date_string,
                    pagination_base=base_path,
                ),
                encoding="utf-8",
            )


def update_sitemap():
    articles = get_articles()
    urlset = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")

    home = ET.SubElement(urlset, "url")
    ET.SubElement(home, "loc").text = SITE_URL.rstrip("/") + "/"
    ET.SubElement(home, "lastmod").text = datetime.now().astimezone().strftime("%Y-%m-%d")
    ET.SubElement(home, "changefreq").text = "hourly"
    ET.SubElement(home, "priority").text = "1.0"

    for item in get_date_groups(articles)[1]:
        url = ET.SubElement(urlset, "url")
        ET.SubElement(url, "loc").text = SITE_URL.rstrip("/") + item["url"]
        ET.SubElement(url, "lastmod").text = item["date"]
        ET.SubElement(url, "changefreq").text = "daily"
        ET.SubElement(url, "priority").text = "0.6"

    for article in articles:
        url = ET.SubElement(urlset, "url")
        sitemap_url = article["url"].removesuffix(".html")
        ET.SubElement(url, "loc").text = SITE_URL.rstrip("/") + sitemap_url
        ET.SubElement(url, "lastmod").text = article["lastmod"]
        ET.SubElement(url, "changefreq").text = "monthly"
        ET.SubElement(url, "priority").text = "0.8"

    ET.indent(ET.ElementTree(urlset), space="  ")
    xml_content = ET.tostring(urlset, encoding="unicode")
    Path("sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_content, encoding="utf-8")


def update_all():
    update_index_json()
    update_search_json()
    update_index_html()
    update_pagination()
    update_date_pages()
    update_sitemap()
