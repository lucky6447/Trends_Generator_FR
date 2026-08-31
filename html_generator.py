import re
from jinja2 import Environment,FileSystemLoader
from pathlib import Path
from config import ROOT,TREND_DIR,SITE_URL,LANGUAGE

env=Environment(loader=FileSystemLoader(ROOT/'templates'),autoescape=True)
template=env.get_template('article.html')

# Optional, isolated AI-image module.
# If unavailable or generation fails, article generation continues without an image.
try:
    from ai_image_cloudflare import generate_article_image
except Exception:
    generate_article_image = None


def _source_homepage(article, news=None):
    """
    Resolve the publisher homepage from production metadata.

    Preferred order:
    1. source_href/source_url/publisher_url already present on the article.
    2. source_href carried by the news items.
    3. A conservative domain inference from the publisher name.

    This keeps the image module isolated: failure to resolve a publisher
    homepage simply means no image is attached.
    """
    for key in ("source_href", "source_url", "publisher_url"):
        value = str(article.get(key, "")).strip()
        if value:
            return value

    for item in news or []:
        if isinstance(item, dict):
            value = str(item.get("source_href", "")).strip()
            if value:
                return value

    source = str(article.get("source", "")).strip()
    if not source:
        return None

    normalized = re.sub(r"\s+", " ", source).strip().lower()

    known = {
        "bbc": "https://www.bbc.co.uk",
        "bbc news": "https://www.bbc.co.uk",
        "the guardian": "https://www.theguardian.com",
        "guardian": "https://www.theguardian.com",
        "sky news": "https://news.sky.com",
        "reuters": "https://www.reuters.com",
        "associated press": "https://apnews.com",
        "ap": "https://apnews.com",
        "bloomberg": "https://www.bloomberg.com",
        "bloomberg.com": "https://www.bloomberg.com",
        "cnn": "https://edition.cnn.com",
        "cnn.com": "https://edition.cnn.com",
        "nbc news": "https://www.nbcnews.com",
        "abc news": "https://abcnews.go.com",
        "cbs news": "https://www.cbsnews.com",
        "fox news": "https://www.foxnews.com",
        "usa today": "https://www.usatoday.com",
        "new york times": "https://www.nytimes.com",
        "the new york times": "https://www.nytimes.com",
        "washington post": "https://www.washingtonpost.com",
        "the washington post": "https://www.washingtonpost.com",
        "politico": "https://www.politico.com",
        "the hill": "https://thehill.com",
        "forbes": "https://www.forbes.com",
        "financial times": "https://www.ft.com",
        "ft": "https://www.ft.com",
        "the independent": "https://www.independent.co.uk",
        "independent": "https://www.independent.co.uk",
        "daily mail": "https://www.dailymail.co.uk",
        "the telegraph": "https://www.telegraph.co.uk",
        "telegraph": "https://www.telegraph.co.uk",
        "mirror": "https://www.mirror.co.uk",
        "daily mirror": "https://www.mirror.co.uk",
    }

    if normalized in known:
        return known[normalized]

    # If the publisher itself is already a hostname, use it directly.
    candidate = normalized
    candidate = re.sub(r"^https?://", "", candidate).strip("/")
    if "." in candidate and " " not in candidate:
        return "https://" + candidate

    return None


def _attach_ai_image(article, news=None):
    """
    Generate one original AI image for the accepted article.
    Fail closed: image failure NEVER blocks article publishing.
    """
    if not generate_article_image:
        return article

    if article.get("image_data"):
        return article

    try:
        image_data = generate_article_image(article, news=news)
    except Exception as exc:
        print(f"[AI IMAGE] FAILED (article continues): {exc}")
        return article

    if image_data:
        article["image_data"] = image_data
        print(
            f"[AI IMAGE] GENERATED: {image_data.get('image', '')} "
            f"| {image_data.get('elapsed', '')}s"
        )
    else:
        print(f"[AI IMAGE] NOT GENERATED: {article.get('title', '')}")

    return article

def render_article(article, news=None):
    article = _attach_ai_image(article, news=news)
    return template.render(
        article=article,
        site_url=SITE_URL,
        language=LANGUAGE,
    )


def save_article(slug,html):
    (TREND_DIR/f'{slug}.html').write_text(html,encoding='utf-8')
