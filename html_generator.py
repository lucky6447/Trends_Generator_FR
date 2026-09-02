import re
from jinja2 import Environment,FileSystemLoader
from pathlib import Path
from config import ROOT,TREND_DIR,SITE_URL,LANGUAGE

env=Environment(loader=FileSystemLoader(ROOT/'templates'),autoescape=True)
template=env.get_template('article.html')


def _attach_source_image(article, news=None):
    """
    Attach a real image declared by a publisher page.

    Runs only when render_article() is called, which in production happens
    after the final Story Quality Gate. The image itself is never downloaded;
    only publisher HTML is fetched to discover the declared image URL.

    We try at most 3 source articles, in their existing evidence order.
    If no valid publisher image is found, the article is published without one.
    """
    if article.get("image_data"):
        return article

    try:
        from news import extract_source_image
    except Exception as exc:
        print(f"[SOURCE IMAGE] module unavailable: {exc}")
        return article

    candidates = []
    seen = set()

    for item in news or []:
        if not isinstance(item, dict):
            continue
        link = str(item.get("link", "")).strip()
        if not link or link in seen:
            continue
        seen.add(link)
        candidates.append((link, str(item.get("source", "")).strip()))

        if len(candidates) >= 3:
            break

    for link, source_name in candidates:
        try:
            image_data = extract_source_image(link)
        except Exception as exc:
            print(f"[SOURCE IMAGE] FAILED: {exc}")
            continue

        if image_data and image_data.get("image"):
            if not image_data.get("source"):
                image_data["source"] = source_name

            article["image_data"] = image_data
            print(
                f"[SOURCE IMAGE] FOUND: {image_data.get('image')} "
                f"| source={image_data.get('source', '')}"
            )
            return article

        print(
            f"[SOURCE IMAGE] NONE: "
            f"{image_data.get('source_url', link) if isinstance(image_data, dict) else link}"
        )

    print(f"[SOURCE IMAGE] NOT FOUND: {article.get('title', '')}")
    return article


def render_article(article, news=None):
    article = _attach_source_image(article, news=news)
    return template.render(
        article=article,
        site_url=SITE_URL,
        language=LANGUAGE,
    )


def save_article(slug,html):
    (TREND_DIR/f'{slug}.html').write_text(html,encoding='utf-8')
