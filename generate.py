import re
import subprocess

from config import MAX_ARTICLES_PER_RUN
from rss import fetch_trends
from news import fetch_news
from prompt import build_prompt
from ollama_client import generate
from html_generator import render_article, save_article
from processed import load_processed, add_processed
from index_generator import update_all

REQUIRED_FIELDS = ["title", "description", "h1", "intro", "sections", "faq"]

MIN_WORDS = 100

SKIP_PATTERNS = [
    " vs ",
    " v ",
    " live",
    " score",
    " result",
    " spielplan",
    " aufstellung",
    " prognose",
    " stream",
    " streaming",
]


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def validate_article(article):
    if not isinstance(article, dict):
        raise Exception("Article is not JSON object")

    for f in REQUIRED_FIELDS:
        if f not in article:
            raise Exception(f"Missing field: {f}")

    if not isinstance(article["sections"], list):
        raise Exception("Sections must be a list")

    if not 3 <= len(article["sections"]) <= 5:
        raise Exception("Article should contain 3-5 well-structured sections.")

    titles = set()
    words = len(article["intro"].split())

    for s in article["sections"]:
        if "title" not in s or "text" not in s:
            raise Exception("Invalid section")

        if s["title"] in titles:
            raise Exception("Duplicate section title")

        titles.add(s["title"])
        words += len(s["text"].split())

    if words < MIN_WORDS:
        raise Exception(f"Article too short ({words} words)")

    return True


def generate_valid_article(prompt, max_attempts=3):
    last = None

    for i in range(max_attempts):
        try:
            article = generate(prompt)
            validate_article(article)
            return article

        except Exception as e:
            last = e
            print(f"Validation failed ({i+1}/{max_attempts}): {e}")

    raise Exception(last)


def run_git(cmd):
    print("\n" + "=" * 60)
    print("Running:", " ".join(cmd))
    print("=" * 60)

    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.stdout:
        print(r.stdout)

    if r.stderr:
        print(r.stderr)

    return r.returncode == 0


def git_push():
    if not run_git(["git", "status"]):
        return

    if not run_git(["git", "add", "."]):
        return

    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print("No changes to commit.")
        return

    if not run_git(["git", "commit", "-m", "Auto update"]):
        return

    if not run_git(["git", "push", "origin", "main"]):
        return

    print("SUCCESS: GitHub updated.")


def main():
    processed = load_processed()
    trends = fetch_trends()

    generated = 0
    new_keywords = []

    for trend in trends:
        keyword = trend["title"]

        keyword_lower = keyword.lower()

        if any(pattern in keyword_lower for pattern in SKIP_PATTERNS):
            print(f"SKIPPED: {keyword}")
            continue

        if keyword in processed:
            continue

        print(f"\nGenerating: {keyword}")

        try:
            news = fetch_news(keyword)

            if len(news) < 5:
                print(f"SKIPPED: only {len(news)} news found")
                continue

            trend["news"] = news

            article = generate_valid_article(build_prompt(trend))

            slug = slugify(keyword)
            article["slug"] = slug

            save_article(slug, render_article(article))

            new_keywords.append(keyword)
            generated += 1

            print(f"OK -> {slug}.html")

            if generated >= MAX_ARTICLES_PER_RUN:
                break

        except Exception as e:
            print(f"ERROR: {keyword}: {e}")

    try:
        update_all()

        for k in new_keywords:
            add_processed(k, processed)

    except Exception as e:
        print("UPDATE ERROR:", e)

    print(f"Finished. Generated {generated} article(s).")

    if generated:
        git_push()


if __name__ == "__main__":
    main()