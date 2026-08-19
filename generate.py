import re
import subprocess
from datetime import date

from config import MAX_ARTICLES_PER_RUN
from rss import fetch_trends
from news import fetch_news
from prompt import build_prompt
from ollama_client import generate
from fact_guard import validate as fact_guard_validate
from fact_guard_repair import repair as fact_guard_repair
import json
from html_generator import render_article, save_article
from processed import load_processed, add_processed
from index_generator import update_all

REQUIRED_FIELDS = ["title", "description", "h1", "intro", "sections"]

MIN_WORDS = 100

SKIP_PATTERNS = [
    " vs ",
    " v ",
    " live",
    " score",
    " result",
    " calendario",
    " alineación",
    " pronóstico",
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

    if not 1 <= len(article["sections"]) <= 5:
        raise Exception("Article should contain 1-5 well-structured sections.")

    titles = set()
    words = len(article["intro"].split())

    for s in article["sections"]:
        if "title" not in s or "text" not in s:
            raise Exception("Invalid section")
        if s["title"] in titles:
            raise Exception("Duplicate section title")
        titles.add(s["title"])
        words += len(s["text"].split())

    min_words = 55
    if words < min_words:
        raise Exception(f"Article too short ({words} words; minimum {min_words})")

    return True


def generate_valid_article(prompt, fact_guard_source, reference_date, max_attempts=1):
    last = None

    for i in range(max_attempts):
        try:
            article = generate(prompt)
            validate_article(article)

            print("[FACT GUARD] Checking generated article...")
            guard = fact_guard_validate(
                fact_guard_source,
                article,
                reference_date=reference_date,
            )

            if guard["status"] != "PASS":
                print("[FACT GUARD] FLAG - article requires repair.")
                print(json.dumps(guard, ensure_ascii=False, indent=2))

                try:
                    print("[FACT GUARD REPAIR] Attempting targeted repair v1.0...")
                    repaired = fact_guard_repair(
                        article,
                        fact_guard_source,
                        guard,
                    )
                    validate_article(repaired)

                    print("[FACT GUARD REPAIR] Re-checking repaired article...")
                    repaired_guard = fact_guard_validate(
                        fact_guard_source,
                        repaired,
                        reference_date=reference_date,
                    )

                    if repaired_guard["status"] != "PASS":
                        print("[FACT GUARD REPAIR] FAIL - repaired article blocked.")
                        print(
                            json.dumps(
                                repaired_guard,
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                        raise Exception(
                            "Fact Guard repair failed re-validation "
                            f"({repaired_guard['blocking_issues']} blocking issue(s))"
                        )

                    print("[FACT GUARD REPAIR] PASS - repaired article accepted.")

                    if repaired_guard.get("review_items", 0):
                        print(
                            f"[FACT GUARD] PASS with "
                            f"{repaired_guard['review_items']} review item(s)."
                        )
                    else:
                        print("[FACT GUARD] PASS")

                    return repaired

                except Exception as repair_error:
                    raise Exception(
                        f"Fact Guard blocked article; repair failed: {repair_error}"
                    ) from repair_error

            if guard.get("review_items", 0):
                print(
                    f"[FACT GUARD] PASS with "
                    f"{guard['review_items']} review item(s)."
                )
            else:
                print("[FACT GUARD] PASS")

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

    # One explicit reference date for the entire production run.
    # This is the date against which event state is evaluated.
    reference_date = date.today()
    print(f"[FACT GUARD] Validation reference date: {reference_date.isoformat()}")

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

            if len(news) < 2:
                print(f"SKIPPED: only {len(news)} news found")
                continue

            trend["news"] = news

            generation_prompt = build_prompt(trend)
            fact_guard_source = json.dumps(
                news,
                ensure_ascii=False,
                indent=2,
            )

            article = generate_valid_article(
                generation_prompt,
                fact_guard_source,
                reference_date,
            )

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
