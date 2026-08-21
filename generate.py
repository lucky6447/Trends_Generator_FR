import re
import os
import subprocess
from datetime import date

from config import MAX_ARTICLES_PER_RUN, LANGUAGE
from rss import fetch_trends
from news import fetch_news
from prompt import build_prompt
from ollama_client import generate, extract_evidence
from ollama import chat
from config import MODEL
from fact_guard import validate as fact_guard_validate
from fact_guard_repair import repair as fact_guard_repair
import json
from html_generator import render_article, save_article
from processed import load_processed, add_processed
from index_generator import update_all
from topic_scorer import rank_trends, score_with_news, select_final_candidates

REQUIRED_FIELDS = ["title", "description", "h1", "intro", "sections"]

MIN_WORDS = 45

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


def _build_fact_guard_source(news):
    """
    Build a deterministic, factual-complete representation for the external
    Fact Guard.

    This intentionally does NOT summarize, truncate, reorder, or deduplicate
    news items. It only removes transport/formatting overhead and omits the
    URL field, which is not factual article content for the semantic audit.

    Summary is retained unless it is substantially redundant with the article
    content. This is conservative: if there is meaningful information in the
    summary that is not present in content, it stays.
    """
    compact_sources = []

    for item in news or []:
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        source = str(item.get("source", "")).strip()
        published = str(item.get("published", "")).strip()
        content = str(item.get("content", "")).strip()

        record = {
            "title": title,
            "source": source,
            "published": published,
            "content": content,
        }

        # Conservative redundancy check. Normalize whitespace and compare
        # whether the complete summary is already contained in article text.
        summary_norm = " ".join(summary.split()).casefold()
        content_norm = " ".join(content.split()).casefold()

        if summary and (
            not content_norm
            or not summary_norm
            or summary_norm not in content_norm
        ):
            record["summary"] = summary

        compact_sources.append(record)

    # No pretty-print indentation and no link field. The factual fields above
    # remain unchanged; only serialization overhead is reduced.
    return json.dumps(
        compact_sources,
        ensure_ascii=False,
        separators=(",", ":"),
    )




HEADLINE_MAX_WORDS = 10
HEADLINE_MAX_CHARS = 65

# Headline-only repair must never invoke the full universal fact-lock pipeline.
HEADLINE_REPAIR_THREADS = max(1, int(os.getenv("HEADLINE_REPAIR_THREADS", "16")))
HEADLINE_REPAIR_CTX = max(2048, int(os.getenv("HEADLINE_REPAIR_CTX", "4096")))
HEADLINE_REPAIR_BATCH = max(64, int(os.getenv("HEADLINE_REPAIR_BATCH", "256")))
HEADLINE_REPAIR_TOKENS = max(48, int(os.getenv("HEADLINE_REPAIR_TOKENS", "96")))


def _normalise_headline(text):
    text = "" if text is None else str(text)
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("|", "-")
    return " ".join(text.split()).strip().casefold()


def _headline_source_suffix(title, news):
    """Return a source suffix if the headline visibly leaks a publisher name."""
    normalized = _normalise_headline(title)
    if not normalized:
        return None

    parts = re.split(r"\s-\s", normalized)
    if len(parts) < 2:
        return None

    tail = parts[-1].strip(" .,:;\u2013\u2014")
    if not tail:
        return None

    for item in news or []:
        source = _normalise_headline(item.get("source", ""))
        if not source:
            continue
        source = source.strip(" .,:;\u2013\u2014")
        if tail == source or tail.endswith(source) or source.endswith(tail):
            return tail

    if re.search(r"\.(?:co|com|id|net|org)(?:\.[a-z]{2,3})?$", tail):
        return tail

    return None


def _headline_violations(article, trend):
    title = " ".join(str(article.get("title", "")).split()).strip()
    h1 = " ".join(str(article.get("h1", "")).split()).strip()
    source_titles = [
        str(x.get("title", "")).strip()
        for x in trend.get("news", [])
    ]

    violations = []

    if not title:
        violations.append("empty title")
    if len(title.split()) > HEADLINE_MAX_WORDS:
        violations.append(f"title exceeds {HEADLINE_MAX_WORDS} words")
    if len(title) > HEADLINE_MAX_CHARS:
        violations.append(f"title exceeds {HEADLINE_MAX_CHARS} characters")

    if not h1:
        violations.append("empty h1")
    if len(h1.split()) > HEADLINE_MAX_WORDS:
        violations.append(f"h1 exceeds {HEADLINE_MAX_WORDS} words")
    if len(h1) > HEADLINE_MAX_CHARS:
        violations.append(f"h1 exceeds {HEADLINE_MAX_CHARS} characters")

    if _headline_source_suffix(title, trend.get("news", [])):
        violations.append("publisher/source suffix in title")
    if _headline_source_suffix(h1, trend.get("news", [])):
        violations.append("publisher/source suffix in h1")

    title_norm = _normalise_headline(title)
    if title_norm and any(
        title_norm == _normalise_headline(x)
        for x in source_titles if x
    ):
        violations.append("title copies source headline")

    h1_norm = _normalise_headline(h1)
    if h1_norm and any(
        h1_norm == _normalise_headline(x)
        for x in source_titles if x
    ):
        violations.append("h1 copies source headline")

    if title_norm and h1_norm and title_norm != h1_norm:
        violations.append("title and h1 differ")

    return violations


def _repair_headline(article, trend):
    """Repair only title/H1 when the model violates the editorial headline policy."""
    current_title = str(article.get("title", "")).strip()
    current_h1 = str(article.get("h1", "")).strip()
    topic = str(trend.get("title", "")).strip()

    source_titles = [
        str(x.get("title", "")).strip()
        for x in trend.get("news", [])
        if str(x.get("title", "")).strip()
    ][:8]

    source_block = "\n".join(f"- {x}" for x in source_titles)

    repair_prompt = f"""
You are a professional {LANGUAGE} news headline editor.

You are NOT rewriting the article. You are repairing ONLY its public headline.
The article has already been generated from source-locked evidence.

MAIN TOPIC:
{topic}

CURRENT TITLE:
{current_title}

CURRENT H1:
{current_h1}

SOURCE HEADLINES FOR CONTEXT:
{source_block}

STRICT HEADLINE RULES:
- Return exactly ONE clean editorial headline in {LANGUAGE}.
- Maximum 10 words.
- Maximum 65 characters.
- Prefer 7-10 words when natural.
- Keep only the core verified entity + core verified event/development.
- Do NOT copy any source headline verbatim.
- Do NOT include a publisher, website, domain, author or source name.
- Do NOT use SEO listicle wording, keyword stuffing or filler.
- Do NOT add any fact that is not already present in the supplied topic/headline context.
- Preserve the factual status of the existing headline; shorten it rather than changing the claim.
- The result must be suitable for a professional news card.

Return ONLY valid JSON:
{{"title":""}}
"""

    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": repair_prompt}],
        options={
            "temperature": 0.0,
            "top_p": 0.85,
            "top_k": 40,
            "num_ctx": HEADLINE_REPAIR_CTX,
            "num_predict": HEADLINE_REPAIR_TOKENS,
            "num_batch": HEADLINE_REPAIR_BATCH,
            "num_thread": HEADLINE_REPAIR_THREADS,
        },
        format={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    )
    raw = response.message.content or ""
    try:
        repaired = json.loads(raw)
    except Exception as exc:
        raise Exception(f"Headline repair returned invalid JSON: {exc}") from exc
    if not isinstance(repaired, dict):
        raise Exception("Headline repair did not return a JSON object")

    new_title = " ".join(str(repaired.get("title", "")).split()).strip()
    if not new_title:
        raise Exception("Headline repair returned an empty title")

    # Deterministic local fallback: never re-call Ollama for a headline
    # that is still over the editorial limits after repair.
    if len(new_title.split()) > HEADLINE_MAX_WORDS or len(new_title) > HEADLINE_MAX_CHARS:
        words = new_title.split()
        new_title = " ".join(words[:HEADLINE_MAX_WORDS]).strip()

        if len(new_title) > HEADLINE_MAX_CHARS:
            new_title = new_title[:HEADLINE_MAX_CHARS].rsplit(" ", 1)[0].strip(" -,:;")

        if not new_title:
            raise Exception("Headline repair produced no usable headline")

    article["title"] = new_title
    article["h1"] = new_title

    violations = _headline_violations(article, trend)
    if violations:
        raise Exception("Headline repair failed: " + "; ".join(violations))

    print(f"[HEADLINE GUARD] PASS -> {new_title}")
    return article


def enforce_headline_policy(article, trend):
    violations = _headline_violations(article, trend)
    if not violations:
        print(f"[HEADLINE GUARD] PASS -> {article.get('title', '')}")
        return article

    print("[HEADLINE GUARD] REPAIR REQUIRED | " + "; ".join(violations))
    return _repair_headline(article, trend)

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

    min_words = 45
    if words < min_words:
        raise Exception(f"Article too short ({words} words; minimum {min_words})")

    return True


def generate_valid_article(prompt, fact_guard_source, reference_date, trend, max_attempts=1, prelocked_evidence=None):
    last = None

    for i in range(max_attempts):
        try:
            article = generate(prompt, evidence=prelocked_evidence)
            validate_article(article)
            article = enforce_headline_policy(article, trend)
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
                    repaired = enforce_headline_policy(repaired, trend)
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
    print(f"[TOPIC FILTER] Raw trends received: {len(trends)}")

    # One explicit reference date for the entire production run.
    # This is the date against which event state is evaluated.
    reference_date = date.today()
    print(f"[FACT GUARD] Validation reference date: {reference_date.isoformat()}")

    generated = 0
    new_keywords = []

    # ============================================================
    # TOPIC PRIORITY FILTER
    # ============================================================
    # IMPORTANT: article generation must NEVER run directly over the raw
    # trend feed. We first rank the entire trend set, then retrieve news only
    # for the small candidate pool, then generate only the final winners.
    #
    # This is deliberately outside Fact Guard. Fact Guard protects factual
    # correctness AFTER a topic has been selected; it does not decide which
    # topics deserve production capacity.
    # ============================================================
    # Keep a wider cheap candidate pool than the final article capacity.
    # This allows the evidence-usability gate to reject a top-ranked topic
    # and fall through to the next-ranked candidate instead of ending the
    # production run with an unused article slot.
    candidate_pool_size = max(
        MAX_ARTICLES_PER_RUN * 4,
        MAX_ARTICLES_PER_RUN + 3,
    )

    candidate_trends = rank_trends(
        trends,
        processed,
        limit=candidate_pool_size,
    )

    source_scored_candidates = []

    for trend in candidate_trends:
        keyword = trend["title"]

        try:
            print(
                f"\n[TOPIC FILTER] Retrieving sources for candidate: "
                f"{keyword}"
            )
            news = fetch_news(keyword)

            if len(news) < 2:
                print(
                    f"[TOPIC FILTER] DROP AFTER NEWS | {keyword} | "
                    f"only {len(news)} usable news result(s)"
                )
                continue

            trend["news"] = news
            topic_final = score_with_news(trend, news)
            trend["_topic_final"] = topic_final
            source_scored_candidates.append(trend)

            print(
                f"[TOPIC FILTER] SOURCE SCORE | "
                f"{topic_final['final_score']:.1f} | {keyword}"
            )

        except Exception as e:
            print(f"[TOPIC FILTER] ERROR: {keyword}: {e}")

    # Evidence usability is part of production candidate selection.
    # Keep the full source-scored ranking available so a top topic that has
    # unusable evidence does not consume the article slot.
    evidence_candidate_pool = select_final_candidates(
        source_scored_candidates,
        limit=len(source_scored_candidates),
    )

    for trend in evidence_candidate_pool:
        if generated >= MAX_ARTICLES_PER_RUN:
            break
        keyword = trend["title"]
        news = trend["news"]

        print(
            f"\n[GENERATION] Selected topic | "
            f"score={trend['_topic_final']['final_score']:.1f} | {keyword}"
        )

        try:
            generation_prompt = build_prompt(trend)
            fact_guard_source = _build_fact_guard_source(news)

            print(
                f"[FACT GUARD] Source prepared | "
                f"news_items={len(news)} | "
                f"source_chars={len(fact_guard_source)}"
            )

            # --------------------------------------------------------
            # EVIDENCE USABILITY GATE
            # --------------------------------------------------------
            # Use the exact same source-grounded extractor as the universal
            # fact-lock pipeline. If this candidate cannot produce at least
            # one provenance-verified fact, skip it and try the next-ranked
            # candidate instead of consuming the article slot.
            print(
                f"[TOPIC FILTER] EVIDENCE USABILITY CHECK | {keyword}"
            )

            try:
                evidence_lock = extract_evidence(generation_prompt)
                trend["_evidence_lock"] = evidence_lock
                print(
                    f"[TOPIC FILTER] EVIDENCE USABILITY PASS | "
                    f"facts={len(evidence_lock.get('facts', []))} | {keyword}"
                )
            except Exception as evidence_error:
                print(
                    f"[TOPIC FILTER] EVIDENCE USABILITY DROP | "
                    f"{keyword} | {evidence_error}"
                )
                continue

            article = generate_valid_article(
                generation_prompt,
                fact_guard_source,
                reference_date,
                trend,
                prelocked_evidence=evidence_lock,
            )

            slug = slugify(keyword)
            article["slug"] = slug

            save_article(slug, render_article(article))

            new_keywords.append(keyword)
            generated += 1

            print(f"OK -> {slug}.html")

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
