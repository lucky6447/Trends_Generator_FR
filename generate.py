print("[TrendCurrent PIPELINE] universal-fact-lock-v2.4-evidence-coverage-stable")
import re
import os
import subprocess
from datetime import date

from config import MAX_ARTICLES_PER_RUN, LANGUAGE
from rss import fetch_trends
from news import fetch_news, extract_article
from prompt import build_prompt
from ollama_client import generate, extract_evidence
from ollama import chat
from config import MODEL
from fact_guard import validate as fact_guard_validate
from fact_guard_repair import repair as fact_guard_repair
import json
import unicodedata
from html_generator import render_article, save_article
from processed import load_processed, add_processed
from index_generator import update_all
from topic_scorer import rank_trends, score_with_news, select_final_candidates

REQUIRED_FIELDS = ["title", "description", "h1", "paragraphs"]

# Article length is determined by the amount of usable verified evidence.
# There is no artificial word-count target or evidence-count-based minimum.

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


def _build_related_story_query(title):
    """Build a compact entity/event query for Google News corroboration.

    Discovery headlines can be long publisher headlines. Re-searching the
    entire headline is too restrictive and often returns no corroborating
    results. Keep the core named entities/event words instead.
    """
    text = " ".join(str(title or "").split()).strip()
    if not text:
        return ""

    # Remove common publisher suffixes when the discovery headline exposes one.
    text = re.sub(
        r"\s*[-|]\s*(?:suara\.com|wolipop|detikcom|detik\.com|"
        r"antaranews|antara|jpnn|liputan6|fimela|inews|idn times|"
        r"kapanlagi(?:\.com)?|haibunda|katadata(?:\.co\.id)?|"
        r"rri(?:\.co\.id)?|sindonews(?:\.com)?|merdeka(?:\.com)?|"
        r"kompas(?:\.com)?|grid\.id)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove obvious source-headline scaffolding that does not identify the
    # underlying story. Keep the entity/event itself.
    text = re.sub(
        r"^(?:sinopsis|daftar|rekomendasi|cara menonton|cara nonton|"
        r"jadwal tayang)\s*[:!-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    words = re.findall(r"[A-Za-z0-9À-ÿ']+", text)
    return " ".join(words[:10])


def _evidence_words(text):
    """Return conservative significant tokens for deterministic story clustering."""
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", text.casefold())
    stop = {
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "yang", "telah",
        "akan", "jadi", "sebuah", "para", "pada", "dalam", "atas", "oleh", "sebagai", "lebih",
        "the", "and", "for", "with", "from", "this", "that", "has", "have", "was", "were", "are",
        "its", "into", "after", "before", "over", "under", "about", "news", "today", "world"
    }
    return {w for w in words if len(w) >= 3 and w not in stop}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _select_evidence_story_sources(news, topic, min_sources=3, max_sources=12):
    """Deterministically rank and retain the strongest source set before Ollama.

    This changes ONLY the evidence-extractor input. The original ``news`` list
    remains untouched for Fact Guard and article generation.
    """
    items = list(news or [])
    if not items:
        return []

    # Always run the deterministic story-selection step.
    # Even exactly 12 retrieved sources can contain multiple stories
    # when the topic is ambiguous (for example, "black panther").
    topic_words = _evidence_words(topic)
    profiles = []
    for idx, item in enumerate(items):
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        words = _evidence_words(title + " " + summary)
        title_words = _evidence_words(title)
        topic_overlap = len(title_words & topic_words)
        profiles.append({
            "idx": idx,
            "words": words,
            "title_words": title_words,
            "topic_overlap": topic_overlap,
            "topic_jaccard": _jaccard(title_words, topic_words),
        })

    # Anchor on the source whose title is most directly tied to the selected topic.
    anchor = max(
        profiles,
        key=lambda x: (x["topic_overlap"], x["topic_jaccard"], -x["idx"]),
    )

    # Rank all retrieved sources by story similarity, but do not discard sources
    # solely because their wording differs from the anchor. The old hard lexical
    # threshold was a major evidence-loss point: useful corroboration, alternate
    # wording and even contradictory status reports could disappear before extraction.
    scored = []
    for prof in profiles:
        if prof["idx"] == anchor["idx"]:
            sim = 1.0
        else:
            sim = _jaccard(anchor["words"], prof["words"])
        scored.append((prof["idx"], sim, prof["topic_overlap"], prof["topic_jaccard"]))

    scored.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))

    # Preserve the strongest sources up to the cap. The evidence extractor is
    # responsible for keeping one coherent event and rejecting unrelated material.
    selected_indices = [x[0] for x in scored[:max_sources]]

    if len(selected_indices) < min_sources:
        selected_indices.extend(
            x[0] for x in scored[len(selected_indices):min_sources]
        )

    selected_indices = sorted(dict.fromkeys(selected_indices))
    selected = [items[i] for i in selected_indices]

    print(
        f"[TOPIC FILTER] Evidence story cluster | anchor={anchor['idx'] + 1} "
        f"| kept={len(selected)}/{len(items)}"
    )
    for item in selected:
        print(f"[TOPIC FILTER] Evidence source -> {str(item.get('title', '')).strip()}")

    return selected


def _build_evidence_source(news, topic=None):
    """Build source-only evidence input from one deterministic story cluster."""
    selected_news = _select_evidence_story_sources(news, topic)
    compact_sources = []

    for item in selected_news:
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        source = str(item.get("source", "")).strip()
        published = str(item.get("published", "")).strip()
        content = str(item.get("content", "")).strip()

        # Preserve usable RSS evidence when publisher extraction is empty/short.
        evidence_text = content if len(content) >= 300 else summary

        compact_sources.append({
            "title": title,
            "source": source,
            "published": published,
            "summary": summary,
            "content": evidence_text[:5000],
        })

    return "\n".join(
        [
            "SOURCE MATERIAL:",
            *[
                "\n".join(
                    [
                        f"SOURCE S{i}",
                        f"ARTICLE {i}",
                        f"Title: {item['title']}",
                        f"Source: {item['source']}",
                        f"Published: {item['published']}",
                        "",
                        "Summary:",
                        item["summary"],
                        "",
                        "Full Article:",
                        item["content"],
                        "---",
                    ]
                )
                for i, item in enumerate(compact_sources, 1)
            ],
        ]
    )

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

    if not isinstance(article["paragraphs"], list):
        raise Exception("paragraphs must be a list")

    if not article["paragraphs"]:
        raise Exception("Article must contain at least one paragraph.")

    for paragraph in article["paragraphs"]:
        if not isinstance(paragraph, str) or not paragraph.strip():
            raise Exception("Invalid paragraph")

    return True



def _enrich_evidence_for_generation(evidence, trend):
    """
    Add deterministic topic context to the already locked evidence.

    The article generator receives the evidence object, not the original trend prompt.
    Keeping the exact selected topic inside that object reduces cross-story contamination
    when Google News returns multiple related pages.
    """
    if not isinstance(evidence, dict):
        raise Exception("Evidence lock is not an object.")

    enriched = dict(evidence)
    topic = str(trend.get("title", "")).strip()

    if topic:
        enriched["selected_topic"] = topic

    # IMPORTANT: Source/publisher attribution is not part of the article narrative.
    # The model must synthesize the locked facts into a continuous news story.
    # Source names/headlines may be used internally for provenance, but must not
    # become "X reported..." / "Y said..." prose unless the attribution itself
    # is an essential verified fact (for example, an official statement).
    enriched["editorial_generation_policy"] = (
        "NARRATIVE SYNTHESIS: Write the article as an original, continuous news "
        "story from the locked verified facts. Do not organize paragraphs by source. "
        "Do not mention publisher names, websites, source headlines, or phrases such "
        "as 'reported by', 'according to [publisher]', 'X has reported', 'Y published', "
        "or similar source-digest wording. State the verified information directly. "
        "Only retain attribution when who made the statement or which official body "
        "confirmed it is itself an essential part of the verified fact. Never use a "
        "source headline to add specificity that the locked evidence does not support."
    )

    # IMPORTANT: Do not pass publisher/source headlines into article generation.
    # Headlines are discovery metadata and can contain claims that are stronger,
    # newer, or more specific than the source body. The locked evidence facts are
    # the sole factual authority for generation.
    # Evidence-density article length policy. When the locked evidence contains
    # only one verified fact, the article must stay naturally concise rather
    # than expanding a single fact into repetitive 150-200 word coverage.
    # This is a generation instruction only; it never changes or weakens the
    # underlying evidence lock or Fact Guard source.
    facts = enriched.get("facts", [])
    if isinstance(facts, list) and len(facts) == 1:
        enriched["article_length_policy"] = (
            "SINGLE-FACT EVIDENCE: Write a naturally short article. "
            "Use only the single verified fact available. Do not pad, repeat, "
            "generalize, speculate, or manufacture context to reach a word count. "
            "The article may be "
            "substantially shorter than a typical article. "
            "Completeness and factual precision take priority over length."
        )

    return enriched


def generate_valid_article(prompt, fact_guard_source, reference_date, trend, max_attempts=2, prelocked_evidence=None):
    last = None

    for i in range(max_attempts):
        try:
            generation_evidence = _enrich_evidence_for_generation(
                prelocked_evidence,
                trend,
            )
            article = generate(prompt, evidence=generation_evidence)
            validate_article(article)

            locked_facts = generation_evidence.get("facts", [])
            paragraph_text = " ".join(
                str(p) for p in article.get("paragraphs", [])
            ).strip()
            if isinstance(locked_facts, list) and len(locked_facts) >= 3:
                print(
                    f"[EVIDENCE COVERAGE] locked_facts={len(locked_facts)} "
                    f"| article_words={len(paragraph_text.split())}"
                )
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

            # A failed Fact Guard repair is terminal for this candidate.
            # Re-generating from the same evidence only repeats the expensive
            # audit/repair cycle instead of improving the underlying condition.
            if "Fact Guard blocked article; repair failed:" in str(e):
                break

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
            # Discovery candidates already come from fresh Google News RSS.
            # Re-query using the category discovery query rather than the full
            # publisher headline; the latter is often too specific and can
            # return zero results. The discovered item is retained as seed
            # evidence so the exact fresh story is not lost.
            discovery_query = trend.get("discovery_query")
            seed = None
            if discovery_query:
                # The discovery headline is already a fresh story. Use it as
                # seed evidence and make a second, entity-focused search for
                # corroboration. Never discard the seed just because Google
                # News cannot find the publisher headline again.
                seed = dict(trend)
                if not seed.get("content") and seed.get("link"):
                    try:
                        seed["content"] = extract_article(seed["link"])
                    except Exception:
                        seed["content"] = ""
                entity_query = _build_related_story_query(keyword)
                if not entity_query:
                    entity_query = keyword.split(" - ")[0].strip()

                # The discovery category is only a discovery hint. Never append it
                # to the corroboration query: doing so can misclassify a local
                # Indonesian artist as K-pop simply because the same headline was
                # returned by a K-pop discovery query. Entity/event terms are safer.
                search_query = entity_query

                print(f"[TOPIC FILTER] Related story search: {search_query}")
                news = fetch_news(search_query)
                # If the seed page extracted very little text, borrow the RSS
                # summary/content from the closest matching search result. This
                # prevents a short publisher extraction from becoming the only
                # evidence source when Google News already has the same story.
                seed_title_norm = _normalise_headline(seed.get("title", ""))
                if len(str(seed.get("content", "")).strip()) < 900 and news:
                    best = None
                    best_overlap = 0
                    seed_words = {w for w in re.findall(r"[a-z0-9]+", seed_title_norm.casefold()) if len(w) >= 4}
                    for candidate in news:
                        cand_norm = _normalise_headline(candidate.get("title", ""))
                        cand_words = {w for w in re.findall(r"[a-z0-9]+", cand_norm.casefold()) if len(w) >= 4}
                        overlap = len(seed_words & cand_words)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best = candidate
                    if best is not None and best_overlap >= 2:
                        if not seed.get("content") and best.get("content"):
                            seed["content"] = best.get("content")
                        if best.get("summary"):
                            seed["summary"] = best.get("summary")

                if seed.get("title") and not any(
                    str(x.get("title", "")).strip().casefold() == str(seed.get("title", "")).strip().casefold()
                    for x in news
                ):
                    news.insert(0, seed)
            else:
                news = fetch_news(keyword)

            if not news:
                print(f"[TOPIC FILTER] DROP AFTER NEWS | {keyword} | no usable news result(s)")
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
                evidence_source = _build_evidence_source(news, keyword)
                print(
                    f"[TOPIC FILTER] Evidence source prepared | "
                    f"source_chars={len(evidence_source)}"
                )
                evidence_lock = extract_evidence(evidence_source)
                evidence_lock = _enrich_evidence_for_generation(
                    evidence_lock,
                    trend,
                )
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

            save_article(slug, render_article(article, news=news))

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
