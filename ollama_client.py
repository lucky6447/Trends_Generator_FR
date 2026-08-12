import json
import os
import re
import time
from datetime import date
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from ollama import chat
from config import MODEL, LANGUAGE

# ============================================================
# TrendCurrent UNIVERSAL FACTUAL PIPELINE v5
# ============================================================
#
# Complete rewrite of the article verification flow.
#
# Public API:
#     generate(prompt) -> article dict
#
# Pipeline:
#     1. SOURCE EVIDENCE MAP
#     2. ARTICLE GENERATION
#     3. EVIDENCE-BOUND AUDIT
#     4. ONE REPAIR (only when necessary)
#     5. FINAL EVIDENCE-BOUND AUDIT
#
# Important design principle:
# The verifier is NOT allowed to invent a temporal requirement.
# A date is judged according to what the article actually asserts:
#
#   publication date != event date
#   announcement date = event date of the announcement
#   purchase date = event date of the purchase
#   match date = event date of the match
#   age = age, NOT a calendar date
#
# Missing an exact date is never an error by itself.
# An explicit date is an error only when the source evidence contradicts
# it or fails to establish it.
#
# This is intentionally language-independent and topic-independent.
# ============================================================

PIPELINE_VERSION = "universal-build-temporal-recovery-v15.4-stable-article-guard-v1"

NUM_THREADS = int(os.getenv("OLLAMA_NUM_THREADS", "16"))
NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "0"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

EVIDENCE_TOKENS = 1800
ARTICLE_TOKENS = 2400
AUDIT_TOKENS = 1800
REPAIR_TOKENS = 2400

MAX_REPAIR_ATTEMPTS = 1

# Article completeness guard. The downstream generator rejects articles below
# 70 words, so keep a small safety margin. This guard runs BEFORE factual audit
# when possible, and again after repair/cleanup. It never adds unsupported facts.
MIN_ARTICLE_WORDS = 82
MAX_ARTICLE_WORDS_TARGET = 150

# ============================================================
# BUILD: TEMPORAL RECOVERY v11
# ============================================================
# When the evidence-bound audit finds a TEMPORAL error, the pipeline
# first tries to resolve it from the existing evidence map. If the
# evidence is insufficient, it performs a targeted web search for the
# exact event/date instead of immediately failing the article.
#
# Web recovery is deliberately narrow: it is only used for temporal
# failures and only for date disambiguation. It does not become a
# general fact source for the article.
# ============================================================

TEMPORAL_RECOVERY_ENABLED = True
TEMPORAL_SEARCH_RESULTS = 5
TEMPORAL_RECOVERY_MAX_QUERY_CHARS = 320
TEMPORAL_RECOVERY_TIMEOUT = 5

print(f"[TrendCurrent PIPELINE] {PIPELINE_VERSION}")


# ------------------------------------------------------------
# JSON / Ollama
# ------------------------------------------------------------

def _clean_json(text):
    text = (text or "").strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]


def _compact_json(value):
    """Compact JSON for downstream prompts without changing its data.

    V1 INPUT COMPRESSION: whitespace/indentation is removed only.
    No fields, values, ordering, or factual content are changed.
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _call(prompt, temperature=0.0, num_predict=2600):
    call_start = time.perf_counter()
    prompt_chars = len(prompt or "")
    prompt_lines = (prompt or "").count("\n") + 1
    print(
        f"[TIMER] Ollama START | predict={num_predict} | temp={temperature} "
        f"| prompt_chars={prompt_chars} | prompt_lines={prompt_lines}"
    )

    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 40,
            "num_ctx": NUM_CTX,
            "num_predict": num_predict,
            "num_thread": NUM_THREADS,
            "num_gpu": NUM_GPU,
        },
        format="json",
    )

    raw = response.message.content
    cleaned = _clean_json(raw)

    call_elapsed = time.perf_counter() - call_start

    # Ollama may expose detailed server-side timings on the response object.
    # Log them when available, without changing pipeline behavior.
    timing_fields = (
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
        "total_duration",
        "prompt_eval_count",
        "eval_count",
    )
    timing = {}
    for field in timing_fields:
        value = getattr(response, field, None)
        if value is not None:
            timing[field] = value

    print(
        f"[TIMER] Ollama END   | elapsed={call_elapsed:.2f}s "
        f"| response_chars={len(raw or '')} | timing={timing}"
    )

    parse_start = time.perf_counter()
    try:
        parsed = json.loads(cleaned)
        print(f"[TIMER] JSON parse     | elapsed={time.perf_counter() - parse_start:.4f}s")
        return parsed
    except json.JSONDecodeError as exc:
        print(f"[TIMER] JSON parse FAIL | elapsed={time.perf_counter() - parse_start:.4f}s")
        raise ValueError(
            f"Ollama returned invalid JSON: {exc}. "
            f"Response prefix: {raw[:600]!r}"
        ) from exc


# ------------------------------------------------------------
# Evidence map
# ------------------------------------------------------------

def _extract_evidence(source_prompt):
    stage_start = time.perf_counter()
    print(
        f"[TIMER] Evidence extraction START | source_chars={len(source_prompt or '')}"
    )
    try:
        return _call(
        f"""
You are TrendCurrent's UNIVERSAL SOURCE EVIDENCE ENGINE.

Convert the supplied SOURCE MATERIAL into a compact, precise evidence map for
another model. DO NOT write an article. SOURCE MATERIAL is authoritative.
Use no outside knowledge.

PRIORITY:
Extract only material facts needed to write and verify the story. Prefer
central, article-relevant facts. Do not repeat the same fact in multiple forms,
do not summarize whole articles, and do not add commentary or filler.

TEMPORAL RULES:
- Publication/update dates are NOT event dates unless the source explicitly
  connects that date to the event/action.
- "published on August 11" -> publication_date.
- "updated on August 11" -> update_date.
- "announced/happened/occurred/scheduled on August 11" -> event_date.
- "last year", "Tuesday", etc. -> relative event time; preserve it as relative.
- Age (for example "at 22") is an age fact, not a missing calendar date.
- Preserve event status exactly: scheduled, ongoing, completed, announced,
  postponed, cancelled, rescheduled, historical, or unknown.
- Never infer a date from publication metadata.
- When multiple source items describe the SAME underlying event/action at
  different points in time, determine whether a later source explicitly
  updates, clarifies, confirms, reschedules, postpones, cancels, or otherwise
  supersedes the earlier information.
- A later source does NOT automatically supersede an earlier source merely
  because its publication date is later. The source must refer to the same
  event/action and provide an explicit update/clarification/status change.
- Preserve both the earlier and later claims when both are relevant, but record
  the supersession relationship in "temporal_updates".
- Prefer the latest supported state when an article describes the CURRENT
  schedule/status of the same event.
- Example: "funeral is this week" followed by a later report saying "funeral is
  today" => record that the later "today" statement supersedes the earlier
  "this week" statement for the same funeral.
- Example: "match scheduled for Tuesday" followed by "match postponed to
  Thursday" => record Thursday as the current schedule and mark Tuesday as
  superseded.
- Do not treat a publication date alone as an update to the underlying event.

CAPTURE WHEN SUPPORTED:
- people/entities, exact roles/titles, relationships
- event/action and status
- exact dates and whether exact/relative/none
- numbers, prices, percentages, rankings, scores
- quotes and speakers
- locations
- uncertainty/dispute
- source attribution
- temporal update/supersession relationships between source items

For "temporal_updates", each item should be an object with:
- "event": the same underlying event/action
- "earlier_evidence_ids": IDs of the earlier evidence items
- "later_evidence_ids": IDs of the later evidence items
- "earlier_claim": the earlier temporal/status claim
- "later_claim": the later temporal/status claim
- "supersedes": true only when the later source explicitly updates/clarifies/
  confirms/reschedules/postpones/cancels or otherwise supersedes the earlier claim
- "reason": a short source-grounded explanation

RELATIONSHIP / ENTITY RULE:
- Record only relationships explicitly supported by the source.
- Keep the two entities and the exact relationship together.
- Do NOT infer a person's club/team, job, title, relationship, nationality,
  location, employer, or affiliation from context, familiarity, or outside
  knowledge.
- If a relationship is not explicitly established, leave it out.
- Do not turn a person's mention into a role or affiliation.

Keep each "fact", "evidence", "event", and "date_evidence" SHORT and atomic.
Do not copy long source passages. Use the minimum wording needed to preserve
the factual meaning. Do not omit a material fact merely to shorten the map.

Return ONLY this JSON object, with exactly this schema:

{{
  "source_facts": [
    {{
      "id": "F1",
      "fact": "",
      "evidence": "",
      "source": "",
      "confidence": "high|medium|low"
    }}
  ],
  "events": [
    {{
      "id": "E1",
      "event": "",
      "event_type": "",
      "status": "scheduled|ongoing|completed|announced|postponed|cancelled|rescheduled|historical|unknown",
      "event_date": "",
      "date_type": "exact|relative|none",
      "date_evidence": "",
      "publication_date": "",
      "source": "",
      "confidence": "high|medium|low"
    }}
  ],
  "entities": [],
  "roles": [],
  "relationships": [],
  "quotes": [],
  "numbers": [],
  "locations": [],
  "uncertainties": [],
  "attributions": [],
  "temporal_updates": []
}}

SOURCE MATERIAL:
{source_prompt}
""",
        temperature=0.0,
        num_predict=EVIDENCE_TOKENS,
        )
    finally:
        print(f"[TIMER] Evidence extraction TOTAL | elapsed={time.perf_counter() - stage_start:.2f}s")


# ------------------------------------------------------------
# Article generation
# ------------------------------------------------------------

def _generate_article(source_prompt, evidence, repair_context=None):
    repair_instruction = ""

    if repair_context:
        repair_instruction = f"""
A previous audit found these specific unsupported or incorrect claims:

{json.dumps(repair_context, ensure_ascii=False, indent=2)}

Repair those claims while preserving everything that is supported.
If a claim cannot be supported, remove it.
Do not add facts merely to keep the article long.
"""

    return _call(
        f"""
You are TrendCurrent's UNIVERSAL ARTICLE GENERATOR.

Write the article exclusively in {LANGUAGE}.

The EVIDENCE MAP is the factual contract.
Use ONLY facts contained in the EVIDENCE MAP.
Do not use outside knowledge.

EVIDENCE MAP:
{_compact_json(evidence)}

NON-NEGOTIABLE FACTUAL RULES:

1. Never invent a fact, date, number, person, role, quote, result, location,
   motive, cause or event.

2. Publication date, update date and event date are different things.

3. Use an exact calendar date ONLY when the evidence map establishes that
   exact date for the same event/action being described.

4. If an event/action is supported but its exact date is not established,
   simply omit the exact date. This is NOT a factual defect.

5. Age is not a calendar date.
   "at age 22" is valid when the source supports the age claim.

6. Communication/reporting actions have their own event dates. If the source
   says something was ANNOUNCED, CONFIRMED, RELEASED, PUBLISHED, PROVIDED,
   REPORTED, HIGHLIGHTED, ISSUED or UPDATED on a date, that date belongs to
   that communication/reporting action. It does NOT need to be the date of
   the underlying event. Never reject a sentence that correctly dates the
   communication action just because the underlying event has another date
   or no known date.

7. Never convert a publication date into an event date.

8. Never use today's date as evidence.

9. Preserve event status:
   scheduled != completed
   completed != scheduled
   postponed/cancelled != completed
   announced != implemented unless the source says so.

10. Preserve exact roles and relationships. Never infer an entity relationship.
    A person being mentioned near a team, organization, place, or role does NOT
    establish membership, employment, affiliation, or position.

11. Preserve quote speaker and attribution.

12. Preserve exact numbers, percentages, scores, rankings and prices.

13. A poll is not an election result.

14. A proposal, intention or prediction is not a completed action.

15. Historical facts may be reported as historical facts. Do not call them
    new developments unless the source explicitly says so.

16. Do not add unsupported psychological claims, motives, causation,
    significance or dramatic adjectives.

17. Do not add "plausible" explanatory language that is not in the evidence.
    This includes claims such as "indicating", "suggesting", "showing that",
    "highlighting", "reflecting", "reinforcing", "demonstrating", "aims to",
    "perfect for", "looking forward to", "remains focused on", or similar
    interpretations unless the evidence explicitly supports that meaning.

17A. NO DERIVED CONSEQUENCES OR SIGNIFICANCE:
    Do not derive a new consequence, impact, implication, importance, readiness,
    likelihood, policy effect, market effect, strategic meaning, or other
    evaluation from a supported fact.
    Example: evidence that "Rwanda suspended imports of more than 50 alcohol
    brands" does NOT support adding "this could have significant impacts on the
    domestic market" unless the evidence explicitly states that impact.
    Example: evidence that "Italy maintains the Libya model" does NOT by itself
    support "this indicates Italy is not ready to follow the Albania model".
    A sentence may state a forecast, consequence, motive, evaluation, or
    interpretation ONLY when that exact meaning is directly present in the
    EVIDENCE MAP.
    Plausibility, common sense, or logical consequence is NOT evidence.

18. Relative-time words such as "today", "tomorrow", "yesterday", "tonight",
    "currently", "soon", "recently", "upcoming", "earlier this week" and
    "this week" are factual temporal claims. Use them only when the evidence
    explicitly supports the same temporal relationship. When in doubt, use
    the concrete event/status supported by the evidence or omit the phrase.

19. TEMPORAL EVIDENCE PRECEDENCE:
    When "temporal_updates" says a later evidence item supersedes an earlier
    item for the SAME event/action, use the later supported state for the
    current story. Do not regress to the superseded wording merely because the
    earlier wording is also factually true historically.
    Do NOT infer supersession from publication order alone.

20. Title, description, H1, intro and section headings are factual output too.
    They must contain only claims supported by the evidence map and must not
    introduce a stronger, newer, or more specific version of the story.

21. If evidence is limited, use the supported facts available rather than filler.
    The finished article should normally be at least 82 words total across title,
    description, H1, intro and sections. Do not stop after only the first one or
    two supported facts when additional material facts are available in the map.
    Never add unsupported filler merely to reach a length.

22. The conclusion must not introduce a new fact.

Return ONLY:

{{
  "title": "",
  "description": "",
  "h1": "",
  "intro": "",
  "sections": [
    {{"title": "", "text": ""}}
  ]
}}

{repair_instruction}
""",
        temperature=0.08,
        num_predict=ARTICLE_TOKENS,
    )


# ------------------------------------------------------------
# Article generation robustness
# ------------------------------------------------------------

def _generate_article_with_schema_retry(source_prompt, evidence):
    """
    Generate an article and, if the model returns a malformed article object,
    make one tightly scoped retry.

    This protects the pipeline from transient Ollama JSON/schema drift without
    changing the factual pipeline or adding a general regeneration loop.
    """
    article = _generate_article(source_prompt, evidence)

    if _schema_ok(article):
        return article

    print("[PIPELINE] Article schema invalid; one strict schema retry...")

    retry = _call(
        f"""
You are TrendCurrent's ARTICLE JSON REPAIR GENERATOR.

The previous article-generation response did not match the required JSON
schema. Generate the article again.

FACTUAL RULE:
Use ONLY the EVIDENCE MAP. Do not add facts or outside knowledge.

REQUIRED JSON OBJECT:
{{
  "title": "string",
  "description": "string",
  "h1": "string",
  "intro": "string",
  "sections": [
    {{
      "title": "string",
      "text": "string"
    }}
  ]
}}

REQUIREMENTS:
- All five top-level keys are mandatory.
- "sections" must be a JSON array.
- Use 1 to 5 section objects.
- Every section object must contain "title" and "text".
- Every value must be valid JSON.
- Do not use markdown code fences.
- Return ONLY the JSON object.
- Keep the article in {LANGUAGE}.
- If evidence is limited, use the supported facts available rather than stopping prematurely.
- Aim for at least 82 words total when the evidence map contains enough material.
- Do not derive consequences, impacts, significance, readiness, likelihood,
  policy effects, market effects, or other interpretations from a base fact
  unless that exact meaning is explicitly present in the EVIDENCE MAP.

EVIDENCE MAP:
{_compact_json(evidence)}
""",
        temperature=0.03,
        num_predict=ARTICLE_TOKENS,
    )

    return retry


# ------------------------------------------------------------
# Article completeness guard
# ------------------------------------------------------------

def _expand_short_article(article, evidence, reason="initial"):
    """Expand only undersized articles using already-supported evidence.

    This is deliberately a targeted completeness pass, not a regeneration loop.
    It runs before an article can be rejected by the downstream 70-word validator.
    The model may only add material facts already present in the evidence map and
    must not invent filler, implications, motives, or interpretations.
    """
    if not _schema_ok(article):
        return article

    words = _article_word_count(article)
    if words >= MIN_ARTICLE_WORDS:
        return article

    print(
        f"[PIPELINE] Article too short ({words} words); "
        f"targeted evidence-only expansion ({reason})..."
    )

    expanded = _call(
        f"""
You are TrendCurrent's ARTICLE COMPLETENESS REPAIR ENGINE.

The supplied article is factually constrained and is too short for publication.
Expand it ONLY by adding directly supported material facts already present in the
EVIDENCE MAP. Do not rewrite supported facts unnecessarily.

NON-NEGOTIABLE RULES:
- Use ONLY the EVIDENCE MAP.
- Do NOT add outside knowledge.
- Do NOT add filler or generic commentary.
- Do NOT add consequences, significance, motives, predictions, emotions,
  implications, or interpretations unless explicitly present in the evidence.
- Preserve dates, status, roles, relationships, numbers, quotes and attribution.
- Do not invent an exact date.
- Keep the original article language.
- Keep the same JSON schema.
- Aim for {MIN_ARTICLE_WORDS}-{MAX_ARTICLE_WORDS_TARGET} total words.
- Add only as many supported details as needed.
- Do not remove supported material merely to rewrite it.

EVIDENCE MAP:
{_compact_json(evidence)}

CURRENT ARTICLE:
{_compact_json(article)}

Return ONLY the repaired article JSON.
""",
        temperature=0.02,
        num_predict=ARTICLE_TOKENS,
    )

    if not _schema_ok(expanded):
        print("[PIPELINE] Short-article expansion returned invalid schema; keeping original.")
        return article

    expanded_words = _article_word_count(expanded)
    print(f"[PIPELINE] Article completeness result: {expanded_words} words")

    # Never replace a usable article with a shorter expansion.
    if expanded_words < words:
        print("[PIPELINE] Expansion was shorter; keeping original article.")
        return article

    return expanded


# ------------------------------------------------------------
# Evidence-bound audit
# ------------------------------------------------------------

def _audit(source_prompt, article, evidence):
    today = date.today().isoformat()

    return _call(
        f"""
You are TrendCurrent's UNIVERSAL EVIDENCE-BOUND FACT AUDITOR.

Your task is to audit the ARTICLE against ONLY the EVIDENCE MAP.

Do not use outside knowledge.
Do not speculate.
Do not demand evidence for something the article does not claim.

CURRENT SYSTEM DATE: {today}

IMPORTANT TEMPORAL LOGIC:

A. PUBLICATION / REPORTING DATE
A date in a source's publication/byline metadata proves the publication date.
If the article describes the source's own communication action with wording
such as "reported on", "published on", "released on", "posted on", or
"the report appeared on", the source publication_date is sufficient evidence
for THAT reporting/publication action. Do NOT demand a separate event_date for
that communication action unless the article instead claims that the underlying
real-world event itself happened on that date.

B. EVENT DATE
If the source explicitly says an action/event was announced, happened,
occurred, began, ended, was scheduled, etc. on a date, that date is the
event date for THAT event.

C. AGE
"at age 22", "aged 22", "when she was 22" etc. are age claims.
They do NOT require an exact calendar date.

D. DATE-FREE EVENT
"visited Glasgow", "bought a home", "spoke about the decision" etc. can be
fully supported without an exact calendar date if the source supports the
action.

E. EXACT DATE
Only flag an exact date when:
- the article asserts an exact calendar date for an event/action, AND
- the evidence map does not establish that date for the same event/action, OR
- the evidence establishes a different date.

IMPORTANT COMMUNICATION-DATE RULE:
If the claim is about a source reporting/publication action (for example,
"The New York Times reported on August 6"), and the evidence map has
publication_date = August 6 for that source, PASS the claim. The publication
date establishes the date of the source's reporting/publication action.
Do not incorrectly treat it as a claim that the underlying subject event
occurred on August 6.

A relative temporal phrase is NOT an exact-date claim, but it IS still a
factual temporal claim. Examples include "before the 2026 WNBA All-Star Game",
"during the tournament", "earlier this week", "recently", "after the
announcement", or "before the match". Do not demand a calendar date merely
because the phrase is relative. Instead, verify that the source/evidence
supports the SAME temporal relationship.

SPECIAL RELATIVE-TIME GUARD:
Treat "today", "tomorrow", "yesterday", "tonight", "currently", "soon",
"recently", "upcoming", "earlier this week", "this week", "this month" and
similar relative expressions as claims. They are errors when the evidence
does not establish that temporal relationship. Do not use the current system
date as a substitute for evidence. If the article can be correct without the
relative expression, flag the unsupported expression and recommend removing
only that expression.

F. ANNOUNCEMENT
"The ruling was announced on August 11" is supported if the source says the
ruling was announced on August 11. The fact that the source was published on
August 11 does NOT make this unsupported if the source text itself reports
the announcement date.

G. NO FALSE POSITIVE
Never return an error whose only reason is:
"the source does not provide an exact event date"
when the article does not assert an exact calendar date.
Never turn a supported relative time relationship into an exact-date requirement.

H. TEMPORAL STATUS
Do not transfer status/date from one event to another related event.

H2. TEMPORAL EVIDENCE PRECEDENCE / SUPERSESSION
When the evidence map contains a "temporal_updates" item with
"supersedes": true, and the article describes the CURRENT state/schedule/status
of that same event, the later evidence item is authoritative for that current
claim. Do not flag the article merely because an earlier evidence item contains
an older temporal state.
If the article explicitly reports the earlier state as historical (for example,
"it was initially reported as this week"), that earlier state may remain valid.
Do not assume that a later publication date alone supersedes an earlier claim;
the evidence map must establish the same-event update relationship.

I. HISTORICAL FACT
A source-supported historical fact remains factual even when it is not a new
development.

J. ENTITY / RELATIONSHIP INTEGRITY
Audit every person/entity relationship independently. A claim such as
"Person X plays for Team Y", "Person X is the coach of Team Y", "Person X is
engaged to Person Y", or "Person X works for Organization Y" requires direct
support for that exact relationship. Co-occurrence in the source is not enough.
Do not import a relationship from general knowledge.

K. UNSUPPORTED INFERENCE / EMBELLISHMENT
Flag claims that add motive, psychology, causation, significance, evaluation,
recommendation, certainty, severity, or narrative meaning beyond the evidence.
Examples include "indicating", "suggesting", "showing that", "highlighting",
"reflecting", "reinforcing", "perfect for", "aims to", "looking forward to",
"remains focused on", "caused a major crisis", "serious environmental
disaster", or similar stronger interpretations unless explicitly supported.

K2. SENTENCE-LEVEL DERIVATION TEST
For EVERY article sentence that goes beyond a direct event/person/number/date/
quote/location statement, ask:
1. What exact evidence item supports the added meaning?
2. Does that evidence explicitly state the consequence, impact, significance,
   interpretation, readiness, likelihood, or evaluation?
3. Or is the article merely deriving it from another fact?

If the sentence is a derived consequence/interpretation and the EVIDENCE MAP
does not explicitly state that consequence/interpretation, flag it as
INFERENCE. Do NOT accept logical plausibility as support.

Examples of claims that MUST be flagged when the evidence contains only the
preceding base fact:
- "This is an important phase that could lead to significant policy changes"
  when evidence only says that negotiations are taking place.
- "This indicates that Italy is not ready to follow the Albania model" when
  evidence only says Italy maintains the Libya model.
- "This decision could have significant impacts on the domestic market or
  international trade" when evidence only says imports were suspended.

These examples illustrate the reasoning error, not a language-specific
keyword rule. Apply the same test to equivalent wording in ANY article
language. If the evidence explicitly states the consequence/interpretation,
PASS it.

L. OUTPUT-FIELD CONSISTENCY
Audit title, description, H1, intro, section headings and section text as
separate factual surfaces. No field may introduce a material claim absent
from the evidence. Title/description must not be stronger or more specific
than the evidence merely because they are metadata.

M. INTERNAL CONSISTENCY
Title, H1, description, intro and body must not contradict one another on
person, team, event, date, status, number, outcome, or attribution. If two
fields describe the same event differently, flag the contradiction even when
the evidence itself is correct.

AUDIT EVERY MATERIAL CLAIM.

Before returning any TEMPORAL error, inspect "temporal_updates". If it explicitly
marks a later evidence item as superseding an earlier item for the same event,
judge a CURRENT article claim against the later state. Do not resurrect the
superseded earlier state as the required correction.

A claim is an ERROR only if:
1. the source/evidence contradicts it, or
2. the article asserts something not supported by the supplied source/evidence.

If a claim is supported, PASS it even if you would have written it differently.

For every error provide an evidence-based reason.
Do not invent corrections.

Return ONLY:

{{
  "passed": true,
  "errors": []
}}

OR:

{{
  "passed": false,
  "errors": [
    {{
      "severity": "CRITICAL|MAJOR|MINOR",
      "category": "FACTUAL|TEMPORAL|ROLE|RELATIONSHIP|ENTITY|EVENT|SPORTS|NUMBER|QUOTE|ATTRIBUTION|LOCATION|CAUSATION|INFERENCE|CONSISTENCY|OTHER",
      "claim": "",
      "reason": "",
      "evidence_ids": ["F1", "E1"]
    }}
  ]
}}

STRICT RULE:
If there is no real factual contradiction or unsupported material claim,
return passed=true.

EVIDENCE MAP:
{_compact_json(evidence)}

ARTICLE:
{_compact_json(article)}
""",
        temperature=0.0,
        num_predict=AUDIT_TOKENS,
    )


# ------------------------------------------------------------
# Repair
# ------------------------------------------------------------

def _repair(source_prompt, article, evidence, audit):
    return _call(
        f"""
You are TrendCurrent's UNIVERSAL FACTUAL REPAIR ENGINE.

Repair the ARTICLE using ONLY the EVIDENCE MAP and AUDIT.

AUDIT ERRORS:
{_compact_json(audit)}

Rules:
1. Correct only the listed factual errors.
2. Preserve all supported material.
3. If a listed claim is unsupported, remove it.
4. Never invent a replacement fact.
5. Never invent an exact date.
6. If an action is supported but its exact date is not, remove only the
   unsupported date and keep the supported action.
7. An age claim does not need a calendar date.
8. An announcement date is an event date when the source explicitly connects
   the announcement to that date.
9. Preserve roles, relationships, status, numbers, quotes and attribution.
10. Remove unsupported entity relationships instead of replacing them with
    a plausible relationship.

11. INFERENCE ERRORS ARE DELETE-FIRST:
    For every AUDIT error with category INFERENCE:
    - delete the unsupported claim;
    - if the whole sentence is the inference, delete the whole sentence;
    - if only one clause is unsupported, delete only that clause;
    - never paraphrase, soften, generalize, summarize, or reinterpret it;
    - never replace it with equivalent wording such as "underscores",
      "highlights", "shows", "reflects", "demonstrates", "suggests",
      "indicates", "points to", "signals", or "reveals";
    - only use a replacement if the EVIDENCE MAP contains a directly supported
      factual claim that is materially different from the rejected inference;
      otherwise DELETE it.
    If a sentence derives a consequence from a supported fact but the evidence
    does not explicitly state that consequence, remove the derived clause or
    sentence. Do not replace it with another speculative consequence.

12. Remove unsupported relative-time words such as "today", "tomorrow",
    "yesterday", "currently", "soon", or "recently" when the evidence does not
    establish the same temporal relationship.
13. Keep title, description and H1 evidence-bound; repair them if they
    introduce unsupported facts or contradict the body.
14. Do not add filler after deleting unsupported material.
    If removing an unsupported inference makes the article unusually short,
    the repair may expand it ONLY by adding additional directly supported,
    material facts already present in the EVIDENCE MAP. It must not invent,
    generalize, speculate, or add generic explanatory prose merely to reach a
    word-count target.
15. When temporal evidence contains conflicting dates or amounts, first check
    "temporal_updates". If a later evidence item explicitly supersedes an earlier
    item for the SAME event/action, repair the article to the later supported
    state rather than reverting to the earlier state. Preserve the earlier state
    only when the article explicitly reports it as historical/initial information.
    If no supersession relationship is established, remove the unsupported
    comparison/sequence rather than inventing an intermediate value, date, or
    "next day" change.
16. Keep the original article language.
17. Return the same JSON schema as the original article.

EVIDENCE MAP:
{_compact_json(evidence)}

ARTICLE:
{_compact_json(article)}

AUDIT:
{_compact_json(audit)}

Return ONLY the repaired article JSON.
""",
        temperature=0.03,
        num_predict=REPAIR_TOKENS,
    )


# ------------------------------------------------------------
# Temporal recovery
# ------------------------------------------------------------

def _is_temporal_error(audit):
    if not isinstance(audit, dict):
        return False
    for error in audit.get("errors", []):
        if str(error.get("category", "")).upper() == "TEMPORAL":
            return True
    return False


def _extract_temporal_errors(audit):
    return [
        e for e in audit.get("errors", [])
        if str(e.get("category", "")).upper() == "TEMPORAL"
    ]


def _temporal_recovery_needed(evidence, audit):
    """
    External recovery is ONLY for a real explicit date/value conflict.
    Missing calendar dates are not recoverable errors when the article itself
    uses a date-free or relative time expression.
    """
    errors = _extract_temporal_errors(audit)
    if not errors:
        return False

    exact_date_re = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*20\d{2})?\b|"
        r"\b20\d{2}-\d{1,2}-\d{1,2}\b",
        re.I,
    )

    for error in errors:
        reason = str(error.get("reason", "")).lower()
        claim = str(error.get("claim", ""))

        # Web recovery is justified only when there is an explicit exact-date
        # or exact-value dispute, not merely an absent date.
        conflict_markers = (
            "contradiction", "conflicting", "conflict", "different date",
            "different amount", "wrong date", "incorrect date", "wrong amount",
            "incorrect amount", "cannot confirm the exact value",
        )
        if any(marker in reason for marker in conflict_markers):
            return True

        # Exact calendar date explicitly asserted in the article and not
        # established by the evidence can justify a narrow recovery.
        if exact_date_re.search(claim) and any(
            marker in reason for marker in (
                "does not establish", "not establish", "does not confirm",
                "fails to establish", "cannot confirm", "unsupported"
            )
        ):
            return True

    return False


def _temporal_recovery_query(source_prompt, audit):
    """
    Build up to three focused queries:
      1) exact audited claim
      2) claim + audit reason
      3) claim + source context
    """
    errors = _extract_temporal_errors(audit)

    claims = " ".join(str(e.get("claim", "")).strip() for e in errors)
    reasons = " ".join(str(e.get("reason", "")).strip() for e in errors)

    claim_q = re.sub(r"\s+", " ", claims).strip()
    reason_q = re.sub(r"\s+", " ", reasons).strip()
    source_q = re.sub(r"\s+", " ", source_prompt).strip()

    # Keep the claim intact as the primary search because it contains the
    # concrete date/value the auditor disputed.
    q1 = claim_q[:220]
    q2 = f"{claim_q} {reason_q}"[:320]
    q3 = f"{claim_q} {source_q}"[:320]

    return [q for q in (q1, q2, q3) if q]


def _web_search_temporal(queries):
    """
    Narrow web fallback for temporal disambiguation.

    Multiple small queries are used because a single long Google query can
    obscure the exact event/date relationship. Results remain scoped to date
    resolution and are never treated as general article evidence.
    """
    if isinstance(queries, str):
        queries = [queries]

    queries = [q.strip() for q in queries if q and q.strip()]
    queries = list(dict.fromkeys(queries))[:3]

    if not queries:
        return []

    results = []

    for query in queries:
        url = "https://www.google.com/search?q=" + quote_plus(query)
        req = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/151 Safari/537.36"
                )
            },
        )

        try:
            with urlopen(req, timeout=TEMPORAL_RECOVERY_TIMEOUT) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            print(f"[TEMPORAL RECOVERY] Search unavailable for query: {exc}")
            continue

        html = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
        html = re.sub(r"<style.*?</style>", " ", html, flags=re.I | re.S)
        visible = re.sub(r"<[^>]+>", " ", html)
        visible = re.sub(r"\s+", " ", visible).strip()

        results.append({
            "query": query,
            "search_text": visible[:10000],
            "source": "Google Search",
        })

    return results




def _resolve_temporal_with_web(source_prompt, evidence, article, audit):
    """
    Resolve ONLY the temporal ambiguity.

    The resolver receives the original evidence, article, audit and targeted
    search material. It returns an evidence update that can be appended to
    the evidence map before the normal repair pass.
    """
    queries = _temporal_recovery_query(source_prompt, audit)
    results = _web_search_temporal(queries)

    if not results:
        return evidence

    recovery = _call(
        f"""
You are TrendCurrent's TEMPORAL DATE RECOVERY ENGINE.

Your ONLY job is to resolve the date ambiguity described by the audit.
Do not fact-check or rewrite the whole article.
Do not introduce unrelated facts.

IMPORTANT:
- Distinguish EVENT_DATE from PUBLICATION_DATE.
- A page publication date is NOT an event date unless the page explicitly
  states that the event/action happened on that date.
- "reported on X" means the reporting/publication action may have date X,
  while the underlying event may have happened on another date.
- Prefer the original supplied evidence when it already establishes the date.
- Web search is a fallback for date disambiguation, not a replacement
  evidence source for the article.
- If sources disagree, do NOT average, interpolate, or guess.
- Determine whether each candidate amount/date belongs to the same drawing,
  jackpot update, publication, announcement, or underlying event.
- A jackpot amount on one date must not be transferred to another drawing/date.
- If the exact value/date cannot be established with high confidence, return
  resolved=false.
- When the evidence supports only a sequence of values (for example $856M,
  then $905M, then $975M), do not invent an intermediate "next day" increase.
- Never guess.

ORIGINAL SOURCE MATERIAL:
{source_prompt}

EXISTING EVIDENCE MAP:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

ARTICLE:
{json.dumps(article, ensure_ascii=False, indent=2)}

TEMPORAL AUDIT:
{json.dumps(audit, ensure_ascii=False, indent=2)}

TARGETED WEB SEARCH:
{json.dumps(results, ensure_ascii=False, indent=2)}

Return ONLY:
{{
  "resolved": true,
  "event_date": "",
  "publication_date": "",
  "date_type": "event|publication|announcement|reporting|update|unknown",
  "event": "",
  "explanation": "",
  "source": ""
}}

or:

{{
  "resolved": false,
  "event_date": "",
  "publication_date": "",
  "date_type": "unknown",
  "event": "",
  "explanation": "",
  "source": ""
}}
""",
        temperature=0.0,
        num_predict=1200,
    )

    if not isinstance(recovery, dict) or not recovery.get("resolved"):
        print("[TEMPORAL RECOVERY] Could not resolve date.")
        return evidence

    # Add a tightly scoped recovery fact. It is explicitly marked as recovery
    # evidence so the normal repair/audit can see exactly where it came from.
    recovered = dict(evidence)

    source_facts = list(recovered.get("source_facts", []))
    source_facts.append({
        "id": f"TR{len(source_facts) + 1}",
        "fact": recovery.get("event", ""),
        "evidence": recovery.get("explanation", ""),
        "source": recovery.get("source", "Targeted temporal web recovery"),
        "confidence": "medium",
    })
    recovered["source_facts"] = source_facts

    events = list(recovered.get("events", []))
    events.append({
        "id": f"TR-E{len(events) + 1}",
        "event": recovery.get("event", ""),
        "event_type": "temporal_recovery",
        "status": "unknown",
        "event_date": recovery.get("event_date", ""),
        "date_type": "exact" if recovery.get("event_date") else "none",
        "date_evidence": recovery.get("explanation", ""),
        "publication_date": recovery.get("publication_date", ""),
        "source": recovery.get("source", "Targeted temporal web recovery"),
        "confidence": "medium",
    })
    recovered["events"] = events
    recovered = _normalize_temporal_updates(recovered)

    print(
        "[TEMPORAL RECOVERY] Resolved:",
        json.dumps(recovery, ensure_ascii=False)
    )

    return recovered


# ------------------------------------------------------------
# Schema validation
# ------------------------------------------------------------

def _schema_ok(article):
    if not isinstance(article, dict):
        return False

    required = ("title", "description", "h1", "intro", "sections")

    if any(field not in article for field in required):
        return False

    if not isinstance(article["title"], str):
        return False
    if not isinstance(article["description"], str):
        return False
    if not isinstance(article["h1"], str):
        return False
    if not isinstance(article["intro"], str):
        return False
    if not isinstance(article["sections"], list):
        return False

    if not (1 <= len(article["sections"]) <= 5):
        return False

    for section in article["sections"]:
        if not isinstance(section, dict):
            return False
        if not isinstance(section.get("title"), str):
            return False
        if not isinstance(section.get("text"), str):
            return False

    return True


def _article_word_count(article):
    parts = [
        article.get("title", ""),
        article.get("description", ""),
        article.get("h1", ""),
        article.get("intro", ""),
    ]

    for section in article.get("sections", []):
        parts.append(section.get("title", ""))
        parts.append(section.get("text", ""))

    return len(" ".join(parts).split())


def _normalize_temporal_updates(evidence):
    """Normalize explicit same-event temporal supersession records only."""
    if not isinstance(evidence, dict):
        return evidence
    updates = evidence.get("temporal_updates", [])
    if not isinstance(updates, list):
        evidence["temporal_updates"] = []
        return evidence
    normalized = []
    for item in updates:
        if not isinstance(item, dict):
            continue
        earlier_ids = item.get("earlier_evidence_ids", [])
        later_ids = item.get("later_evidence_ids", [])
        if isinstance(earlier_ids, str):
            earlier_ids = [earlier_ids]
        if isinstance(later_ids, str):
            later_ids = [later_ids]
        earlier_ids = [str(x).strip() for x in earlier_ids if str(x).strip()]
        later_ids = [str(x).strip() for x in later_ids if str(x).strip()]
        event = str(item.get("event", "")).strip()
        if not event or not earlier_ids or not later_ids:
            continue
        normalized.append({
            "event": event,
            "earlier_evidence_ids": earlier_ids,
            "later_evidence_ids": later_ids,
            "earlier_claim": str(item.get("earlier_claim", "")).strip(),
            "later_claim": str(item.get("later_claim", "")).strip(),
            "supersedes": bool(item.get("supersedes", False)),
            "reason": str(item.get("reason", "")).strip(),
        })
    evidence["temporal_updates"] = normalized
    return evidence


def _normalize_audit(audit, evidence=None):
    """
    Deterministic normalization of the LLM audit.

    Temporal validation is deliberately source-aware:
    - a publication/reporting date validates the source's own reporting action;
    - it never validates the underlying real-world event;
    - dates from one source can never validate a claim about another source;
    - relative time expressions do not require calendar dates.
    """
    if not isinstance(audit, dict):
        raise ValueError("Auditor returned a non-object JSON response.")

    evidence = evidence or {}
    audit.pop("corrections", None)
    errors = audit.get("errors", [])
    if not isinstance(errors, list):
        raise ValueError("Auditor returned invalid errors field.")

    clean_errors = []
    for error in errors:
        if not isinstance(error, dict):
            continue
        claim = str(error.get("claim", "")).strip()
        reason = str(error.get("reason", "")).strip()
        if not claim or not reason:
            continue
        clean_errors.append({
            "severity": str(error.get("severity", "MAJOR")).upper(),
            "category": str(error.get("category", "OTHER")).upper(),
            "claim": claim,
            "reason": reason,
            "evidence_ids": error.get("evidence_ids", []),
        })

    exact_date_re = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*20\d{2})?\b|"
        r"\b20\d{2}-\d{1,2}-\d{1,2}\b",
        re.I,
    )
    temporal_date_missing_markers = (
        "does not provide an exact date", "does not provide an exact event date",
        "does not provide a specific date", "does not provide a calendar date",
        "does not establish an exact date", "does not establish the exact date",
        "fails to establish an exact date", "cannot confirm the exact date",
        "no exact date", "missing exact date",
    )
    communication_verbs = (
        "reported on", "reported", "published on", "published", "released on",
        "released", "posted on", "posted", "the report appeared on",
        "report appeared on", "article appeared on"
    )
    month_names = (
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December"
    )

    def _date_variants(value):
        raw = str(value or "").strip()
        if not raw:
            return set()
        out = {raw.lower()}
        m = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|"
            r"November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(20\d{2}))?",
            raw, re.I,
        )
        if m and m.group(3):
            month, day, year = m.group(1), int(m.group(2)), m.group(3)
            out.add(f"{month} {day}, {year}".lower())
            out.add(f"{year}-{month_names.index(month.capitalize())+1:02d}-{day:02d}".lower())
        return out

    def _norm_source(text):
        text = str(text or "").lower()
        text = re.sub(r"https?://", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # Explicit aliases cover common publisher names/domains. Generic token
    # matching below handles other sources without topic-specific patches.
    SOURCE_ALIASES = {
        "new york times": {"new york times", "nytimes", "nytimes com"},
        "new york post": {"new york post", "nypost", "nypost com"},
        "sny": {"sny", "sportsnet new york", "sny tv", "sny tv com"},
        "espn": {"espn", "espn com"},
        "sky sports": {"sky sports", "skysports", "skysports com"},
        "bbc": {"bbc", "bbc com"},
        "reuters": {"reuters", "reuters com"},
        "associated press": {"associated press", "ap news", "apnews"},
    }

    def _source_forms(text):
        n = _norm_source(text)
        forms = {n, n.replace(" ", "")}
        for canonical, aliases in SOURCE_ALIASES.items():
            if any(a in n or a.replace(" ", "") in n.replace(" ", "") for a in aliases):
                forms.add(canonical)
                forms.update(aliases)
        return {x for x in forms if x}

    def _source_matches(claim_source, evidence_source):
        a = _source_forms(claim_source)
        b = _source_forms(evidence_source)
        if not a or not b:
            return False
        if a & b:
            return True
        # Known publishers must NEVER fall through to generic token matching.
        # This prevents "New York Times" from matching "New York Post"
        # merely because both contain "new" and "york".
        known_canonical = set(SOURCE_ALIASES.keys())
        claim_norm = _norm_source(claim_source)
        evidence_norm = _norm_source(evidence_source)
        claim_known = [c for c in known_canonical if c in _source_forms(claim_source)]
        evidence_known = [c for c in known_canonical if c in _source_forms(evidence_source)]
        if claim_known or evidence_known:
            return bool(set(claim_known) & set(evidence_known))

        # Generic fallback is only for publishers not covered by the explicit
        # alias table. Require a distinctive shared token; never use common
        # geographic/organizational words such as "new" or "york" alone.
        stop_tokens = {"new", "york", "the", "news", "post", "times", "sports", "network"}
        ta = {t for t in claim_norm.split() if len(t) >= 4 and t not in stop_tokens}
        tb = {t for t in evidence_norm.split() if len(t) >= 4 and t not in stop_tokens}
        return len(ta & tb) >= 2 or any(len(t) >= 6 and t in tb for t in ta)

    # Build publication records WITH source identity. Never flatten dates from
    # different publishers into one global date set.
    publication_records = []
    for fact in evidence.get("source_facts", []):
        src = str(fact.get("source", "")).strip()
        ev = str(fact.get("evidence", ""))
        if not src:
            continue
        if "publication" in ev.lower() or "published" in ev.lower() or "reported" in ev.lower():
            for d in re.findall(
                r"(?:January|February|March|April|May|June|July|August|September|October|"
                r"November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*20\d{2})?",
                ev, re.I,
            ):
                publication_records.append((src, _date_variants(d)))

    for event in evidence.get("events", []):
        src = str(event.get("source", "")).strip()
        d = str(event.get("publication_date", "")).strip()
        if src and d:
            publication_records.append((src, _date_variants(d)))

    def _claim_sources(claim):
        """Extract the publisher phrase immediately before a communication verb."""
        out = []
        for verb in communication_verbs:
            m = re.search(r"(?:the\s+)?(.{2,100}?)\s+" + re.escape(verb) + r"\b", claim, re.I)
            if m:
                candidate = m.group(1).strip(" ,;:-")
                candidate = re.sub(r"^(both|also|and)\s+", "", candidate, flags=re.I)
                # For "SNY and the New York Post reported", split publishers.
                parts = re.split(r"\s+(?:and|&)\s+", candidate, flags=re.I)
                out.extend(p.strip() for p in parts if p.strip())
        return out

    filtered_errors = []
    for err in clean_errors:
        if err["category"] == "TEMPORAL":
            reason_l = err["reason"].lower()
            claim_l = err["claim"].lower()
            claim_dates = re.findall(
                r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*20\d{2})?",
                err["claim"], re.I,
            )
            communication_claim = any(v in claim_l for v in communication_verbs)

            # HARD publication/reporting rule: if the claim is explicitly about
            # a publisher's reporting action, only that publisher's publication
            # date may validate it. Underlying event dates remain separate.
            if communication_claim and claim_dates and publication_records:
                claim_sources = _claim_sources(err["claim"])
                matched = False
                for claim_source in claim_sources:
                    for pub_source, pub_dates in publication_records:
                        if not _source_matches(claim_source, pub_source):
                            continue
                        if any(any(cd.lower() in pd for pd in pub_dates) for cd in claim_dates):
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    # The only valid objection in this situation would be a
                    # genuine contradiction (e.g. evidence says the source was
                    # published on a different date). A mere absence of an
                    # underlying event date is NOT an error.
                    contradiction_words = (
                        "contradict", "conflict", "different date", "wrong date",
                        "incorrect date", "impossible date"
                    )
                    if not any(w in reason_l for w in contradiction_words):
                        continue

            only_missing_exact_date = any(marker in reason_l for marker in temporal_date_missing_markers)
            has_exact_date_in_claim = bool(exact_date_re.search(err["claim"]))
            contradiction_words = (
                "contradict", "conflict", "different date", "wrong date",
                "incorrect date", "impossible date"
            )
            has_real_conflict = any(w in reason_l for w in contradiction_words)
            if only_missing_exact_date and not has_exact_date_in_claim and not has_real_conflict:
                continue

        filtered_errors.append(err)

    if not filtered_errors:
        return {"passed": True, "errors": []}
    return {"passed": False, "errors": filtered_errors}


# ------------------------------------------------------------
# Final inference cleanup guard
# ------------------------------------------------------------

def _final_inference_cleanup(article, evidence, audit):
    """Remove inference claims discovered only by the final audit.

    This is intentionally a narrow post-repair guard. It is NOT a general
    regeneration loop and it is invoked only when the final audit contains
    INFERENCE errors and no other error category. The cleanup receives the
    exact final-audit claims and is forbidden to paraphrase or replace them.
    """
    inference_errors = [
        e for e in (audit or {}).get("errors", [])
        if str(e.get("category", "")).upper() == "INFERENCE"
    ]
    if not inference_errors:
        return article

    return _call(
        f"""
You are TrendCurrent's FINAL INFERENCE CLEANUP ENGINE.

The article has already passed the normal repair stage, but the FINAL AUDIT
found unsupported INFERENCE claims that were missed earlier.

Your ONLY task is to REMOVE those unsupported inference claims.

NON-NEGOTIABLE:
- Use ONLY the EVIDENCE MAP.
- Remove every claim listed in the INFERENCE ERRORS.
- If an error claim is a complete sentence, DELETE that sentence.
- If it is a clause inside a supported sentence, DELETE ONLY that unsupported
  clause and preserve the supported factual material.
- NEVER paraphrase, soften, generalize, summarize, reinterpret, or replace an
  unsupported inference.
- NEVER replace it with equivalent wording such as suggests, indicates,
  highlights, underscores, reflects, demonstrates, shows, signals, or reveals.
- Do not add any new fact, explanation, consequence, significance, opinion,
  motive, or filler.
- Preserve all supported facts, dates, roles, relationships, numbers,
  attribution, and the original article language.
- Keep title/description/H1 valid and factual. If an unsupported inference is
  present there, remove it rather than replacing it with another claim.
- Return the SAME JSON schema. Return ONLY JSON.

INFERENCE ERRORS:
{_compact_json(inference_errors)}

EVIDENCE MAP:
{_compact_json(evidence)}

ARTICLE:
{_compact_json(article)}
""",
        temperature=0.0,
        num_predict=REPAIR_TOKENS,
    )


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def generate(prompt, retries=0):
    """
    Universal TrendCurrent generation entry point.

    Normal path:
        evidence -> article -> audit

    If the audit finds real errors:
        evidence -> article -> audit -> one repair -> final audit

    There is NO regeneration loop and NO recursive retry loop.
    A single trend normally uses three calls, with targeted recovery only when
    necessary. A short-article completeness pass is used only when the article is
    below the publication safety margin:
        - 3 normal calls: evidence -> article -> audit
        - +1 short-article expansion only when needed
        - +1 temporal recovery only when needed
        - +1 repair when needed
        - +1 final audit after repair
        - +1 inference cleanup only when the final audit contains inference-only errors
        - +1 final audit after that cleanup
    """

    pipeline_start = time.perf_counter()
    print(f"[TIMER] PIPELINE START | prompt_chars={len(prompt or '')}")

    print("[PIPELINE] Evidence extraction...")
    evidence = _extract_evidence(prompt)
    evidence = _normalize_temporal_updates(evidence)
    print(f"[TIMER] Evidence stage complete | elapsed={time.perf_counter() - pipeline_start:.2f}s")

    print("[PIPELINE] Article generation...")
    article_stage_start = time.perf_counter()
    article = _generate_article_with_schema_retry(prompt, evidence)
    print(f"[TIMER] Article stage complete | elapsed={time.perf_counter() - article_stage_start:.2f}s")

    if not _schema_ok(article):
        raise ValueError("Article generation returned invalid article schema after schema retry.")

    # Prevent the common failure where factual audit passes but the downstream
    # publisher rejects the article for being under 70 words. Expand first, then audit.
    article = _expand_short_article(article, evidence, reason="pre-audit")

    print("[PIPELINE] Evidence-bound audit...")
    audit_stage_start = time.perf_counter()
    audit = _normalize_audit(_audit(prompt, article, evidence), evidence)
    print(f"[TIMER] Initial audit complete | elapsed={time.perf_counter() - audit_stage_start:.2f}s")

    if audit["passed"]:
        print("[PIPELINE] FACT CHECK PASSED")
        print(f"[TIMER] PIPELINE TOTAL | elapsed={time.perf_counter() - pipeline_start:.2f}s")
        return article

    print("[UNIVERSAL FACT/TEMPORAL CHECK FAILED]")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    # --------------------------------------------------------
    # BUILD TEMPORAL RECOVERY v11
    # --------------------------------------------------------
    # Do not immediately fail a recoverable date mismatch.
    # First use the existing evidence map. Only if that is insufficient,
    # perform a narrow web search dedicated to resolving the date.
    # --------------------------------------------------------
    if TEMPORAL_RECOVERY_ENABLED and _is_temporal_error(audit):
        if _temporal_recovery_needed(evidence, audit):
            print("[TEMPORAL RECOVERY] Temporal conflict detected; searching exact date/value...")
            recovered_evidence = _resolve_temporal_with_web(
                prompt, evidence, article, audit
            )
            evidence = recovered_evidence
            print("[TEMPORAL RECOVERY] Evidence updated; repairing temporal claim.")
        else:
            print("[TEMPORAL RECOVERY] Existing evidence is sufficient; repairing from evidence.")

    print("[PIPELINE] One factual repair...")
    repair_stage_start = time.perf_counter()
    repaired = _repair(prompt, article, evidence, audit)
    print(f"[TIMER] Repair stage complete | elapsed={time.perf_counter() - repair_stage_start:.2f}s")

    if not _schema_ok(repaired):
        print("[PIPELINE] Repair schema invalid; one strict repair-schema retry...")
        repaired = _call(
            f"""
Return ONLY a valid JSON article object using exactly this schema:

{{
  "title": "string",
  "description": "string",
  "h1": "string",
  "intro": "string",
  "sections": [
    {{"title": "string", "text": "string"}}
  ]
}}

Repair ONLY the factual errors listed in the AUDIT.
Use ONLY the EVIDENCE MAP and AUDIT.
Do not invent facts or dates.
If dates or amounts conflict, remove the unsupported comparison instead of
creating an intermediate value or date.
For every INFERENCE error, DELETE the unsupported inference rather than
paraphrasing, softening, or replacing it with another interpretation. Delete the
whole sentence when the whole sentence is unsupported; otherwise delete only the
unsupported clause. Never substitute equivalent wording such as "underscores",
"highlights", "shows", "reflects", "demonstrates", "suggests", or "indicates".
If removing such language makes the article short, add only additional material
facts already present in the EVIDENCE MAP; never add filler or speculation.
Keep the original article language.
Do not use markdown fences.

EVIDENCE MAP:
{_compact_json(evidence)}

ORIGINAL ARTICLE:
{_compact_json(article)}

AUDIT:
{_compact_json(audit)}
""",
            temperature=0.02,
            num_predict=REPAIR_TOKENS,
        )

    if not _schema_ok(repaired):
        raise ValueError("Repair returned invalid article schema after schema retry.")

    # Repair may delete an unsupported inference and accidentally leave the article
    # below the publisher minimum. Expand only from the existing evidence map.
    repaired = _expand_short_article(repaired, evidence, reason="post-repair")

    if _article_word_count(repaired) < 70:
        raise ValueError("Repair produced an article below the publication minimum after evidence-only expansion.")

    print("[PIPELINE] Final evidence-bound audit...")
    final_audit_stage_start = time.perf_counter()
    final_audit = _normalize_audit(_audit(prompt, repaired, evidence), evidence)
    print(f"[TIMER] Final audit complete | elapsed={time.perf_counter() - final_audit_stage_start:.2f}s")
    print(f"[TIMER] PIPELINE TOTAL SO FAR | elapsed={time.perf_counter() - pipeline_start:.2f}s")

    if final_audit["passed"]:
        print("[PIPELINE] FACT CHECK PASSED AFTER REPAIR")
        print(f"[TIMER] PIPELINE TOTAL | elapsed={time.perf_counter() - pipeline_start:.2f}s")
        return repaired

    print("[UNIVERSAL FINAL FACT/TEMPORAL CHECK FAILED]")
    print(json.dumps(final_audit, ensure_ascii=False, indent=2))

    # --------------------------------------------------------
    # V15.3 FINAL INFERENCE CLEANUP GUARD
    # --------------------------------------------------------
    # The initial auditor can occasionally miss an inference that becomes
    # visible only after a temporal/factual repair. Do one narrow cleanup when
    # the FINAL audit contains INFERENCE errors and nothing else. This prevents
    # the model from merely rephrasing the inference while keeping the main
    # one-repair architecture intact for ordinary factual/temporal failures.
    # --------------------------------------------------------
    final_errors = final_audit.get("errors", [])
    if final_errors and all(
        str(e.get("category", "")).upper() == "INFERENCE"
        for e in final_errors
    ):
        print("[PIPELINE] Final inference cleanup guard...")
        cleanup_stage_start = time.perf_counter()
        cleaned = _final_inference_cleanup(repaired, evidence, final_audit)
        print(
            f"[TIMER] Final inference cleanup complete | elapsed="
            f"{time.perf_counter() - cleanup_stage_start:.2f}s"
        )

        if not _schema_ok(cleaned):
            raise ValueError("Final inference cleanup returned invalid article schema.")

        cleaned = _expand_short_article(cleaned, evidence, reason="post-cleanup")
        if _article_word_count(cleaned) < 70:
            raise ValueError("Final inference cleanup produced an article below the publication minimum after evidence-only expansion.")

        print("[PIPELINE] Final evidence-bound audit after inference cleanup...")
        cleanup_audit_stage_start = time.perf_counter()
        cleanup_audit = _normalize_audit(
            _audit(prompt, cleaned, evidence), evidence
        )
        print(
            f"[TIMER] Cleanup final audit complete | elapsed="
            f"{time.perf_counter() - cleanup_audit_stage_start:.2f}s"
        )

        if cleanup_audit["passed"]:
            print("[PIPELINE] FACT CHECK PASSED AFTER FINAL INFERENCE CLEANUP")
            print(f"[TIMER] PIPELINE TOTAL | elapsed={time.perf_counter() - pipeline_start:.2f}s")
            return cleaned

        print("[UNIVERSAL POST-CLEANUP FACT/TEMPORAL CHECK FAILED]")
        print(json.dumps(cleanup_audit, ensure_ascii=False, indent=2))

    raise ValueError(
        "Article failed source-grounded validation after factual repair and final inference cleanup."
    )
