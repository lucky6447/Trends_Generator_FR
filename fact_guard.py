import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Dict, List

from ollama import chat
from config import MODEL


# ============================================================
# TrendCurrent FACT GUARD
# External post-generation factual validation layer.
#
# IMPORTANT:
#   This file does NOT modify ollama_client.py.
#
# Input:
#   source material + generated article
#
# Output:
#   PASS / FLAG with concrete factual issues.
#
# Design:
#   1) deterministic checks for obvious contradictions
#   2) independent Ollama audit for semantic/source-grounding issues
#   3) no automatic article rewriting
# ============================================================

FACT_GUARD_VERSION = "fact-guard-v1.1.9-conservative-event-date-source-id"

# Performance configuration:
# Threads and batch are intentionally left to Ollama by default.
# This mirrors the AUTO runner strategy already used elsewhere in TrendCurrent.
# Explicit environment overrides remain supported for controlled testing.
FACT_GUARD_NUM_THREADS = os.getenv("FACT_GUARD_NUM_THREADS", "").strip()
FACT_GUARD_NUM_BATCH = os.getenv("FACT_GUARD_NUM_BATCH", "").strip()

# Benchmark only: when enabled, independent focused audits run concurrently.
# Default is OFF so the normal production behavior remains unchanged.
FACT_GUARD_PARALLEL_AUDITS = (
    os.getenv("FACT_GUARD_PARALLEL_AUDITS", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)

NUM_CTX = max(4096, int(os.getenv("FACT_GUARD_NUM_CTX", "8192")))
AUDIT_TOKENS = max(300, int(os.getenv("FACT_GUARD_AUDIT_TOKENS", "520")))

print(f"[FACT GUARD] {FACT_GUARD_VERSION}")


def _fact_guard_ollama_options(
    *,
    temperature: float,
    num_predict: int,
) -> Dict[str, Any]:
    """Build Ollama options while preserving AUTO defaults."""
    options: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": 0.85,
        "top_k": 40,
        "num_ctx": NUM_CTX,
        "num_predict": num_predict,
    }

    # Only send thread/batch when explicitly overridden.
    # Omitting them lets Ollama choose automatically.
    if FACT_GUARD_NUM_THREADS:
        options["num_thread"] = max(1, int(FACT_GUARD_NUM_THREADS))

    if FACT_GUARD_NUM_BATCH:
        options["num_batch"] = max(64, int(FACT_GUARD_NUM_BATCH))

    return options


def _fact_guard_timer_label(label: str, started: float) -> None:
    elapsed = time.perf_counter() - started
    print(f"[FACT GUARD TIMER] {label} END | elapsed={elapsed:.2f}s")


# ============================================================
# JSON / text helpers
# ============================================================

def _extract_json_object(text: str) -> Dict[str, Any]:
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
        raise ValueError("No JSON object returned.")

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
                return json.loads(text[start:i + 1])

    raise ValueError("Incomplete JSON object.")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _article_text(article: Dict[str, Any]) -> str:
    parts = [
        article.get("title", ""),
        article.get("description", ""),
        article.get("h1", ""),
        article.get("intro", ""),
    ]

    for section in article.get("sections", []):
        if isinstance(section, dict):
            parts.append(section.get("title", ""))
            parts.append(section.get("text", ""))

    return "\n".join(str(x) for x in parts if x)


def _source_contains(source: str, text: str) -> bool:
    text_norm = _normalize(text)
    return bool(text_norm) and text_norm in _normalize(source)


# ============================================================

def _temporal_consistency_signals(article: Dict[str, Any]) -> Dict[str, bool]:
    """
    Detect whether a focused cross-fact temporal audit is warranted.

    This is only a trigger. It does not decide factual correctness.
    """
    text_norm = _normalize(_article_text(article))

    temporal_patterns = [
        r"\bround\s+\d+\b", r"\bround\b",
        r"\bweek\s+\d+\b", r"\bmatchday\s+\d+\b", r"\bday\s+\d+\b",
        r"\bopening round\b", r"\bsecond round\b", r"\bthird round\b",
        r"\bfinal round\b", r"\bquarter[- ]final\b", r"\bsemi[- ]final\b",
        r"\bfinal\b", r"\btoday\b", r"\byesterday\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b",
        r"\b[a-z]+\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]

    score_patterns = [
        r"\b(?:score|scored|shot|fired|carded|finished|recorded)\b.{0,45}\b\d{1,3}\b",
        r"\b\d{1,3}\s*(?:under|over)\b",
        r"\b\d{1,3}[-–]\d{1,3}\b",
        r"\b\d{1,3}\b.{0,12}\b(?:goals?|points?|runs?|yards?|strokes?|shots?)\b",
    ]

    relation_patterns = [
        r"\bin contrast\b", r"\bwhile\b", r"\bcompared with\b",
        r"\bcompared to\b", r"\bversus\b", r"\bvs\.?\b",
        r"\bwhereas\b", r"\bmeanwhile\b", r"\bbut\b",
        r"\bafter\b", r"\bbefore\b", r"\bsame (?:day|round|week|match)\b",
    ]

    return {
        "temporal": any(re.search(p, text_norm) for p in temporal_patterns),
        "score": any(re.search(p, text_norm) for p in score_patterns),
        "relation": any(re.search(p, text_norm) for p in relation_patterns),
    }


# ============================================================
# Event-date consistency trigger
# ============================================================

def _event_date_consistency_signals(article: Dict[str, Any]) -> Dict[str, bool]:
    """
    Trigger for the dedicated event-date/freshness audit.
    This function does not infer or compare dates.
    """
    text_norm = _normalize(_article_text(article))

    date_patterns = [
        r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b",
        r"\b[a-z]+\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",
    ]

    event_patterns = [
        r"\b(win|wins|won|defeated|beat|beats|lost|lose|final|match|game|race|event)\b",
        r"\b(gewann|gewinnt|besiegte|schlug|verlor|finale|spiel|rennen|veranstaltung)\b",
        r"\b(gagné|gagne|battu|finale|match|course|événement)\b",
        r"\b(ganó|gana|venció|final|partido|carrera|evento)\b",
        r"\b(vinto|vince|battuto|finale|partita|gara|evento)\b",
        r"\b(venceu|vence|derrotou|final|partida|corrida|evento)\b",
        r"\b(signed|signs|joined|appointed|elected|announced|released|launched)\b",
        r"\b(unterschrieb|unterzeichnete|wechselte|ernannt|gewählt|angekündigt|veröffentlicht|gestartet)\b",
        # Entertainment/media release events.
        r"\b(trailer|teaser|film|movie|series|album|single|premiere|release|released|debut)\b",
        r"\b(bande[- ]annonce|teaser|film|série|album|single|première|sortie|sorti|début)\b",
        r"\b(trailer|teaser|película|serie|álbum|sencillo|estreno|lanzamiento|lanzó|debut)\b",
        r"\b(trailer|teaser|film|serie|album|singolo|anteprima|uscita|uscito|debutto)\b",
        r"\b(trailer|teaser|filme|série|álbum|single|estreia|lançamento|lançou)\b",
    ]

    # Relative/current temporal language must also trigger the audit.
    # v1.1.2 required an explicit calendar date, so "récemment ..." could
    # bypass the dedicated event-date audit completely.
    # Explicit event-status language also triggers the focused audit.
    event_status_patterns = [
        r"\b(?:made|makes|making)\s+landfall\b",
        r"\blandfall\b",
        r"\b(?:without|did not|didn't|never)\s+(?:make|making)\s+landfall\b",
        r"\b(?:passed|passes|moved|moving)\s+(?:offshore|off shore)\b",
        r"\b(?:weakened|strengthened|downgraded|upgraded|intensified|dissipated)\b",
        r"\b(?:cancelled|canceled|postponed|called off|went ahead)\b",
        r"\b(?:landete|landet)\b", r"\bohne\s+landfall\b",
        r"\b(?:touché|a touché)\s+terre\b", r"\bsans\s+atteindre\s+les\s+côtes\b",
        r"\btocó\s+tierra\b", r"\bsin\s+tocar\s+tierra\b",
        r"\b(?:atterrato|approdato)\b", r"\bsem\s+(?:tocar|atingir)\s+terra\b",
    ]

    freshness_patterns = [
        r"\b(recent|recently|latest|new|newly|current|currently|today|yesterday|just)\b",
        r"\b(récemment|récent|récente|dernier|dernière|nouveau|nouvelle|actuel|actuelle|aujourd'hui|hier|vient de)\b",
        r"\b(recientemente|reciente|último|última|nuevo|nueva|actual|actualmente|hoy|ayer|acaba de)\b",
        r"\b(recentemente|recente|ultimo|ultima|nuovo|nuova|attuale|attualmente|oggi|ieri|ha appena)\b",
        r"\b(recentemente|recente|último|última|novo|nova|atual|atualmente|hoje|ontem|acabou de)\b",
        r"\b(kürzlich|neu|neue|neuest|aktuell|derzeit|heute|gestern|gerade)\b",
    ]

    return {
        "explicit_date": any(re.search(p, text_norm) for p in date_patterns),
        "event_language": any(re.search(p, text_norm) for p in event_patterns),
        "event_status_language": any(re.search(p, text_norm) for p in event_status_patterns),
        "freshness_language": any(re.search(p, text_norm) for p in freshness_patterns),
    }



# ============================================================
# Current-state / upcoming-event deterministic checks
# ============================================================

# These checks are intentionally narrow. They only operate when the
# article itself contains an explicit calendar date and/or an unambiguous
# future/upcoming event-status phrase. They never infer completion merely
# because a date is old: postponement/cancellation/rescheduling language
# must be considered by the focused semantic audit.
#
# The validator's reference date is supplied by FACT_GUARD_REFERENCE_DATE
# when available. If it is absent, no deterministic past-vs-upcoming check
# is performed; the existing source-grounded Ollama audit remains active.

_FUTURE_EVENT_STATUS_PATTERNS = [
    r"\bwill\s+(?:play|face|meet|take\s+on|host|visit|travel|compete|return|appear)\b",
    r"\b(?:is|are)\s+(?:set|scheduled|due)\s+to\s+(?:play|face|meet|take\s+on|host|visit|compete|return|appear)\b",
    r"\b(?:is|are)\s+set\s+for\b",
    r"\b(?:will|is|are)\s+(?:battle|clash|go)\b",
    r"\b(?:set|scheduled)\s+to\s+battle\b",
    r"\b(?:upcoming|forthcoming)\b",
    r"\b(?:tomorrow|tonight)\b",
    r"\b(?:se\s+enfrentar[aá]n|jugar[aá]n|disputar[aá]n)\b",
    r"\b(?:s'affronteront|joueront)\b",
    r"\b(?:si affronteranno|giocheranno)\b",
    r"\b(?:werden\s+spielen|werden\s+antreten|treten\s+gegeneinander\s+an)\b",
    r"\b(?:vão\s+jogar|irão\s+jogar|se\s+enfrentarão)\b",
]

def _parse_reference_date() -> date | None:
    """
    Optional validator reference date.

    Priority:
      1) FACT_GUARD_REFERENCE_DATE=YYYY-MM-DD
      2) no deterministic reference date

    We deliberately do not use the machine's wall-clock date implicitly.
    This keeps validation reproducible and avoids timezone-related drift.
    """
    raw = os.getenv("FACT_GUARD_REFERENCE_DATE", "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        print(
            f"[FACT GUARD] Invalid FACT_GUARD_REFERENCE_DATE={raw!r}; "
            "deterministic current-state date check disabled.",
            file=sys.stderr,
        )
        return None


def _extract_article_dates(article: Dict[str, Any]) -> List[date]:
    """
    Extract explicit YYYY-MM-DD, D Month YYYY, or Month D, YYYY dates
    from the article text. Natural-language month parsing is intentionally
    limited to English month names here; the semantic audit handles
    multilingual dates.
    """
    text = _article_text(article)
    dates: List[date] = []

    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            dates.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass

    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    for m in re.finditer(
        r"\b(\d{1,2})\s+([a-z]+)\s+(\d{4})\b",
        _normalize(text),
    ):
        month = months.get(m.group(2))
        if month:
            try:
                dates.append(date(int(m.group(3)), month, int(m.group(1))))
            except ValueError:
                pass

    for m in re.finditer(
        r"\b([a-z]+)\s+(\d{1,2}),\s+(\d{4})\b",
        _normalize(text),
    ):
        month = months.get(m.group(1))
        if month:
            try:
                dates.append(date(int(m.group(3)), month, int(m.group(2))))
            except ValueError:
                pass

    # Preserve order while removing duplicates.
    unique: List[date] = []
    seen = set()
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _source_has_multiple_event_dates(source: str) -> bool:
    """
    Conservative trigger for multi-event temporal auditing.

    The source often contains several dates that are ONLY publication/update
    dates. Counting all dates was too broad and could trigger a temporal audit
    for an article whose source material contains no evidence of multiple
    event instances.

    We therefore count a date only when it appears near explicit event-language
    in the source. This remains a trigger only: the focused temporal audit is
    still the authority for deciding whether facts were actually conflated.
    """
    source_norm = _normalize(source)

    event_context_patterns = [
        r"\b(game|match|race|event|round|matchday|final|semifinal|quarterfinal)\b",
        r"\b(win|wins|won|defeated|beat|beats|lost|lose|scored|finished|played|faced)\b",
        r"\b(signed|signs|joined|appointed|elected|announced|released|launched|debut)\b",
        r"\b(trailer|teaser|film|movie|series|album|single|premiere|release|released)\b",
        r"\b(cancelled|canceled|postponed|rescheduled|scheduled|landfall|weakened|strengthened)\b",
        r"\b(gewann|gewonnen|besiegte|spiel|rennen|runde|angekündigt|veröffentlicht)\b",
        r"\b(gagné|battu|match|course|finale|sortie|annoncé)\b",
        r"\b(ganó|venció|partido|carrera|final|lanzamiento|anunció)\b",
        r"\b(vinto|battuto|partita|gara|finale|uscita|annunciato)\b",
        r"\b(venceu|derrotou|partida|corrida|final|lançamento|anunciou)\b",
    ]

    event_spans = []
    for pattern in event_context_patterns:
        event_spans.extend(m.span() for m in re.finditer(pattern, source_norm))

    if not event_spans:
        return False

    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    date_occurrences = []

    for m in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", source_norm):
        date_occurrences.append((m.start(), m.end(), m.group(1)))

    for m in re.finditer(
        r"\b(\d{1,2})\s+([a-z]+)\s+(\d{4})\b", source_norm
    ):
        month = months.get(m.group(2))
        if month:
            try:
                date_occurrences.append((
                    m.start(), m.end(),
                    date(int(m.group(3)), month, int(m.group(1))).isoformat(),
                ))
            except ValueError:
                pass

    for m in re.finditer(
        r"\b([a-z]+)\s+(\d{1,2}),\s+(\d{4})\b", source_norm
    ):
        month = months.get(m.group(1))
        if month:
            try:
                date_occurrences.append((
                    m.start(), m.end(),
                    date(int(m.group(3)), month, int(m.group(2))).isoformat(),
                ))
            except ValueError:
                pass

    event_dates: set[str] = set()
    CONTEXT_WINDOW = 180

    publication_context_patterns = [
        r"\bpublished\b", r"\bupdated\b", r"\bposted\b",
        r"\bpublication\b", r"\bsource date\b", r"\bdate published\b",
        r"\bpublished on\b", r"\bupdated on\b",
        r"\bveröffentlicht\b", r"\baktualisiert\b",
        r"\bpublié\b", r"\bmis à jour\b",
        r"\bpublicado\b", r"\bactualizado\b",
        r"\bpubblicato\b", r"\baggiornato\b",
        r"\bpublicado\b", r"\batualizado\b",
    ]

    for start, end, iso_date in date_occurrences:
        # Do not classify an explicit publication/update date as an event date
        # merely because an event word appears elsewhere in the same source
        # block. A short local context is enough to catch common feed metadata.
        local_before = source_norm[max(0, start - 90):start]
        local_after = source_norm[end:min(len(source_norm), end + 40)]
        publication_context = any(
            re.search(pattern, local_before) or re.search(pattern, local_after)
            for pattern in publication_context_patterns
        )
        if publication_context:
            continue

        near_event = any(
            abs(event_pos - start) <= CONTEXT_WINDOW
            or abs(event_pos - end) <= CONTEXT_WINDOW
            for span_start, span_end in event_spans
            for event_pos in (span_start, span_end)
        )
        if near_event:
            event_dates.add(iso_date)

    return len(event_dates) >= 2


def _deterministic_current_state_checks(
    article: Dict[str, Any],
    reference_date: date | None,
) -> List[Dict[str, Any]]:
    """
    Detect the narrow, high-confidence case:
      explicit article event date < validation reference date
      AND article uses explicit future/upcoming event language.

    This does NOT infer that every past-dated event is completed.
    It only blocks the particularly clear contradiction where the article
    itself dates the event in the past but still presents it as upcoming.
    The focused Event Status audit remains responsible for source-grounded
    cases involving postponed/rescheduled/cancelled events and cases where
    the article has no explicit event date.
    """
    if reference_date is None:
        return []

    text_norm = _normalize(_article_text(article))
    if not any(re.search(p, text_norm) for p in _FUTURE_EVENT_STATUS_PATTERNS):
        return []

    article_dates = _extract_article_dates(article)
    past_dates = [d for d in article_dates if d < reference_date]
    if not past_dates:
        return []

    # Avoid assuming that an arbitrary historical date in the article is
    # necessarily the event date. The semantic audit still performs the
    # source-grounded event/date binding.
    dates_display = ", ".join(d.isoformat() for d in past_dates[:3])
    matched_status = next(
        (
            re.search(p, text_norm).group(0)
            for p in _FUTURE_EVENT_STATUS_PATTERNS
            if re.search(p, text_norm)
        ),
        "future/upcoming event language",
    )

    return [{
        "severity": "REVIEW",
        "type": "past_dated_event_presented_as_upcoming",
        "claim": f"{matched_status} with article date(s) before reference date",
        "reason": (
            f"The article contains an explicit date before the validation "
            f"reference date ({reference_date.isoformat()}) while also using "
            f"future/upcoming event wording. This is a targeted temporal "
            f"signal; the source-grounded Event Status audit must determine "
            f"whether the dated item is the relevant event and whether it "
            f"was postponed, cancelled, or rescheduled."
        ),
        "evidence": [dates_display],
        "deterministic": True,
    }]


# Deterministic checks
# ============================================================

def _deterministic_checks(source: str, article: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    These checks are deliberately conservative.

    They do NOT attempt to prove that every sentence is true.
    They catch obvious situations where the article contains a
    suspiciously strong claim or a contradiction that can be
    demonstrated directly from the supplied source.
    """
    issues: List[Dict[str, Any]] = []
    text = _article_text(article)
    source_norm = _normalize(source)
    text_norm = _normalize(text)

    def add(severity: str, issue_type: str, claim: str, reason: str):
        issues.append({
            "severity": severity,
            "type": issue_type,
            "claim": claim,
            "reason": reason,
            "evidence": [],
            "deterministic": True,
        })

    # --------------------------------------------------------
    # Absolute / superlative claims
    # --------------------------------------------------------
    superlative_patterns = [
        (r"\blowest\b.{0,50}\brecord\b", "lowest on record"),
        (r"\bhighest\b.{0,50}\brecord\b", "highest on record"),
        (r"\brecord[- ]breaking\b", "record-breaking"),
        (r"\brecord\b.{0,50}\bintensit", "record intensity"),
        (r"\bunprecedented\b", "unprecedented"),
        (r"\bhistoric(?:al)?\b.{0,40}\bintens", "historic intensity"),
        (r"\ball[- ]time\b", "all-time"),
        (r"\bfirst[- ]ever\b", "first-ever"),
        (r"\bnever before\b", "never before"),
    ]

    for pattern, label in superlative_patterns:
        if re.search(pattern, text_norm):
            # Only flag automatically when the source itself does not
            # contain a comparable explicit formulation.
            if not re.search(pattern, source_norm):
                add(
                    "HIGH",
                    "unsupported_superlative",
                    label,
                    "The article makes a strong superlative/record claim that "
                    "is not explicitly present in the supplied source."
                )

    # --------------------------------------------------------
    # Nationwide / universal availability claims
    # --------------------------------------------------------
    availability_patterns = [
        r"\bnationwide\b",
        r"\bacross the country\b",
        r"\bthroughout the country\b",
        r"\bin all cinemas\b",
        r"\bat cinemas across\b",
        r"\beverywhere\b",
    ]

    for pattern in availability_patterns:
        if re.search(pattern, text_norm) and not re.search(pattern, source_norm):
            add(
                "HIGH",
                "scope_inflation",
                re.search(pattern, text_norm).group(0),
                "The article broadens the source to a nationwide/universal scope "
                "that is not explicitly supported."
            )

    # --------------------------------------------------------
    # Causal / motive language
    # --------------------------------------------------------
    causal_patterns = [
        r"\bbecause of\b",
        r"\bdue to\b",
        r"\bcaused by\b",
        r"\bcausing\b",
        r"\bled to\b",
        r"\bresulted in\b",
        r"\bconnected to\b",
        r"\blinked to\b",
        r"\bbehind (?:his|her|their) death\b",
        r"\bpossible connection\b",
    ]

    # Do not automatically reject every causal phrase. The LLM audit
    # determines whether the specific causal claim is actually supported.
    # We only mark it for the semantic audit.
    if any(re.search(pattern, text_norm) for pattern in causal_patterns):
        add(
            "REVIEW",
            "causal_claim",
            "One or more causal/connection claims detected.",
            "Causal or motive language requires explicit source support and "
            "must not be inferred merely from chronology or context."
        )

    # --------------------------------------------------------
    # Quotation / attribution guard
    # --------------------------------------------------------
    quote_chars = ['"', "“", "”", "„", "«", "»"]
    quoted_segments = []

    for match in re.finditer(r'"([^"]{8,300})"|“([^”]{8,300})”|„([^“]{8,300})“|«([^»]{8,300})»', text):
        quoted = next((g for g in match.groups() if g), "")
        if quoted:
            quoted_segments.append(quoted)

    for quoted in quoted_segments:
        if not _source_contains(source, quoted):
            add(
                "HIGH",
                "unsupported_quote",
                quoted,
                "The quoted wording cannot be traced directly to the supplied source."
            )

    # --------------------------------------------------------
    # Obvious role/title substitutions
    #
    # Keep this deliberately narrow: role/title consistency is handled
    # by the independent audit, but we explicitly mark it as a required
    # high-priority comparison. We do not maintain a hard-coded person list.
    # --------------------------------------------------------

    role_patterns = [
        r"\bmanager\b", r"\bcoach\b", r"\bhead coach\b",
        r"\bplayer\b", r"\bmidfielder\b", r"\bdefender\b",
        r"\bforward\b", r"\bpresident\b", r"\bceo\b",
        r"\bowner\b", r"\bminister\b", r"\bmayor\b",
    ]
    article_has_role_claim = any(re.search(p, text_norm) for p in role_patterns)

    if article_has_role_claim:
        add(
            "REVIEW",
            "role_attribution_check",
            "Explicit person/entity role or title detected.",
            "The independent audit must verify that each named person's role/title "
            "matches the source and is not merely inferred from team/entity association."
        )

    # --------------------------------------------------------
    # Current-state / temporal consistency
    #
    # A historical event must not be silently presented as a new/current
    # event. The semantic audit is responsible for deciding the actual
    # chronology from the supplied source.
    # --------------------------------------------------------

    temporal_patterns = [
        r"\b(joined|joins|signed|signs|moved|moves|transferred|transfer|departed|leaves|left)\b",
        r"\b(on|since|from|as of)\s+\d{1,2}\s+[a-z]+\s+\d{4}\b",
        r"\b(on|since|from|as of)\s+[a-z]+\s+\d{1,2},\s+\d{4}\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(first|debut|latest|current|currently|today|yesterday)\b",
    ]

    if any(re.search(p, text_norm) for p in temporal_patterns):
        add(
            "REVIEW",
            "temporal_state_check",
            "Explicit transfer/event/current-state claim detected.",
            "The independent audit must reconstruct the event chronology from the "
            "source and must not treat an old event as a new/current event."
        )


    # Cross-fact temporal relationship trigger. Individual facts can each be
    # true while their combined temporal relationship is false.
    temporal_signals = _temporal_consistency_signals(article)
    if (
        temporal_signals["temporal"]
        and temporal_signals["score"]
        and temporal_signals["relation"]
    ):
        add(
            "REVIEW",
            "cross_fact_temporal_consistency",
            "Multiple time-bound facts are connected by a comparative or "
            "relational statement.",
            "The focused temporal audit must verify that the connected facts "
            "belong to the same relevant event state rather than merely being "
            "individually true."
        )

    return issues


# ============================================================
# Focused event-date / current-state audit
# ============================================================

_EVENT_DATE_AUDIT_FORMAT = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "type": {"type": "string"},
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                    "source_excerpt": {"type": "string"},
                },
                "required": [
                    "severity", "type", "claim", "reason", "source_excerpt"
                ],
            },
        },
    },
    "required": ["passed", "issues"],
}


def _event_date_audit_prompt(source: str, article: Dict[str, Any], reference_date: date | None = None) -> str:
    reference_text = reference_date.isoformat() if reference_date else "NOT PROVIDED"
    return f"""
You are a focused event-date consistency validator for TrendCurrent.

Your ONLY task is to detect whether the ARTICLE gives an EVENT an
incorrect temporal status or date compared with the SOURCE MATERIAL.

This includes THREE distinct failure modes:
1) explicit date mismatch: the article assigns the event to the wrong date;
2) temporal freshness mismatch: the source establishes that the event is
   historical/old, but the article presents the same event as recent, new,
   current, just announced, newly released, or otherwise happening now;
3) event-status mismatch: the source explicitly establishes one event state
   or transition, but the article states a materially different state.
   Examples include "made landfall" vs "passed offshore without landfall",
   "weakened to a tropical storm" vs "strengthened to a hurricane", or
   "was cancelled" vs "went ahead".

SOURCE MATERIAL is the ONLY factual authority.

STRICT RULES:
- Do not use outside knowledge.
- Do not rewrite the article.
- Do not reject normal paraphrasing.
- Distinguish EVENT DATE from SOURCE PUBLICATION DATE, UPDATE DATE, and
  ARTICLE GENERATION DATE.
- A source being published on date B does NOT mean that the event happened on
  date B.
- If the source explicitly states or clearly establishes that an event happened
  on date A, while the article states that the event happened on date B, flag
  the mismatch.
- Pay special attention to results, finals, matches, races, elections,
  appointments, transfers, signings, releases, launches, landfalls, storm
  strength changes, cancellations, departures, arrivals, and other completed
  events.
- For event-status claims, compare the actual event relationship/state, not just
  whether the same event and entities are mentioned.
- If the source explicitly says an event did NOT happen or happened in a
  materially different way, a contrary article statement is HIGH.
- Do not infer a negative event status from silence. A HIGH event-status
  mismatch requires explicit source support for the contrary state.
- Prefer the date explicitly tied to the EVENT itself over a nearby publication
  or update date.
- If the source contains multiple dates, identify what each date refers to
  before judging the article.
- Do NOT flag a date merely because it differs from the source publication date.
- If the article gives a date but the source does not provide enough information
  to establish the event date, use REVIEW rather than HIGH.
- Do NOT require the article to contain an explicit calendar date. Relative
  temporal wording such as "recently", "today", "newly", "current",
  "just announced", "récemment", "nouvelle", "actuel", etc. is a temporal
  claim and must be checked against the source-supported event date.
- If the source clearly establishes that the event is old/historical and the
  article presents that event as recent/new/current, flag HIGH even when the
  article contains no explicit calendar date.
- If the article gives no explicit date and makes no relative/current temporal
  claim, return no event-date issue.
- VALIDATION REFERENCE DATE: {reference_text}
- When a validation reference date is provided, use it only as the temporal
  point at which the article is being judged. If the source establishes that
  the relevant event occurred before that reference date, the article must
  not present that same event as upcoming, scheduled to occur, or otherwise
  still pending unless the source explicitly establishes a postponement,
  cancellation, rescheduling, delay, or another later event date.
- Distinguish an EVENT DATE from the publication date and the validation
  reference date. Never treat the reference date itself as the event date.
- A past event may legitimately be discussed in the present tense as a
  historical/current fact (for example, "won" or "defeated"). Only future/
  upcoming status applied to an already completed past event is a mismatch.
- If the source gives an earlier scheduled date but explicitly says the event
  was postponed/rescheduled/cancelled and provides a later state, judge the
  later supported state rather than assuming completion.
- HIGH requires either a clear date mismatch OR a clear source-supported
  historical event being presented with materially false current/recent framing.
- REVIEW is appropriate when the source has ambiguous or conflicting temporal
  information that prevents a confident determination.
- Ignore style, grammar, and harmless wording.

IMPORTANT:
Catch BOTH:
A) the failure mode where an article copies the SOURCE PUBLICATION DATE and
   presents it as the EVENT DATE;
B) the failure mode where an old event is presented as recent/current without
   an explicit date.

For every issue, provide a short exact source excerpt supporting the conclusion.
If no source excerpt can be identified, leave it empty.

Return ONLY JSON:
{{
  "passed": true,
  "issues": []
}}

or:
{{
  "passed": false,
  "issues": [
    {{
      "severity": "HIGH",
      "type": "event_date_mismatch",
      "claim": "short description of the article's event/date/status claim",
      "reason": "The article assigns the event a materially different date or event status than the source-supported event state.",
      "source_excerpt": "short exact excerpt"
    }}
  ]
}}

SOURCE MATERIAL:
{source}

VALIDATION REFERENCE DATE:
{reference_text}

ARTICLE:
{json.dumps(article, ensure_ascii=False)}
"""



def _downgrade_publication_date_only_event_issues(
    issues: List[Dict[str, Any]],
    source: str,
    article: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Keep publication/update dates from being treated as proof that the
    underlying event is historical.

    If the article has no explicit event date and the source does not provide
    multiple event dates, a HIGH event_date_mismatch backed only by a
    publication/update timestamp is a REVIEW signal, not a publication block.
    """
    if not issues:
        return issues

    if _extract_article_dates(article) or _source_has_multiple_event_dates(source):
        return issues

    publication_markers = (
        "published",
        "publication",
        "date published",
        "published on",
        "updated",
        "updated on",
        "posted",
        "source date",
    )

    cleaned = []
    downgraded = 0

    for issue in issues:
        if (
            issue.get("severity") == "HIGH"
            and issue.get("type") == "event_date_mismatch"
        ):
            excerpt = _normalize(issue.get("source_excerpt", ""))
            reason = _normalize(issue.get("reason", ""))
            if (
                any(marker in excerpt for marker in publication_markers)
                or any(marker in reason for marker in publication_markers)
            ):
                item = dict(issue)
                item["severity"] = "REVIEW"
                item["reason"] = (
                    str(issue.get("reason", "")).strip()
                    + " Downgraded to REVIEW because the supplied evidence "
                    "identifies a publication/update date but does not establish "
                    "the underlying event date."
                ).strip()
                item["audit_layer"] = "focused_event_date_conservative"
                cleaned.append(item)
                downgraded += 1
                continue
        cleaned.append(issue)

    if downgraded:
        print(
            "[FACT GUARD] Event-date publication-only issue(s) "
            f"downgraded to REVIEW: {downgraded}"
        )

    return cleaned


def _ollama_event_date_audit(source: str, article: Dict[str, Any], reference_date: date | None = None) -> Dict[str, Any]:
    _fact_guard_started = time.perf_counter()
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": _event_date_audit_prompt(source, article, reference_date)}],
        options=_fact_guard_ollama_options(
            temperature=0.0,
            num_predict=AUDIT_TOKENS,
        ),
        format=_EVENT_DATE_AUDIT_FORMAT,
    )

    raw = response.message.content or ""
    result = _extract_json_object(raw)

    if not isinstance(result, dict):
        raise ValueError("Event-date Fact Guard audit returned invalid JSON.")

    issues = result.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    issues = _downgrade_publication_date_only_event_issues(
        issues,
        source,
        article,
    )

    clean = []
    for item in issues:
        if not isinstance(item, dict):
            continue

        claim = str(item.get("claim", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not claim or not reason:
            continue

        severity = str(item.get("severity", "REVIEW")).upper()
        if severity not in {"HIGH", "MEDIUM", "LOW", "REVIEW"}:
            severity = "REVIEW"

        clean.append({
            "severity": severity,
            "type": str(item.get("type", "event_date_mismatch")).strip()
                    or "event_date_mismatch",
            "claim": claim,
            "reason": reason,
            "source_excerpt": str(item.get("source_excerpt", "")).strip(),
            "deterministic": False,
            "audit_layer": "focused_event_date",
        })

    _fact_guard_timer_label("event_date", _fact_guard_started)
    return {
        "passed": bool(result.get("passed", False)) and not clean,
        "issues": clean,
    }


# ============================================================
# Focused entity-attribution audit
# ============================================================

_ENTITY_ATTRIBUTION_FORMAT = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "type": {"type": "string"},
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                    "source_excerpt": {"type": "string"},
                },
                "required": ["severity", "type", "claim", "reason", "source_excerpt"],
            },
        },
    },
    "required": ["passed", "issues"],
}


def _entity_attribution_signals(article: Dict[str, Any]) -> bool:
    """Conservative trigger; ordinary entity mentions do not trigger this audit."""
    text_norm = _normalize(_article_text(article))
    patterns = [
        r"\b(said|says|told|admitted|denied|announced|confirmed|revealed|warned|appointed|elected)\b",
        r"\b(signed|agreed|rejected|accepted|joined|left|returns?|returned|injured|operated|will join|will sign|set to join|set to sign)\b",
        r"\b(has|have|had)\s+\w+\s+(?:days?|weeks?|months?)\b",
        r"\b(?:is|was|became|remains|will be|has been)\s+(?:the|a|an)?\s*(?:manager|coach|player|midfielder|defender|forward|president|ceo|minister|mayor)\b",
        r"\b(sagte|sagt|erklärte|bestätigte|kündigte|warnte|ernannt|gewählt|unterschrieb|vereinbarte|wechselte|verließ|verletzt|operiert)\b",
        r"\b(?:ist|war|wurde|bleibt|wird)\s+(?:der|die|das|ein|eine)?\s*(?:trainer|cheftrainer|spieler|mittelfeldspieler|verteidiger|stürmer|präsident|minister|bürgermeister)\b",
        r"\b(a déclaré|a confirmé|a annoncé|a révélé|a averti|nommé|élu|a signé|a accepté|a refusé|a rejoint|a quitté|blessé|opéré)\b",
        r"\b(?:est|était|devient|reste|sera)\s+(?:le|la|un|une)?\s*(?:entraîneur|joueur|milieu|défenseur|attaquant|président|ministre|maire)\b",
        r"\b(dijo|dice|declaró|confirmó|anunció|reveló|advirtió|nombrado|elegido|firmó|aceptó|rechazó|dejó|regresa|lesionado|operado)\b",
        r"\b(?:es|era|se convirtió en|sigue siendo|será)\s+(?:el|la|un|una)?\s*(?:entrenador|jugador|centrocampista|defensa|delantero|presidente|ministro|alcalde)\b",
        r"\b(ha detto|ha dichiarato|ha confermato|ha annunciato|ha rivelato|ha avvertito|nominato|eletto|ha firmato|ha accettato|ha rifiutato|ha lasciato|torna|infortunato|operato)\b",
        r"\b(?:è|era|diventa|rimane|sarà)\s+(?:il|la|un|una)?\s*(?:allenatore|giocatore|centrocampista|difensore|attaccante|presidente|ministro|sindaco)\b",
        r"\b(disse|declarou|confirmou|anunciou|revelou|alertou|nomeado|eleito|assinou|aceitou|rejeitou|deixou|retorna|lesionado|operado)\b",
        r"\b(?:é|era|tornou-se|continua sendo|será)\s+(?:o|a|um|uma)?\s*(?:treinador|jogador|meio-campista|defensor|atacante|presidente|ministro|prefeito)\b",
    ]
    return any(re.search(pattern, text_norm) for pattern in patterns)


def _entity_attribution_audit_prompt(source: str, article: Dict[str, Any]) -> str:
    return f"""
You are a focused entity-attribution consistency validator for TrendCurrent.

Your ONLY task is to detect material errors where the ARTICLE assigns a factual
ACTION, STATEMENT, ROLE, STATUS, DEADLINE, DECISION, TRANSFER STATE, or other
meaning-bearing relationship to the WRONG PERSON, ORGANIZATION, TEAM, or OTHER
ENTITY compared with the SOURCE MATERIAL.

SOURCE MATERIAL is the ONLY factual authority.

STRICT RULES:
- Do not use outside knowledge.
- Do not rewrite the article.
- Do not reject normal paraphrasing.
- Identify the actual subject of each important claim. Verify SUBJECT +
  PREDICATE/ACTION + OBJECT/RELATIONSHIP together.
- A fact is materially wrong when the source supports it for entity A but the
  article assigns the same fact to entity B.
- Proximity does NOT transfer an action or status from one entity to another.
- Pay special attention to transfers, appointments, injuries, departures,
  signings, deadlines, quotations, decisions, and roles.
- Do NOT flag an omitted source entity or harmless pronoun use.
- If the source is ambiguous about the subject, use REVIEW.
- HIGH requires a clear source-supported attribution mismatch that materially
  changes the factual meaning.
- REVIEW is appropriate when entity linkage cannot be established confidently.
- Ignore style, grammar and harmless wording.

For every issue, provide a short exact source excerpt directly supporting the
attribution conclusion.

Return ONLY JSON:
{{
  "passed": true,
  "issues": []
}}
or:
{{
  "passed": false,
  "issues": [{{
    "severity": "HIGH",
    "type": "wrong_entity_attribution",
    "claim": "short description",
    "reason": "The source assigns this action/status/relationship to a different entity.",
    "source_excerpt": "short exact excerpt"
  }}]
}}

SOURCE MATERIAL:
{source}

ARTICLE:
{json.dumps(article, ensure_ascii=False)}
"""


def _ollama_entity_attribution_audit(source: str, article: Dict[str, Any]) -> Dict[str, Any]:
    _fact_guard_started = time.perf_counter()
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": _entity_attribution_audit_prompt(source, article)}],
        options=_fact_guard_ollama_options(
            temperature=0.0,
            num_predict=AUDIT_TOKENS,
        ),
        format=_ENTITY_ATTRIBUTION_FORMAT,
    )
    raw = response.message.content or ""
    result = _extract_json_object(raw)
    if not isinstance(result, dict):
        raise ValueError("Entity-attribution Fact Guard audit returned invalid JSON.")
    issues = result.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    clean = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not claim or not reason:
            continue
        severity = str(item.get("severity", "REVIEW")).upper()
        if severity not in {"HIGH", "MEDIUM", "LOW", "REVIEW"}:
            severity = "REVIEW"
        clean.append({
            "severity": severity,
            "type": str(item.get("type", "wrong_entity_attribution")).strip() or "wrong_entity_attribution",
            "claim": claim,
            "reason": reason,
            "source_excerpt": str(item.get("source_excerpt", "")).strip(),
            "deterministic": False,
            "audit_layer": "focused_entity_attribution",
        })
    _fact_guard_timer_label("entity_attribution", _fact_guard_started)
    return {"passed": bool(result.get("passed", False)) and not clean, "issues": clean}


# Independent semantic audit
# ============================================================

_AUDIT_FORMAT = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "type": {"type": "string"},
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                    "source_excerpt": {"type": "string"},
                },
                "required": [
                    "severity",
                    "type",
                    "claim",
                    "reason",
                    "source_excerpt",
                ],
            },
        },
    },
    "required": ["passed", "issues"],
}


def _audit_prompt(source: str, article: Dict[str, Any]) -> str:
    return f"""
You are an independent factual validator for TrendCurrent.

Your job is NOT to rewrite the article.
Your job is to determine whether the ARTICLE contains material factual
claims that are unsupported or contradicted by the SOURCE MATERIAL.

SOURCE MATERIAL is the ONLY factual authority.

STRICT RULES:
- Do not use outside knowledge.
- Do not assume that a claim is true because it sounds plausible.
- Do not require identical wording.
- Normal journalistic paraphrasing is allowed.
- Flag a claim when the source does not support it.
- Flag wrong names, roles, dates, numbers, locations, event status,
  attribution, quotations, causal claims and exaggerated scope.
- Reconstruct the chronology of important events before judging them.
- When two or more factual claims are linked by "while", "in contrast",
  "compared with", "versus", "whereas", "meanwhile", or similar wording,
  verify the RELATION between the facts, not only each fact in isolation.
- For time-bound results/statistics, bind each fact to its relevant event,
  round, matchday, week, date, or status whenever the source provides it.
  Do not combine a true fact from one temporal state with a true fact from
  another temporal state as if they occurred in the same state.
- A fact such as "Player A scored 61" and "Player B scored 74" can both be
  source-supported while the sentence connecting them is still materially
  wrong if 61 belongs to Round 2 and 74 belongs to Round 1.
- Treat cross-round/cross-day/cross-match/cross-week conflation as HIGH when
  the source explicitly establishes the conflicting temporal assignments.
  Use REVIEW when the source does not contain enough temporal information.
- For every person mentioned with a job/team role, verify the ROLE itself,
  not merely that the person is associated with the club/company/entity.
- Treat PLAYER vs MANAGER/COACH and similar role substitutions as material
  factual errors when the source establishes the person's actual role.
- For transfers, appointments, departures, signings and debuts, determine
  the event date and the person's state at the article's claimed time.
  An old/historical transfer must NOT be accepted as a new/current transfer
  merely because the same person and destination still appear in the source.
- Be especially careful with:
  * "record", "lowest/highest ever", "unprecedented", "first-ever"
  * "nationwide", "all cinemas", "everywhere"
  * causes, motives or connections inferred from chronology
  * people whose job title/role may have been changed
  * old information presented as current
  * historical transfers/events presented as newly announced
   * results/statistics from different rounds, dates, matchdays or weeks
     accidentally combined into one comparison or same-day narrative
  * reported/expected/planned information presented as confirmed
- A contextual sentence is NOT automatically wrong merely because it is
  not word-for-word in the source. Flag it only when it makes a factual
  claim that the source cannot reasonably support.
- If a claim is uncertain, prefer severity REVIEW rather than HIGH.
- HIGH means a clear material factual error or unsupported concrete claim.
- A clearly wrong person-role attribution is HIGH.
- A clearly wrong event chronology, such as an old transfer presented as a
  new/current transfer, is HIGH.
- REVIEW means the wording requires human/secondary verification.
- Ignore style, grammar and harmless editorial wording.

For every issue, provide a short source excerpt that directly supports
your conclusion. If no source excerpt can be identified, leave it empty
and explain why.

Return ONLY JSON.

Format:
{{
  "passed": true,
  "issues": []
}}

or:

{{
  "passed": false,
  "issues": [
    {{
      "severity": "HIGH",
      "type": "wrong_role",
      "claim": "example claim",
      "reason": "The source identifies the person differently.",
      "source_excerpt": "short exact excerpt"
    }}
  ]
}}

SOURCE MATERIAL:
{source}

ARTICLE:
{json.dumps(article, ensure_ascii=False)}
"""



_TEMPORAL_AUDIT_FORMAT = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "type": {"type": "string"},
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                    "source_excerpt": {"type": "string"},
                },
                "required": [
                    "severity", "type", "claim", "reason", "source_excerpt"
                ],
            },
        },
    },
    "required": ["passed", "issues"],
}


def _temporal_audit_prompt(source: str, article: Dict[str, Any]) -> str:
    return f"""
You are a focused temporal-consistency validator for TrendCurrent.

Your ONLY task is to detect factual errors caused by combining individually
true facts that belong to different temporal/event states.

SOURCE MATERIAL is the ONLY factual authority.

STRICT RULES:
- Do not use outside knowledge.
- Do not rewrite the article.
- Do not reject normal paraphrasing.
- Bind each relevant result/statistic/event to the source-supported event,
  round, matchday, week, date, or status whenever available.
- Inspect whether the ARTICLE connects facts as if they belong to the same
  temporal state.
- Pay special attention to "while", "in contrast", "compared with",
  "versus", "whereas", "meanwhile", "but", and similar relational wording.
- A fact being individually true is NOT enough. The relationship between
  connected facts must also be source-supported.
- Example:
  Source: A = 61 in Round 2; B = 74 in Round 1; B = 70 in Round 2.
  Article: "A shot 61, while B struggled with 74."
  This is HIGH if the wording presents 61 and 74 as results from the same
  relevant round/day.
- Do NOT infer a same-round, same-day, same-match, or same-event relationship
  merely because facts appear in the same source or article.
- Treat different event instances involving the same person, team, club,
  organisation, or topic as separate unless the source explicitly supports
  their connection. This includes separate games, matches, races, appearances,
  performances, announcements, incidents, or developments on adjacent dates.
- A fact from Event A must not be attached to Event B merely because the same
  entity appears in both events.
- If the article explicitly identifies different rounds/dates/events, that is fine.
- If the source lacks enough temporal information to decide, use REVIEW.
- HIGH requires a clear temporal contradiction established by the source.
- Ignore style and grammar.

For every issue, provide a short exact source excerpt supporting the
temporal conclusion. If none can be identified, leave it empty.

Return ONLY JSON:
{{
  "passed": true,
  "issues": []
}}

or:
{{
  "passed": false,
  "issues": [
    {{
      "severity": "HIGH",
      "type": "cross_round_conflation",
      "claim": "short description of the connected claim",
      "reason": "why the article combines different temporal states",
      "source_excerpt": "short exact excerpt"
    }}
  ]
}}

SOURCE MATERIAL:
{source}

ARTICLE:
{json.dumps(article, ensure_ascii=False)}
"""


def _ollama_temporal_audit(source: str, article: Dict[str, Any]) -> Dict[str, Any]:
    _fact_guard_started = time.perf_counter()
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": _temporal_audit_prompt(source, article)}],
        options=_fact_guard_ollama_options(
            temperature=0.0,
            num_predict=AUDIT_TOKENS,
        ),
        format=_TEMPORAL_AUDIT_FORMAT,
    )

    raw = response.message.content or ""
    result = _extract_json_object(raw)

    if not isinstance(result, dict):
        raise ValueError("Temporal Fact Guard audit returned invalid JSON.")

    issues = result.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    clean = []
    for item in issues:
        if not isinstance(item, dict):
            continue

        claim = str(item.get("claim", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not claim or not reason:
            continue

        severity = str(item.get("severity", "REVIEW")).upper()
        if severity not in {"HIGH", "MEDIUM", "LOW", "REVIEW"}:
            severity = "REVIEW"

        clean.append({
            "severity": severity,
            "type": str(
                item.get("type", "cross_fact_temporal_consistency")
            ).strip() or "cross_fact_temporal_consistency",
            "claim": claim,
            "reason": reason,
            "source_excerpt": str(item.get("source_excerpt", "")).strip(),
            "deterministic": False,
            "audit_layer": "focused_temporal",
        })

    _fact_guard_timer_label("temporal", _fact_guard_started)
    return {
        "passed": bool(result.get("passed", False)) and not clean,
        "issues": clean,
    }


def _ollama_audit(source: str, article: Dict[str, Any]) -> Dict[str, Any]:
    _fact_guard_started = time.perf_counter()
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": _audit_prompt(source, article)}],
        options=_fact_guard_ollama_options(
            temperature=0.0,
            num_predict=AUDIT_TOKENS,
        ),
        format=_AUDIT_FORMAT,
    )

    raw = response.message.content or ""
    result = _extract_json_object(raw)

    if not isinstance(result, dict):
        raise ValueError("Fact Guard audit returned invalid JSON.")

    issues = result.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    clean = []

    for item in issues:
        if not isinstance(item, dict):
            continue

        claim = str(item.get("claim", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not claim or not reason:
            continue

        severity = str(item.get("severity", "REVIEW")).upper()
        if severity not in {"HIGH", "MEDIUM", "LOW", "REVIEW"}:
            severity = "REVIEW"

        clean.append({
            "severity": severity,
            "type": str(item.get("type", "unsupported_claim")).strip()
                     or "unsupported_claim",
            "claim": claim,
            "reason": reason,
            "source_excerpt": str(item.get("source_excerpt", "")).strip(),
            "deterministic": False,
        })

    _fact_guard_timer_label("semantic", _fact_guard_started)
    return {
        "passed": bool(result.get("passed", False)) and not clean,
        "issues": clean,
    }


# ============================================================
# Result normalization
# ============================================================

# ============================================================
# Correlated blocking-issue normalization
# ============================================================

# The broad semantic audit and focused entity-attribution audit are
# intentionally independent. The same factual incident can therefore be
# reported twice by two audit layers. That is useful for detection, but the
# downstream targeted repair contract expects ONE blocking issue per
# underlying repairable incident.
#
# This normalization is deliberately narrow:
#   - only wrong_role + wrong_entity_attribution pairs are correlated;
#   - only when their claims clearly describe the same action/relationship;
#   - unrelated role/attribution failures remain separate blockers.
#
# We do NOT use an LLM here. The purpose is normalization, not adjudication.

_ISSUE_STOPWORDS = {
    "the", "and", "are", "was", "were", "is", "be", "been", "being",
    "to", "of", "a", "an", "for", "in", "on", "with", "from", "as", "at",
    "by", "that", "this", "it", "they", "their", "his", "her", "its",
    "set", "season",
}

_RELATION_TOKENS = {
    # Roles / appointments
    "host", "hosts", "hosting", "hosted",
    "coach", "manager", "player", "president", "ceo", "minister", "mayor",
    "director", "captain", "chairman", "chairwoman",
    # Actions / status commonly audited by entity attribution
    "appointed", "appoint", "appointed",
    "elected", "elect", "elected",
    "signed", "sign", "signing",
    "joined", "join", "joining",
    "left", "leave", "leaving",
    "returned", "return", "returns",
    "announced", "announce", "announcing",
    "confirmed", "confirm", "confirming",
    "revealed", "reveal", "revealing",
    "said", "says", "told",
    "injured", "injury",
    "rejected", "reject",
    "accepted", "accept",
    "transferred", "transfer",
}

def _issue_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalize(text))
    return {
        token for token in tokens
        if len(token) >= 3 and token not in _ISSUE_STOPWORDS
    }


def _correlated_role_attribution(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    types = {str(a.get("type", "")).strip(), str(b.get("type", "")).strip()}
    if types != {"wrong_role", "wrong_entity_attribution"}:
        return False

    claim_a = _issue_tokens(str(a.get("claim", "")))
    claim_b = _issue_tokens(str(b.get("claim", "")))
    if not claim_a or not claim_b:
        return False

    shared = claim_a & claim_b
    relation_shared = shared & _RELATION_TOKENS

    # Same action/role + at least two additional shared claim tokens is a
    # conservative indication that both audit layers describe one incident.
    if relation_shared and len(shared) >= 3:
        return True

    # Source excerpts can be more precise than the generated claim. If both
    # layers independently point to substantially overlapping evidence, that
    # is also enough to treat them as correlated.
    excerpt_a = _issue_tokens(str(a.get("source_excerpt", "")))
    excerpt_b = _issue_tokens(str(b.get("source_excerpt", "")))
    if excerpt_a and excerpt_b:
        excerpt_shared = excerpt_a & excerpt_b
        if len(excerpt_shared) >= 4:
            return True

    return False


def _deduplicate_correlated_blockers(
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Collapse only duplicate reports of the same role/entity-attribution
    incident. Every other issue remains untouched.

    The first issue is retained so the broad semantic audit remains the
    primary blocking record. The duplicate is removed from the top-level
    issue list so downstream repair sees exactly one blocker.
    """
    result: List[Dict[str, Any]] = []
    removed = 0

    for issue in issues:
        if issue.get("severity") not in {"HIGH", "MEDIUM"}:
            result.append(issue)
            continue

        duplicate_index = None
        for idx, existing in enumerate(result):
            if existing.get("severity") not in {"HIGH", "MEDIUM"}:
                continue
            if _correlated_role_attribution(existing, issue):
                duplicate_index = idx
                break

        if duplicate_index is None:
            result.append(issue)
        else:
            removed += 1

    if removed:
        print(
            f"[FACT GUARD] Correlated duplicate blockers collapsed: {removed}"
        )

    return result


def validate(
    source: str,
    article: Dict[str, Any],
    reference_date: date | None = None,
) -> Dict[str, Any]:
    _fact_guard_total_started = time.perf_counter()
    if not isinstance(article, dict):
        raise ValueError("Article must be a JSON object.")

    # Production callers pass the run's explicit validation date.
    # CLI/manual callers may still use FACT_GUARD_REFERENCE_DATE.
    if reference_date is None:
        reference_date = _parse_reference_date()

    deterministic = _deterministic_checks(source, article)
    deterministic += _deterministic_current_state_checks(article, reference_date)

    temporal_signals = _temporal_consistency_signals(article)

    event_date_signals = _event_date_consistency_signals(article)

    # Existing focused temporal trigger remains unchanged.
    #
    # Additional narrow trigger:
    # If the article contains an explicit event date and event language, and
    # the supplied source contains multiple distinct dates that are each near
    # explicit event language, run the temporal audit even when the article has
    # no score/comparison wording. Publication/update dates alone do not trigger
    # this path.
    #
    # This specifically covers adjacent-event conflation, where two individually
    # true facts about the same person/team can belong to different event
    # instances (for example, separate games on consecutive days).
    #
    # The helper is ONLY a trigger. The focused temporal audit remains the
    # authority for deciding whether the facts were actually conflated.
    run_temporal = (
        (
            temporal_signals["temporal"]
            and temporal_signals["score"]
            and temporal_signals["relation"]
        )
        or
        (
            event_date_signals["event_language"]
            and event_date_signals["explicit_date"]
            and _source_has_multiple_event_dates(source)
        )
    )

    run_event_date = (
        (
            event_date_signals["event_language"]
            and (
                event_date_signals["explicit_date"]
                or event_date_signals["freshness_language"]
            )
        )
        or event_date_signals.get("event_status_language", False)
    )

    run_entity = _entity_attribution_signals(article)

    # The broad semantic audit is always independent and remains mandatory.
    # The focused audits are also independent of one another. In benchmark
    # mode they may execute concurrently; their prompts, models, options,
    # triggers, result parsing and issue normalization are unchanged.
    if FACT_GUARD_PARALLEL_AUDITS:
        jobs = {
            "semantic": lambda: _ollama_audit(source, article),
        }
        if run_temporal:
            jobs["temporal"] = lambda: _ollama_temporal_audit(source, article)
        if run_event_date:
            jobs["event_date"] = lambda: _ollama_event_date_audit(
                source, article, reference_date
            )
        if run_entity:
            jobs["entity_attribution"] = lambda: _ollama_entity_attribution_audit(
                source, article
            )

        parallel_started = time.perf_counter()
        results: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {
                executor.submit(job): name
                for name, job in jobs.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()

        print(
            f"[FACT GUARD TIMER] parallel audit group END | "
            f"elapsed={time.perf_counter() - parallel_started:.2f}s | "
            f"audits={len(jobs)}"
        )

        semantic = results["semantic"]
        temporal = results.get("temporal", {"issues": []})
        event_date = results.get("event_date", {"issues": []})
        entity_attribution = results.get(
            "entity_attribution", {"issues": []}
        )
    else:
        semantic = _ollama_audit(source, article)

        temporal = {"issues": []}
        if run_temporal:
            temporal = _ollama_temporal_audit(source, article)

        event_date = {"issues": []}
        if run_event_date:
            event_date = _ollama_event_date_audit(
                source, article, reference_date
            )

        entity_attribution = {"issues": []}
        if run_entity:
            entity_attribution = _ollama_entity_attribution_audit(
                source, article
            )

    all_issues = (
        deterministic
        + semantic.get("issues", [])
        + temporal.get("issues", [])
        + event_date.get("issues", [])
        + entity_attribution.get("issues", [])
    )

    # Normalize only correlated duplicate reports before counting blockers.
    # Detection remains independent; this step only makes one underlying
    # repairable incident count as one blocking issue.
    all_issues = _deduplicate_correlated_blockers(all_issues)

    # REVIEW alone does not automatically reject an article.
    # HIGH / MEDIUM are publication blockers.
    blocking = [
        issue for issue in all_issues
        if issue.get("severity") in {"HIGH", "MEDIUM"}
    ]

    status = "FLAG" if blocking else "PASS"

    _fact_guard_timer_label("TOTAL", _fact_guard_total_started)
    return {
        "fact_guard_version": FACT_GUARD_VERSION,
        "status": status,
        "passed": status == "PASS",
        "blocking_issues": len(blocking),
        "review_items": len([
            x for x in all_issues if x.get("severity") == "REVIEW"
        ]),
        "issues": all_issues,
    }


# ============================================================
# CLI
# ============================================================

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: python fact_guard.py source.txt article.json",
            file=sys.stderr,
        )
        return 2

    source_path = sys.argv[1]
    article_path = sys.argv[2]

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    article = _load_json(article_path)
    result = validate(source, article)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
