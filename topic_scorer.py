"""
TrendCurrent Topic Priority Filter v1.2

Purpose:
    Select the small number of trends that deserve expensive news retrieval
    and article-generation capacity.

This module is intentionally independent from:
    - universal-fact-lock / ollama_client
    - Fact Guard
    - article generation
    - index generation

It is a deterministic ranking layer. It does NOT generate articles.

Pipeline:
    ALL TRENDS
        -> cheap pre-score
        -> top candidate pool
        -> fetch_news only for candidates
        -> source-quality re-score
        -> article generation only for final winners
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

try:
    from config import COUNTRY
except Exception:
    COUNTRY = os.getenv("COUNTRY", "")

# These defaults are deliberately conservative. They can be changed with
# environment variables without touching the stable factual pipeline.
TOPIC_CANDIDATE_MULTIPLIER = max(
    2, int(os.getenv("TOPIC_CANDIDATE_MULTIPLIER", "4"))
)
TOPIC_MAX_CANDIDATES = max(
    8, int(os.getenv("TOPIC_MAX_CANDIDATES", "24"))
)

# Production topic selection is score-neutral: pre_score is ranking priority only.
# Deterministic editorial exclusions and downstream evidence/factual/editorial
# gates are the actual admission controls.

# Production GEO priority is now market-aware. An explicit environment
# override still wins, preserving the existing deployment control.
COUNTRY_ALIASES = {
    "US": ("us", "usa", "united states", "united states of america", "america"),
    "UK": ("uk", "gb", "gbr", "united kingdom", "great britain", "britain", "england", "scotland", "wales"),
    "DE": ("de", "deu", "germany", "deutschland", "german", "deutsch"),
    "IT": ("it", "ita", "italy", "italia", "italian", "italiano"),
    "FR": ("fr", "fra", "france", "french", "français", "francais"),
    "ES": ("es", "esp", "spain", "españa", "espana", "spanish", "español", "espanol"),
    "PT": ("pt", "prt", "portugal", "portuguese", "português", "portugues"),
    "BR": ("br", "bra", "brazil", "brasil", "brazilian", "brasileiro"),
    "ID": ("id", "idn", "indonesia", "indonesian", "bahasa indonesia"),
    "CA": ("ca", "canada", "canadian"),
    "AU": ("au", "aus", "australia", "australian"),
}

# Keep the existing English signals and add only high-confidence localized
# equivalents for the production markets. These remain deterministic regex
# signals; they do not invoke an LLM or alter the pipeline flow.
SKIP_PATTERNS_BY_LANG = {
    "en": (" vs ", " v ", " live", " score", " result", "calendario", "alineación", "alineacion", "prognóstico", "pronostico", "stream", "streaming"),
    "de": (" vs ", " v ", " live", " ergebnis", " spielstand", "stream", "streaming"),
    "it": (" vs ", " v ", " live", " risultato", "punteggio", "stream", "streaming"),
    "fr": (" vs ", " v ", " en direct", " score", " résultat", "resultat", "stream", "streaming"),
    "es": (" vs ", " v ", " en vivo", " resultado", " marcador", "stream", "streaming", "calendario", "alineación", "alineacion", "pronóstico", "pronostico"),
    "pt": (" vs ", " v ", " ao vivo", " resultado", " placar", "stream", "streaming", "calendário", "calendario"),
    "id": (" vs ", " v ", " live", " skor", " hasil", "stream", "streaming", "jadwal"),
}


# Concrete sports match / fixture topics are excluded from article generation.
#
# Intentionally narrow: ordinary sports news remains eligible. A title must
# contain BOTH a matchup-style separator and a strong sports signal.
SPORTS_MATCH_SEPARATORS = (
    r"\s+vs\.?\s+",
    r"\s+v\.?\s+",
    r"\s[-–—]\s",
)

SPORTS_MATCH_TERMS = (
    # Strong league / sport signals
    r"\bmlb\b", r"\bnfl\b", r"\bnba\b", r"\bnhl\b", r"\bmls\b",
    r"\bbaseball\b", r"\bfootball\b", r"\bbasketball\b", r"\bhockey\b",

    # Explicit fixture language
    r"\bmatch\b", r"\bmatchup\b", r"\bfixture\b", r"\bgame\b",

    # Competition language
    r"\bplayoff(?:s)?\b", r"\bsemifinal(?:s)?\b", r"\bquarterfinal(?:s)?\b",
    r"\bchampionship\b", r"\bfinal(?:s)?\b",

    # Major competition names
    r"\bpremier league\b", r"\bchampions league\b", r"\beuropa league\b",
    r"\bla liga\b", r"\bserie a\b", r"\bbundesliga\b",

    # High-confidence team/entity signals for fixture-style titles.
    # Keep these specific to avoid false positives from generic words such as
    # "giants", "united", "city", or "rangers".
    r"\bmariners\b", r"\bred sox\b", r"\byankees\b",
    r"\bdodgers\b", r"\bcubs\b", r"\bcardinals\b",
    r"\bpadres\b", r"\bphillies\b", r"\bbraves\b",
    r"\bastros\b", r"\btwins\b", r"\bguardians\b", r"\btigers\b",
    r"\broyals\b", r"\bblue jays\b", r"\borioles\b",
    r"\brays\b", r"\bbrewers\b", r"\bmets\b", r"\bnationals\b",
    r"\bathletics\b", r"\bangels\b", r"\breds\b",
    r"\blakers\b", r"\bceltics\b", r"\bwarriors\b",
    r"\bknicks\b", r"\bnets\b", r"\bbulls\b", r"\bheat\b",
    r"\bsuns\b", r"\bnuggets\b", r"\b76ers\b", r"\bclippers\b",
    r"\bchiefs\b", r"\beagles\b", r"\bcowboys\b", r"\bpackers\b",
    r"\b49ers\b", r"\bseahawks\b", r"\bbills\b", r"\bravens\b",
    r"\bpatriots\b", r"\bsteelers\b", r"\bjets\b", r"\bdolphins\b",
    r"\bgiants\b",
    r"\bbarcelona\b", r"\breal madrid\b", r"\batletico madrid\b",
    r"\bmanchester united\b", r"\bmanchester city\b",
    r"\bliverpool\b", r"\barsenal\b", r"\bchelsea\b",
    r"\btottenham\b", r"\bbayern munich\b", r"\bborussia dortmund\b",
    r"\bjuventus\b", r"\bac milan\b", r"\binter milan\b",
    # Additional production-relevant European / international teams.
    r"\bgalatasaray\b", r"\bbasaksehir\b", r"\bbasakşehir\b",
    r"\bauxerre\b", r"\blyon\b", r"\bvilzing\b",
    r"\b1860 munich\b", r"\b1860 münchen\b", r"\bistanbul bb\b",
    r"\babha\b", r"\bal[- ]ettifaq\b",
)

def _is_sports_match_topic(title: str) -> bool:
    t = _norm(title)

    # A separator is mandatory, so "Mariners news" and "Red Sox trade"
    # remain valid candidates.
    if not any(
        re.search(pattern, t, re.IGNORECASE)
        for pattern in SPORTS_MATCH_SEPARATORS
    ):
        return False

    # A strong sports signal is also mandatory.
    return any(
        re.search(pattern, t, re.IGNORECASE)
        for pattern in SPORTS_MATCH_TERMS
    )

NOISE_PATTERNS_BY_LANG = {
    "en": (
        r"\bhoroscope(?:s)?\b", r"\blottery results?\b", r"\bweather forecast\b",
        r"\bexchange rate\b", r"\bprice forecast\b", r"\bstock forecast\b", r"\bflight status\b",
    ),
    "de": (
        r"\bhoroskop(?:e)?\b", r"\blotto(?:zahlen|ergebnisse)?\b", r"\bwetter(?:vorhersage|prognose)\b",
        r"\bwechselkurs(?:e)?\b", r"\bpreisprognose(?:n)?\b", r"\baktienprognose(?:n)?\b", r"\bflugstatus\b",
    ),
    "it": (
        r"\boroscopo(?:i)?\b", r"\brisultati della lotteria\b", r"\bprevisioni? del tempo\b",
        r"\btasso di cambio\b", r"\bprevisioni? di prezzo\b", r"\bprevisioni? azionari[ae]\b", r"\bstato del volo\b",
    ),
    "fr": (
        r"\bhoroscope(?:s)?\b", r"\brésultats? de loterie\b", r"\bprévisions? météo\b", r"\bprevisions? meteo\b",
        r"\btaux de change\b", r"\bprévision(?:s)? de prix\b", r"\bprevision(?:s)? de prix\b", r"\bprévision(?:s)? boursière(?:s)?\b", r"\bstatut du vol\b",
    ),
    "es": (
        r"\bhoróscopo(?:s)?\b", r"\bhoroscopo(?:s)?\b", r"\bresultados? de lotería\b", r"\bresultados? de loteria\b",
        r"\bpronóstico del tiempo\b", r"\bpronostico del tiempo\b", r"\btipo de cambio\b", r"\bprevisión de precio\b", r"\bprevision de precio\b", r"\bpronóstico bursátil\b", r"\bpronostico bursatil\b", r"\bestado del vuelo\b",
    ),
    "pt": (
        r"\bhoróscopo(?:s)?\b", r"\bhoroscopo(?:s)?\b", r"\bresultados? da loteria\b", r"\bprevisão do tempo\b", r"\bprevisao do tempo\b",
        r"\btaxa de câmbio\b", r"\btaxa de cambio\b", r"\bprevisão de preço\b", r"\bprevisao de preco\b", r"\bprevisão de ações?\b", r"\bprevisao de acoes?\b", r"\bstatus do voo\b",
    ),
    "id": (
        r"\bramalan\b", r"\bhasil lotre\b", r"\bprakiraan cuaca\b", r"\bkurs valuta asing\b",
        r"\bprediksi harga\b", r"\bprediksi saham\b", r"\bstatus penerbangan\b",
    ),
}

NEWS_PATTERNS_BY_LANG = {
    "en": (
        r"\bannounc(?:e|ed|es|ement)\b", r"\bconfirm(?:ed|s|ation)?\b", r"\bappoint(?:ed|ment|s)?\b",
        r"\bsign(?:ed|s|ing)?\b", r"\bjoin(?:ed|s|ing)?\b", r"\bwin(?:s|ner|ning)?\b", r"\bwon\b",
        r"\bdefeat(?:ed|s)?\b", r"\bkill(?:ed|s)?\b", r"\bdie(?:d|s)?\b", r"\bdeath\b",
        r"\barrest(?:ed|s)?\b", r"\bresign(?:ed|s|ation)?\b", r"\blaunch(?:ed|es)?\b", r"\brelease(?:d|s)?\b",
        r"\bcrash(?:ed|es)?\b", r"\bearthquake\b", r"\bhurricane\b", r"\bstorm\b", r"\bfire\b",
        r"\bshooting\b", r"\battack\b", r"\belection\b", r"\bdecision\b", r"\bcourt\b", r"\btrial\b",
        r"\bpolice\b", r"\bgovernment\b", r"\bminister\b", r"\bpresident\b",
    ),
    "de": (
        r"\bankündig(?:t|ung|en)\b", r"\bbestätig(?:t|ung|en)\b", r"\bernenn(?:t|ung|en)\b",
        r"\bunterschreib(?:t|en)\b", r"\bwechsel(?:t|n)?\b", r"\bsieg(?:t|e|en)?\b", r"\bgewinn(?:t|en)?\b",
        r"\bniederlag(?:e|en)\b", r"\bgetötet\b", r"\bgestorben\b", r"\btod\b", r"\bfestgenommen\b",
        r"\brücktritt\b", r"\bveröffentlicht\b", r"\bfreigelassen\b", r"\bunfall\b", r"\berdbeben\b",
        r"\bsturm\b", r"\bbrand\b", r"\bschießerei\b", r"\bangriff\b", r"\bwahl\b", r"\bentscheidung\b",
        r"\bgericht\b", r"\bprozess\b", r"\bpolizei\b", r"\bregierung\b", r"\bminister\b", r"\bpräsident\b",
    ),
    "it": (
        r"\bannunc(?:ia|iato|iati|iamento)\b", r"\bconferm(?:a|ato|ata|are)\b", r"\bnomina(?:to|ta|re)?\b",
        r"\bfirm(?:a|ato|ata|are)\b", r"\bader(?:isce|ito|ire)\b", r"\bvinc(?:e|ente|ere|ono)\b", r"\bsconfitt(?:a|o|e)\b",
        r"\buccis(?:o|a|i|e)\b", r"\bmort(?:o|a|i|e)\b", r"\bdecesso\b", r"\barrest(?:ato|ata|ati|ate|o)\b",
        r"\bdimissioni?\b", r"\blanc(?:io|iato|iata|iare)\b", r"\buscit(?:a|o|i|e)\b", r"\bincidente\b",
        r"\bterremoto\b", r"\btempesta\b", r"\bincendio\b", r"\battacco\b", r"\belezi(?:one|oni)\b",
        r"\bdecisione\b", r"\bcorte\b", r"\bprocesso\b", r"\bpolizia\b", r"\bgoverno\b", r"\bministro\b", r"\bpresidente\b",
    ),
    "fr": (
        r"\bannonc(?:e|é|ée|ées|és|ement)\b", r"\bconfirm(?:e|é|ée|ation)\b", r"\bnomm(?:e|é|ée|ation)\b",
        r"\bsign(?:e|é|ée|er)\b", r"\brejoint(?:e|s|re)?\b", r"\bgagn(?:e|é|ée|er)\b", r"\bvainc(?:u|re|queur)\b",
        r"\bdéfait(?:e|es)?\b", r"\btu(?:é|ée|és|ées)\b", r"\bmort(?:e|s|es)?\b", r"\bdécès\b",
        r"\barrest(?:é|ée|és|ées|ation)\b", r"\bdémission(?:s)?\b", r"\blanc(?:e|é|ée|ement)\b", r"\bsort(?:i|ie|ies)?\b",
        r"\baccident\b", r"\bséisme\b", r"\btempête\b", r"\bincendie\b", r"\btir\b", r"\battaque\b",
        r"\bélection(?:s)?\b", r"\bdécision\b", r"\bcour\b", r"\bprocès\b", r"\bpolice\b", r"\bgouvernement\b", r"\bministre\b", r"\bprésident\b",
    ),
    "es": (
        r"\banunci(?:a|ó|ado|ada|amiento)\b", r"\bconfirm(?:a|ó|ado|ada|ación)\b", r"\bnombr(?:a|ó|ado|ada|amiento)\b",
        r"\bfirm(?:a|ó|ado|ada|ar)\b", r"\bse une\b", r"\bgan(?:a|ó|ado|adora|ar)\b", r"\bvenc(?:e|ió|ido|ida|er)\b",
        r"\bderrot(?:a|ó|ado|ada)\b", r"\bmat(?:ó|ado|ada|aron)\b", r"\bmuert(?:o|a|os|as)\b", r"\bmuerte\b",
        r"\barrest(?:a|ó|ado|ada|o)\b", r"\bdimisi(?:ón|ones)\b", r"\blanz(?:a|ó|ado|ada|ar)\b", r"\bliber(?:a|ó|ado|ada)\b",
        r"\baccidente\b", r"\bterremoto\b", r"\btormenta\b", r"\bincendio\b", r"\btiroteo\b", r"\bataque\b",
        r"\belecci(?:ón|ones)\b", r"\bdecisi(?:ón|ones)\b", r"\bcorte\b", r"\bjuicio\b", r"\bpolicía\b", r"\bgobierno\b", r"\bministro\b", r"\bpresidente\b",
    ),
    "pt": (
        r"\banunci(?:a|ou|ado|ada|amento)\b", r"\bconfirm(?:a|ou|ado|ada|ação)\b", r"\bnome(?:a|ou|ado|ada|ação)\b",
        r"\bassina(?:la|ou|do|da|r)\b", r"\bentra(?:r|ou|do|da)\b", r"\bvence(?:u|r|dor|dora)\b", r"\bganh(?:a|ou|o|adora|ador)\b",
        r"\bderrot(?:a|ou|ado|ada)\b", r"\bmort(?:o|a|os|as)\b", r"\bmorte\b", r"\bpres(?:o|a|os|as)\b",
        r"\bpris(?:ão|oes)\b", r"\brenúncia\b", r"\brenuncia\b", r"\blanç(?:a|ou|ado|ada)\b", r"\bliber(?:a|ou|ado|ada)\b",
        r"\bacidente\b", r"\bterremoto\b", r"\btempestade\b", r"\bincêndio\b", r"\bincendio\b", r"\btiroteio\b", r"\bataque\b",
        r"\beleição(?:ões)?\b", r"\beleicao(?:oes)?\b", r"\bdecisão(?:ões)?\b", r"\bdecisao(?:oes)?\b", r"\btribunal\b", r"\bjulgamento\b", r"\bpolícia\b", r"\bpolicia\b", r"\bgoverno\b", r"\bministro\b", r"\bpresidente\b",
    ),
    "id": (
        r"\bpengumuman\b", r"\bkonfirmasi\b", r"\bpenunjukan\b", r"\bmenandatangani\b", r"\bbergabung\b",
        r"\bmenang\b", r"\bkemenangan\b", r"\bkalah\b", r"\bmembunuh\b", r"\btewas\b", r"\bkematian\b",
        r"\bditangkap\b", r"\bpenangkapan\b", r"\bmengundurkan diri\b", r"\bpeluncuran\b", r"\bdirilis\b",
        r"\bkecelakaan\b", r"\bgempa\b", r"\bbadai\b", r"\bkebakaran\b", r"\bpenembakan\b", r"\bserangan\b",
        r"\bpemilu\b", r"\bkeputusan\b", r"\bpengadilan\b", r"\bpolisi\b", r"\bpemerintah\b", r"\bmenteri\b", r"\bpresiden\b",
    ),
}

GEO_TERMS = {
    "US": ("united states", "u.s.", "u.s", "us", "america", "american", "usa"),
    "UK": ("united kingdom", "uk", "great britain", "britain", "british", "england", "scotland", "wales"),
    "DE": ("germany", "german", "deutschland", "deutsch"),
    "IT": ("italy", "italia", "italian", "italiano"),
    "FR": ("france", "french", "français", "francais"),
    "ES": ("spain", "españa", "espana", "spanish", "español", "espanol"),
    "PT": ("portugal", "portuguese", "português", "portugues"),
    "BR": ("brazil", "brasil", "brazilian", "brasileiro"),
    "ID": ("indonesia", "indonesian", "bahasa indonesia"),
    "CA": ("canada", "canadian"),
    "AU": ("australia", "australian"),
}

LANGUAGE_BY_COUNTRY = {
    "US": "en", "UK": "en", "CA": "en", "AU": "en",
    "DE": "de", "IT": "it", "FR": "fr", "ES": "es",
    "PT": "pt", "BR": "pt", "ID": "id",
}

def _localized_language() -> str:
    return LANGUAGE_BY_COUNTRY.get(CURRENT_COUNTRY, "en")

def _localized_patterns(mapping: Dict[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    language = _localized_language()
    localized = mapping.get(language, mapping["en"])
    if language == "en":
        return localized
    # Keep the original English signals active and add localized equivalents.
    # Trend titles can remain English even on non-English markets.
    return tuple(dict.fromkeys(mapping["en"] + localized))

# Backward-compatible English aliases for any existing imports.
SKIP_PATTERNS = SKIP_PATTERNS_BY_LANG["en"]
NOISE_PATTERNS = NOISE_PATTERNS_BY_LANG["en"]
NEWS_PATTERNS = NEWS_PATTERNS_BY_LANG["en"]

def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def _norm(text: Any) -> str:
    return _clean(text).casefold()

def _country_code(value: Any) -> str:
    text = _norm(value)
    if not text:
        return ""
    for code, aliases in COUNTRY_ALIASES.items():
        if text == code.casefold() or any(text == alias.casefold() for alias in aliases):
            return code
    return text.upper() if len(text) == 2 else ""

CURRENT_COUNTRY = _country_code(COUNTRY)
DEFAULT_PRIORITY_GEOS = (CURRENT_COUNTRY,) if CURRENT_COUNTRY else ("US", "UK")
_priority_geos_env = os.getenv("TOPIC_PRIORITY_GEOS", "")
PRIORITY_GEOS = tuple(
    x.strip().upper()
    for x in (_priority_geos_env.split(",") if _priority_geos_env else DEFAULT_PRIORITY_GEOS)
    if x.strip()
)

def _tokens(text: Any) -> Set[str]:
    return {
        t for t in re.findall(r"[a-z0-9']+", _norm(text))
        if len(t) >= 3
    }

def _traffic_value(trend: Dict[str, Any]) -> float:
    keys = (
        "traffic", "approx_traffic", "ht_approx_traffic",
        "approxTraffic", "traffic_value", "search_volume",
    )
    for key in keys:
        value = trend.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        m = re.search(r"([\d,.]+)\s*([KMB])?", str(value), re.I)
        if not m:
            continue
        try:
            n = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        suffix = (m.group(2) or "").upper()
        return n * {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix, 1)
    return 0.0

def _published_age_hours(trend: Dict[str, Any]) -> float | None:
    value = (
        trend.get("published")
        or trend.get("pubDate")
        or trend.get("published_at")
        or trend.get("updated")
    )
    if not value:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        # RSS / ISO formats used by feedparser and common trend feeds.
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(text)
        except Exception:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except Exception:
                return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600)

def _term_in_text(text: str, term: str) -> bool:
    # GEO terms include short tokens such as "uk" and "us". Require a
    # word/phrase boundary so entity names such as "chukwuemeka" cannot
    # accidentally become UK matches.
    pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None

def _geo_score(title: str, trend: Dict[str, Any] | None = None) -> Tuple[float, str]:
    t = _norm(title)

    # Controlled Indonesia discovery candidates belong to the ID market even
    # when the story itself is about an international entertainment entity.
    if CURRENT_COUNTRY == "ID" and isinstance(trend, dict) and trend.get("id_discovery"):
        return 15.0, "ID"

    # Strong explicit current production GEO preference, now tied to the
    # current market unless TOPIC_PRIORITY_GEOS explicitly overrides it.
    for geo in PRIORITY_GEOS:
        if any(_term_in_text(t, term) for term in GEO_TERMS.get(geo, ())):
            return 15.0, geo

    # If the current config country is explicitly represented in the title,
    # give the smaller local signal. This is now COUNTRY-aware across the
    # supported production markets rather than hard-coded to US/UK/CA/AU.
    if CURRENT_COUNTRY:
        local_terms = GEO_TERMS.get(CURRENT_COUNTRY, ())
        if any(_term_in_text(t, term) for term in local_terms):
            return 12.0, CURRENT_COUNTRY

    return 5.0, "GLOBAL"

# Routine weather/forecast is a hard topic exclusion.
# Genuine major/extreme weather events remain eligible.
ROUTINE_WEATHER_PATTERNS_BY_LANG = {
    "en": (
        r"\bweather\b", r"\bforecast\b", r"\bmeteo\b",
        r"\btemperature forecast\b", r"\btemperatures? (?:hit|reach|rise|fall|drop|soar)\b",
    ),
    "de": (
        r"\bwetter\b", r"\bwettervorhersage\b", r"\bwetterprognose\b",
        r"\btemperatur(?:en)?(?:prognose)?\b",
    ),
    "it": (
        r"\bmeteo\b", r"\bprevisioni? del tempo\b",
        r"\bprevisioni? della temperatura\b",
    ),
    "fr": (
        r"\bmétéo\b", r"\bmeteo\b", r"\bprévisions? météo\b",
        r"\bprevisions? meteo\b", r"\bprévisions? de température\b",
        r"\bprevisions? de temperature\b",
    ),
    "es": (
        r"\btiempo\b", r"\bpronóstico del tiempo\b", r"\bpronostico del tiempo\b",
        r"\bpronóstico de temperatura\b", r"\bpronostico de temperatura\b",
    ),
    "pt": (
        r"\btempo\b", r"\bprevisão do tempo\b", r"\bprevisao do tempo\b",
        r"\bprevisão de temperatura\b", r"\bprevisao de temperatura\b",
    ),
    "id": (
        r"\bcuaca\b", r"\bprakiraan cuaca\b", r"\bprakiraan suhu\b",
    ),
}

WEATHER_EVENT_PATTERNS_BY_LANG = {
    "en": (
        r"\bhurricane\b", r"\bcyclone\b", r"\btyphoon\b", r"\bflood(?:ing)?\b",
        r"\bwildfire\b", r"\bheat emergency\b", r"\bheatwave\b",
        r"\bweather disaster\b", r"\bmajor storm\b", r"\bsevere storm\b",
        r"\bstorm (?:hits|strikes|causes|leaves|kills|forces)\b",
    ),
    "de": (
        r"\bhurrikan\b", r"\bzyklon\b", r"\btaifun\b", r"\bhochwasser\b",
        r"\büberschwemmung\b", r"\bwaldbrand\b", r"\bhitzewelle\b",
        r"\bhitze-notstand\b", r"\bunwetter\b", r"\bschwerer sturm\b",
        r"\bstarker sturm\b",
    ),
    "it": (
        r"\buragano\b", r"\bciclone\b", r"\btaifone\b", r"\balluvion[ei]\b",
        r"\binondazion[ei]\b", r"\bincendio boschivo\b", r"\bondata di calore\b",
        r"\bemergenza caldo\b", r"\bforte tempesta\b", r"\btempesta grave\b",
    ),
    "fr": (
        r"\bouragan\b", r"\bcyclone\b", r"\btyphon\b", r"\binondation(?:s)?\b",
        r"\bincendie de forêt\b", r"\bcanicule\b", r"\burgence chaleur\b",
        r"\btempête majeure\b", r"\btempête sévère\b", r"\bforte tempête\b",
    ),
    "es": (
        r"\bhuracán\b", r"\bhuracan\b", r"\bciclón\b", r"\bciclon\b",
        r"\btsunami\b", r"\binundación(?:es)?\b", r"\bincendio forestal\b",
        r"\bola de calor\b", r"\bemergencia por calor\b",
        r"\btormenta grave\b", r"\btormenta severa\b",
    ),
    "pt": (
        r"\bfuracão\b", r"\bfuracao\b", r"\bciclone\b", r"\btaifão\b",
        r"\btaifao\b", r"\binundação(?:ões)?\b", r"\bincêndio florestal\b",
        r"\bonda de calor\b", r"\bemergência de calor\b",
        r"\btempestade severa\b", r"\bforte tempestade\b",
    ),
    "id": (
        r"\bbadai\b", r"\btopan\b", r"\bsiklon\b", r"\bbanjir\b",
        r"\bkebakaran hutan\b", r"\bgelombang panas\b",
        r"\bkeadaan darurat panas\b",
    ),
}

def _all_localized_patterns(mapping: Dict[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(
        pattern
        for patterns in mapping.values()
        for pattern in patterns
    ))

def _is_routine_weather(title: str) -> bool:
    t = _norm(title)
    weather = _all_localized_patterns(ROUTINE_WEATHER_PATTERNS_BY_LANG)
    events = _all_localized_patterns(WEATHER_EVENT_PATTERNS_BY_LANG)
    return (
        any(re.search(p, t, re.IGNORECASE) for p in weather)
        and not any(re.search(p, t, re.IGNORECASE) for p in events)
    )

def _noise_penalty(title: str) -> float:
    t = _norm(title)
    patterns = _localized_patterns(NOISE_PATTERNS_BY_LANG)
    if any(re.search(p, t, re.IGNORECASE) for p in patterns):
        return 30.0
    return 0.0

def _skip_reason(title: str) -> str | None:
    t = _norm(title)
    if _is_routine_weather(title):
        return "routine weather/forecast topic"
    if _is_sports_match_topic(title):
        return "concrete sports match/fixture topic"
    patterns = _localized_patterns(SKIP_PATTERNS_BY_LANG)
    for pattern in patterns:
        if pattern.startswith(" ") or pattern.endswith(" "):
            if pattern in t:
                return f"existing skip pattern: {pattern.strip()}"
        elif _term_in_text(t, pattern):
            return f"existing skip pattern: {pattern.strip()}"
    if _noise_penalty(title) >= 30:
        return "low-value utility/noise topic"
    return None

def _pre_score(trend: Dict[str, Any]) -> Dict[str, Any]:
    title = _clean(trend.get("title"))
    traffic = _traffic_value(trend)
    age = _published_age_hours(trend)

    # 0-30: trend momentum. If the feed exposes traffic, use it. Otherwise
    # use a neutral baseline plus a small news-language signal.
    if traffic > 0:
        momentum = min(30.0, 8.0 + 8.0 * math.log10(max(1.0, traffic)))
    else:
        momentum = 12.0

    # 0-20: freshness.
    if age is None:
        freshness = 12.0
    elif age <= 3:
        freshness = 20.0
    elif age <= 6:
        freshness = 18.0
    elif age <= 12:
        freshness = 16.0
    elif age <= 24:
        freshness = 13.0
    elif age <= 48:
        freshness = 8.0
    else:
        freshness = 3.0

    # ID discovery candidates are already scoped by rss.py to the controlled
    # Indonesian discovery case. Their headlines are often Korean/global
    # entertainment stories, so requiring the literal word "Indonesia" in the
    # headline would incorrectly classify them as GLOBAL.
    #
    # Preserve the normal GEO scoring for the generic pipeline, but explicitly
    # mark controlled ID discovery candidates as ID-local for ranking.
    if CURRENT_COUNTRY == "ID" and trend.get("id_discovery"):
        geo, geo_name = 15.0, "ID"
    else:
        geo, geo_name = _geo_score(title, trend)

    news_patterns = _localized_patterns(NEWS_PATTERNS_BY_LANG)
    newsworthiness = min(
        15.0,
        7.0 + 2.0 * sum(bool(re.search(p, _norm(title), re.IGNORECASE)) for p in news_patterns),
    )

    # Clear, entity-rich topics tend to be more usable than vague one-word
    # trends. This is a quality signal, not an article-fact claim.
    token_count = len(_tokens(title))
    clarity = min(10.0, 3.0 + token_count * 1.5)

    penalty = _noise_penalty(title)

    score = max(0.0, min(100.0, momentum + freshness + geo + newsworthiness + clarity - penalty))

    return {
        "title": title,
        "pre_score": round(score, 2),
        "momentum": round(momentum, 2),
        "freshness": round(freshness, 2),
        "geo": round(geo, 2),
        "geo_name": geo_name,
        "newsworthiness": round(newsworthiness, 2),
        "clarity": round(clarity, 2),
        "penalty": round(penalty, 2),
        "traffic": traffic,
        "age_hours": round(age, 2) if age is not None else None,
    }

def rank_trends(trends: Sequence[Dict[str, Any]], processed: Iterable[str], limit: int) -> List[Dict[str, Any]]:
    """
    Cheap first-pass ranking.

    This function does NOT fetch news and does NOT invoke Ollama.
    """
    processed_norm = {_norm(x) for x in processed}
    scored: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()

    for trend in trends:
        if not isinstance(trend, dict):
            continue

        title = _clean(trend.get("title"))
        if not title:
            continue

        norm_title = _norm(title)
        if norm_title in processed_norm or norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)

        reason = _skip_reason(title)
        if reason:
            print(f"[TOPIC FILTER] DROP | {title} | {reason}")
            continue

        data = _pre_score(trend)

        # pre_score is PRIORITY ONLY. There is no numeric admission threshold.
        # Deterministic editorial exclusions above remain the actual topic
        # eligibility gate; evidence/factual/editorial gates decide production.
        item = dict(trend)
        item["_topic"] = data
        scored.append(item)

    scored.sort(key=lambda x: x["_topic"]["pre_score"], reverse=True)

    # pre_score is used only to order otherwise eligible topics. The caller
    # controls capacity through `limit`; there is no score-based admission gate,
    # tier gate, fallback gate, or hidden retrieval cap here.
    selected = scored[:max(0, limit)]

    print(
        f"[TOPIC FILTER] QUALITY GATE | deterministic_eligible={len(scored)} "
        f"| score_gate=DISABLED | score_used=PRIORITY_ONLY"
    )

    print(
        f"[TOPIC FILTER] {len(trends)} trends -> "
        f"{len(scored)} viable -> {len(selected)} candidates before news retrieval"
    )

    for rank, item in enumerate(selected, 1):
        d = item["_topic"]
        print(
            f"[TOPIC FILTER] #{rank:02d} "
            f"{d['pre_score']:5.1f} | {d['geo_name']:6s} | {d['title']}"
        )

    return selected

def _normalize_topic_match_text(text: Any) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    text = re.sub(r"[-–—_/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


GERMAN_STATES = (
    "baden wuerttemberg", "bayern", "berlin", "brandenburg", "bremen",
    "hamburg", "hessen", "mecklenburg vorpommern", "niedersachsen",
    "nordrhein westfalen", "rheinland pfalz", "saarland", "sachsen",
    "sachsen anhalt", "schleswig holstein", "thueringen",
)


def _topic_state(topic: Any) -> str:
    text = _normalize_topic_match_text(topic)
    for state in sorted(GERMAN_STATES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(state)}(?!\w)", text):
            return state
    return ""


def _topic_anchor_tokens(topic: Any) -> Set[str]:
    text = _normalize_topic_match_text(topic)
    stop = {
        "wahl", "wahlprognose", "prognose", "umfrage", "nachrichten", "news",
        "latest", "today", "update", "bericht", "berlin", "deutschland",
        "germany", "german", "deutsch", "gala", "live", "ergebnis",
        "entscheidung", "bombenentscharfung", "entscharfung",
    }
    return {
        token for token in re.findall(r"[a-z0-9]+", text)
        if len(token) >= 4 and token not in stop
    }


def filter_relevant_news(trend: Dict[str, Any], news: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic topic/news gate; regional identity has priority."""
    topic = _clean(trend.get("title"))
    state = _topic_state(topic)
    anchors = _topic_anchor_tokens(topic)
    valid_news = [x for x in news if isinstance(x, dict)]

    relevant = []
    for item in valid_news:
        haystack = _normalize_topic_match_text(" ".join([
            str(item.get("title", "")),
            str(item.get("summary", "")),
            str(item.get("content", ""))[:4000],
        ]))

        # If the topic names a German federal state, the state is a hard
        # geographic identity requirement. This is the exact protection needed
        # for Sachsen-Anhalt vs Mecklenburg-Vorpommern.
        if state:
            if state not in haystack:
                continue
            relevant.append(item)
            continue

        # Multi-word named entities (e.g. Katja Kipping): require all name
        # tokens, avoiding unrelated stories that happen to share one word.
        if len(anchors) >= 2:
            if all(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", haystack) for a in anchors):
                relevant.append(item)
            continue

        # Single distinctive entity/topic anchor.
        if anchors and any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", haystack) for a in anchors):
            relevant.append(item)
            continue

        # Generic topics are intentionally left to Story Concentration.
        if not anchors:
            relevant.append(item)

    print(
        f"[TOPIC FILTER] TOPIC/NEWS RELEVANCE | topic={topic} | "
        f"anchors={','.join(sorted(anchors)) or '-'} | "
        f"state={state or '-'} | kept={len(relevant)}/{len(valid_news)}"
    )
    return relevant


def score_with_news(trend: Dict[str, Any], news: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Second-stage sourceability score.

    Uses only the news already fetched for this candidate.
    No additional network calls and no LLM.
    """
    base = dict(trend.get("_topic") or _pre_score(trend))
    news = [x for x in news if isinstance(x, dict)]

    count = len(news)

    def _source_identity(item: Dict[str, Any]) -> str:
        # Prefer publisher/domain identity when the fetcher exposes a URL.
        # This collapses www/mobile/news subdomains conservatively while
        # keeping unrelated domains independent. Fall back to the existing
        # source name when no URL is available.
        # Prefer the publisher URL exposed by Google News RSS. The item
        # `link` is normally a news.google.com redirect and must never be
        # used as the publisher identity when a real publisher URL exists.
        url = _clean(
            item.get("source_url")
            or item.get("sourceUrl")
            or item.get("source_href")
            or item.get("url")
            or item.get("link")
        )
        if url:
            try:
                host = (urlparse(url).hostname or "").casefold().strip(".")
            except Exception:
                host = ""
            if host:
                host = re.sub(r"^(?:www|m|mobile|amp)\.", "", host)
                return f"domain:{host}"

        source = _norm(item.get("source"))
        source = re.sub(r"[^a-z0-9]+", " ", source).strip()
        return f"name:{source}" if source else ""

    source_names = {
        identity
        for identity in (_source_identity(x) for x in news)
        if identity
    }
    content_count = sum(1 for x in news if len(_clean(x.get("content"))) >= 700)

    # Keep the existing 0-20 sourceability formula intact, but make the
    # independent-source component reflect publisher/domain identity rather
    # than raw source-name spelling. This improves independence estimation
    # without changing the scoring architecture or adding network/LLM work.
    sourceability = min(20.0, count * 2.0 + min(5.0, len(source_names)) + min(5.0, content_count))

    # Strong penalty for a topic that has only one weak/duplicate source.
    evidence_penalty = 0.0
    if count < 2:
        # A controlled ID discovery candidate carries a fresh Google News seed
        # from rss.py. Do not treat that seed as an ordinary generic topic with
        # a hard single-source collapse; Fact Guard still remains the factual
        # gate after generation. Keep the normal penalty for every other market.
        if not (CURRENT_COUNTRY == "ID" and trend.get("id_discovery")):
            evidence_penalty = 30.0
    elif len(source_names) <= 1:
        evidence_penalty = 6.0
    elif content_count == 0:
        evidence_penalty = 5.0

    final_score = max(
        0.0,
        min(
            100.0,
            base["pre_score"] * 0.78
            + sourceability
            - evidence_penalty,
        ),
    )

    result = {
        **base,
        "news_count": count,
        "unique_sources": len(source_names),
        "usable_content": content_count,
        "sourceability": round(sourceability, 2),
        "evidence_penalty": round(evidence_penalty, 2),
        "final_score": round(final_score, 2),
        "id_discovery": bool(trend.get("id_discovery")),
    }
    return result

def select_final_candidates(
    candidates: Sequence[Dict[str, Any]],
    limit: int,
    min_score: float | None = None,
) -> List[Dict[str, Any]]:
    # final_score is PRIORITY ONLY. No numeric score blocks a candidate after
    # news retrieval. Evidence, Fact Guard and Story Quality are the substantive
    # production gates. `min_score` is retained in the signature for compatibility
    # but is intentionally ignored.
    viable = list(candidates)

    viable.sort(
        key=lambda x: (
            x["_topic_final"]["final_score"],
            x["_topic_final"].get("sourceability", 0),
            x["_topic_final"].get("freshness", 0),
        ),
        reverse=True,
    )

    selected = viable[:max(0, limit)]

    print(
        f"[TOPIC FILTER] {len(candidates)} candidates -> "
        f"{len(viable)} viable after source scoring -> "
        f"{len(selected)} article slots"
    )

    for rank, item in enumerate(selected, 1):
        d = item["_topic_final"]
        print(
            f"[TOPIC FILTER] FINAL #{rank:02d} "
            f"{d['final_score']:5.1f} | {d['geo_name']:6s} | "
            f"sources={d['unique_sources']} | {d['title']}"
        )

    return selected
