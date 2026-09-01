"""
TrendCurrent AI image backend: Cloudflare Workers AI / FLUX.1 Schnell.

Drop-in replacement for ai_image_hf.py.

Environment:
    CLOUDFLARE_ACCOUNT_ID   required
    CLOUDFLARE_API_TOKEN    required
    TC_AI_IMAGE_ENABLED     default "1"
    TC_AI_IMAGE_STEPS       default "4"
    TC_AI_IMAGE_TIMEOUT     default "180"
    TC_AI_IMAGE_WEBP_QUALITY default "80"
    CLOUDFLARE_ACCOUNT_ID_2 optional second Cloudflare account ID
    CLOUDFLARE_API_TOKEN_2 optional second Cloudflare API token
    CLOUDFLARE_ACCOUNT_ID_3 optional third Cloudflare account ID
    CLOUDFLARE_API_TOKEN_3 optional third Cloudflare API token

The image layer is isolated from article generation.
It receives the article headline as the primary visual source, with description/intro
as secondary context. Named people are never used as visual subjects.
Cloudflare's FLUX endpoint does not accept width/height, so the returned
image is prepared locally as a natural 3:2, 1024x683 WebP format.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
import urllib.parse
import io
import re
from pathlib import Path
from typing import Any

from PIL import Image


from config import ROOT, SITE_URL


# Cloudflare account pool.
# All three accounts are required for production rotation. Do not silently
# continue with only one account: that would defeat quota failover.
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

ACCOUNT_ID_2 = os.getenv("CLOUDFLARE_ACCOUNT_ID_2", "").strip()
API_TOKEN_2 = os.getenv("CLOUDFLARE_API_TOKEN_2", "").strip()

ACCOUNT_ID_3 = os.getenv("CLOUDFLARE_ACCOUNT_ID_3", "").strip()
API_TOKEN_3 = os.getenv("CLOUDFLARE_API_TOKEN_3", "").strip()

ENABLED = os.getenv("TC_AI_IMAGE_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}

WIDTH = 1024
HEIGHT = 683
STEPS = int(os.getenv("TC_AI_IMAGE_STEPS", "4"))
QUALITY = int(os.getenv("TC_AI_IMAGE_WEBP_QUALITY", "80"))
TIMEOUT = int(os.getenv("TC_AI_IMAGE_TIMEOUT", "180"))

MODEL = "@cf/black-forest-labs/flux-1-schnell"



IMAGE_DIR = Path(ROOT) / "assets" / "articles"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _clean(text: Any, limit: int = 1200) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit]


def _infer_visual_lock(
    title: str,
    description: str,
    intro: str,
) -> dict[str, str]:
    """Infer a deterministic, multi-signal visual subject lock.

    The classifier is intentionally local: it consumes no AI/Cloudflare neurons.
    It scores multiple visual domains instead of using "first keyword wins".

    Output:
        type       = stable story domain
        label      = compact visual-domain label for logging/prompting
        subject    = concrete visual direction selected from the article
        forbidden  = category-specific negative visual directions

    Design goals:
    - headline gets the strongest weight
    - description/intro provide supporting evidence
    - multi-word phrases beat isolated generic words
    - multiple matching domains are resolved by weighted score
    - concrete sub-subjects are selected where safely inferable
    - named people remain context, never the visual subject
    - broad roundups intentionally fall back to a broad but coherent visual anchor
    """

    title = _clean(title, 320)
    description = _clean(description, 500)
    intro = _clean(intro, 500)

    title_text = title.lower()
    desc_text = description.lower()
    intro_text = intro.lower()
    full_text = " ".join((title_text, desc_text, intro_text))

    # Each rule:
    # (story_type, label, broad_subject, forbidden, phrase_weights, word_weights)
    #
    # Phrase weights are deliberately higher than single-word weights.
    # This prevents generic words such as "final", "show", "market", "study",
    # "court" or "policy" from dominating a story merely because they appear once.
    rules = [
        {
            "type": "weather",
            "label": "WEATHER / ATMOSPHERIC EVENT",
            "broad_subject": (
                "the specific weather condition or atmospheric event described by "
                "the article, shown through its affected environment, landscape, "
                "infrastructure or sky"
            ),
            "forbidden": (
                "weather presenters, generic people, unrelated sports, unrelated "
                "events, generic city stock imagery"
            ),
            "phrases": {
                "tropical storm": 10, "tropical cyclone": 10, "winter storm": 10,
                "severe weather": 9, "heat wave": 9, "cold snap": 9,
                "flash flood": 10, "flood warning": 9, "storm surge": 10,
                "storm warning": 9, "weather warning": 9, "hurricane warning": 10,
                "tornado warning": 10, "blizzard warning": 10,
                "heavy rainfall": 8, "record rainfall": 8,
            },
            "words": {
                "hurricane": 7, "tornado": 7, "thunderstorm": 7, "blizzard": 7,
                "snowfall": 6, "flooding": 6, "flood": 5, "wildfire": 6,
                "drought": 6, "heatwave": 6, "rainfall": 5, "snow": 4,
            },
        },
        {
            "type": "sports",
            "label": "SPORT / MATCH",
            "broad_subject": (
                "the specific sport, playing surface, equipment, venue, match or "
                "competition identified by the article"
            ),
            "forbidden": (
                "unrelated sports, generic business scenes, generic celebrity "
                "portraits, unrelated people"
            ),
            "phrases": {
                "tennis match": 10, "football match": 10, "soccer match": 10,
                "basketball game": 10, "baseball game": 10, "hockey game": 10,
                "golf tournament": 10, "grand prix": 10, "formula 1": 10,
                "formula one": 10, "world cup": 9, "champions league": 10,
                "playoff game": 9, "playoff series": 9, "final match": 8,
                "quarterfinal match": 9, "semifinal match": 9,
            },
            "words": {
                "tennis": 7, "football": 7, "soccer": 7, "basketball": 7,
                "baseball": 7, "hockey": 7, "golf": 7, "cricket": 7,
                "rugby": 7, "boxing": 7, "ufc": 7, "marathon": 6,
                "tournament": 5, "playoffs": 5, "playoff": 5, "championship": 5,
                "race": 4, "match": 4, "game": 2,
            },
        },
        {
            "type": "theatre",
            "label": "THEATRE / STAGE",
            "broad_subject": (
                "the theatre production, stage, theatrical set, venue interior, "
                "stage lighting or clearly theatrical production elements"
            ),
            "forbidden": (
                "tennis, football, basketball, baseball, stadiums, athletes, "
                "unrelated sports, generic celebrity portraits, generic people"
            ),
            "phrases": {
                "west end": 10, "broadway musical": 10, "theatre production": 10,
                "theater production": 10, "stage production": 9,
                "musical production": 9, "theatrical production": 10,
            },
            "words": {
                "theatre": 7, "theater": 7, "broadway": 7, "stage": 6,
                "musical": 6, "production": 3, "revival": 5, "barbican": 6,
                "cast": 3,
            },
        },
        {
            "type": "court",
            "label": "COURT / LEGAL PROCEEDING",
            "broad_subject": (
                "a concrete courtroom, legal hearing, jury setting, judge's bench, "
                "court building or legal proceeding environment described by the story"
            ),
            "forbidden": (
                "portraits of named people, unrelated sports, generic offices, "
                "generic crowds, unrelated government scenes"
            ),
            "phrases": {
                "court ruling": 9, "court decision": 9, "court hearing": 10,
                "court case": 9, "court battle": 9, "legal battle": 8,
                "supreme court": 10, "court of appeals": 10,
                "criminal trial": 10, "civil trial": 10, "jury trial": 10,
                "lawsuit against": 8, "lawsuit over": 8,
            },
            "words": {
                "courtroom": 8, "court": 5, "jury": 7, "trial": 7, "verdict": 7,
                "lawsuit": 6, "hearing": 6, "judge": 6, "sentenced": 7,
                "convicted": 7, "acquitted": 7, "litigation": 6,
            },
        },
        {
            "type": "politics",
            "label": "GOVERNMENT / POLITICS",
            "broad_subject": (
                "the relevant government building, parliament, official chamber, "
                "ballot box, election setting, campaign environment, policy setting "
                "or other concrete political setting described by the article"
            ),
            "forbidden": (
                "portraits of named politicians, unrelated sports, generic business "
                "meetings, generic people, unrelated entertainment"
            ),
            "phrases": {
                "midterm elections": 12, "midterm election": 12,
                "presidential election": 12, "general election": 11,
                "election campaign": 11, "political campaign": 11,
                "campaign funding": 10, "campaign finance": 10,
                "election results": 10, "election race": 10,
                "white house": 12, "u.s. capitol": 12, "us capitol": 12,
                "house of representatives": 11, "senate race": 11,
                "senate election": 11, "congressional race": 11,
                "congressional election": 11, "trade policy": 8,
                "foreign policy": 8, "executive order": 9,
            },
            "words": {
                "president": 6, "presidential": 6, "prime": 3, "minister": 6,
                "government": 6, "parliament": 7, "congress": 7, "senate": 7,
                "election": 7, "vote": 4, "voting": 5, "political": 5,
                "policy": 4, "cabinet": 6, "legislation": 6, "republican": 5,
                "democrat": 5, "midterms": 8, "pac": 5,
            },
        },
        {
            "type": "business",
            "label": "BUSINESS / FINANCE",
            "broad_subject": (
                "the specific business, financial market, transaction, workplace, "
                "store, factory, bank or physical business setting described"
            ),
            "forbidden": (
                "generic corporate handshakes, generic meetings, unrelated sports, "
                "portraits, unrelated political scenes"
            ),
            "phrases": {
                "stock market": 11, "financial markets": 10, "share price": 9,
                "stock prices": 9, "quarterly earnings": 10, "earnings report": 10,
                "merger agreement": 10, "acquisition deal": 10,
                "business deal": 8, "interest rates": 9, "central bank": 10,
                "banking crisis": 10, "market crash": 11, "economic growth": 8,
            },
            "words": {
                "company": 4, "corporate": 5, "business": 5, "shares": 6,
                "stock": 6, "stocks": 6, "market": 4, "markets": 4,
                "earnings": 6, "revenue": 6, "profit": 6, "acquisition": 7,
                "merger": 7, "investment": 6, "investor": 6, "bank": 5,
                "banking": 6, "economy": 5, "economic": 5, "inflation": 6,
            },
        },
        {
            "type": "transport",
            "label": "TRANSPORT / INFRASTRUCTURE",
            "broad_subject": (
                "the specific aircraft, train, airport, railway, road, vehicle or "
                "transport infrastructure involved in the story"
            ),
            "forbidden": (
                "generic people, unrelated sports, generic city stock imagery, "
                "unrelated business scenes"
            ),
            "phrases": {
                "flight cancellations": 10, "flight delays": 10,
                "travel disruption": 10, "air traffic": 9, "train service": 9,
                "rail service": 9, "road closure": 10, "highway closure": 10,
                "traffic accident": 9, "car crash": 10, "plane crash": 10,
                "train crash": 10,
            },
            "words": {
                "flight": 5, "flights": 5, "airline": 6, "airport": 7,
                "aircraft": 7, "plane": 6, "train": 7, "railway": 7,
                "rail": 6, "subway": 7, "metro": 7, "bus": 5, "road": 4,
                "highway": 6, "traffic": 5, "crash": 6, "collision": 7,
                "vehicle": 4, "transport": 5,
            },
        },
        {
            "type": "science_space",
            "label": "SCIENCE / SPACE",
            "broad_subject": (
                "the specific scientific subject, spacecraft, laboratory, instrument, "
                "experiment or astronomical phenomenon described"
            ),
            "forbidden": (
                "generic scientists posing, unrelated people, unrelated sports, "
                "generic offices, unrelated technology stock imagery"
            ),
            "phrases": {
                "space mission": 10, "rocket launch": 10, "lunar mission": 10,
                "mars mission": 10, "climate study": 9, "scientific study": 9,
                "medical study": 8, "research team": 7, "space telescope": 10,
                "deep space": 9,
            },
            "words": {
                "nasa": 8, "space": 7, "rocket": 7, "launch": 5, "satellite": 7,
                "astronaut": 7, "moon": 6, "mars": 7, "planet": 6,
                "telescope": 7, "mission": 4, "scientist": 5, "research": 6,
                "study": 4, "laboratory": 7, "lab": 6, "experiment": 6,
                "discovery": 4,
            },
        },
        {
            "type": "technology",
            "label": "TECHNOLOGY / DIGITAL",
            "broad_subject": (
                "the specific device, software concept, computing hardware, digital "
                "interface or technology environment described"
            ),
            "forbidden": (
                "generic people using laptops, unrelated sports, generic office "
                "scenes, unrelated business meetings"
            ),
            "phrases": {
                "artificial intelligence": 11, "ai model": 9, "machine learning": 10,
                "social media": 9, "smartphone app": 9, "mobile app": 9,
                "cyber attack": 10, "cyberattack": 10, "data breach": 10,
                "computer chip": 10, "semiconductor chip": 10,
                "operating system": 9, "generative ai": 11,
            },
            "words": {
                "ai": 5, "software": 6, "app": 5, "iphone": 7, "android": 7,
                "google": 5, "microsoft": 5, "apple": 5, "chip": 7,
                "processor": 7, "robot": 6, "robotics": 7, "cyber": 6,
                "internet": 5, "technology": 5, "tech": 4, "computer": 6,
                "smartphone": 7,
            },
        },
        {
            "type": "disaster",
            "label": "DISASTER / EMERGENCY",
            "broad_subject": (
                "the specific disaster, damaged environment, emergency infrastructure "
                "or response setting, without graphic injury"
            ),
            "forbidden": (
                "graphic injury, gore, unrelated sports, generic people, unrelated "
                "city stock imagery"
            ),
            "phrases": {
                "natural disaster": 10, "emergency response": 9,
                "mass evacuation": 10, "building collapse": 10,
                "structural collapse": 10, "rescue operation": 9,
                "search and rescue": 9, "volcanic eruption": 11,
                "forest fire": 10, "house fire": 10,
            },
            "words": {
                "earthquake": 8, "wildfire": 8, "volcanic": 7, "volcano": 8,
                "eruption": 8, "disaster": 6, "emergency": 6, "evacuation": 7,
                "rescue": 6, "explosion": 7, "collapse": 7, "fire": 5,
            },
        },
        {
            "type": "entertainment",
            "label": "ENTERTAINMENT",
            "broad_subject": (
                "the specific film, television, music, concert, award, exhibition "
                "or entertainment production setting described by the article"
            ),
            "forbidden": (
                "portraits or likenesses of named people, unrelated sports, generic "
                "crowds, unrelated business or political scenes"
            ),
            "phrases": {
                "red carpet": 10, "film premiere": 10, "movie premiere": 10,
                "music festival": 10, "award ceremony": 10, "awards ceremony": 10,
                "television series": 9, "tv series": 9, "streaming series": 9,
                "box office": 9, "concert tour": 9,
            },
            "words": {
                "film": 5, "movie": 6, "cinema": 6, "actor": 5, "actress": 5,
                "singer": 5, "album": 6, "song": 5, "concert": 7, "music": 5,
                "premiere": 7, "television": 5, "series": 5, "show": 3,
                "grammy": 7, "oscars": 7, "award": 5, "awards": 5,
            },
        },
    ]

    def occurrences(text: str, term: str) -> int:
        # Word-boundary matching for short/general words. Multi-word phrases
        # are naturally matched as exact normalized substrings.
        if " " in term or "-" in term:
            return text.count(term)
        return len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", text))

    def score_rule(rule: dict) -> tuple[float, int, int]:
        score = 0.0
        phrase_hits = 0
        word_hits = 0

        # Headline is intentionally dominant because it normally represents the
        # article's actual central story better than incidental body context.
        for term, weight in rule["phrases"].items():
            hits = occurrences(title_text, term)
            if hits:
                score += hits * weight * 3.0
                phrase_hits += hits
            hits = occurrences(desc_text, term)
            if hits:
                score += hits * weight * 1.35
                phrase_hits += hits
            hits = occurrences(intro_text, term)
            if hits:
                score += hits * weight
                phrase_hits += hits

        for term, weight in rule["words"].items():
            hits = occurrences(title_text, term)
            if hits:
                score += hits * weight * 2.0
                word_hits += hits
            hits = occurrences(desc_text, term)
            if hits:
                score += hits * weight * 1.0
                word_hits += hits
            hits = occurrences(intro_text, term)
            if hits:
                score += hits * weight * 0.75
                word_hits += hits

        return score, phrase_hits, word_hits

    scored = []
    for rule in rules:
        score, phrase_hits, word_hits = score_rule(rule)
        if score > 0:
            scored.append((score, phrase_hits, word_hits, rule))

    # No recognized domain: retain the safe general route.
    if not scored:
        return {
            "type": "general",
            "label": "GENERAL NEWS",
            "subject": (
                "the most concrete non-person object, place, event, physical setting "
                "or visible consequence explicitly described by the headline"
            ),
            "forbidden": (
                "generic stock people, generic business meetings, generic conferences, "
                "unrelated sports, unrelated celebrities, unrelated scenes"
            ),
        }

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_score, best_phrases, best_words, best_rule = scored[0]

    # If two domains are close, prefer the one with stronger headline evidence.
    # This is deliberately deterministic and avoids arbitrary rule-order wins.
    if len(scored) > 1:
        second_score, second_phrases, second_words, second_rule = scored[1]
        if second_score >= best_score * 0.88:
            # Headline-only evidence gets priority in close cases.
            best_headline_score = (
                sum(
                    occurrences(title_text, term) * weight * 3.0
                    for term, weight in best_rule["phrases"].items()
                )
                + sum(
                    occurrences(title_text, term) * weight * 2.0
                    for term, weight in best_rule["words"].items()
                )
            )
            second_headline_score = (
                sum(
                    occurrences(title_text, term) * weight * 3.0
                    for term, weight in second_rule["phrases"].items()
                )
                + sum(
                    occurrences(title_text, term) * weight * 2.0
                    for term, weight in second_rule["words"].items()
                )
            )

            if second_headline_score > best_headline_score:
                best_rule = second_rule
                best_score = second_score
                best_phrases = second_phrases
                best_words = second_words

    story_type = best_rule["type"]
    label = best_rule["label"]
    subject = best_rule["broad_subject"]
    forbidden = best_rule["forbidden"]

    # Concrete visual-subject refinements. These are intentionally conservative:
    # only use them when strong phrases in the article support them.
    if story_type == "politics":
        if occurrences(full_text, "white house") > 0:
            subject = (
                "the White House and its surrounding official government setting, "
                "with architecture and grounds as the dominant visual subject"
            )
        elif occurrences(full_text, "u.s. capitol") > 0 or occurrences(full_text, "us capitol") > 0:
            subject = (
                "the U.S. Capitol and surrounding congressional setting, with the "
                "Capitol building as the dominant visual subject"
            )
        elif occurrences(full_text, "house of representatives") > 0:
            subject = (
                "the U.S. House of Representatives / congressional chamber setting, "
                "with government architecture as the dominant visual subject"
            )
        elif occurrences(full_text, "senate") > 0 and (
            occurrences(full_text, "race") > 0
            or occurrences(full_text, "election") > 0
            or occurrences(full_text, "elections") > 0
        ):
            subject = (
                "a U.S. Senate election or congressional setting, represented through "
                "the Capitol, Senate chamber or official election environment"
            )
        elif (
            occurrences(full_text, "election campaign") > 0
            or occurrences(full_text, "political campaign") > 0
            or occurrences(full_text, "campaign funding") > 0
            or occurrences(full_text, "campaign finance") > 0
            or occurrences(full_text, "midterm elections") > 0
            or occurrences(full_text, "midterm election") > 0
        ):
            subject = (
                "a U.S. election campaign setting, such as campaign infrastructure, "
                "ballot boxes, polling-place environment or official election signage "
                "without readable text"
            )

    elif story_type == "weather":
        if occurrences(full_text, "hurricane") or occurrences(full_text, "tropical storm"):
            subject = (
                "the hurricane or tropical storm itself, shown through the affected "
                "coastline, buildings, infrastructure, rain, wind and dramatic sky"
            )
        elif occurrences(full_text, "tornado"):
            subject = (
                "the tornado and affected landscape or infrastructure, shown safely "
                "without people or graphic damage"
            )
        elif occurrences(full_text, "flood"):
            subject = (
                "the flooding and its affected streets, buildings, roads or landscape, "
                "with water and environmental impact as the dominant subject"
            )
        elif occurrences(full_text, "snow") or occurrences(full_text, "blizzard"):
            subject = (
                "the snow or blizzard conditions affecting the described environment, "
                "roads, buildings or landscape"
            )

    elif story_type == "sports":
        sport_subjects = [
            ("tennis", "a tennis-specific court, racket, net and match environment"),
            ("football", "a football-specific pitch, stadium or match environment"),
            ("soccer", "a soccer-specific pitch, stadium or match environment"),
            ("basketball", "a basketball-specific court, arena or game environment"),
            ("baseball", "a baseball-specific field, stadium or game environment"),
            ("hockey", "an ice hockey-specific rink, goal or game environment"),
            ("golf", "a golf course and golf-specific competition environment"),
            ("cricket", "a cricket-specific pitch, stadium or match environment"),
            ("rugby", "a rugby-specific pitch, stadium or match environment"),
            ("boxing", "a boxing ring and boxing-specific competition environment"),
            ("formula 1", "a Formula 1 racing circuit, car and race environment"),
            ("formula one", "a Formula 1 racing circuit, car and race environment"),
        ]
        for term, refined in sport_subjects:
            if occurrences(full_text, term):
                subject = refined
                break

    elif story_type == "transport":
        if occurrences(full_text, "airport") or occurrences(full_text, "flight"):
            subject = (
                "the specific airport, aircraft or aviation environment involved in "
                "the story, with aviation infrastructure as the dominant subject"
            )
        elif occurrences(full_text, "train") or occurrences(full_text, "rail"):
            subject = (
                "the specific train, railway or rail infrastructure involved in the "
                "story, shown as a concrete transport setting"
            )
        elif occurrences(full_text, "road") or occurrences(full_text, "highway"):
            subject = (
                "the specific road or highway infrastructure and its described traffic "
                "or disruption, without generic city stock imagery"
            )

    elif story_type == "science_space":
        if occurrences(full_text, "nasa") or occurrences(full_text, "rocket launch"):
            subject = (
                "the specific NASA space mission, rocket or launch environment described "
                "by the story, with the spacecraft or launch infrastructure dominant"
            )
        elif occurrences(full_text, "mars"):
            subject = (
                "the Mars-related scientific or space subject described by the story, "
                "represented through spacecraft, planetary terrain or scientific equipment"
            )
        elif occurrences(full_text, "laboratory") or occurrences(full_text, "lab"):
            subject = (
                "the specific laboratory, scientific equipment or research environment "
                "described by the article"
            )

    elif story_type == "technology":
        if occurrences(full_text, "artificial intelligence") or occurrences(full_text, "generative ai"):
            subject = (
                "the specific artificial-intelligence technology or computing environment "
                "described by the story, shown through relevant hardware or digital systems"
            )
        elif occurrences(full_text, "cyber attack") or occurrences(full_text, "cyberattack") or occurrences(full_text, "data breach"):
            subject = (
                "the cybersecurity or data-security environment described by the story, "
                "shown through computing infrastructure and security systems"
            )

    elif story_type == "disaster":
        if occurrences(full_text, "earthquake"):
            subject = (
                "the earthquake's affected built environment, infrastructure or landscape, "
                "without graphic injury"
            )
        elif occurrences(full_text, "wildfire") or occurrences(full_text, "forest fire"):
            subject = (
                "the wildfire and affected landscape, smoke, vegetation or infrastructure, "
                "without graphic injury"
            )
        elif occurrences(full_text, "volcanic eruption"):
            subject = (
                "the volcanic eruption, volcano and affected landscape, shown without "
                "graphic injury"
            )

    elif story_type == "court":
        if occurrences(full_text, "supreme court"):
            subject = (
                "the U.S. Supreme Court building and formal judicial setting, with "
                "architecture and institutional context as the dominant subject"
            )
        elif occurrences(full_text, "courtroom") or occurrences(full_text, "court hearing"):
            subject = (
                "a courtroom or formal legal hearing setting, with the judge's bench, "
                "jury area and legal environment as the dominant subject"
            )

    # If the winning domain is supported only weakly, retain its broad subject
    # rather than fabricating a more specific object.
    if best_score < 8 and best_phrases == 0 and best_words <= 1:
        story_type = "general"
        label = "GENERAL NEWS"
        subject = (
            "the most concrete non-person object, place, event, physical setting "
            "or visible consequence explicitly described by the headline"
        )
        forbidden = (
            "generic stock people, generic business meetings, generic conferences, "
            "unrelated sports, unrelated celebrities, unrelated scenes"
        )

    return {
        "type": story_type,
        "label": label,
        "subject": subject,
        "forbidden": forbidden,
    }


def _build_prompt(article: dict, news: list | None = None) -> str:
    """Build a locked, headline-first visual prompt within Cloudflare's 2048-char limit.

    A lightweight story-type classifier creates a visual subject lock before FLUX
    generation. The classifier is local and consumes no AI/Cloudflare neurons.
    """

    title = _clean(article.get("title") or article.get("h1"), 260)
    description = _clean(article.get("description"), 300)
    intro = _clean(
        article.get("intro")
        or (article.get("paragraphs") or [""])[0],
        100,
    )

    if not title:
        title = "the central subject or event described by the article"

    lock = _infer_visual_lock(title, description, intro)

    prompt = f"""
Create ONE photorealistic editorial news photograph in a natural 3:2 composition.

ARTICLE HEADLINE — PRIMARY SOURCE:
{title}

STORY TYPE — VISUAL SUBJECT LOCK:
{lock["label"]}

PRIMARY VISUAL SUBJECT — LOCKED:
{lock["subject"]}

This visual lock is mandatory. Build the image around the PRIMARY VISUAL SUBJECT.
Do not reinterpret the story into another category. Do not substitute a generic stock
image. The image must be immediately recognizable as belonging to the locked story type.

STRICT SUBJECT CONTROL:
- Depict the concrete non-person subject, place, object, event or setting described.
- If the headline names a person, the person is CONTEXT ONLY, not the visual subject.
- Do not create a portrait, likeness, celebrity recreation or recognizable face of a named person.
- People are NOT allowed merely to make the image look like a news photograph.
- If people are not essential to the locked subject, use ZERO HUMAN FIGURES.
- Never introduce an unrelated human activity or unrelated visual category.

LOCKED NEGATIVE SUBJECTS:
{lock["forbidden"]}

COMBINED HEADLINES:
If the headline contains a secondary condition such as weather, delays, disruption or
cancellation, keep the LOCKED PRIMARY SUBJECT dominant. The secondary condition may
support it but must never replace it.

EXAMPLE:
If the story is a theatre production, show the theatre stage, set, venue or production
environment. Do NOT turn a theatre story into a sports scene, city stock photo or portrait.
If the story is a tennis match, show tennis-specific court/equipment/action. Do NOT replace
it with another sport or generic people.

SECONDARY ARTICLE CONTEXT:
{description}
{intro}

Use secondary context only to clarify the locked subject. Do not allow names of people,
secondary stories or incidental details to override the visual subject lock.
Create ONE specific, coherent real-world scene directly representing the article.
Do not invent unsupported facts, objects, locations or activities.

PEOPLE RULE:
No named person may be depicted. No celebrity likeness. No recognizable face.
For stories that do not intrinsically require people, use ZERO people, silhouettes,
crowds or visible human figures.

NO TEXT OR MARKINGS:
No words, letters, numbers, captions, headlines, logos, watermarks, signs, billboards,
banners, posters, documents, readable screens, scoreboards, written clothing, jersey
names, jersey numbers or brand marks.

STYLE:
Photorealistic editorial photography, realistic perspective, natural lighting, realistic
materials and proportions, one clear focal point, natural depth of field. No collage,
infographic, poster, illustration, cartoon, painting, fantasy or obvious AI-art look.
This is an AI-generated editorial reconstruction, not a claim that this is a real photograph
of the exact event. No graphic gore.
""".strip()

    if len(prompt) > 2048:
        fixed = """
Create ONE photorealistic editorial news photograph in a natural 3:2 composition.

HEADLINE:
{title}

VISUAL SUBJECT LOCK:
{label}

PRIMARY SUBJECT:
{subject}

MANDATORY: the image must depict the locked subject above. Do not reinterpret it as
another category or generic stock imagery. Named people are context only and MUST NOT
be depicted. Do not generate portraits, likenesses, recognizable faces or celebrity
recreations.

NEGATIVE SUBJECTS:
{forbidden}

If the story does not intrinsically require people, use ZERO human figures, silhouettes
or crowds. Never introduce unrelated people or unrelated activities.

SECONDARY CONTEXT:
{description}
{intro}

Use secondary context only to clarify the locked subject. Keep one coherent scene.
No text, letters, numbers, logos, signs, watermarks or readable markings.
Photorealistic editorial photography, natural lighting, realistic materials, one clear
focal point, no collage, infographic, illustration, cartoon, painting or fantasy.
No graphic gore.
""".strip()

        overhead = len(
            fixed.format(
                title="",
                label="",
                subject="",
                forbidden="",
                description="",
                intro="",
            )
        )
        available = max(0, 2048 - overhead)

        title_trimmed = title[:available]
        remaining = max(0, available - len(title_trimmed))

        label = lock["label"]
        label_trimmed = label[:remaining]
        remaining = max(0, remaining - len(label_trimmed))

        subject = lock["subject"]
        subject_trimmed = subject[:remaining]
        remaining = max(0, remaining - len(subject_trimmed))

        forbidden = lock["forbidden"]
        forbidden_trimmed = forbidden[:remaining]
        remaining = max(0, remaining - len(forbidden_trimmed))

        description_trimmed = description[:remaining]
        remaining = max(0, remaining - len(description_trimmed))
        intro_trimmed = intro[:remaining]

        prompt = fixed.format(
            title=title_trimmed,
            label=label_trimmed,
            subject=subject_trimmed,
            forbidden=forbidden_trimmed,
            description=description_trimmed,
            intro=intro_trimmed,
        )

        if len(prompt) > 2048:
            prompt = prompt[:2048]

    print(
        f"[AI IMAGE] Visual lock: {lock['type']} | "
        f"subject={lock['label']}"
    )

    return prompt


def _api_request_once(
    prompt: str,
    account_id: str,
    api_token: str,
    account_label: str,
) -> bytes:
    if not account_id:
        raise RuntimeError(f"{account_label}: account ID is not set.")
    if not api_token:
        raise RuntimeError(f"{account_label}: API token is not set.")

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{MODEL}"
    )

    payload = {
        "prompt": prompt,
        "steps": STEPS,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{account_label}: HTTP {exc.code}: {body[:3000]}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"{account_label}: Cloudflare image request failed: {exc}"
        ) from exc

    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"{account_label}: Cloudflare returned invalid JSON ({len(raw)} bytes)."
        ) from exc

    if not data.get("success"):
        raise RuntimeError(
            f"{account_label}: Cloudflare image API returned an error: "
            + json.dumps(data, ensure_ascii=False)[:3000]
        )

    result = data.get("result") or {}
    image_b64 = result.get("image")

    if not isinstance(image_b64, str) or not image_b64:
        raise RuntimeError(
            f"{account_label}: Cloudflare response did not contain result.image: "
            + json.dumps(data, ensure_ascii=False)[:3000]
        )

    try:
        return base64.b64decode(image_b64)
    except Exception as exc:
        raise RuntimeError(
            f"{account_label}: Invalid Base64 image returned by Cloudflare."
        ) from exc


def _is_quota_exhausted_error(exc: Exception) -> bool:
    """
    Return True only for errors that strongly indicate a depleted account
    allocation, not merely a transient HTTP 429/rate-limit condition.

    This distinction matters because a temporary rate limit must not permanently
    remove a healthy account from the pool for the rest of the run.
    """
    message = str(exc).lower()

    # Cloudflare Workers AI daily free allocation / neuron exhaustion.
    daily_allocation_markers = (
        "used up your daily free allocation",
        "daily free allocation",
        "daily allocation",
        "daily quota",
        "daily limit",
        "quota exceeded",
        "neurons",
    )
    if any(marker in message for marker in daily_allocation_markers):
        return True

    # Other explicit exhaustion/payment signals.
    exhausted_markers = (
        "account limited",
        "credits exhausted",
        "allocation exhausted",
        "limit exceeded",
    )
    return any(marker in message for marker in exhausted_markers)


_CF_ACCOUNTS = [
    (ACCOUNT_ID, API_TOKEN, "Cloudflare account 1"),
    (ACCOUNT_ID_2, API_TOKEN_2, "Cloudflare account 2"),
    (ACCOUNT_ID_3, API_TOKEN_3, "Cloudflare account 3"),
]

# Production must start with all three Cloudflare accounts configured.
# Previously, incomplete pairs were silently filtered out, causing the process
# to run with only account 1 and making the fallback appear broken.
def _validate_cf_accounts() -> None:
    missing: list[str] = []

    for account_no, (account_id, token, _label) in enumerate(
        _CF_ACCOUNTS, start=1
    ):
        suffix = "" if account_no == 1 else f"_{account_no}"

        if not account_id:
            missing.append(f"CLOUDFLARE_ACCOUNT_ID{suffix}")
        if not token:
            missing.append(f"CLOUDFLARE_API_TOKEN{suffix}")

    if missing:
        raise RuntimeError(
            "Cloudflare image rotation requires all 3 accounts. "
            "Missing environment variables: "
            + ", ".join(missing)
        )

    print(
        "[AI IMAGE] Cloudflare account pool: "
        "account 1=CONFIGURED, account 2=CONFIGURED, account 3=CONFIGURED"
    )


_CF_NEXT_ACCOUNT = 0

# Accounts that have returned a quota/rate exhaustion error during this
# process run. Once exhausted, skip them for all subsequent generation
# requests instead of retrying a known-depleted account.
_CF_EXHAUSTED_ACCOUNTS: set[int] = set()


def _configured_cf_accounts() -> list[tuple[str, str, str]]:
    # Validation above guarantees that all three production accounts are
    # complete. Return the full pool in fixed order.
    return list(_CF_ACCOUNTS)


def _api_request(prompt: str) -> bytes:
    """
    Generate an image using the configured Cloudflare accounts.

    Accounts that return quota/rate exhaustion are marked exhausted for the
    current process run and skipped on every subsequent call. This prevents
    regeneration from repeatedly retrying an account whose daily allocation
    is already depleted.
    """
    global _CF_NEXT_ACCOUNT

    _validate_cf_accounts()
    accounts = _configured_cf_accounts()

    available = [
        index for index in range(len(accounts))
        if index not in _CF_EXHAUSTED_ACCOUNTS
    ]
    if not available:
        raise RuntimeError(
            "All configured Cloudflare accounts are exhausted for this run."
        )

    start_index = _CF_NEXT_ACCOUNT % len(accounts)
    last_error = None

    for offset in range(len(accounts)):
        index = (start_index + offset) % len(accounts)

        if index in _CF_EXHAUSTED_ACCOUNTS:
            continue

        account_id, token, label = accounts[index]

        try:
            print(f"[AI IMAGE] Generating with {label}...")
            result = _api_request_once(
                prompt,
                account_id,
                token,
                label,
            )

            # Next successful generation starts with the other account.
            _CF_NEXT_ACCOUNT = (index + 1) % len(accounts)
            return result

        except Exception as exc:
            last_error = exc
            print(f"[AI IMAGE] {label} failed: {exc}")

            if _is_quota_exhausted_error(exc):
                _CF_EXHAUSTED_ACCOUNTS.add(index)
                print(
                    f"[AI IMAGE] {label} marked EXHAUSTED for this run; "
                    "skipping it on subsequent requests."
                )
                continue

            # Non-quota errors are not silently masked by account rotation.
            raise

    raise RuntimeError(
        "All available Cloudflare accounts failed for image generation. "
        f"Last error: {last_error}"
    )





def _save_webp(image_bytes: bytes, slug: str) -> str:
    source = IMAGE_DIR / f".{slug}.cloudflare.jpg"
    target = IMAGE_DIR / f"{slug}.webp"

    source.write_bytes(image_bytes)

    try:
        with Image.open(source) as im:
            im = im.convert("RGB")
            if im.size != (WIDTH, HEIGHT):
                target_ratio = WIDTH / HEIGHT
                source_ratio = im.width / im.height

                if source_ratio > target_ratio:
                    # Source is wider than target: crop the sides.
                    new_width = int(im.height * target_ratio)
                    left = (im.width - new_width) // 2
                    im = im.crop((left, 0, left + new_width, im.height))
                elif source_ratio < target_ratio:
                    # Source is taller than target: crop the top/bottom.
                    new_height = int(im.width / target_ratio)
                    top = (im.height - new_height) // 2
                    im = im.crop((0, top, im.width, top + new_height))

                im = im.resize(
                    (WIDTH, HEIGHT),
                    Image.Resampling.LANCZOS,
                )
            im.save(
                target,
                "WEBP",
                quality=QUALITY,
                method=6,
            )
    finally:
        try:
            source.unlink()
        except OSError:
            pass

    return f"{SITE_URL.rstrip('/')}/assets/articles/{target.name}"




def generate_article_image(
    article: dict,
    news: list | None = None,
) -> dict | None:
    if not ENABLED:
        return None

    slug = _clean(article.get("slug"), 120)
    if not slug:
        raise RuntimeError("Article has no slug")

    target = IMAGE_DIR / f"{slug}.webp"
    public_url = f"{SITE_URL.rstrip('/')}/assets/articles/{target.name}"

    if target.exists() and target.stat().st_size > 0:
        return {
            "image": public_url,
            "source": "",
            "ai_generated": True,
            "elapsed": "0.00",
        }

    started = time.perf_counter()

    # UNIVERSAL TOPIC-ONLY ROUTE:
    # Named people are context only and are NEVER generated as visual subjects.
    # Cloudflare is the ONLY image generator. The existing three-account
    # Cloudflare pool rotates/fails over through _api_request().
    #
    # No post-generation AI vision/ is used. A successful
    # Cloudflare generation is accepted directly. Semantic control is handled
    # by the headline-first topic-only prompt above.
    prompt = _build_prompt(article, news=news)

    image_bytes = None
    generated_source = ""

    for cf_attempt in range(1, 4):
        try:
            print(
                f"[AI IMAGE] Cloudflare generation attempt "
                f"{cf_attempt}/3..."
            )
            image_bytes = _api_request(prompt)
            generated_source = "Cloudflare / FLUX.1 Schnell"
            print(
                f"[AI IMAGE] Cloudflare generation attempt "
                f"{cf_attempt}/3 succeeded | slug={slug}"
            )
            break

        except Exception as exc:
            print(
                f"[AI IMAGE] Cloudflare generation attempt "
                f"{cf_attempt}/3 failed | slug={slug} | error={exc}"
            )

    if image_bytes is None:
        print(
            f"[AI IMAGE] ALL CLOUDFLARE GENERATION ATTEMPTS FAILED "
            f"-> no image | slug={slug}"
        )
        return None

    public_url = _save_webp(image_bytes, slug)

    elapsed = round(time.perf_counter() - started, 2)

    print(
        f"[AI IMAGE] ACCEPTED | {slug} | source={generated_source} "
        f"| elapsed={elapsed:.2f}s"
    )

    return {
        "image": public_url,
        "source": "",
        "ai_generated": True,
        "elapsed": f"{elapsed:.2f}",
    }

