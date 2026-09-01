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

The image layer is isolated from article generation.
It receives only visual prose (description/intro), never source headlines.
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
from pathlib import Path
from typing import Any

from PIL import Image

from config import ROOT, SITE_URL

# Strict post-generation relevance gate.
# Uses the existing Cloudflare account rotation; no extra provider is introduced.
IMAGE_RELEVANCE_ENABLED = os.getenv(
    "TC_IMAGE_RELEVANCE_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
IMAGE_RELEVANCE_MODEL = os.getenv(
    "TC_IMAGE_RELEVANCE_MODEL",
    "@cf/meta/llama-3.2-11b-vision-instruct",
).strip()
IMAGE_RELEVANCE_TIMEOUT = int(
    os.getenv("TC_IMAGE_RELEVANCE_TIMEOUT", "120")
)

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

# Optional second Cloudflare account used only after the primary account
# reaches its Workers AI daily free quota.
ACCOUNT_ID_2 = os.getenv("CLOUDFLARE_ACCOUNT_ID_2", "").strip()
API_TOKEN_2 = os.getenv("CLOUDFLARE_API_TOKEN_2", "").strip()

ENABLED = os.getenv("TC_AI_IMAGE_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}

WIDTH = 1024
HEIGHT = 683
STEPS = int(os.getenv("TC_AI_IMAGE_STEPS", "4"))
QUALITY = int(os.getenv("TC_AI_IMAGE_WEBP_QUALITY", "80"))
TIMEOUT = int(os.getenv("TC_AI_IMAGE_TIMEOUT", "180"))

MODEL = "@cf/black-forest-labs/flux-1-schnell"

# Person-image route: only used when the article slug resolves to a verified
# human on Wikipedia/Wikidata. All other images continue through Cloudflare.
HF_TOKENS = [
    os.getenv("HF_TOKEN_1", "").strip(),
    os.getenv("HF_TOKEN_2", "").strip(),
    os.getenv("HF_TOKEN_3", "").strip(),
    os.getenv("HF_TOKEN_4", "").strip(),
]
# Backward compatibility with the old single-token variable.
if not any(HF_TOKENS):
    legacy_token = os.getenv("HF_TOKEN", "").strip()
    if legacy_token:
        HF_TOKENS = [legacy_token]

HF_PERSON_MODEL = os.getenv(
    "TC_HF_PERSON_MODEL",
    "black-forest-labs/FLUX.1-Kontext-dev",
).strip()
HF_PROVIDER = os.getenv("TC_HF_PERSON_PROVIDER", "fal-ai").strip()
HF_PERSON_TIMEOUT = int(os.getenv("TC_HF_PERSON_TIMEOUT", "300"))
PERSON_IMAGE_ENABLED = os.getenv(
    "TC_HF_PERSON_IMAGE_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
PERSON_REF_DIR = Path(ROOT) / "assets" / "person_refs"
PERSON_REF_DIR.mkdir(parents=True, exist_ok=True)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_UA = (
    "TrendCurrent/1.0 (AI editorial image reference retrieval; "
    "https://trendcurrent.today)"
)

IMAGE_DIR = Path(ROOT) / "assets" / "articles"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _clean(text: Any, limit: int = 1200) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit]


def _build_prompt(article: dict, news: list | None = None) -> str:
    """Build a headline-first visual prompt within Cloudflare's 2048-char limit.

    The headline is the primary semantic source. Description and intro are
    secondary context only. The generator must depict the specific subject/event
    expressed by the headline, rather than a generic image representing the topic.
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

    prompt = f"""
Create ONE photorealistic editorial news photograph in a natural 3:2 composition.

ARTICLE HEADLINE — PRIMARY VISUAL SOURCE:
{title}

The headline is the primary semantic source for the image. Identify the concrete
subject, person, place, object, event, action, condition or development expressed
by the headline and depict THAT specific subject directly.

Do NOT illustrate merely the broad topic. Do NOT substitute a generic person,
generic business meeting, generic conference, generic weather presenter, generic
stadium, generic city, or generic stock image when the headline identifies a more
specific subject.

If the headline describes WEATHER, a FORECAST, STORM, HURRICANE, RAIN, SNOW, HEAT,
or another atmospheric event, the image must visibly depict the weather/atmospheric
condition itself, not a generic person or presenter.

If the headline describes MULTIPLE PEOPLE, GROUPS, FIGURES, LEADERS, SPEAKERS or
ATTENDEES, depict a scene with the relevant group or multiple people when that is
the natural meaning of the headline. Do not reduce a plural subject to one
unidentified generic person.

If the headline describes a CONFERENCE, FORUM, SUMMIT, MEETING or EVENT, depict the
actual type of event and its setting, rather than merely a person who could be
attending it.

If the headline names a specific PERSON, that person is the primary subject.
If it names a PRODUCT or OBJECT, make that product/object the primary subject.
If it names a PLACE, make that location visually prominent.
If it describes a SPORT or MATCH, depict the relevant sporting action.
If it describes a DISASTER or EMERGENCY, depict the relevant event and setting
without graphic injury.

SECONDARY ARTICLE CONTEXT:
{description}
{intro}

Use the secondary context only to clarify or enrich the headline-defined subject.
Never let secondary context replace the subject defined by the headline.
Create ONE specific, coherent real-world scene that directly represents the headline.
Do not invent unsupported facts, people, actions, locations or objects.

Use realistic perspective, natural lighting, realistic anatomy, correct proportions,
natural hands and fingers, depth of field and one clear focal point.

NO TEXT OR MARKINGS: no words, letters, numbers, captions, headlines, logos,
watermarks, signs, billboards, banners, posters, documents, readable screens,
scoreboards, written clothing, jersey names, jersey numbers or brand marks.

No collage, infographic, poster, illustration, cartoon, painting, fantasy or obvious
AI-art look. This is an AI-generated editorial reconstruction, not a claim that
this is a real photograph of the exact event. No graphic gore.
""".strip()

    if len(prompt) > 2048:
        fixed = """
Create ONE photorealistic editorial news photograph in a natural 3:2 composition.

ARTICLE HEADLINE — PRIMARY VISUAL SOURCE:
{title}

Depict the specific subject, person, place, object, event or condition expressed by
the headline. The headline is the primary semantic source and must not be ignored.

Do NOT illustrate merely the broad topic or substitute a generic stock subject.
Weather/forecast headlines require visible weather or atmospheric conditions.
Plural people/groups require a group or multiple relevant people when appropriate.
Conference/forum/summit headlines require the actual event setting, not just a generic
person. Named people, products, places, sports and disasters must be depicted directly.

SECONDARY CONTEXT:
{description}
{intro}

Use secondary context only to clarify the headline. Create one specific, coherent
real-world scene. Do not invent unsupported facts or replace the headline-defined
subject.

No text, letters, numbers, logos, signs, watermarks or readable markings. No collage,
infographic, illustration, cartoon, painting or fantasy. Photorealistic editorial
photography, natural lighting, realistic anatomy, one clear focal point, no graphic
gore.
""".strip()

        overhead = len(fixed.format(title="", description="", intro=""))
        available = max(0, 2048 - overhead)

        # Preserve the headline first. Secondary context is expendable.
        title_trimmed = title[:available]
        remaining = max(0, available - len(title_trimmed))
        description_trimmed = description[:remaining]
        remaining = max(0, remaining - len(description_trimmed))
        intro_trimmed = intro[:remaining]

        prompt = fixed.format(
            title=title_trimmed,
            description=description_trimmed,
            intro=intro_trimmed,
        )

        if len(prompt) > 2048:
            prompt = prompt[:2048]

    return prompt

def _http_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": WIKIPEDIA_UA},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _wikidata_is_human(entity_id: str) -> bool:
    """Return True only when Wikidata explicitly classifies the entity as human."""
    data = _http_json(
        "https://www.wikidata.org/w/api.php?"
        + urllib.parse.urlencode(
            {
                "action": "wbgetentities",
                "ids": entity_id,
                "props": "claims",
                "format": "json",
            }
        )
    )
    entity = (data.get("entities") or {}).get(entity_id) or {}
    claims = entity.get("claims") or {}
    for claim in claims.get("P31", []):
        mainsnak = claim.get("mainsnak") or {}
        datavalue = mainsnak.get("datavalue") or {}
        value = datavalue.get("value") or {}
        if value.get("id") == "Q5":
            return True
    return False


def _person_reference_from_slug(slug: str) -> tuple[bytes, str] | None:
    """
    Resolve a trend slug to a Wikipedia page that Wikidata explicitly marks
    as a human, then return its thumbnail bytes and display name.

    This is deliberately conservative: if the slug does not clearly resolve
    to a human, the normal Cloudflare path is used.
    """
    if not PERSON_IMAGE_ENABLED or not any(HF_TOKENS):
        return None

    slug = _clean(slug, 160).strip("-")
    if not slug:
        return None

    cache_path = PERSON_REF_DIR / f"{slug}.jpg"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            meta_path = cache_path.with_suffix(".json")
            person_name = slug.replace("-", " ").strip()
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                person_name = str(meta.get("title") or person_name)
            return cache_path.read_bytes(), person_name
        except Exception:
            pass

    # The slug is intentionally the primary signal. This prevents arbitrary
    # people mentioned deep in an article from hijacking the image subject.
    search = slug.replace("-", " ")
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": search,
            "gsrnamespace": "0",
            "gsrlimit": "5",
            "prop": "pageimages|pageprops",
            "piprop": "thumbnail",
            "pithumbsize": "640",
            "ppprop": "wikibase_item",
            "format": "json",
            "formatversion": "2",
        }
    )

    try:
        data = _http_json(f"{WIKIPEDIA_API}?{query}")
    except Exception:
        return None

    pages = (data.get("query") or {}).get("pages") or []
    if isinstance(pages, dict):
        pages = list(pages.values())

    for page in pages:
        entity_id = ((page.get("pageprops") or {}).get("wikibase_item") or "").strip()
        thumbnail = (page.get("thumbnail") or {}).get("source")
        title = (page.get("title") or "").strip()

        if not entity_id or not thumbnail or not title:
            continue

        try:
            if not _wikidata_is_human(entity_id):
                continue
        except Exception:
            continue

        try:
            request = urllib.request.Request(
                thumbnail,
                headers={"User-Agent": WIKIPEDIA_UA},
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                image_bytes = response.read()
            if not image_bytes:
                continue

            # Validate and normalize the reference before caching it.
            with Image.open(io.BytesIO(image_bytes)) as im:
                im = im.convert("RGB")
                im.thumbnail((480, 480), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                im.save(buffer, "JPEG", quality=92)
                image_bytes = buffer.getvalue()

            cache_path.write_bytes(image_bytes)
            cache_path.with_suffix(".json").write_text(
                json.dumps(
                    {"title": title, "entity_id": entity_id},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return image_bytes, title
        except Exception:
            continue

    return None


def _person_prompt(article: dict, person_name: str) -> str:
    """Build the person-specific Kontext prompt."""
    description = _clean(article.get("description"), 500)
    intro = _clean(
        article.get("intro")
        or (article.get("paragraphs") or [""])[0],
        250,
    )

    if not description:
        description = "a current news story involving the referenced person"

    return f"""
Create ONE photorealistic editorial news photograph using input_image_0 as the
identity reference for {person_name}.

IDENTITY:
Keep the exact same person from the reference image. Preserve recognizable facial
identity, facial structure, eyes, nose, mouth, hair, age and overall appearance.
Do not replace the person with a generic person, lookalike or another public figure.

PRIMARY NEWS EVENT:
{description}

SUPPORTING CONTEXT:
{intro}

Create one specific, coherent real-world editorial scene that directly represents
the article. The referenced person must be the unmistakable visual focal point.
Change the setting, pose and composition naturally while preserving identity.

Use realistic skin texture, natural anatomy, realistic hands and fingers, complete
visible limbs, correct body proportions, natural lighting and professional
photography. Natural 3:2 composition.

Do not invent unsupported events or actions. No text, captions, logos, watermarks,
readable signs, billboards, posters, documents or readable screens. No collage,
infographic, illustration, cartoon or obvious AI-art look.
""".strip()


def _hf_person_request(image_bytes: bytes, prompt: str) -> bytes:
    if InferenceClient is None:
        raise RuntimeError(
            "huggingface_hub is not installed; install it with "
            "'python -m pip install -U huggingface_hub'."
        )

    if not any(HF_TOKENS):
        raise RuntimeError(
            "No Hugging Face tokens are set. "
            "Set HF_TOKEN_1, HF_TOKEN_2, HF_TOKEN_3 and HF_TOKEN_4."
        )

    last_error = None

    # Same tested failover behavior as the standalone 3-token module:
    # token 1 -> token 2 -> token 3 only when the current token is exhausted.
    for token_index, token in enumerate(HF_TOKENS, start=1):
        if not token:
            continue

        print(
            f"[HF PERSON] Trying Hugging Face token "
            f"{token_index}/{len(HF_TOKENS)}..."
        )

        client = InferenceClient(
            provider=HF_PROVIDER,
            api_key=token,
            timeout=HF_PERSON_TIMEOUT,
        )

        try:
            image = client.image_to_image(
                image=image_bytes,
                prompt=prompt,
                model=HF_PERSON_MODEL,
            )
            print(f"[HF PERSON] Token {token_index} succeeded.")

            output = io.BytesIO()
            image.save(output, format="JPEG", quality=95)
            return output.getvalue()

        except Exception as exc:
            last_error = exc
            error_text = str(exc)

            # Hugging Face returns HTTP 402 when the token's included
            # Inference Provider credits are depleted. Only then rotate.
            is_credit_exhausted = (
                "402" in error_text
                or "Payment Required" in error_text
                or "depleted your monthly included credits" in error_text
                or "monthly included credits" in error_text
            )

            if is_credit_exhausted:
                print(
                    f"[HF PERSON] Token {token_index} exhausted. "
                    "Rotating to the next token..."
                )
                continue

            raise RuntimeError(
                f"Hugging Face token {token_index} failed: {error_text}"
            ) from exc

    raise RuntimeError(
        "All configured Hugging Face tokens are exhausted or unavailable. "
        f"Last error: {last_error}"
    )


def _try_person_image(article: dict) -> tuple[bool, bytes | None]:
    """
    Attempt the HF identity-preserving route only for a verified human slug.

    Returns:
        (False, None): not a verified person -> normal Cloudflare route.
        (True, bytes): verified person and HF generation succeeded.
        (True, None): verified person but reference/generation failed -> NO
                      Cloudflare fallback. This keeps person failures safe.
    """
    if not PERSON_IMAGE_ENABLED or not any(HF_TOKENS):
        return False, None

    slug = _clean(article.get("slug"), 120)
    if not slug:
        return False, None

    reference = _person_reference_from_slug(slug)
    if not reference:
        return False, None

    image_bytes, person_name = reference
    prompt = _person_prompt(article, person_name)

    try:
        result = _hf_person_request(image_bytes, prompt)
        print(
            f"[AI IMAGE] PERSON ROUTE: {person_name} "
            "via Hugging Face / FLUX.1 Kontext"
        )
        return True, result
    except Exception as exc:
        print(
            f"[AI IMAGE] PERSON ROUTE FAILED ({person_name}); "
            f"NO IMAGE FALLBACK: {exc}"
        )
        return True, None


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
    """Return True only for errors that look like account quota/rate exhaustion."""
    message = str(exc).lower()
    quota_markers = (
        "account limited",
        "daily limit",
        "daily quota",
        "quota exceeded",
        "rate limit",
        "rate_limit",
        "too many requests",
        "http 429",
        "http 402",
    )
    return any(marker in message for marker in quota_markers)


def _api_request(prompt: str) -> bytes:
    if not ACCOUNT_ID:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is not set.")
    if not API_TOKEN:
        raise RuntimeError("CLOUDFLARE_API_TOKEN is not set.")

    # Use account 1 until Cloudflare reports quota/rate exhaustion.
    try:
        return _api_request_once(
            prompt,
            ACCOUNT_ID,
            API_TOKEN,
            "Cloudflare account 1",
        )
    except Exception as first_exc:
        # Only fail over for quota/rate exhaustion. Authentication,
        # malformed requests, network errors, etc. should not silently
        # switch accounts because they indicate a different problem.
        if not _is_quota_exhausted_error(first_exc):
            raise

        if not ACCOUNT_ID_2 or not API_TOKEN_2:
            raise RuntimeError(
                "Cloudflare account 1 quota/rate limit reached, "
                "but second account credentials are not configured. "
                f"Original error: {first_exc}"
            ) from first_exc

        try:
            return _api_request_once(
                prompt,
                ACCOUNT_ID_2,
                API_TOKEN_2,
                "Cloudflare account 2",
            )
        except Exception as second_exc:
            raise RuntimeError(
                "Cloudflare account 1 quota/rate limit reached and "
                "account 2 also failed. "
                f"Account 1: {first_exc}; Account 2: {second_exc}"
            ) from second_exc


def _build_relevance_prompt(article: dict) -> str:
    """Build a strict headline-first article-vs-image relevance judge prompt."""
    title = _clean(article.get("title") or article.get("h1"), 300)
    description = _clean(article.get("description"), 500)
    intro = _clean(
        article.get("intro")
        or (article.get("paragraphs") or [""])[0],
        650,
    )

    return f"""
You are a STRICT editorial image relevance judge for a professional news website.

ARTICLE TITLE — PRIMARY CRITERION:
{title}

ARTICLE DESCRIPTION:
{description}

ARTICLE INTRODUCTION:
{intro}

Look at the supplied image.

The image is acceptable ONLY when it clearly and directly represents the SPECIFIC
MAIN SUBJECT or MAIN NEWS EVENT expressed by the article title.

The title is the primary criterion. The image must represent what the headline is
actually about, not merely the broad topic or atmosphere.

A merely attractive, photorealistic, professional, plausible, or broadly related
image is NOT acceptable.

SPECIFICITY RULES:
- If the title identifies a specific person, the pictured person must be that person.
  A generic person, lookalike, or another person from the same profession is FAIL.
- If the title identifies a product or object, that specific product/object must be
  visually central. A generic object from the same category is FAIL.
- If the title identifies a place, the image must provide meaningful visual evidence
  of that specific place, not merely a generic scene of the same type.
- If the title identifies a specific event, conference, forum, summit, match or
  incident, the image must visually represent that specific event or its defining
  setting/context. A generic person who could be attending is FAIL.
- If the title refers to multiple people, figures, leaders, speakers or attendees,
  a single unidentified generic person is FAIL unless the title clearly makes one
  specific person the sole main subject.
- If the title is about weather, a forecast, storm, hurricane, rain, snow, heat or
  another atmospheric condition, the image must visibly depict the weather/condition.
  A presenter, meteorologist, generic person or unrelated indoor scene is FAIL.
- If the title is about sport, the relevant sport/action must be visible.
- If the title is about a disaster or emergency, the relevant event/scene must be
  visible without requiring graphic injury.
- If the title describes a policy, rule, market development, announcement or other
  non-person event, a generic person is FAIL unless the person is itself the main
  subject described by the title.

FAIL if the image:
- could illustrate many unrelated articles;
- would make a reasonable reader think the article is about a different subject;
- represents only a broad topic while missing the headline's specific subject/event;
- lacks enough visual evidence to connect the image to the headline;
- relies on clothing, a lanyard, a generic office/conference setting, or a generic
  human alone as proof of relevance.

Do NOT judge whether the image is beautiful. Judge only editorial relevance and
specificity to the headline.

When uncertain, return FAIL.

Return exactly:
VERDICT: PASS or FAIL
CONFIDENCE: 0-100
REASON: one short sentence
""".strip()

def _image_relevance_check(article: dict, image_bytes: bytes) -> bool:
    """
    Strict post-generation gate.

    PASS only on an explicit PASS verdict. Any API error, malformed response,
    uncertainty, or explicit FAIL rejects the image. No regeneration is done.
    """
    if not IMAGE_RELEVANCE_ENABLED:
        print("[IMAGE RELEVANCE] DISABLED")
        return True

    prompt = _build_relevance_prompt(article)
    accounts = [
        (ACCOUNT_ID, API_TOKEN, "Cloudflare account 1"),
        (ACCOUNT_ID_2, API_TOKEN_2, "Cloudflare account 2"),
    ]

    last_error = None

    for account_id, token, label in accounts:
        if not account_id or not token:
            continue

        try:
            url = (
                f"https://api.cloudflare.com/client/v4/accounts/"
                f"{account_id}/ai/run/{IMAGE_RELEVANCE_MODEL}"
            )
            payload = {
                "prompt": prompt,
                "image": base64.b64encode(image_bytes).decode("ascii"),
                "max_tokens": 160,
                "temperature": 0,
            }

            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            print(f"[IMAGE RELEVANCE] Checking with {label}...")
            with urllib.request.urlopen(
                request, timeout=IMAGE_RELEVANCE_TIMEOUT
            ) as response:
                data = json.loads(response.read().decode("utf-8"))

            if not data.get("success"):
                raise RuntimeError(json.dumps(data, ensure_ascii=False))

            result = data.get("result") or {}
            raw = str(result.get("response") or "").strip()
            normalized = raw.upper().replace("*", "").replace("`", "").replace("_", "")

            if "VERDICT: PASS" in normalized:
                print(f"[IMAGE RELEVANCE] PASS -> {raw}")
                return True

            if "VERDICT: FAIL" in normalized:
                print(f"[IMAGE RELEVANCE] FAIL -> {raw}")
                return False

            # Fail closed on malformed/ambiguous judge output.
            print(f"[IMAGE RELEVANCE] FAIL CLOSED -> {raw}")
            return False

        except Exception as exc:
            last_error = exc
            print(f"[IMAGE RELEVANCE] {label} failed: {exc}")

            # Rotate only when the current Cloudflare account cannot perform
            # the judge call. Never accept an unvalidated image.
            continue

    print(
        "[IMAGE RELEVANCE] FAIL CLOSED: no validation account available"
        + (f" | last_error={last_error}" if last_error else "")
    )
    return False


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

    # PERSON ROUTE:
    # Verified human slugs use Hugging Face + reference image.
    # A verified person never falls back to Cloudflare if HF fails.
    is_person, image_bytes = _try_person_image(article)

    if is_person:
        if image_bytes is None:
            print(
                "[AI IMAGE] PERSON IMAGE REJECTED -> publishing without image"
            )
            return None

        generated_source = "Hugging Face / FLUX.1 Kontext"
    else:
        prompt = _build_prompt(article, news=news)
        image_bytes = _api_request(prompt)
        generated_source = "Cloudflare / FLUX.1 Schnell"

    # Strict semantic gate applies to every newly generated image.
    if not _image_relevance_check(article, image_bytes):
        print(
            f"[AI IMAGE] REJECTED BY RELEVANCE GATE -> no image | slug={slug}"
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

