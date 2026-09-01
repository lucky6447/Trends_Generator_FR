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
    """Build a focused visual prompt within Cloudflare's 2048-character limit."""

    description = _clean(article.get("description"), 350)
    intro = _clean(
        article.get("intro")
        or (article.get("paragraphs") or [""])[0],
        50,
    )

    if not description and not intro:
        description = "Show the central real-world subject or event as an editorial news photograph."

    prompt = f"""
Create ONE photorealistic editorial news photograph in a natural 3:2 composition.

PRIMARY NEWS EVENT:
{description}

SUPPORTING CONTEXT:
{intro}

Make the PRIMARY NEWS EVENT the clear visual subject. Create one specific,
coherent real-world scene that directly represents what the article reports,
not merely its broad topic. Use supporting context only to improve the setting.
Do not combine unrelated events or let secondary details replace the main event.

IDENTITY ACCURACY:
When a specific real person is named or clearly described in the PRIMARY NEWS EVENT,
that exact person must be the visual subject. Do not replace the named person with
a generic person, lookalike, unrelated public figure, or another person from the
same profession. Preserve supported identity cues such as age range, sex, facial
structure, hair and overall appearance as closely as possible. If multiple named
people are present, keep their identities distinct and never merge or swap them.
Do not invent additional people unless the PRIMARY NEWS EVENT requires them.

Choose the most appropriate realistic scene. For a specific person, make that
person the unmistakable focal point. For sport, show the relevant match. For
finance, show the specific market or business development. For politics, show
the relevant figure or official setting. For geopolitics, show the supported location
and context. For emergencies or disasters, show the relevant scene without graphic
injury. For a place, make the location the primary subject.

Do not invent facts, events, people, actions, locations or objects not supported
by the PRIMARY NEWS EVENT. Avoid generic stock images when a specific scene can
be inferred. Use realistic perspective, natural lighting, natural anatomy,
complete visible limbs, realistic hands and fingers, correct body proportions,
depth of field and one clear focal point. Avoid awkward cropping of important
body parts.

NO TEXT OR MARKINGS: no words, letters, numbers, captions, headlines, logos,
watermarks, signs, billboards, banners, posters, documents, readable screens,
scoreboards, written clothing, jersey names, jersey numbers or brand marks.

No collage, infographic, poster, illustration, cartoon, painting, fantasy or
obvious AI-art look. This is an AI-generated editorial reconstruction, not a
claim that this is a real photograph of the exact event. No graphic gore.
""".strip()

    # Cloudflare FLUX accepts a maximum of 2048 characters for /prompt.
    # If an unusually long prompt exceeds the limit, preserve the complete
    # instruction block and trim only the article-supplied description/context.
    if len(prompt) > 2048:
        fixed = """
Create ONE photorealistic editorial news photograph in a natural 3:2 composition.

PRIMARY NEWS EVENT:
{description}

SUPPORTING CONTEXT:
{intro}

Make the PRIMARY NEWS EVENT the clear visual subject. Create one specific,
coherent real-world scene that directly represents what the article reports,
not merely its broad topic. Use supporting context only to improve the setting.
Do not combine unrelated events or let secondary details replace the main event.

IDENTITY ACCURACY:
When a specific real person is named or clearly described in the PRIMARY NEWS EVENT,
that exact person must be the visual subject. Do not replace the named person with
a generic person, lookalike, unrelated public figure, or another person from the
same profession. Preserve supported identity cues such as age range, sex, facial
structure, hair and overall appearance as closely as possible. If multiple named
people are present, keep their identities distinct and never merge or swap them.
Do not invent additional people unless the PRIMARY NEWS EVENT requires them.

Choose the most appropriate realistic scene. For a specific person, make that
person the unmistakable focal point. For sport, show the relevant match. For
finance, show the specific market or business development. For politics, show
the relevant figure or official setting. For geopolitics, show the supported
location and context. For emergencies or disasters, show the relevant scene
without graphic injury. For a place, make the location the primary subject.

Do not invent facts, events, people, actions, locations or objects not supported
by the PRIMARY NEWS EVENT. Avoid generic stock images when a specific scene can
be inferred. Use realistic perspective, natural lighting, natural anatomy,
complete visible limbs, realistic hands and fingers, correct body proportions,
depth of field and one clear focal point. Avoid awkward cropping of important
body parts.

NO TEXT OR MARKINGS: no words, letters, numbers, captions, headlines, logos,
watermarks, signs, billboards, banners, posters, documents, readable screens,
scoreboards, written clothing, jersey names, jersey numbers or brand marks.

No collage, infographic, poster, illustration, cartoon, painting, fantasy or
obvious AI-art look. This is an AI-generated editorial reconstruction, not a
claim that this is a real photograph of the exact event. No graphic gore.
""".strip()

        # Keep the full instruction block and trim only supplied article text.
        # The normal 350/50 limits are already small; this is a final safety path.
        overhead = len(fixed.format(description="", intro=""))
        available = max(0, 2048 - overhead)

        # Preserve the description first; use remaining space for intro.
        description_limit = min(len(description), available)
        description_trimmed = description[:description_limit]
        remaining = max(0, available - len(description_trimmed))
        intro_trimmed = intro[:remaining]

        prompt = fixed.format(
            description=description_trimmed,
            intro=intro_trimmed,
        )

        # Absolute final guard. This can only activate in an unforeseen
        # formatting/encoding edge case and guarantees Cloudflare's limit.
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


def _try_person_image(article: dict) -> bytes | None:
    """
    Attempt the HF identity-preserving route only for a trend whose slug
    resolves to a verified human. Any lookup/generation failure falls back
    to Cloudflare so one person-image failure never breaks publication.
    """
    if not PERSON_IMAGE_ENABLED or not any(HF_TOKENS):
        return None

    slug = _clean(article.get("slug"), 120)
    if not slug:
        return None

    reference = _person_reference_from_slug(slug)
    if not reference:
        return None

    image_bytes, person_name = reference
    prompt = _person_prompt(article, person_name)

    try:
        result = _hf_person_request(image_bytes, prompt)
        print(
            f"[AI IMAGE] PERSON ROUTE: {person_name} "
            "via Hugging Face / FLUX.1 Kontext"
        )
        return result
    except Exception as exc:
        print(
            f"[AI IMAGE] PERSON ROUTE FAILED ({person_name}); "
            f"falling back to Cloudflare: {exc}"
        )
        return None


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
    # Everything else remains on the existing Cloudflare pipeline.
    image_bytes = _try_person_image(article)

    if image_bytes is not None:
        public_url = _save_webp(image_bytes, slug)
    else:
        prompt = _build_prompt(article, news=news)
        image_bytes = _api_request(prompt)
        public_url = _save_webp(image_bytes, slug)

    elapsed = round(time.perf_counter() - started, 2)

    return {
        "image": public_url,
        "source": "",
        "ai_generated": True,
        "elapsed": f"{elapsed:.2f}",
    }
