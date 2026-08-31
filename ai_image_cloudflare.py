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
from pathlib import Path
from typing import Any

from PIL import Image

from config import ROOT, SITE_URL


ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

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

    return f"""
Create ONE photorealistic editorial news photograph in a natural 3:2 composition.

PRIMARY NEWS EVENT:
{description}

SUPPORTING CONTEXT:
{intro}

Make the PRIMARY NEWS EVENT the clear visual subject. Create one specific,
coherent real-world scene that directly represents what the article reports,
not merely its broad topic. Use supporting context only to improve the setting.
Do not combine unrelated events or let secondary details replace the main event.

Choose the most appropriate realistic scene. For a person, make them the focal
point. For sport, show the relevant match. For finance, show the specific
market or business development. For politics, show the relevant figure or
official setting. For geopolitics, show the supported location and context.
For emergencies or disasters, show the relevant scene without graphic injury.
For a place, make the location the primary subject.

Do not invent facts, events, people, actions, locations or objects not supported
by the PRIMARY NEWS EVENT. Avoid generic stock images when a specific scene can
be inferred. Use realistic perspective, natural lighting, depth of field and
one clear focal point.

NO TEXT OR MARKINGS: no words, letters, numbers, captions, headlines, logos,
watermarks, signs, billboards, banners, posters, documents, readable screens,
scoreboards, written clothing, jersey names, jersey numbers or brand marks.

No collage, infographic, poster, illustration, cartoon, painting, fantasy or
obvious AI-art look. This is an AI-generated editorial reconstruction, not a
claim that this is a real photograph of the exact event. No graphic gore.
""".strip()


def _api_request(prompt: str) -> bytes:
    if not ACCOUNT_ID:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is not set.")
    if not API_TOKEN:
        raise RuntimeError("CLOUDFLARE_API_TOKEN is not set.")

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{ACCOUNT_ID}/ai/run/{MODEL}"
    )

    payload = {
        "prompt": prompt,
        "steps": STEPS,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
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
            f"Cloudflare image API HTTP {exc.code}: {body[:2000]}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Cloudflare image request failed: {exc}") from exc

    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Cloudflare returned invalid JSON ({len(raw)} bytes)."
        ) from exc

    if not data.get("success"):
        raise RuntimeError(
            "Cloudflare image API returned an error: "
            + json.dumps(data, ensure_ascii=False)[:3000]
        )

    result = data.get("result") or {}
    image_b64 = result.get("image")

    if not isinstance(image_b64, str) or not image_b64:
        raise RuntimeError(
            "Cloudflare response did not contain result.image: "
            + json.dumps(data, ensure_ascii=False)[:3000]
        )

    try:
        return base64.b64decode(image_b64)
    except Exception as exc:
        raise RuntimeError("Invalid Base64 image returned by Cloudflare.") from exc


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

    prompt = _build_prompt(article, news=news)

    started = time.perf_counter()
    image_bytes = _api_request(prompt)
    public_url = _save_webp(image_bytes, slug)
    elapsed = round(time.perf_counter() - started, 2)

    return {
        "image": public_url,
        "source": "",
        "ai_generated": True,
        "elapsed": f"{elapsed:.2f}",
    }
