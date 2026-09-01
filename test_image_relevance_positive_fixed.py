import os
import json
import base64
import urllib.request
from pathlib import Path

MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"

ACCOUNTS = [
    (os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip(),
     os.getenv("CLOUDFLARE_API_TOKEN", "").strip(),
     "Cloudflare account 1"),
    (os.getenv("CLOUDFLARE_ACCOUNT_ID_2", "").strip(),
     os.getenv("CLOUDFLARE_API_TOKEN_2", "").strip(),
     "Cloudflare account 2"),
]

ROOT = Path.cwd()

# We deliberately use locally generated/reference images that should be
# obviously relevant to their matching article descriptions.
CANDIDATES = [
    {
        "name": "jennifer-generated",
        "files": [
            ROOT / "jennifer-aniston-hf-3token-test.webp",
            ROOT / "jennifer_huggingface_kontext_test.png",
        ],
        "title": "Jennifer Aniston Appears in New Public Event",
        "summary": (
            "A news article about actress Jennifer Aniston and her public "
            "appearance, with Jennifer Aniston as the central subject."
        ),
        "expected": "PASS",
    },
    {
        "name": "jennifer-reference",
        "files": [
            ROOT / "licensed-image.jpg",
        ],
        "title": "Jennifer Aniston Remains in the Spotlight",
        "summary": (
            "A news article about Jennifer Aniston, with the actress herself "
            "as the central visual subject."
        ),
        "expected": "PASS",
    },
]

def get_image_path(candidate):
    for path in candidate["files"]:
        if path.exists() and path.is_file():
            return path
    return None

def cf_call(account_id, token, prompt, image_bytes):
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{MODEL}"
    )

    payload = {
        "prompt": prompt,
        "image": base64.b64encode(image_bytes).decode("ascii"),
        "max_tokens": 120,
        "temperature": 0,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    if not data.get("success"):
        raise RuntimeError(json.dumps(data, ensure_ascii=False))

    return data["result"].get("response", "")

def agree(account_id, token):
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{MODEL}"
    )
    payload = {"prompt": "agree"}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60):
            pass
    except Exception:
        pass

def prompt_for(candidate):
    return f"""
You are a STRICT editorial image relevance judge for a news website.

ARTICLE TITLE:
{candidate["title"]}

ARTICLE SUMMARY:
{candidate["summary"]}

Look at the supplied image.

The image should PASS only if it clearly and directly represents the
main subject of the article. The subject must be visually identifiable
and central to the image.

For this test, do NOT require the image to show the exact event described
in the article. Judge whether the image is an appropriate editorial image
for the stated article subject.

Do not reject an image merely because the setting or clothing differs.

Return exactly:

VERDICT: PASS or FAIL
CONFIDENCE: 0-100
REASON: one short sentence

Be strict about subject mismatch, but do not reject an obviously correct
subject merely because the scene is different.
""".strip()

def main():
    available = [a for a in ACCOUNTS if a[0] and a[1]]
    if not available:
        raise RuntimeError(
            "Set CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN first."
        )

    print("TrendCurrent Image Relevance Validator - POSITIVE TEST")
    print(f"Model: {MODEL}")
    print()

    for account_id, token, label in available:
        print(f"Preparing {label}...")
        agree(account_id, token)

    results = []

    for candidate in CANDIDATES:
        path = get_image_path(candidate)

        print(f"\n=== {candidate['name']} ===")
        print(f"Expected: {candidate['expected']}")

        if path is None:
            print("SKIPPED: matching local image was not found.")
            print("Looked for:")
            for p in candidate["files"]:
                print(f"  {p}")
            results.append((candidate["name"], "SKIPPED", "SKIPPED"))
            continue

        print(f"Image: {path}")
        image_bytes = path.read_bytes()
        prompt = prompt_for(candidate)

        response = None
        last_error = None

        for account_id, token, label in available:
            try:
                print(f"Trying {label}...")
                response = cf_call(account_id, token, prompt, image_bytes)
                print(f"Judge response:\n{response}")
                break
            except Exception as exc:
                last_error = exc
                print(f"{label} failed: {exc}")

        if response is None:
            raise RuntimeError(
                f"Validator failed for {candidate['name']}: {last_error}"
            )

        # Models may wrap the verdict in Markdown, e.g. **VERDICT:** PASS.
        # Normalize common Markdown punctuation before parsing.
        normalized = response.upper()
        for marker in ("*", "`", "_"):
            normalized = normalized.replace(marker, "")

        if "VERDICT: PASS" in normalized:
            verdict = "PASS"
        elif "VERDICT: FAIL" in normalized:
            verdict = "FAIL"
        else:
            verdict = "UNKNOWN"

        test_ok = verdict == candidate["expected"]
        results.append((candidate["name"], candidate["expected"], verdict))

        print(f"Detected verdict: {verdict}")
        print("TEST: " + ("PASS" if test_ok else "FAIL"))

    print("\n=== FINAL RESULT ===")

    tested = [r for r in results if r[1] != "SKIPPED"]

    if not tested:
        print("No positive images were found locally.")
        print("Copy the Jennifer test image into this folder and run again.")
        return

    all_ok = all(expected == actual for _, expected, actual in tested)

    for name, expected, actual in results:
        print(f"{name}: expected {expected}, got {actual}")

    print()
    if all_ok and len(tested) == len(CANDIDATES):
        print("SUCCESS - positive images were accepted.")
    elif all_ok:
        print("PARTIAL SUCCESS - available positive images were accepted.")
    else:
        print("WARNING - at least one positive image was rejected.")

if __name__ == "__main__":
    main()
