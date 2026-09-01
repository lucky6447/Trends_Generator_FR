#!/usr/bin/env python3
"""
Fix article SEO URLs in the current trends/ folder.

What it does:
- Scans all .html files recursively under the folder where this script is run.
- Removes ONLY the trailing ".html" from article URLs in:
    1) <link rel="canonical" ...>
    2) <meta property="og:url" ...>
    3) JSON-LD "url":"..." for Article schema
- Does NOT touch:
    - .webp/.jpg/.png image URLs
    - normal file names
    - URLs outside those SEO fields
    - the article's own filename
- Creates a .bak backup for every changed file.
- Prints every changed file and number of fixes.
- Safe to run multiple times.
"""

from pathlib import Path
import re
import shutil

ROOT = Path.cwd()

# Only target SEO URL fields. The regex deliberately requires the URL
# to be inside one of the known article SEO tags/fields.
PATTERNS = [
    (
        re.compile(
            r'(<link\b[^>]*\brel=["\']canonical["\'][^>]*\bhref=["\'][^"\']+?)\.html(["\'])',
            re.IGNORECASE,
        ),
        r'\1\2',
        "canonical",
    ),
    (
        re.compile(
            r'(<meta\b[^>]*\bproperty=["\']og:url["\'][^>]*\bcontent=["\'][^"\']+?)\.html(["\'])',
            re.IGNORECASE,
        ),
        r'\1\2',
        "og:url",
    ),
    (
        re.compile(
            r'("url"\s*:\s*"[^"]+?)\.html(")',
            re.IGNORECASE,
        ),
        r'\1\2',
        "JSON-LD url",
    ),
]

def process_file(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[SKIP] Non-UTF8: {path}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {path}: {exc}")
        return 0

    new_text = text
    total = 0
    labels = []

    for pattern, replacement, label in PATTERNS:
        new_text, count = pattern.subn(replacement, new_text)
        if count:
            total += count
            labels.append(f"{label}={count}")

    if total == 0:
        return 0

    # Backup before modifying.
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    path.write_text(new_text, encoding="utf-8")
    print(f"[FIXED] {path} | {total} fix(es) | {', '.join(labels)}")
    return total

def main():
    print("=" * 70)
    print("TrendCurrent SEO URL fixer")
    print("=" * 70)
    print(f"Scanning: {ROOT}")
    print()

    files = sorted(ROOT.rglob("*.html"))

    if not files:
        print("No .html files found.")
        return

    changed_files = 0
    total_fixes = 0

    for path in files:
        fixes = process_file(path)
        if fixes:
            changed_files += 1
            total_fixes += fixes

    print()
    print("=" * 70)
    print(f"Scanned files : {len(files)}")
    print(f"Changed files : {changed_files}")
    print(f"Total fixes   : {total_fixes}")
    print("=" * 70)

    if changed_files:
        print()
        print("Backups were created as: filename.html.bak")
        print("The script is safe to run again; already-fixed URLs will not change.")
    else:
        print()
        print("Nothing to fix. No targeted SEO URL ending in .html was found.")

if __name__ == "__main__":
    main()
