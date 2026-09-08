#!/usr/bin/env python3
"""
TrendCurrent post-generation Article Cleaner.

Purpose:
    A safety-net quality layer that runs AFTER article generation. It does not
    change generation, evidence selection, word floors, retries, or prompts.

Default behaviour is conservative:
    PASS        -> keep
    QUARANTINE  -> move rejected files to a sibling quarantine directory
    DELETE      -> only when ARTICLE_CLEANER_DELETE=1 / --delete is supplied

Supported languages (the current seven-language production set):
    en, de, es, it, fr, pt, id

Usage:
    python article_cleaner.py
    python article_cleaner.py --root .
    python article_cleaner.py --dirs en de es it fr pt id
    python article_cleaner.py --delete
    python article_cleaner.py --dry-run

Environment:
    ARTICLE_CLEANER_ROOTS   comma-separated article directories
    ARTICLE_CLEANER_DELETE  1/true to permanently delete rejected files
    ARTICLE_CLEANER_LLM     0/false to disable the LLM judge
    ARTICLE_CLEANER_MIN_WORDS  deterministic "very short" threshold (default 90)
    ARTICLE_CLEANER_LOG     optional JSON log path

The cleaner intentionally does NOT use a hard publication word floor:
a short article can still PASS when it is information-dense and coherent.
"""

import argparse
import difflib
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

try:
    from ollama import chat
except Exception:
    chat = None

try:
    from config import MODEL
except Exception:
    MODEL = os.getenv("MODEL", "ministral")

LANGUAGE_DIR_ALIASES = {
    "en": {"en", "en-us", "en-gb", "english"},
    "de": {"de", "german", "deutsch"},
    "es": {"es", "spanish", "español"},
    "it": {"it", "italian", "italiano"},
    "fr": {"fr", "french", "français"},
    "pt": {"pt", "pt-br", "portuguese", "português"},
    "id": {"id", "indonesian", "bahasa-indonesia", "bahasa_indonesia"},
}

LANGUAGE_WORDS = {
    "en": {"the","and","of","to","in","for","on","with","from","that","this","was","were","has","have","are","is","as","by","at","after","before","will","said","about","their","they","which","also","more","than","its","who","what","when","new"},
    "de": {"der","die","das","und","von","zu","den","dem","des","ein","eine","einer","einem","einen","ist","sind","war","wurde","wurden","mit","auf","für","im","in","aus","nach","über","auch","nicht","sich","als","bei","hat","haben","wie","dass","durch","werden","wird"},
    "es": {"el","la","los","las","del","de","un","una","unos","unas","y","que","en","con","por","para","sobre","desde","entre","al","es","son","era","fue","han","ha","no","también","como","más","menos","después","antes","esta","este","estos","estas","según"},
    "it": {"il","lo","la","i","gli","le","di","del","della","dei","degli","delle","un","uno","una","e","che","in","con","per","su","da","al","alla","agli","alle","è","sono","era","ha","hanno","non","anche","come","dopo","prima","nel","nella","nelle","questa","questo","quello","secondo"},
    "fr": {"le","la","les","des","du","de","un","une","et","en","dans","pour","sur","avec","par","est","sont","était","ont","a","au","aux","ce","cette","ces","qui","que","pas","plus","mais","comme","après","avant","leur","leurs","entre","vers","selon"},
    "pt": {"o","a","os","as","um","uma","uns","umas","e","de","do","da","dos","das","em","no","na","nos","nas","para","por","com","que","é","são","foi","era","tem","têm","não","também","como","mais","depois","antes","esta","este","estes","estas","segundo"},
    "id": {"yang","dan","di","ke","dari","untuk","dengan","pada","dalam","ini","itu","akan","telah","adalah","sebagai","oleh","lebih","juga","tidak","dapat","bisa","setelah","sebelum","tentang","menjadi","sudah","masih","mereka","para","karena","hingga","tersebut","bahwa","atau","serta","terhadap"},
}

MORPHOLOGY = {
    "en": (r"\b\w+(?:ing|ed|ly)\b",),
    "de": (r"\b\w+(?:ung|keit|heit|lich|ischen|isch)\b",),
    "fr": (r"\b\w+(?:ment|tion|ique|eur|euse|aient|ées|és)\b",),
    "it": (r"\b\w+(?:zione|zioni|mente|ità|ismo|are|ere|ire)\b",),
    "es": (r"\b\w+(?:ción|ciones|mente|ando|iendo|ado|ido|ación)\b",),
    "pt": (r"\b\w+(?:ção|ções|mente|ando|endo|ado|ido|idade)\b",),
    "id": (r"\b\w+(?:kan|nya|lah|kah|pun|per|ber|ter|meng|mem|men)\b",),
}

PLACEHOLDER_TERMS = (
    "lorem ipsum", "placeholder", "insert article", "write article here",
    "return only the required json", "locked evidence:", "final entitlement check",
    "you are a professional", "article generator", "do not pad, speculate",
)

# These are generic boilerplate phrases that should not by themselves kill an article.
# They contribute only to the deterministic repetition/boilerplate signal.
BOILERPLATE_PATTERNS = (
    r"\bmore details (?:are|will be) expected\b",
    r"\bthe situation remains (?:fluid|developing|unclear)\b",
    r"\bthe development comes as\b",
    r"\baccording to reports\b",
    r"\bthe latest development\b",
)

class ArticleHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.h1 = ""
        self._capture = None
        self._parts = []
        self.paragraphs = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._capture = "title"
            self._parts = []
        elif tag == "h1":
            self._capture = "h1"
            self._parts = []
        elif tag == "p":
            self._capture = "p"
            self._parts = []

    def handle_data(self, data):
        if self._skip_depth or not self._capture:
            return
        self._parts.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if self._capture == tag:
            value = re.sub(r"\s+", " ", " ".join(self._parts)).strip()
            if value:
                if tag == "title":
                    self.title = value
                elif tag == "h1":
                    self.h1 = value
                elif tag == "p":
                    self.paragraphs.append(value)
            self._capture = None
            self._parts = []

def normalize_text(value):
    value = html.unescape(str(value or ""))
    value = value.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()

def tokens(value):
    return re.findall(r"[^\W_]+", normalize_text(value).casefold(), flags=re.UNICODE)

def significant_tokens(value):
    out = []
    for token in tokens(value):
        if len(token) >= 3 and not token.isdigit():
            out.append(token)
    return out

def word_count(article):
    return len(significant_tokens(" ".join(article["paragraphs"])))

def sentence_list(text):
    text = normalize_text(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])(?:[\"”»’')\]]+)?\s+", text) if s.strip()]

def sentence_similarity(a, b):
    return difflib.SequenceMatcher(None, a.casefold(), b.casefold()).ratio()

def jaccard(a, b):
    sa, sb = set(significant_tokens(a)), set(significant_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def normalize_for_hash(text):
    return " ".join(significant_tokens(text))

def content_hash(article):
    normalized = normalize_for_hash(" ".join(article["paragraphs"]))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

def fingerprint(article):
    # Small, stable signature used to avoid O(n²) comparisons over every article.
    toks = significant_tokens(" ".join(article["paragraphs"]))
    if not toks:
        return ""
    return " ".join(toks[:18])

def detect_language(article):
    text = " ".join([article["title"], article["h1"], *article["paragraphs"]])
    all_tokens = tokens(text)
    if len(all_tokens) < 12:
        return None, 0.0
    scores = {}
    for lang, words in LANGUAGE_WORDS.items():
        hits = sum(1 for t in all_tokens if t in words)
        unique_hits = len(set(all_tokens) & words)
        morph = sum(min(len(re.findall(p, text.casefold())), 6) * 0.45 for p in MORPHOLOGY[lang])
        scores[lang] = unique_hits + min(hits, 12) * 0.35 + morph
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return None, 0.0
    best, score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = max(0.0, min(1.0, (score - second + 1.0) / max(score, 1.0)))
    return best, confidence

def language_from_path(path):
    parts = [p.casefold() for p in path.parts]
    for lang, aliases in LANGUAGE_DIR_ALIASES.items():
        if any(part in aliases for part in parts):
            return lang
    return None

def parse_article(path):
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"read_error:{exc}"
    parser = ArticleHTMLParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception as exc:
        return None, f"html_parse_error:{exc}"
    title = normalize_text(parser.title or parser.h1)
    h1 = normalize_text(parser.h1 or parser.title)
    paragraphs = [normalize_text(p) for p in parser.paragraphs if normalize_text(p)]
    return {
        "path": path,
        "title": title,
        "h1": h1,
        "paragraphs": paragraphs,
        "raw_size": len(raw),
        "language": language_from_path(path),
    }, None

def deterministic_quality(article, min_words):
    paragraphs = article["paragraphs"]
    body = " ".join(paragraphs)
    wc = word_count(article)
    sentences = sentence_list(body)
    reasons = []
    score = 100.0

    if not article["title"] or not body:
        return {"decision": "REJECT", "reasons": ["empty_or_missing_core_content"], "score": 0}

    searchable = " ".join(
        [article.get("title", ""), article.get("h1", ""), body]
    )
    lower = searchable.casefold()
    placeholders = [x for x in PLACEHOLDER_TERMS if x in lower]
    if placeholders:
        return {"decision": "REJECT", "reasons": ["generator_or_placeholder_leakage"], "score": 0}

    # Very short is a trigger, not an automatic rejection.
    if wc < min_words:
        reasons.append(f"very_short:{wc}")
        score -= 30

    if len(sentences) < 3:
        reasons.append(f"too_few_sentences:{len(sentences)}")
        score -= 30

    if len(paragraphs) == 0:
        reasons.append("no_paragraphs")
        score -= 40

    # Repeated paragraphs / sentences.
    duplicate_sentence_pairs = []
    seen_exact = {}
    for idx, sent in enumerate(sentences):
        key = normalize_for_hash(sent)
        if len(key) >= 30 and key in seen_exact:
            duplicate_sentence_pairs.append((seen_exact[key], idx))
        else:
            seen_exact[key] = idx

    semantic_pairs = []
    # Adjacent and nearby comparisons catch model loops without quadratic cost.
    for i in range(len(sentences)):
        for j in range(i + 1, min(len(sentences), i + 6)):
            a, b = sentences[i], sentences[j]
            if min(len(a), len(b)) < 35:
                continue
            sim = sentence_similarity(a, b)
            jac = jaccard(a, b)
            if sim >= 0.88 or (sim >= 0.72 and jac >= 0.58):
                semantic_pairs.append((i, j, round(sim, 3), round(jac, 3)))

    if duplicate_sentence_pairs:
        reasons.append(f"duplicate_sentences:{len(duplicate_sentence_pairs)}")
        score -= min(35, 15 * len(duplicate_sentence_pairs))

    if semantic_pairs:
        reasons.append(f"semantic_sentence_repetition:{len(semantic_pairs)}")
        score -= min(35, 12 * len(semantic_pairs))

    # Paragraph-level repetition.
    duplicate_paragraphs = 0
    for i in range(len(paragraphs)):
        for j in range(i + 1, len(paragraphs)):
            if len(paragraphs[i]) < 60 or len(paragraphs[j]) < 60:
                continue
            if sentence_similarity(paragraphs[i], paragraphs[j]) >= 0.84 or jaccard(paragraphs[i], paragraphs[j]) >= 0.70:
                duplicate_paragraphs += 1
    if duplicate_paragraphs:
        reasons.append(f"repeated_paragraphs:{duplicate_paragraphs}")
        score -= min(40, 20 * duplicate_paragraphs)

    # Boilerplate density.
    boilerplate_hits = sum(len(re.findall(p, lower)) for p in BOILERPLATE_PATTERNS)
    if boilerplate_hits >= 2:
        reasons.append(f"boilerplate_repetition:{boilerplate_hits}")
        score -= min(20, boilerplate_hits * 5)

    # Information density proxy: too many tiny sentences/paragraphs with little lexical variety.
    unique = len(set(significant_tokens(body)))
    lexical_ratio = unique / max(1, len(significant_tokens(body)))
    if wc >= 120 and lexical_ratio < 0.42:
        reasons.append(f"low_lexical_variety:{lexical_ratio:.2f}")
        score -= 20

    # A very short article is only hard-rejected when it also shows another defect.
    hard_reject = (
        # 35 words is only an absolute emergency floor for malformed/thin
        # fragments. The configurable min_words value is a REVIEW trigger, not
        # a publication floor.
        wc < 35
        or len(sentences) < 2
        or duplicate_paragraphs >= 1
        or len(duplicate_sentence_pairs) >= 1
        or len(semantic_pairs) >= 2
        or boilerplate_hits >= 3
        or placeholders
    )

    if hard_reject:
        decision = "REJECT"
    elif reasons:
        decision = "REVIEW"
    else:
        decision = "PASS"

    return {
        "decision": decision,
        "reasons": reasons,
        "score": round(max(0.0, score), 1),
        "word_count": wc,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "lexical_ratio": round(lexical_ratio, 3),
    }

def llm_judge(article, deterministic, expected_language=None):
    if chat is None:
        return {"decision": "PASS", "reason": "llm_unavailable"}

    # Only call the model for borderline/review cases. Obvious deterministic rejects
    # never need an expensive inference.
    if deterministic["decision"] != "REVIEW":
        return {"decision": "PASS", "reason": "not_needed"}

    lang = expected_language or article.get("language") or "the target language"
    body = "\n\n".join(article["paragraphs"])
    prompt = f"""
You are TrendCurrent's post-generation quality gate.

Review ONE already-generated news article in {lang}.

Your job is NOT to rewrite it and NOT to fact-check it against outside knowledge.
Judge only whether the article is publishable based on its own information density,
coherence, repetition and substantive value.

PASS if:
- it is a coherent standalone news story;
- it contains useful, concrete information;
- repetition is limited and does not materially reduce value;
- a concise article is still acceptable when it is information-dense.

REJECT if:
- it is mostly filler or generic commentary;
- the same information is repeated without a meaningful new detail;
- paragraphs/sentences loop or restate each other;
- it is so short that it fails to communicate a meaningful standalone story;
- it contains obvious generation leakage/placeholders.

Do NOT reject merely because it is short.
Do NOT demand a fixed word count.
Do NOT invent missing facts.
Return ONLY JSON:
{{"decision":"PASS" or "REJECT","reason":"short explanation"}}

TITLE:
{article["title"]}

ARTICLE:
{body}
"""
    try:
        started = time.perf_counter()
        response = chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "top_p": 0.85, "top_k": 40, "num_ctx": 4096, "num_predict": 220},
            format="json",
        )
        raw = response.message.content or ""
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return {"decision": "PASS", "reason": "invalid_llm_response"}
        result = json.loads(raw[start:end + 1])
        decision = str(result.get("decision", "PASS")).upper()
        if decision not in {"PASS", "REJECT"}:
            decision = "PASS"
        return {
            "decision": decision,
            "reason": str(result.get("reason", ""))[:500],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:
        # Quality tooling must never take production down.
        return {"decision": "PASS", "reason": f"llm_error:{exc}"}

def title_similarity(a, b):
    a = normalize_for_hash(a["title"])
    b = normalize_for_hash(b["title"])
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def duplicate_story_candidates(articles):
    """
    Find high-confidence duplicate stories within the SAME language.

    Candidate generation uses an inverted index over meaningful title tokens
    instead of one brittle title bucket. This catches paraphrased headlines that
    do not share the same first/sorted tokens while keeping comparisons bounded.
    """
    # Generic tokens are poor duplicate signals. Keep this list deliberately small
    # and language-neutral; entity/event words should remain available.
    generic = {
        "news", "latest", "update", "updates", "report", "reports", "breaking",
        "today", "new", "after", "before", "amid", "says", "said",
    }

    index = defaultdict(set)
    article_tokens = {}
    for idx, item in enumerate(articles):
        lang = item.get("language") or "unknown"
        words = {
            w for w in significant_tokens(item["title"])
            if len(w) >= 4 and w not in generic
        }
        article_tokens[idx] = words
        # Use several meaningful title tokens as candidate anchors.
        for word in words:
            index[(lang, word)].add(idx)

    candidate_pairs = set()
    for i, words in article_tokens.items():
        lang = articles[i].get("language") or "unknown"
        counts = defaultdict(int)
        for word in words:
            for j in index.get((lang, word), ()):
                if j > i:
                    counts[j] += 1

        # Require at least two shared meaningful title tokens before the more
        # expensive body comparison. A very high title similarity can qualify
        # with one shared token (or even zero) as a separate path below.
        for j, shared in counts.items():
            if shared >= 2:
                candidate_pairs.add((i, j))

    # Also consider near-identical titles even when tokenization differs.
    # This remains O(n²) only for a bounded recent/title inventory in practice;
    # avoid it for very large repositories.
    if len(articles) <= 3000:
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                if (articles[i].get("language") or "unknown") != (articles[j].get("language") or "unknown"):
                    continue
                ts = title_similarity(articles[i], articles[j])
                if ts >= 0.86:
                    candidate_pairs.add((i, j))

    pairs = []
    for i, j in sorted(candidate_pairs):
        a, b = articles[i], articles[j]
        ts = title_similarity(a, b)
        body_sim = jaccard(" ".join(a["paragraphs"]), " ".join(b["paragraphs"]))

        # Strong duplicate evidence. These thresholds are deliberately
        # conservative because related stories must not be collapsed.
        if ts >= 0.86 or (ts >= 0.72 and body_sim >= 0.55) or body_sim >= 0.80:
            pairs.append((a, b, round(ts, 3), round(body_sim, 3)))
    return pairs

def choose_duplicate_winner(a, b, quality_by_path):
    qa = quality_by_path.get(str(a["path"]), {})
    qb = quality_by_path.get(str(b["path"]), {})
    sa = float(qa.get("score", 0)) + min(word_count(a), 800) * 0.02
    sb = float(qb.get("score", 0)) + min(word_count(b), 800) * 0.02
    if sa > sb:
        return a, b
    if sb > sa:
        return b, a
    # Stable tie-breaker: keep the older file and remove the newer duplicate.
    try:
        return (a, b) if a["path"].stat().st_mtime <= b["path"].stat().st_mtime else (b, a)
    except OSError:
        return a, b

def discover_dirs(root, explicit_dirs=None):
    if explicit_dirs:
        return [Path(x).resolve() for x in explicit_dirs if Path(x).is_dir()]

    env_roots = os.getenv("ARTICLE_CLEANER_ROOTS", "").strip()
    if env_roots:
        result = [Path(x.strip()).resolve() for x in env_roots.split(",") if x.strip()]
        result = [x for x in result if x.is_dir()]
        if result:
            return result

    # Prefer the configured current-language TREND_DIR when available.
    result = []
    try:
        from config import TREND_DIR
        p = Path(TREND_DIR).resolve()
        if p.is_dir():
            result.append(p)
    except Exception:
        pass

    # Also discover the seven standard language directories in the repository.
    root = Path(root).resolve()
    for child in root.iterdir() if root.is_dir() else []:
        if child.is_dir() and child.name.casefold() in LANGUAGE_DIR_ALIASES:
            result.append(child.resolve())

    # If language folders are nested, find them one level deeper.
    for lang in LANGUAGE_DIR_ALIASES:
        for p in root.glob(f"**/{lang}"):
            if p.is_dir():
                result.append(p.resolve())

    # If no language folders exist, scan root itself as the article directory.
    if not result:
        result = [root]

    return list(dict.fromkeys(result))

def quarantine_path(path):
    qdir = path.parent / "_cleaner_quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    candidate = qdir / path.name
    if candidate.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = qdir / f"{path.stem}.{stamp}{path.suffix}"
    return candidate

def apply_action(path, delete=False, dry_run=False):
    if dry_run:
        return "DRY_RUN"
    if delete:
        try:
            path.unlink()
            return "DELETED"
        except OSError as exc:
            return f"DELETE_ERROR:{exc}"
    target = quarantine_path(path)
    try:
        shutil.move(str(path), str(target))
        return f"QUARANTINED:{target}"
    except OSError as exc:
        return f"QUARANTINE_ERROR:{exc}"

def clean(dirs, delete=False, dry_run=False, min_words=90, use_llm=True):
    all_paths = []
    for directory in dirs:
        try:
            all_paths.extend(p for p in directory.rglob("*.html") if "_cleaner_quarantine" not in p.parts)
        except OSError:
            continue

    # Do not treat site indexes/category pages as articles.
    paths = [
        p for p in sorted(set(all_paths))
        if p.name.casefold() not in {"index.html", "sitemap.html", "404.html"}
    ]

    print(f"[ARTICLE CLEANER] directories={len(dirs)} articles={len(paths)}")
    quality_by_path = {}
    parsed = []
    results = []

    for path in paths:
        article, error = parse_article(path)
        if error:
            results.append({"file": str(path), "decision": "REJECT", "reason": error})
            continue

        lang, lang_conf = detect_language(article)
        expected = article.get("language")
        if expected and lang and lang != expected and lang_conf >= 0.70:
            det = {"decision": "REJECT", "reasons": [f"wrong_language:{lang} expected:{expected}"], "score": 0}
        else:
            det = deterministic_quality(article, min_words)

        llm = {"decision": "PASS", "reason": "not_needed"}
        if use_llm and det["decision"] == "REVIEW":
            llm = llm_judge(article, det, expected_language=expected)
            final_decision = llm["decision"]
        else:
            final_decision = det["decision"]

        if final_decision == "REVIEW":
            # LLM disabled/unavailable: conservative keep.
            final_decision = "PASS"

        item = {
            "file": str(path),
            "language": expected or lang,
            "language_confidence": round(lang_conf, 3),
            "decision": final_decision,
            "deterministic": det,
            "llm": llm,
            "word_count": word_count(article),
            "title": article["title"],
        }
        results.append(item)

        # Duplicate selection must use the FINAL quality decision, not only the
        # deterministic score. A borderline article rejected by the LLM must never
        # become the winner of a duplicate pair.
        final_score = float(det.get("score", 0) or 0)
        if final_decision == "REJECT":
            final_score = 0.0
        quality_by_path[str(path)] = {
            "score": final_score,
            "decision": final_decision,
        }

        # Do not allow an already-rejected article to participate in duplicate
        # selection. It will be handled by the normal rejection pass below.
        if final_decision != "REJECT":
            parsed.append(article)

    # Cross-article duplicate stories.
    #
    # First build connected duplicate clusters. This prevents a chain such as
    # A~B and B~C from accidentally deleting B and then trying to use the already
    # removed B as the winner for C. Each cluster gets exactly ONE survivor.
    duplicates = duplicate_story_candidates(parsed)
    duplicate_actions = []
    if duplicates:
        parent = {}
        rank = {}

        def find_path(x):
            parent.setdefault(x, x)
            rank.setdefault(x, 0)
            if parent[x] != x:
                parent[x] = find_path(parent[x])
            return parent[x]

        def union_path(a, b):
            ra, rb = find_path(a), find_path(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        by_path = {str(a["path"]): a for a in parsed}
        for a, b, _, _ in duplicates:
            union_path(str(a["path"]), str(b["path"]))

        clusters = defaultdict(list)
        for path_str in parent:
            clusters[find_path(path_str)].append(by_path[path_str])

        for cluster in clusters.values():
            if len(cluster) < 2:
                continue

            # Select the single best survivor using final quality + a small
            # bounded length tie-breaker. Never select a rejected article because
            # rejected articles were excluded from `parsed`.
            winner = cluster[0]
            for candidate in cluster[1:]:
                winner, _loser = choose_duplicate_winner(
                    winner, candidate, quality_by_path
                )

            winner_path = str(winner["path"])
            for loser in cluster:
                loser_path = str(loser["path"])
                if loser_path == winner_path:
                    continue

                # Find the strongest direct duplicate evidence for logging.
                evidence = next(
                    (
                        (ts, bs)
                        for a, b, ts, bs in duplicates
                        if {
                            str(a["path"]), str(b["path"])
                        } == {winner_path, loser_path}
                    ),
                    (0.0, 0.0),
                )
                action = apply_action(loser["path"], delete=delete, dry_run=dry_run)
                duplicate_actions.append({
                    "winner": winner_path,
                    "loser": loser_path,
                    "title_similarity": evidence[0],
                    "body_similarity": evidence[1],
                    "action": action,
                })
                for item in results:
                    if item["file"] == loser_path:
                        item["decision"] = "DUPLICATE"
                        item["duplicate_of"] = winner_path
                        item["action"] = action
                        break

    # Apply deterministic/LLM rejections.
    for item in results:
        if item.get("decision") not in {"REJECT"}:
            continue
        # Already handled as a duplicate.
        if item.get("action"):
            continue
        action = apply_action(Path(item["file"]), delete=delete, dry_run=dry_run)
        item["action"] = action

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "directories": [str(x) for x in dirs],
        "articles_scanned": len(paths),
        "pass": sum(1 for x in results if x.get("decision") == "PASS"),
        "rejected": sum(1 for x in results if x.get("decision") == "REJECT"),
        "duplicates": sum(1 for x in results if x.get("decision") == "DUPLICATE"),
        "actions": {
            # Each affected article is represented once in results, so count from
            # results only. Counting duplicate_actions too would double-count every
            # duplicate action.
            "deleted": sum(1 for x in results if str(x.get("action","")).startswith("DELETED")),
            "quarantined": sum(1 for x in results if str(x.get("action","")).startswith("QUARANTINED")),
            "dry_run": sum(1 for x in results if x.get("action") == "DRY_RUN"),
        },
        "duplicate_pairs": duplicate_actions,
        "results": results,
    }

    print(
        f"[ARTICLE CLEANER] PASS={summary['pass']} "
        f"REJECT={summary['rejected']} DUPLICATE={summary['duplicates']} "
        f"SCANNED={summary['articles_scanned']}"
    )

    log_path = os.getenv("ARTICLE_CLEANER_LOG", "").strip()
    if log_path:
        lp = Path(log_path)
    else:
        base = dirs[0].parent if dirs else Path(".")
        lp = base / "cleaner_logs" / f"article_cleaner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ARTICLE CLEANER] log={lp}")
    except OSError as exc:
        print(f"[ARTICLE CLEANER] log write failed: {exc}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="TrendCurrent post-generation article cleaner")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--dirs", nargs="*", help="explicit article directories")
    parser.add_argument("--delete", action="store_true", help="permanently delete rejected/duplicate files")
    parser.add_argument("--dry-run", action="store_true", help="scan and report without changing files")
    parser.add_argument("--no-llm", action="store_true", help="disable Ollama judge for borderline articles")
    parser.add_argument("--min-words", type=int, default=int(os.getenv("ARTICLE_CLEANER_MIN_WORDS", "90")))
    args = parser.parse_args()

    delete = args.delete or os.getenv("ARTICLE_CLEANER_DELETE", "").casefold() in {"1", "true", "yes", "on"}
    dirs = discover_dirs(args.root, args.dirs)
    if not dirs:
        print("[ARTICLE CLEANER] No article directories found.")
        return 0

    print("[ARTICLE CLEANER] roots:")
    for d in dirs:
        print(f"  - {d}")

    clean(
        dirs,
        delete=delete,
        dry_run=args.dry_run,
        min_words=max(35, args.min_words),
        use_llm=not args.no_llm,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
