"""TrendCurrent generator observability logger.

Logging only. This module must never influence pipeline decisions or generation.
All logger failures are swallowed so production behavior continues unchanged.
"""
import json
import os
import threading
import uuid
from datetime import datetime

_LOCK = threading.Lock()
_CURRENT = None
_RUN_ID = None

def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def _safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return str(value)

def _write(record):
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        filename = "generator_%s.jsonl" % datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(log_dir, filename)
        line = json.dumps(_safe(record), ensure_ascii=False, separators=(",", ":"))
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Observability must never break production.
        pass

def start_run(language=None, model=None, pipeline=None, max_articles=None):
    global _RUN_ID
    try:
        _RUN_ID = uuid.uuid4().hex[:12]
        _write({
            "record": "run",
            "event": "run_start",
            "timestamp": _now(),
            "run_id": _RUN_ID,
            "language": language,
            "model": model,
            "pipeline": pipeline,
            "max_articles": max_articles,
        })
        return _RUN_ID
    except Exception:
        return None

def end_run(generated=None, status="FINISHED", error=None):
    try:
        _write({
            "record": "run",
            "event": "run_end",
            "timestamp": _now(),
            "run_id": _RUN_ID,
            "generated": generated,
            "status": status,
            "error": error,
        })
    except Exception:
        pass

def start_candidate(keyword, trend=None):
    global _CURRENT
    try:
        cid = uuid.uuid4().hex[:12]
        _CURRENT = {"candidate_id": cid, "keyword": str(keyword or "")}
        _write({
            "record": "candidate",
            "event": "candidate_start",
            "timestamp": _now(),
            "run_id": _RUN_ID,
            "candidate_id": cid,
            "keyword": keyword,
            "trend": trend,
        })
        return cid
    except Exception:
        return None

def clear_candidate():
    global _CURRENT
    _CURRENT = None

def candidate_event(event, **data):
    try:
        current = _CURRENT or {}
        _write({
            "record": "candidate",
            "event": event,
            "timestamp": _now(),
            "run_id": _RUN_ID,
            "candidate_id": current.get("candidate_id"),
            "keyword": current.get("keyword"),
            **data,
        })
    except Exception:
        pass

def finish_candidate(status, reason=None, slug=None):
    try:
        candidate_event(
            "candidate_end",
            status=status,
            reason=reason,
            slug=slug,
        )
    finally:
        clear_candidate()
