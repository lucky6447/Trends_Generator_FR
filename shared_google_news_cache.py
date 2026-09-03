from __future__ import annotations
import hashlib, os, sqlite3, threading, time
from pathlib import Path
from urllib.parse import urlencode
import requests

CACHE_TTL_SECONDS=max(60,int(os.environ.get("TREND_GOOGLE_CACHE_TTL","900")))
GLOBAL_MIN_INTERVAL_SECONDS=max(0.0,float(os.environ.get("TREND_GOOGLE_MIN_INTERVAL","3.0")))
CACHE_DB_PATH=Path(os.environ.get("TREND_GOOGLE_CACHE_DB",str(Path.home()/"Documents"/"TrendCurrent_Shared"/"google_news_cache.sqlite3"))).expanduser()
CACHE_DB_PATH.parent.mkdir(parents=True,exist_ok=True)
_LOCAL_LOCK=threading.Lock()

def _connect():
    c=sqlite3.connect(str(CACHE_DB_PATH),timeout=30)
    c.execute("PRAGMA busy_timeout=30000")
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS rss_cache(cache_key TEXT PRIMARY KEY,fetched_at REAL NOT NULL,body BLOB NOT NULL,url TEXT NOT NULL)")
    c.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    c.commit(); return c

def _key(url,params):
    q=urlencode(sorted((str(k),str(v)) for k,v in (params or {}).items()),doseq=True)
    return hashlib.sha256((url+"?"+q).encode("utf-8")).hexdigest()

def _cached(k):
    c=_connect()
    try:
        r=c.execute("SELECT fetched_at,body,url FROM rss_cache WHERE cache_key=?",(k,)).fetchone()
        if not r:return None
        age=time.time()-float(r[0])
        if age>CACHE_TTL_SECONDS:return None
        return bytes(r[1]),age,r[2]
    finally:c.close()

def _last():
    c=_connect()
    try:
        r=c.execute("SELECT value FROM meta WHERE key='last_google_request_at'").fetchone()
        return float(r[0]) if r else 0.0
    finally:c.close()

def _set(v):
    c=_connect()
    try:
        c.execute("INSERT INTO meta(key,value) VALUES('last_google_request_at',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(v),));c.commit()
    finally:c.close()

def _wait():
    with _LOCAL_LOCK:
        while True:
            w=GLOBAL_MIN_INTERVAL_SECONDS-(time.time()-_last())
            if w<=0:break
            time.sleep(min(w,0.5))
        _set(time.time())

def get_rss(*,url,params,headers=None,timeout=20):
    k=_key(url,params); c=_cached(k)
    if c:return c[0],True,c[1],None
    _wait()
    try:
        r=requests.get(url,params=params,headers=headers or {},timeout=timeout,allow_redirects=True)
        r.raise_for_status();body=bytes(r.content);now=time.time();db=_connect()
        try:
            db.execute("INSERT INTO rss_cache(cache_key,fetched_at,body,url) VALUES(?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET fetched_at=excluded.fetched_at,body=excluded.body,url=excluded.url",(k,now,sqlite3.Binary(body),r.url));db.commit()
        finally:db.close()
        return body,False,0.0,None
    except Exception as exc:
        return b"",False,0.0,f"{type(exc).__name__}: {exc}"
