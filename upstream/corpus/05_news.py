import os, re, time, html, datetime, hashlib
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from common import *

OUT = os.path.join(RAW, "news"); os.makedirs(OUT, exist_ok=True)


def google_news():
    rows = []
    editions = [("en-US", "US", "US:en", NEWS_QUERIES_EN), ("ko", "KR", "KR:ko", NEWS_QUERIES_KO)]
    for hl, gl, ceid, queries in editions:
        for q in queries:
            for a, b in month_windows():
                b1 = b + datetime.timedelta(days=1)
                phrase = q in ("fatty liver disease", "steatotic liver disease", "대사이상 지방간")
                qq = f'"{q}" after:{a:%Y-%m-%d} before:{b1:%Y-%m-%d}' if phrase else f'{q} after:{a:%Y-%m-%d} before:{b1:%Y-%m-%d}'
                qid = hashlib.md5(q.encode("utf-8")).hexdigest()[:8]
                fn = os.path.join(OUT, f"gnews_{gl}_{qid}_{a:%Y%m}.xml")
                if not os.path.exists(fn):
                    r = http("GET", "https://news.google.com/rss/search",
                             params={"q": qq, "hl": hl, "gl": gl, "ceid": ceid}, backoff=10)
                    open(fn, "wb").write(r.content); time.sleep(1.5)
                try:
                    root = ET.parse(fn).getroot()
                except ET.ParseError:
                    continue
                for it in root.iter("item"):
                    src = it.find("source")
                    rows.append({"stream": "google_news", "edition": gl, "query": q,
                                 "title": html.unescape(it.findtext("title") or ""),
                                 "url": it.findtext("link"), "outlet": src.text if src is not None else None,
                                 "outlet_url": src.get("url") if src is not None else None,
                                 "published": it.findtext("pubDate"), "slice": f"{a:%Y-%m}"})
        log_event("news.google.edition", edition=gl, rows_so_far=len(rows))
    return rows


BREAKER = 5


def gdelt(cache_only=False):
    rows, ok, fail, consecutive_fail = [], 0, 0, 0
    q = '(MASLD OR NAFLD OR MAFLD OR "fatty liver" OR "steatotic liver" OR "steatohepatitis")'
    for a, b in month_windows():
        if consecutive_fail >= BREAKER:
            fail += 2
            continue
        for half in ((a, a.replace(day=15)), (a.replace(day=16), b)):
            fn = os.path.join(OUT, f"gdelt_{half[0]:%Y%m%d}.json")
            if not os.path.exists(fn) and cache_only:
                fail += 1; continue
            if not os.path.exists(fn):
                got = None
                for i in range(3):
                    time.sleep((6, 20, 60)[i])
                    try:
                        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", timeout=90, params={
                            "query": q, "mode": "artlist", "format": "json", "maxrecords": 250, "sort": "datedesc",
                            "startdatetime": f"{half[0]:%Y%m%d}000000", "enddatetime": f"{half[1]:%Y%m%d}235959"})
                        if r.status_code == 200 and r.text.strip().startswith("{"):
                            got = r.text; break
                        if r.status_code == 200:
                            got = '{"articles": []}'; break
                    except requests.RequestException:
                        pass
                if got is None:
                    fail += 1; consecutive_fail += 1; continue
                open(fn, "w", encoding="utf-8").write(got)
            ok += 1; consecutive_fail = 0
            import json
            try:
                arts = json.load(open(fn, encoding="utf-8")).get("articles", [])
            except Exception:
                arts = []
            for x in arts:
                rows.append({"stream": "gdelt", "edition": x.get("sourcecountry"), "query": q,
                             "title": x.get("title"), "url": x.get("url"), "outlet": x.get("domain"),
                             "outlet_url": x.get("domain"), "published": x.get("seendate"),
                             "language": x.get("language"), "slice": f"{half[0]:%Y-%m}"})
    log_event("news.gdelt", windows_ok=ok, windows_failed=fail, rows=len(rows), query=q)
    return rows


def parse_date(s):
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y%m%dT%H%M%SZ"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


if __name__ == "__main__":
    import sys
    log_event("news.start", env=environment_info())
    if "--gdelt-download-only" in sys.argv:
        BREAKER = 10 ** 9
        for _pass in range(3):
            gdelt()
        sys.exit(0)
    rows = google_news()
    if "--no-gdelt" not in sys.argv:
        rows += gdelt(cache_only="--gdelt-cache-only" in sys.argv)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.published.map(lambda s: parse_date(s) if isinstance(s, str) else None), errors="coerce")
    n0 = len(df)
    df = df[(df.date >= DATE_START) & (df.date <= DATE_END)]
    df["title_norm"] = df.title.fillna("").str.replace(r"\s+-\s+[^-]+$", "", regex=True).str.lower().str.replace(r"[^a-z0-9가-힣]+", " ", regex=True).str.strip()
    n1 = len(df)
    df = df.sort_values("date").drop_duplicates(["title_norm"])
    df.to_parquet(os.path.join(PROC, "news.parquet"), index=False)
    log_event("news.done", raw_items=n0, in_window=n1, deduplicated=len(df),
              google=int((df.stream == "google_news").sum()), gdelt=int((df.stream == "gdelt").sum()))
