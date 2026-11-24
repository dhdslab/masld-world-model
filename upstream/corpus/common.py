import os, json, time, gzip, datetime, hashlib, platform, sys
import requests

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.abspath(os.environ.get("MASLD_CORPUS_DIR", os.path.join(_REPO, "data", "corpus")))
RAW = os.path.join(ROOT, "raw")
PROC = os.path.join(ROOT, "processed")
FULLTEXT = os.path.join(ROOT, "fulltext")
FIG = os.path.join(ROOT, "figures")
LOGS = os.path.join(ROOT, "logs")
for d in (RAW, PROC, FULLTEXT, FIG, LOGS):
    os.makedirs(d, exist_ok=True)

DATE_START = "2020-01-01"
DATE_END = "2026-06-30"

TERMS = [
    '"fatty liver"', 'NAFLD', 'NAFL',
    '"nonalcoholic steatohepatitis"', '"non-alcoholic steatohepatitis"', 'NASH',
    'MAFLD', '"metabolic dysfunction-associated fatty liver"', '"metabolic associated fatty liver"',
    'MASLD', '"metabolic dysfunction-associated steatotic liver"', 'MASH',
    '"metabolic dysfunction-associated steatohepatitis"', 'MetALD',
    '"steatotic liver disease"', '"hepatic steatosis"', '"liver steatosis"',
]

PUBMED_QUERY_CORE = ("(" + " OR ".join(t + "[tiab]" for t in TERMS)
                     + ' OR "Non-alcoholic Fatty Liver Disease"[MeSH])')

EPMC_QUERY_CORE = "(" + " OR ".join(
    f"TITLE_ABS:{t}" if t.startswith('"') else f'TITLE_ABS:"{t}"' for t in TERMS) + ")"

OPENALEX_SEARCH = " OR ".join([
    '"fatty liver"', 'NAFLD', '"nonalcoholic steatohepatitis"', '"non-alcoholic steatohepatitis"',
    'MAFLD', 'MASLD', '"metabolic dysfunction-associated steatohepatitis"', 'MetALD',
    '"steatotic liver disease"', '"hepatic steatosis"'])

NEWS_QUERIES_EN = ['MASLD', 'MASH liver', 'NAFLD', 'NASH liver', 'fatty liver disease',
                   'MAFLD', 'steatotic liver disease']
NEWS_QUERIES_KO = ['지방간', '대사이상 지방간', 'MASLD']

USER_AGENT = "MASLD-evidence-collection/1.0 (academic research; python-requests)"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log_event(step, **kw):
    rec = {"utc": now(), "step": step, **kw}
    with open(os.path.join(LOGS, "pipeline_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{rec['utc']}] {step}: " + ", ".join(f"{k}={v}" for k, v in kw.items() if k != "query")[:400], flush=True)


def environment_info():
    import pandas
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "requests": requests.__version__, "pandas": pandas.__version__}


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def http(method, url, retries=6, backoff=5, ok=(200,), **kw):
    kw.setdefault("timeout", 120)
    last = None
    for i in range(retries):
        try:
            r = SESSION.request(method, url, **kw)
            if r.status_code in ok:
                return r
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (400, 404):
                return r
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"failed after {retries} tries: {url} :: {last}")


def write_jsonl_gz(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def month_windows(start=DATE_START, end=DATE_END):
    s = datetime.date.fromisoformat(start); e = datetime.date.fromisoformat(end)
    cur = s.replace(day=1)
    while cur <= e:
        nxt = (cur.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        yield cur, min(nxt - datetime.timedelta(days=1), e)
        cur = nxt


import re
NASH_NOISE = re.compile(r"\bnash[\s\-]+(equilibri\w*|bargain\w*|sutcliffe|solution|product|game|welfare|"
                        r"embedding|inequalit\w*|theorem|map|demand|cournot|stable)\b|\bjohn\s+nash\b", re.I)
DISEASE_CS = re.compile(r"\b(NAFLD|NAFL|NASH|MAFLD|MASLD|MASH|MetALD)\b")
DISEASE_CI = re.compile(r"fatty[\s\-]+liver|steatohepatitis|steatotic[\s\-]+liver|hepatic[\s\-]+steatosis|"
                        r"liver[\s\-]+steatosis|steatosis[\s\-]+of[\s\-]+the[\s\-]+liver|"
                        r"metabolic[\s\-]+(dysfunction[\s\-]+)?associated[\s\-]+(fatty|steatotic)|liver fat|"
                        r"hepatosteatosis|fatty[\s\-]+hepatocyte|hepatic[\s\-]+lipid[\s\-]+accumulation", re.I)


DISEASE_KO = re.compile(r"지방간|지방간염|간\s*지방|대사이상\s*지방|MASLD|NAFLD|MASH")


def is_relevant(text, korean=False):
    t = NASH_NOISE.sub(" ", str(text or ""))
    if DISEASE_CI.search(t) or DISEASE_CS.search(t):
        return True
    return bool(korean and DISEASE_KO.search(t))
