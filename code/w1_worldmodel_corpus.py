from __future__ import annotations

import gzip
import json
import ssl
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from common import PKG, log

OUT = PKG / "analysis" / "world_model"
RAW = OUT / "pubtator_txt"
EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PT = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/pubtator"

TERMS = ['"fatty liver"', 'NAFLD', 'NAFL', '"nonalcoholic steatohepatitis"', '"non-alcoholic steatohepatitis"', 'NASH',
         'MAFLD', '"metabolic dysfunction-associated fatty liver"', '"metabolic associated fatty liver"', 'MASLD',
         '"metabolic dysfunction-associated steatotic liver"', 'MASH', '"metabolic dysfunction-associated steatohepatitis"',
         'MetALD', '"steatotic liver disease"', '"hepatic steatosis"', '"liver steatosis"']
QUERY = ("(" + " OR ".join(t + "[tiab]" for t in TERMS) + ' OR "Non-alcoholic Fatty Liver Disease"[MeSH])'
         ' NOT ("Published Erratum"[pt] OR "Retraction of Publication"[pt] OR Comment[pt] OR Editorial[pt]'
         ' OR Letter[pt] OR News[pt])')
TYPES = {"Gene", "Chemical", "Disease", "Variant"}
VARIANT_TYPES = {"Variant", "SNP", "DNAMutation", "ProteinMutation"}


def session():
    import truststore

    class TA(HTTPAdapter):
        def init_poolmanager(self, *a, **kw):
            kw["ssl_context"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            return super().init_poolmanager(*a, **kw)
    s = requests.Session()
    s.mount("https://", TA(max_retries=3))
    s.headers["User-Agent"] = "research script; no contact"
    return s


def _get(s, method, url, **kw):
    for attempt in range(6):
        try:
            r = s.request(method, url, timeout=120, **kw)
            if r.status_code == 200:
                return r
            time.sleep(2 + 3 * attempt)
        except requests.RequestException:
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"failed: {url}")


def esearch(s, a: date, b: date):
    term = f'{QUERY} AND ("{a:%Y/%m/%d}"[dp] : "{b:%Y/%m/%d}"[dp])'
    r = _get(s, "POST", EU + "esearch.fcgi", data={"db": "pubmed", "term": term, "retmax": 9999, "retmode": "json"})
    js = r.json()["esearchresult"]
    time.sleep(0.4)
    return int(js["count"]), js.get("idlist", [])


def collect_pmids(s):
    rows = []
    for y in range(2000, 2027):
        end = date(2026, 6, 30) if y == 2026 else date(y, 12, 31)
        stack = [(date(y, 1, 1), end)]
        while stack:
            a, b = stack.pop()
            n, ids = esearch(s, a, b)
            if n >= 9999 and (b - a).days > 0:
                mid = a + timedelta(days=(b - a).days // 2)
                stack += [(a, mid), (mid + timedelta(days=1), b)]
                continue
            rows += [{"pmid": int(i), "year": y} for i in ids]
        log("w1.esearch", year=y, n=sum(r["year"] == y for r in rows))
    p = pd.DataFrame(rows).drop_duplicates("pmid")
    p.to_csv(OUT / "pmids.csv", index=False)
    return p


def fetch_pubtator(s, pmids):
    RAW.mkdir(parents=True, exist_ok=True)
    pm = sorted(pmids)
    for k in range(0, len(pm), 100):
        f = RAW / f"batch_{k // 100:05d}.txt.gz"
        if f.exists():
            continue
        chunk = pm[k:k + 100]
        r = _get(s, "GET", PT, params={"pmids": ",".join(map(str, chunk))})
        with gzip.open(f, "wt", encoding="utf-8") as fh:
            fh.write(r.text)
        time.sleep(0.35)
        if (k // 100) % 50 == 0:
            log("w1.pubtator", batch=k // 100, of=len(pm) // 100 + 1)


def parse(pm_year):
    docs, anns = [], []
    for f in sorted(RAW.glob("batch_*.txt.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        title, abstract, raw = {}, {}, []
        for line in lines:
            if "|t|" in line[:20]:
                pm, _, t = line.split("|", 2); title[int(pm)] = t
            elif "|a|" in line[:20]:
                pm, _, a = line.split("|", 2); abstract[int(pm)] = a
            else:
                fl = line.split("	")
                if len(fl) >= 6:
                    raw.append(fl)
        for pm in title:
            docs.append((pm, title[pm], abstract.get(pm, "")))
        for fl in raw:
            pm, s0, e0, men, t, ident = int(fl[0]), int(fl[1]), int(fl[2]), fl[3], fl[4], fl[5]
            t = "Variant" if t in VARIANT_TYPES else t
            if t not in TYPES or not ident or ident in ("-", "None"):
                continue
            tl = len(title.get(pm, ""))
            part, off = ("title", 0) if s0 < tl else ("abstract", tl + 1)
            anns.append((pm, part, s0 - off, e0 - off, t, f"{t}:{ident}", men))
    D = pd.DataFrame(docs, columns=["pmid", "title", "abstract"]).drop_duplicates("pmid")
    D = D.merge(pm_year, on="pmid", how="left")
    A = pd.DataFrame(anns, columns=["pmid", "part", "start", "end", "type", "entity", "mention"])
    D.to_parquet(OUT / "docs.parquet", index=False)
    A.to_parquet(OUT / "ann.parquet", index=False)
    log("w1.parse", docs=len(D), annotations=len(A), entities=A.entity.nunique(),
        missing_from_pubtator=int(len(pm_year) - len(D)))
    return D, A


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = session()
    if "--parse-only" not in sys.argv:
        p = pd.read_csv(OUT / "pmids.csv") if (OUT / "pmids.csv").exists() else collect_pmids(s)
        fetch_pubtator(s, p.pmid.tolist())
    if "--fetch-only" not in sys.argv:
        parse(pd.read_csv(OUT / "pmids.csv"))


if __name__ == "__main__":
    main()
