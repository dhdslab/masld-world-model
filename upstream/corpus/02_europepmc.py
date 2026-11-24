import os, json
import pandas as pd
from common import *

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OUT = os.path.join(RAW, "europepmc"); os.makedirs(OUT, exist_ok=True)
QUERY = f"{EPMC_QUERY_CORE} AND FIRST_PDATE:[{DATE_START} TO {DATE_END}]"


def harvest():
    cursor, page, n = "*", 0, 0
    total = None
    while True:
        fn = os.path.join(OUT, f"page_{page:04d}.jsonl.gz")
        r = http("GET", API, params={"query": QUERY, "format": "json", "pageSize": 1000,
                                     "resultType": "core", "cursorMark": cursor})
        js = r.json()
        total = js["hitCount"] if total is None else total
        res = js.get("resultList", {}).get("result", [])
        if not res:
            break
        write_jsonl_gz(fn, res)
        n += len(res); page += 1
        print("epmc page", page, n, "/", total, flush=True)
        nxt = js.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
    log_event("europepmc.harvest", hitCount=total, retrieved=n, pages=page, query=QUERY)
    return page


def sj(xs, sep=";"):
    return sep.join(str(x) for x in (xs or []) if x)


def flat(rec):
    g = lambda *ks: _get(rec, ks)
    ft = rec.get("fullTextUrlList", {}).get("fullTextUrl", [])
    affs = []
    for a in rec.get("authorList", {}).get("author", []) or []:
        for af in (a.get("authorAffiliationDetailsList", {}) or {}).get("authorAffiliation", []) or []:
            affs.append(af.get("affiliation"))
    return {
        "epmc_id": rec.get("id"), "epmc_source": rec.get("source"),
        "pmid": rec.get("pmid"), "pmcid": rec.get("pmcid"),
        "doi": (rec.get("doi") or "").lower() or None,
        "title": rec.get("title"), "abstract": rec.get("abstractText"),
        "journal": g("journalInfo", "journal", "title"),
        "pub_year": rec.get("pubYear"), "first_pub_date": rec.get("firstPublicationDate"),
        "pub_types": sj(rec.get("pubTypeList", {}).get("pubType", [])),
        "language": rec.get("language"),
        "is_open_access": rec.get("isOpenAccess"), "in_epmc": rec.get("inEPMC"),
        "in_pmc": rec.get("inPMC"), "has_pdf": rec.get("hasPDF"), "license": rec.get("license"),
        "cited_by": rec.get("citedByCount"),
        "fulltext_urls": ";".join(f"{u.get('availability')}|{u.get('documentStyle')}|{u.get('url')}" for u in ft),
        "keywords": sj(rec.get("keywordList", {}).get("keyword", [])),
        "mesh": ";".join(m.get("descriptorName", "") for m in rec.get("meshHeadingList", {}).get("meshHeading", []) or []),
        "affiliation": rec.get("affiliation"),
        "all_affiliations": sj(affs, " | ")[:20000],
        "author_string": (rec.get("authorString") or "")[:2000],
        "grants": ";".join((x.get("agency") or "") for x in rec.get("grantsList", {}).get("grant", []) or []),
        "preprint_server": g("bookOrReportDetails", "publisher") if rec.get("source") == "PPR" else None,
    }


def _get(d, ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


if __name__ == "__main__":
    import sys, glob
    log_event("europepmc.start", env=environment_info())
    if "--parse-only" in sys.argv:
        pages = len(glob.glob(os.path.join(OUT, "page_*.jsonl.gz")))
        log_event("europepmc.parse_only", pages=pages)
    else:
        pages = harvest()
    rows = []
    for p in range(pages):
        rows += [flat(r) for r in read_jsonl_gz(os.path.join(OUT, f"page_{p:04d}.jsonl.gz"))]
    df = pd.DataFrame(rows).drop_duplicates(["epmc_source", "epmc_id"])
    df.to_parquet(os.path.join(PROC, "europepmc.parquet"), index=False)
    log_event("europepmc.done", rows=len(df), open_access=int((df.is_open_access == "Y").sum()),
              preprints=int((df.epmc_source == "PPR").sum()), with_abstract=int(df.abstract.notna().sum()))
