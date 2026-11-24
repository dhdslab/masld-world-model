import os
import pandas as pd
from common import *

API = "https://api.openalex.org/works"
OUT = os.path.join(RAW, "openalex"); os.makedirs(OUT, exist_ok=True)
FILTER = (f"title_and_abstract.search:{OPENALEX_SEARCH},"
          f"from_publication_date:{DATE_START},to_publication_date:{DATE_END}")
SELECT = ("id,doi,ids,title,publication_date,publication_year,type,language,open_access,"
          "cited_by_count,primary_location,best_oa_location,authorships,primary_topic,"
          "abstract_inverted_index,is_retracted,fwci")


def harvest():
    key = os.environ["OPENALEX_API_KEY"]
    cursor, page, n, total = "*", 0, 0, None
    while cursor:
        fn = os.path.join(OUT, f"page_{page:05d}.jsonl.gz")
        r = http("GET", API, params={"filter": FILTER, "per_page": 200, "cursor": cursor,
                                     "select": SELECT, "api_key": key})
        js = r.json()
        total = js["meta"]["count"] if total is None else total
        res = js.get("results", [])
        if not res:
            break
        write_jsonl_gz(fn, res)
        n += len(res); page += 1
        if page % 25 == 0:
            print("openalex page", page, n, "/", total, flush=True)
        cursor = js["meta"].get("next_cursor")
    log_event("openalex.harvest", count=total, retrieved=n, pages=page, filter=FILTER)
    return page


def deinvert(inv):
    if not inv:
        return None
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))


def flat(w):
    ids = w.get("ids") or {}
    pl = w.get("primary_location") or {}
    src = pl.get("source") or {}
    oa = w.get("open_access") or {}
    countries, insts = set(), []
    for a in w.get("authorships") or []:
        countries.update(a.get("countries") or [])
    first = (w.get("authorships") or [{}])[0] if w.get("authorships") else {}
    last = (w.get("authorships") or [{}])[-1] if w.get("authorships") else {}
    pt = w.get("primary_topic") or {}
    pmid = (ids.get("pmid") or "").replace("https://pubmed.ncbi.nlm.nih.gov/", "") or None
    pmcid = (ids.get("pmcid") or "").rstrip("/").split("/")[-1] or None
    return {
        "openalex_id": w["id"].split("/")[-1],
        "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower() or None,
        "pmid": pmid, "pmcid": pmcid,
        "title": w.get("title"), "abstract": deinvert(w.get("abstract_inverted_index")),
        "pub_date": w.get("publication_date"), "pub_year": w.get("publication_year"),
        "oa_type": w.get("type"), "language": w.get("language"),
        "journal": src.get("display_name"), "source_type": src.get("type"),
        "is_oa": oa.get("is_oa"), "oa_status": oa.get("oa_status"), "oa_url": oa.get("oa_url"),
        "cited_by": w.get("cited_by_count"), "fwci": w.get("fwci"),
        "is_retracted": w.get("is_retracted"),
        "countries": ";".join(sorted(countries)),
        "first_author_countries": ";".join(first.get("countries") or []),
        "last_author_countries": ";".join(last.get("countries") or []),
        "n_authors": len(w.get("authorships") or []),
        "topic": pt.get("display_name"), "subfield": (pt.get("subfield") or {}).get("display_name"),
        "field": (pt.get("field") or {}).get("display_name"),
    }


if __name__ == "__main__":
    log_event("openalex.start", env=environment_info())
    pages = harvest()
    rows = []
    for p in range(pages):
        rows += [flat(w) for w in read_jsonl_gz(os.path.join(OUT, f"page_{p:05d}.jsonl.gz"))]
    df = pd.DataFrame(rows).drop_duplicates("openalex_id")
    df.to_parquet(os.path.join(PROC, "openalex.parquet"), index=False)
    log_event("openalex.done", rows=len(df), with_abstract=int(df.abstract.notna().sum()),
              is_oa=int(df.is_oa.fillna(False).sum()), with_pmid=int(df.pmid.notna().sum()))
