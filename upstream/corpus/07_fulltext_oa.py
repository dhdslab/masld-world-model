import os, gzip, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from common import *

I = pd.read_parquet(os.path.join(PROC, "literature_included.parquet"))
el = I[I.open_access & I.pmcid.notna()][["record_id", "pmcid", "license"]].drop_duplicates("pmcid")


def get(row):
    pmc = row.pmcid
    fn = os.path.join(FULLTEXT, f"{pmc}.xml.gz")
    if os.path.exists(fn) and os.path.getsize(fn) > 200:
        return (row.record_id, pmc, "downloaded", os.path.getsize(fn))
    r = None
    for i in range(3):
        try:
            r = SESSION.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML", timeout=90)
            if r.status_code not in (429, 502, 503, 504):
                break
        except Exception:
            r = None
        time.sleep(3 * (i + 1))
    if r is None:
        return (row.record_id, pmc, "error_network", 0)
    if r.status_code == 200 and b"<article" in r.content[:5000]:
        with gzip.open(fn, "wb") as f:
            f.write(r.content)
        return (row.record_id, pmc, "downloaded", os.path.getsize(fn))
    return (row.record_id, pmc, f"not_available_http_{r.status_code}", 0)


if __name__ == "__main__":
    log_event("fulltext.start", eligible=len(el))
    out = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        for k, res in enumerate(ex.map(get, el.itertuples(index=False))):
            out.append(res)
            if k % 1000 == 0:
                print("fulltext", k, "/", len(el), flush=True)
    M = pd.DataFrame(out, columns=["record_id", "pmcid", "status", "bytes_gz"])
    M.to_parquet(os.path.join(PROC, "fulltext_manifest.parquet"), index=False)
    log_event("fulltext.done", eligible=len(el), **{f"status_{k}": int(v) for k, v in M.status.value_counts().items()},
              total_gb=round(M.bytes_gz.sum() / 1e9, 2))
