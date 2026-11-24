import os, gzip, json, time
import xml.etree.ElementTree as ET
import pandas as pd
from common import *

EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OUT = os.path.join(RAW, "pubmed"); os.makedirs(OUT, exist_ok=True)
DELAY = 0.4


def esearch_ids(term):
    r = http("POST", EU + "esearch.fcgi", data={"db": "pubmed", "term": term, "retmax": 9999, "retmode": "json"})
    js = r.json()["esearchresult"]
    time.sleep(DELAY)
    return int(js["count"]), js.get("idlist", [])


def harvest_ids():
    full = f'{PUBMED_QUERY_CORE} AND ("2020/01/01"[dp] : "2026/06/30"[dp])'
    total, _ = esearch_ids(full)
    log_event("pubmed.esearch.total", count=total, query=full)
    ids, slices = set(), []
    for a, b in month_windows():
        q = f'{PUBMED_QUERY_CORE} AND ("{a:%Y/%m/%d}"[dp] : "{b:%Y/%m/%d}"[dp])'
        n, lst = esearch_ids(q)
        if n > 9999:
            raise RuntimeError(f"slice {a} has {n} > 9999")
        ids.update(lst); slices.append({"from": str(a), "to": str(b), "count": n})
    json.dump(slices, open(os.path.join(OUT, "esearch_slices.json"), "w"), indent=1)
    log_event("pubmed.esearch.union", union=len(ids), total=total, missing=total - len(ids))
    if len(ids) < total:
        log_event("pubmed.esearch.note", msg="union below total; residual recovered by EDAT sweep")
        for a, b in list(month_windows("2019-01-01", "2026-06-30")):
            q = f'{full} AND ("{a:%Y/%m/%d}"[edat] : "{b:%Y/%m/%d}"[edat])'
            n, lst = esearch_ids(q); ids.update(lst)
        q = f'{full} AND ("2026/07/01"[edat] : "3000"[edat])'
        n, lst = esearch_ids(q); ids.update(lst)
        q = f'{full} AND ("1900/01/01"[edat] : "2018/12/31"[edat])'
        n, lst = esearch_ids(q); ids.update(lst)
        log_event("pubmed.esearch.union_after_edat", union=len(ids), total=total)
    ids = sorted(ids, key=int)
    with open(os.path.join(OUT, "pmids.txt"), "w") as f:
        f.write("\n".join(ids))
    return ids, total


def efetch(ids):
    files = []
    for i in range(0, len(ids), 200):
        fn = os.path.join(OUT, f"efetch_{i // 200:05d}.xml.gz")
        files.append(fn)
        if os.path.exists(fn):
            continue
        r = http("POST", EU + "efetch.fcgi", data={"db": "pubmed", "id": ",".join(ids[i:i + 200]), "retmode": "xml"})
        with gzip.open(fn, "wb") as f:
            f.write(r.content)
        time.sleep(DELAY)
        if (i // 200) % 25 == 0:
            print("efetch", i, "/", len(ids), flush=True)
    return files


def txt(el):
    return "".join(el.itertext()).strip() if el is not None else None


def parse(files):
    rows = []
    for fn in files:
        root = ET.parse(gzip.open(fn)).getroot()
        for art in root.findall("PubmedArticle"):
            mc = art.find("MedlineCitation"); a = mc.find("Article")
            ids = {x.get("IdType"): x.text for x in art.findall("PubmedData/ArticleIdList/ArticleId")}
            abs_parts = []
            for t in a.findall("Abstract/AbstractText"):
                lab = t.get("Label"); s = txt(t)
                abs_parts.append(f"{lab}: {s}" if lab else s)
            pd_el = a.find("Journal/JournalIssue/PubDate")
            year = txt(pd_el.find("Year")) if pd_el is not None else None
            medline = txt(pd_el.find("MedlineDate")) if pd_el is not None else None
            month = txt(pd_el.find("Month")) if pd_el is not None else None
            ad = a.find("ArticleDate")
            epub = None
            if ad is not None:
                epub = f"{txt(ad.find('Year'))}-{txt(ad.find('Month'))}-{txt(ad.find('Day'))}"
            affs = [txt(x) for x in a.findall("AuthorList/Author/AffiliationInfo/Affiliation")]
            first_aff = None
            au = a.find("AuthorList/Author")
            if au is not None:
                first_aff = txt(au.find("AffiliationInfo/Affiliation"))
            last_aff = None
            aus = a.findall("AuthorList/Author")
            if aus:
                last_aff = txt(aus[-1].find("AffiliationInfo/Affiliation"))
            rows.append({
                "pmid": txt(mc.find("PMID")),
                "doi": (ids.get("doi") or "").lower() or None,
                "pmcid": ids.get("pmc"),
                "title": txt(a.find("ArticleTitle")),
                "abstract": "\n".join(p for p in abs_parts if p) or None,
                "journal": txt(a.find("Journal/Title")),
                "journal_iso": txt(a.find("Journal/ISOAbbreviation")),
                "pub_year": year or (medline[:4] if medline else None),
                "pub_month": month, "epub_date": epub,
                "language": ";".join(x.text for x in a.findall("Language")),
                "pub_types": ";".join(txt(x) for x in a.findall("PublicationTypeList/PublicationType")),
                "mesh": ";".join(txt(x.find("DescriptorName")) for x in mc.findall("MeshHeadingList/MeshHeading")),
                "keywords": ";".join(txt(x) or "" for x in mc.findall("KeywordList/Keyword")),
                "n_authors": len(aus),
                "first_author_aff": first_aff, "last_author_aff": last_aff,
                "all_affiliations": " | ".join(x for x in affs if x)[:20000],
                "grants": ";".join(txt(x.find("Agency")) or "" for x in a.findall("GrantList/Grant")),
            })
        for bk in root.findall("PubmedBookArticle"):
            bd = bk.find("BookDocument")
            ids = {x.get("IdType"): x.text for x in bk.findall("PubmedBookData/ArticleIdList/ArticleId")}
            rows.append({"pmid": txt(bd.find("PMID")), "doi": (ids.get("doi") or "").lower() or None, "pmcid": None,
                         "title": txt(bd.find("ArticleTitle")) or txt(bd.find("Book/BookTitle")),
                         "abstract": "\n".join(txt(t) or "" for t in bd.findall("Abstract/AbstractText")) or None,
                         "journal": txt(bd.find("Book/BookTitle")), "journal_iso": None,
                         "pub_year": txt(bd.find("Book/PubDate/Year")), "pub_month": None, "epub_date": None,
                         "language": ";".join(x.text for x in bd.findall("Language")),
                         "pub_types": "Book Chapter;" + ";".join(txt(x) for x in bd.findall("PublicationType")),
                         "mesh": "", "keywords": "", "n_authors": len(bd.findall("AuthorList/Author")),
                         "first_author_aff": None, "last_author_aff": None, "all_affiliations": None, "grants": ""})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    log_event("pubmed.start", env=environment_info())
    if "--parse-only" in sys.argv:
        ids = open(os.path.join(OUT, "pmids.txt")).read().split(); total = len(ids)
    else:
        ids, total = harvest_ids()
    files = efetch(ids)
    df = parse(files)
    df = df.drop_duplicates("pmid")
    df.to_parquet(os.path.join(PROC, "pubmed.parquet"), index=False)
    log_event("pubmed.done", esearch_total=total, pmids=len(ids), parsed=len(df),
              with_abstract=int(df.abstract.notna().sum()), with_pmcid=int(df.pmcid.notna().sum()))
