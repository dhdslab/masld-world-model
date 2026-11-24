import os, json, re
import pandas as pd
from common import *

OUT = os.path.join(RAW, "registries"); os.makedirs(OUT, exist_ok=True)

CT_COND = ('NAFLD OR NASH OR MASLD OR MASH OR MAFLD OR MetALD OR "fatty liver" OR '
           '"steatotic liver disease" OR "hepatic steatosis" OR "nonalcoholic steatohepatitis" OR '
           '"non-alcoholic fatty liver disease" OR "metabolic dysfunction-associated steatohepatitis"')


def clinicaltrials():
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.cond": CT_COND, "pageSize": 1000, "countTotal": "true",
              "filter.advanced": f"AREA[StudyFirstSubmitDate]RANGE[{DATE_START},{DATE_END}]"}
    studies, token, total = [], None, None
    while True:
        p = dict(params); 
        if token: p["pageToken"] = token
        js = http("GET", url, params=p).json()
        total = js.get("totalCount", total)
        studies += js.get("studies", [])
        token = js.get("nextPageToken")
        if not token: break
    write_jsonl_gz(os.path.join(OUT, "clinicaltrials_raw.jsonl.gz"), studies)
    rows = []
    for s in studies:
        ps = s["protocolSection"]; idm = ps.get("identificationModule", {}); st = ps.get("statusModule", {})
        dm = ps.get("designModule", {}); sp = ps.get("sponsorCollaboratorsModule", {})
        arms = ps.get("armsInterventionsModule", {}); cl = ps.get("contactsLocationsModule", {})
        rows.append({
            "nct_id": idm.get("nctId"), "title": idm.get("officialTitle") or idm.get("briefTitle"),
            "brief_summary": ps.get("descriptionModule", {}).get("briefSummary"),
            "conditions": ";".join(ps.get("conditionsModule", {}).get("conditions", [])),
            "study_type": dm.get("studyType"), "phases": ";".join(dm.get("phases", []) or []),
            "allocation": (dm.get("designInfo") or {}).get("allocation"),
            "enrollment": (dm.get("enrollmentInfo") or {}).get("count"),
            "status": st.get("overallStatus"), "first_submitted": st.get("studyFirstSubmitDate"),
            "start_date": (st.get("startDateStruct") or {}).get("date"),
            "completion_date": (st.get("completionDateStruct") or {}).get("date"),
            "has_results": s.get("hasResults"),
            "lead_sponsor": (sp.get("leadSponsor") or {}).get("name"),
            "sponsor_class": (sp.get("leadSponsor") or {}).get("class"),
            "interventions": ";".join(f"{i.get('type')}:{i.get('name')}" for i in arms.get("interventions", []) or []),
            "countries": ";".join(sorted({l.get("country") for l in cl.get("locations", []) or [] if l.get("country")})),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(PROC, "clinicaltrials.parquet"), index=False)
    log_event("clinicaltrials.done", totalCount=total, rows=len(df), query=CT_COND)


def nih_reporter():
    url = "https://api.reporter.nih.gov/v2/projects/search"
    text = ("NAFLD OR NASH OR MASLD OR MASH OR MAFLD OR MetALD OR \"fatty liver\" OR "
            "\"steatotic liver\" OR \"hepatic steatosis\" OR \"steatohepatitis\"")
    allr = []
    for fy in range(2020, 2027):
        off = 0
        while True:
            body = {"criteria": {"fiscal_years": [fy], "advanced_text_search": {
                        "operator": "advanced", "search_field": "projecttitle,abstracttext", "search_text": text}},
                    "offset": off, "limit": 500}
            js = http("POST", url, json=body).json()
            res = js.get("results", []) or []
            allr += res
            tot = js["meta"]["total"]
            off += 500
            if off >= tot or not res or off >= 14999:
                break
            time.sleep(1)
        log_event("nih_reporter.fy", fiscal_year=fy, total=tot)
        time.sleep(1)
    write_jsonl_gz(os.path.join(OUT, "nih_reporter_raw.jsonl.gz"), allr)
    rows = [{"appl_id": r.get("appl_id"), "project_num": r.get("project_num"),
             "core_project_num": r.get("core_project_num"), "fiscal_year": r.get("fiscal_year"),
             "title": r.get("project_title"), "abstract": r.get("abstract_text"),
             "activity_code": r.get("activity_code"), "award_amount": r.get("award_amount"),
             "org": (r.get("organization") or {}).get("org_name"),
             "org_country": (r.get("organization") or {}).get("org_country"),
             "ic": (r.get("agency_ic_admin") or {}).get("abbreviation"),
             "project_start": r.get("project_start_date"), "award_notice_date": r.get("award_notice_date"),
             "pref_terms": (r.get("pref_terms") or "")[:4000]} for r in allr]
    df = pd.DataFrame(rows).drop_duplicates("appl_id")
    df = df[pd.to_datetime(df.award_notice_date, errors="coerce").fillna(pd.Timestamp("2000-01-01")) <= pd.Timestamp(DATE_END)
            ] if df.award_notice_date.notna().any() else df
    df.to_parquet(os.path.join(PROC, "nih_reporter.parquet"), index=False)
    log_event("nih_reporter.done", rows=len(df), unique_core_projects=df.core_project_num.nunique(), query=text)


def openfda():
    url = "https://api.fda.gov/drug/label.json"
    q = ('(indications_and_usage:("steatohepatitis" OR "MASH" OR "NASH" OR "fatty liver" OR "steatotic liver") '
         'OR warnings_and_cautions:("steatohepatitis" OR "MASH" OR "NASH" OR "steatotic liver")) '
         f'AND effective_time:[{DATE_START.replace("-", "")} TO {DATE_END.replace("-", "")}]')
    res, skip = [], 0
    while True:
        js = http("GET", url, params={"search": q, "limit": 100, "skip": skip}).json()
        if "results" not in js: break
        res += js["results"]; skip += 100
        if skip >= js["meta"]["results"]["total"] or skip >= 25000: break
    write_jsonl_gz(os.path.join(OUT, "openfda_labels_raw.jsonl.gz"), res)
    rows = []
    for r in res:
        of = r.get("openfda", {})
        ind = " ".join(r.get("indications_and_usage", []) or [])
        rows.append({"set_id": r.get("set_id"), "effective_time": r.get("effective_time"),
                     "brand_name": ";".join(of.get("brand_name", []) or []),
                     "generic_name": ";".join(of.get("generic_name", []) or []),
                     "manufacturer": ";".join(of.get("manufacturer_name", []) or []),
                     "indication_mentions_disease": bool(re.search(r"steatohepatitis|\bMASH\b|\bNASH\b|fatty liver|steatotic", ind, re.I)),
                     "indications_excerpt": ind[:1500]})
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(PROC, "openfda_labels.parquet"), index=False)
    log_event("openfda.done", rows=len(df), indication_rows=int(df.indication_mentions_disease.sum()) if len(df) else 0, query=q)


if __name__ == "__main__":
    log_event("registries.start", env=environment_info())
    clinicaltrials(); nih_reporter(); openfda()
