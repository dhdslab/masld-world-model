from __future__ import annotations

import json
import re
import sys
import time
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ctgov import ADV_COND, MASLD_TERM, get
from common import to_group
from common import (ANA, TOPICS_ANA, TOPICS_EMB, CORPUS_PROC, GROUP_OF_TYPE, PKG, REF_RUN, SEED, SRC, log, now,
                    update_results, wilson)

URL = "https://clinicaltrials.gov/api/v2/studies"
CACHE = ANA / "registry_landscape.json"
PM_CACHE = ANA / "registry_landscape_pubmed.json"
FIELDS = ",".join(["protocolSection.identificationModule", "protocolSection.statusModule",
                   "protocolSection.sponsorCollaboratorsModule.leadSponsor", "protocolSection.conditionsModule",
                   "protocolSection.designModule", "protocolSection.armsInterventionsModule.interventions",
                   "protocolSection.outcomesModule.primaryOutcomes", "protocolSection.eligibilityModule.eligibilityCriteria",
                   "LocationCountry", "protocolSection.referencesModule.references", "hasResults"])
Q1 = {"query.term": MASLD_TERM, "filter.advanced": "AREA[StudyFirstSubmitDate]RANGE[2010-01-01,2025-12-31] AND AREA[StudyType]INTERVENTIONAL"}
Q2 = {"query.cond": ADV_COND, "filter.advanced": "AREA[StudyFirstSubmitDate]RANGE[2020-01-01,2025-12-31] AND AREA[StudyType]INTERVENTIONAL"}

MASLD_RX = re.compile(r"\b(nafld|nash|masld|mash|mafld|metald)\b|fatty liver|steatohepatitis|steatotic liver|"
                      r"metabolic[- ]dysfunction[- ]associated|non[- ]?alcoholic (fatty|steato)", re.I)
NEG = r"(?:no|not|without|absence of|free of|never|excluding|other than|non)[- ]"
CIRR_NEG = re.compile(r"non[- ]?cirrhotic|without (?:evidence of |any )?(?:clinical |histological )?cirrhosis|no (?:evidence of |history of )?"
                      r"cirrhosis|absence of cirrhosis|not cirrhotic|exclud\w* cirrhosis|cirrhosis (?:is|are|will be) excluded|"
                      r"no (?:clinical )?(?:signs?|features?) of cirrhosis", re.I)
CIRR = re.compile(r"\bcirrho(?:sis|tic)\b|\bf4\b|stage 4 fibrosis|fibrosis stage 4|metavir f4|ishak (?:5|6)", re.I)
DECOMP_NEG = re.compile(r"(?:no|without|absence of|free of|never|no history of|no prior|no previous)\s+(?:\w+\s+){0,5}?"
                        r"(?:decompensat\w*|ascites|encephalopathy|variceal (?:bleed\w*|haemorrhage|hemorrhage))", re.I)
DECOMP = re.compile(r"decompensat\w*|\bascites\b|hepatic encephalopathy|variceal (?:bleed\w*|haemorrhage|hemorrhage)|"
                    r"child[- ]?(?:turcotte[- ])?pugh (?:class |score )?(?:b|c)\b|child[- ]?pugh (?:score )?(?:of )?(?:[7-9]|1[0-5])\b|"
                    r"acute[- ]on[- ]chronic liver failure|\baclf\b|hepatorenal", re.I)
HCC = re.compile(r"hepatocellular carcinoma|\bhcc\b|liver cancer|hepatoma|primary liver (?:cancer|tumou?r)", re.I)
HCC_NEG = re.compile(r"(?:no|without|history of|absence of|free of)\s+(?:\w+\s+){0,4}?(?:hepatocellular carcinoma|\bhcc\b|liver cancer)", re.I)
TRANSPLANT = re.compile(r"liver transplant\w*|transplant recipients?|post[- ]?transplant", re.I)
FIBROSIS = re.compile(r"\bf[1-3]\b|f[1-3]\s*(?:-|–|to)\s*f?[2-3]|stage [1-3] fibrosis|fibrosis stage [1-3]|steatohepatitis|\bnash\b|\bmash\b|"
                      r"nafld activity score|\bnas\b|significant fibrosis|advanced fibrosis|at[- ]risk (?:nash|mash)", re.I)
HEP_IMP = re.compile(r"hepatic impairment|renal impairment", re.I)
HEALTHY = re.compile(r"healthy (?:volunteers?|participants?|subjects?|adults?)", re.I)

EP_CLIN = re.compile(r"\bdeath\b|mortality|overall survival|\bsurvival\b|decompensat\w*|transplant\w*|liver[- ]related (?:event|outcome)|"
                     r"clinical (?:event|outcome)|cardiovascular events?|\bmace\b|hospitali[sz]\w*|variceal bleed\w*|"
                     r"(?:incidence|occurrence|development) of (?:hcc|hepatocellular carcinoma|ascites|cirrhosis)|progression to cirrhosis|"
                     r"time to (?:first )?(?:event|decompensation)", re.I)
EP_HIST = re.compile(r"histolog\w*|biops\w+|nafld activity score|\bnas\b|fibrosis (?:stage|improvement|regression)|improvement in fibrosis|"
                     r"(?:resolution|remission) of (?:nash|mash|steatohepatitis)|(?:nash|mash) resolution|kleiner|ishak|\bsaf\b", re.I)
EP_SURR = re.compile(r"mri[- ]?pdff|\bpdff\b|proton density fat|liver fat|hepatic fat|intrahepatic (?:lipid|triglyceride|fat)|steatosis|"
                     r"controlled attenuation|\bcap\b|elastograph\w*|stiffness|fibroscan|\bmre\b|\balt\b|\bast\b|aminotransferase|"
                     r"transaminase|fib-?4|\belf\b|ck-?18|biomarker|hba1c|insulin (?:resistance|sensitivity)|homa|body weight|weight loss|"
                     r"\bbmi\b|lipid|triglyceride|cholesterol|objective response|response rate|\borr\b|progression[- ]free|\bpfs\b|"
                     r"hepatic venous pressure|\bhvpg\b|\bmeld\b|child[- ]pugh score|ammonia|liver (?:function|enzymes?)", re.I)
EP_PK = re.compile(r"pharmacokinetic|\bauc\b|\bcmax\b|adverse events?|safety|tolerab\w+|maximum tolerated|dose[- ]limiting|"
                   r"treatment[- ]emergent", re.I)

STAGES = ["Steatosis or unspecified", "Steatohepatitis or fibrosis (non-cirrhotic)", "Compensated cirrhosis enrolled",
          "Decompensated cirrhosis", "Liver cancer (HCC)", "Liver transplantation", "Hepatic impairment or healthy volunteers"]
ENDPOINTS = ["Clinical outcome", "Histology", "Imaging or biomarker", "Pharmacokinetics or safety", "Other"]

REG_RX = re.compile(r"\b(NCT\d{8}|ISRCTN\d{8}|ChiCTR[-\w]{6,20}\d|IRCT\d{6,}N\d+|CTRI/\d{4}/\d{2,3}/\d{6}|"
                    r"(?:EudraCT[ :]*(?:No\.?|number)?[ :]*)?\d{4}-\d{6}-\d{2}|ACTRN\d{14}|jRCT[a-z]?\d{9,10}|UMIN\d{9}|"
                    r"KCT\d{7}|DRKS\d{8}|NTR\d{3,5}|NL\d{4,5}|PACTR\d{15}|TCTR\d{11}|RBR-\w{6,8}|SLCTR/\d{4}/\d{3})", re.I)


def registry_of(i):
    i = i.upper()
    for k in ("NCT", "ISRCTN", "CHICTR", "IRCT", "CTRI", "ACTRN", "JRCT", "UMIN", "KCT", "DRKS", "NTR", "PACTR", "TCTR", "RBR", "SLCTR"):
        if i.startswith(k):
            return {"CHICTR": "ChiCTR", "JRCT": "jRCT"}.get(k, k)
    if re.match(r"(EUDRACT)?.*\d{4}-\d{6}-\d{2}", i):
        return "EudraCT"
    return "NL" if i.startswith("NL") else "Other"


def split_elig(t):
    t = t or ""
    m = re.search(r"exclusion criteria", t, re.I)
    return (t[:m.start()], t[m.start():]) if m else (t, "")


def masld_status(r):
    head = " ".join([r["title"], r["official_title"], r["conditions"], r["keywords"]])
    inc, exc = split_elig(r["eligibility"])
    if MASLD_RX.search(head):
        return "Targeted"
    if MASLD_RX.search(inc):
        return "Listed as eligible"
    if MASLD_RX.search(exc):
        return "Excluded"
    return "Not mentioned"


LINE_NEG = re.compile(r"^\s*(?:\d+[.)]\s*)?(?:no|not|without|absence|never|free of|must not|should not|exclu\w*|"
                      r"patients? (?:with|who have) no)\b", re.I)
HCC_NOT_TARGET = re.compile(r"risk|prevent\w*|surveillance|screening|predict\w*|chemoprevention", re.I)


def positive_inclusion(inc):
    lines = re.split(r"\n|\*|•|;", inc or "")
    return " \n ".join(x for x in lines if x.strip() and not LINE_NEG.search(x))


def stage_of(r):
    title = r["title"] + " " + r["official_title"]
    head = " ".join([title, r["conditions"], r["keywords"]])
    inc, _ = split_elig(r["eligibility"])
    pos = positive_inclusion(inc)
    t = title + " " + r["conditions"] + " \n " + pos
    if HEP_IMP.search(title) or HEALTHY.search(title):
        return STAGES[6]
    if TRANSPLANT.search(title) or re.search(r"(?:recipients? of|underwent|post)[- ]?(?:a )?liver transplant", pos, re.I):
        return STAGES[5]
    if (HCC.search(title) and not HCC_NOT_TARGET.search(title)) or re.search(
            r"(?:diagnos\w+|confirmed|documented|proven)\s+(?:\w+\s+){0,3}(?:hcc|hepatocellular carcinoma)|\bbclc\b|unresectable",
            HCC_NEG.sub(" ", pos), re.I):
        return STAGES[4]
    t_dec = DECOMP_NEG.sub(" ", t)
    t_cir = CIRR_NEG.sub(" ", t)
    if DECOMP.search(t_dec) and CIRR.search(t_cir + " " + t_dec):
        return STAGES[3]
    if re.search(r"decompensated", head, re.I):
        return STAGES[3]
    if CIRR.search(t_cir):
        return STAGES[2]
    if FIBROSIS.search(t):
        return STAGES[1]
    return STAGES[0]


def endpoint_of(r):
    txt = " ".join(r["primary_outcomes"])
    for lab, rx in (("Clinical outcome", EP_CLIN), ("Histology", EP_HIST), ("Imaging or biomarker", EP_SURR),
                    ("Pharmacokinetics or safety", EP_PK)):
        if rx.search(txt):
            return lab
    return "Other"


def _compact(s):
    ps = s.get("protocolSection", {})
    idm, st, cond = ps.get("identificationModule", {}), ps.get("statusModule", {}), ps.get("conditionsModule", {})
    des = ps.get("designModule", {})
    pc = st.get("primaryCompletionDateStruct", {})
    return {"nct_id": idm.get("nctId"), "title": idm.get("briefTitle", ""), "official_title": idm.get("officialTitle", "") or "",
            "conditions": "; ".join(cond.get("conditions", [])), "keywords": "; ".join(cond.get("keywords", [])),
            "phases": ";".join(des.get("phases", []) or []), "allocation": (des.get("designInfo") or {}).get("allocation", ""),
            "purpose": (des.get("designInfo") or {}).get("primaryPurpose", ""),
            "enrollment": (des.get("enrollmentInfo") or {}).get("count"), "enrollment_type": (des.get("enrollmentInfo") or {}).get("type", ""),
            "status": st.get("overallStatus", ""), "first_submitted": st.get("studyFirstSubmitDate", ""),
            "start": (st.get("startDateStruct") or {}).get("date", ""), "primary_completion": pc.get("date", ""),
            "primary_completion_type": pc.get("type", ""), "results_first_post": (st.get("resultsFirstPostDateStruct") or {}).get("date", ""),
            "results_first_submit": st.get("resultsFirstSubmitDate", "") or "",
            "has_results": bool(s.get("hasResults")),
            "sponsor_class": ((ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}).get("class", ""),
            "sponsor": ((ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}).get("name", ""),
            "interventions": "; ".join(f"{i.get('type', '')}:{i.get('name', '')}" for i in (ps.get("armsInterventionsModule") or {}).get("interventions", [])),
            "primary_outcomes": [f"{o.get('measure', '')} {o.get('description', '')}" for o in (ps.get("outcomesModule") or {}).get("primaryOutcomes", [])],
            "eligibility": (ps.get("eligibilityModule") or {}).get("eligibilityCriteria", ""),
            "countries": sorted({l.get("country", "") for l in (ps.get("contactsLocationsModule") or {}).get("locations", []) if l.get("country")}),
            "references": [(x.get("pmid"), x.get("type")) for x in (ps.get("referencesModule") or {}).get("references", []) if x.get("pmid")]}


def fetch(params):
    out, token = [], None
    while True:
        p = dict(params, pageSize=500, fields=FIELDS)
        if token:
            p["pageToken"] = token
        r = get(URL, p); r.raise_for_status(); js = r.json(); time.sleep(0.4)
        out += [_compact(s) for s in js.get("studies", [])]
        token = js.get("nextPageToken")
        if not token:
            return out


def registry(refresh=False):
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    a, b = fetch(Q1), fetch(Q2)
    recs = {r["nct_id"]: dict(r, in_q1=True, in_q2=False) for r in a}
    for r in b:
        recs.setdefault(r["nct_id"], dict(r, in_q1=False, in_q2=True))["in_q2"] = True
    out = {"retrieved_utc": now(), "q1": Q1, "q2": Q2, "records": list(recs.values())}
    CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def pubmed_meta(pmids, refresh=False):
    cache = json.loads(PM_CACHE.read_text(encoding="utf-8")) if PM_CACHE.exists() and not refresh else {}
    need = [p for p in sorted(set(map(str, pmids))) if p.isdigit() and p not in cache]
    for i in range(0, len(need), 200):
        r = get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                {"db": "pubmed", "id": ",".join(need[i:i + 200]), "retmode": "xml", "tool": "masld_evidence_map"})
        r.raise_for_status(); time.sleep(0.45)
        root = ET.fromstring(r.content)
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//MedlineCitation/PMID")
            d = None
            ad = art.find(".//Article/ArticleDate")
            if ad is not None:
                d = f"{ad.findtext('Year')}-{int(ad.findtext('Month') or 1):02d}-{int(ad.findtext('Day') or 1):02d}"
            if d is None:
                ph = art.find(".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']")
                if ph is not None:
                    d = f"{ph.findtext('Year')}-{int(ph.findtext('Month') or 1):02d}-{int(ph.findtext('Day') or 1):02d}"
            cache[pmid] = {"date": d, "title": art.findtext(".//Article/ArticleTitle") or "",
                           "pub_types": [x.text for x in art.findall(".//PublicationType")]}
    PM_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def km(t, e, at=(12, 24, 36, 48)):
    t, e = np.asarray(t, float), np.asarray(e, int)
    order = np.argsort(t); t, e = t[order], e[order]
    S, var, out, times = 1.0, 0.0, {}, np.unique(t[e == 1])
    curve = [(0.0, 0.0, 0.0, 0.0)]
    for u in times:
        n_at = (t >= u).sum(); d = ((t == u) & (e == 1)).sum()
        if n_at > d:
            var += d / (n_at * (n_at - d))
        S *= 1 - d / n_at
        se = S * np.sqrt(var)
        curve.append((float(u), 1 - S, float(max(0, 1 - S - 1.96 * se)), float(min(1, 1 - S + 1.96 * se))))
    for a in at:
        Sa, va = 1.0, 0.0
        for u in times[times <= a]:
            n_at = (t >= u).sum(); d = ((t == u) & (e == 1)).sum()
            Sa *= 1 - d / n_at
            if n_at > d:
                va += d / (n_at * (n_at - d))
        se = Sa * np.sqrt(va)
        out[str(a)] = {"published": float(1 - Sa), "ci": [float(max(0, 1 - Sa - 1.96 * se)), float(min(1, 1 - Sa + 1.96 * se))],
                       "n_at_risk": int((t >= a).sum())}
    out["n_at_risk"] = {str(a): int((t >= a).sum()) for a in (0, 12, 24, 36, 48, 60)}
    return out, curve


def months_between(a, b):
    return (b - a).days / 30.4375


def to_date(s):
    s = (s or "").strip()
    if not s:
        return pd.NaT
    if len(s) == 7:
        s += "-15"
    return pd.to_datetime(s, errors="coerce")


def main(refresh=False):
    reg = registry(refresh)
    D = pd.DataFrame(reg["records"])
    cutoff = pd.to_datetime(reg["retrieved_utc"][:10])
    D["first_year"] = D.first_submitted.str[:4].astype(int)
    D["masld_status"] = D.apply(masld_status, axis=1)
    D["stage"] = D.apply(stage_of, axis=1)
    D["endpoint"] = D.apply(endpoint_of, axis=1)
    D["randomized"] = D.allocation.eq("RANDOMIZED")
    D["phase_group"] = np.select([D.phases.str.contains("PHASE3|PHASE4"), D.phases.str.contains("PHASE2"),
                                  D.phases.str.contains("PHASE1|EARLY_PHASE1")], ["Phase 3-4", "Phase 2", "Phase 1"], "Not applicable")
    D["enrollment"] = pd.to_numeric(D.enrollment, errors="coerce")
    D["region"] = D.countries.apply(lambda c: "Multinational" if len(c) > 1 else (c[0] if c else "Unknown"))
    out = {"retrieved_utc": reg["retrieved_utc"], "n_q1": int(D.in_q1.sum()), "n_q2": int(D.in_q2.sum()), "n_union": int(len(D))}

    T = D[D.in_q1 & D.masld_status.eq("Targeted")].copy()
    T["period"] = pd.cut(T.first_year, [2009, 2014, 2019, 2025], labels=["2010-2014", "2015-2019", "2020-2025"])
    out["masld_targeted_n"] = int(len(T))
    out["masld_status_q1"] = D[D.in_q1].masld_status.value_counts().to_dict()
    tab = T.groupby(["period", "stage"], observed=False).size().unstack(fill_value=0).reindex(columns=STAGES, fill_value=0)
    out["stage_by_period"] = {str(k): v for k, v in tab.T.to_dict().items()}
    out["stage_total"] = T.stage.value_counts().reindex(STAGES, fill_value=0).to_dict()
    enr = T.groupby("stage").enrollment.sum().reindex(STAGES, fill_value=0)
    out["enrollment_by_stage"] = enr.to_dict()
    out["enrollment_share_by_stage"] = (enr / enr.sum()).to_dict()
    ep = T.groupby(["stage", "endpoint"]).size().unstack(fill_value=0).reindex(index=STAGES, columns=ENDPOINTS, fill_value=0)
    out["endpoint_by_stage"] = ep.T.to_dict()
    out["endpoint_total"] = T.endpoint.value_counts().reindex(ENDPOINTS, fill_value=0).to_dict()
    adv_mask = T.stage.isin(STAGES[2:6])
    out["advanced_share_of_targeted"] = {"k": int(adv_mask.sum()), "n": int(len(T)), "share": float(adv_mask.mean()),
                                         "ci": wilson(int(adv_mask.sum()), int(len(T)))}
    out["advanced_share_by_period"] = {str(p): float(g.stage.isin(STAGES[2:6]).mean()) for p, g in T.groupby("period", observed=False)}
    out["decomp_or_hcc_share_of_targeted"] = float(T.stage.isin(STAGES[3:5]).mean())
    out["clinical_endpoint_share"] = {"all": float((T.endpoint == "Clinical outcome").mean()),
                                      "phase3": float((T[T.phase_group == "Phase 3-4"].endpoint == "Clinical outcome").mean())}
    out["phase3_n"] = int((T.phase_group == "Phase 3-4").sum())
    drug = T.phase_group.isin(["Phase 2", "Phase 3-4"]) & T.interventions.str.contains("DRUG|BIOLOGICAL", na=False)
    out["advanced_share_phase2_4_drug"] = {"k": int((drug & adv_mask).sum()), "n": int(drug.sum()),
                                           "share": float((drug & adv_mask).sum() / drug.sum()),
                                           "ci": wilson(int((drug & adv_mask).sum()), int(drug.sum()))}
    out["industry_share"] = float((T.sponsor_class == "INDUSTRY").mean())
    T["adv"] = adv_mask.astype(int); T["yc"] = T.first_year - 2010
    m = smf.glm("adv ~ yc", data=T, family=sm.families.Binomial()).fit()
    out["advanced_share_or_per_year"] = {"or": float(np.exp(m.params["yc"])), "ci": np.exp(m.conf_int().loc["yc"]).tolist()}
    T.drop(columns=["eligibility"]).to_csv(SRC / "trial_landscape_masld_targeted.csv", index=False)

    A = D[D.in_q2].copy()
    out["advanced_any_masld_status"] = A.masld_status.value_counts().to_dict()
    out["advanced_any_masld_status_share"] = A.masld_status.value_counts(normalize=True).to_dict()
    As = A[A.stage.isin(STAGES[2:6])]
    out["advanced_staged_n"] = int(len(As))
    out["advanced_staged_masld_status"] = As.masld_status.value_counts().to_dict()
    out["advanced_staged_masld_status_share"] = As.masld_status.value_counts(normalize=True).to_dict()
    out["advanced_staged_targeted_share_by_stage"] = As.groupby("stage").masld_status.apply(lambda s_: float((s_ == "Targeted").mean())).to_dict()
    out["advanced_any_stage"] = A.stage.value_counts().to_dict()
    A.drop(columns=["eligibility"]).to_csv(SRC / "trial_landscape_advanced_any_aetiology.csv", index=False)

    L = pd.read_parquet(CORPUS_PROC / "literature_included_final.parquet", columns=["record_id", "pmid", "year", "title", "abstract"])
    L["ids"] = [sorted(set(x.upper() for x in REG_RX.findall(f"{a} {b}"))) for a, b in zip(L.title.fillna(""), L.abstract.fillna(""))]
    nct_pmids = {}
    for pm, ids in zip(L.pmid.astype(str), L.ids):
        for i in ids:
            if i.startswith("NCT") and pm.isdigit():
                nct_pmids.setdefault(i, set()).add(pm)
    C = T[T.status.isin(["COMPLETED", "TERMINATED"]) & T.primary_completion_type.eq("ACTUAL")].copy()
    C["pc_date"] = C.primary_completion.apply(to_date)
    C = C[(C.pc_date >= "2010-01-01") & (C.pc_date <= cutoff - pd.Timedelta(days=365))]
    linked = {r.nct_id: {str(p) for p, ty in r.references if ty in ("RESULT", "DERIVED")} | nct_pmids.get(r.nct_id, set())
              for r in C.itertuples()}
    meta = pubmed_meta(set().union(*linked.values()) if linked else set(), refresh)
    PROTO = re.compile(r"protocol|rationale and design|design and rationale|study design|baseline characteristics|"
                       r"statistical analysis plan|methodology of", re.I)
    first_pub, n_links = [], []
    for r in C.itertuples():
        ds = []
        for p in linked.get(r.nct_id, ()):
            mm = meta.get(p)
            if not mm or not mm.get("date"):
                continue
            if "Clinical Trial Protocol" in mm["pub_types"] or PROTO.search(mm["title"]):
                continue
            d = pd.to_datetime(mm["date"], errors="coerce")
            if pd.notna(d) and d >= r.pc_date - pd.Timedelta(days=31):
                ds.append(d)
        first_pub.append(min(ds) if ds else pd.NaT); n_links.append(len(ds))
    C["first_pub"] = first_pub
    C["event"] = C.first_pub.notna().astype(int)
    C["months"] = np.where(C.event == 1, (C.first_pub - C.pc_date).dt.days / 30.4375, (cutoff - C.pc_date).dt.days / 30.4375)
    C["months"] = C.months.clip(lower=0.01)
    C["results_submitted_12m"] = [bool(rp) and (to_date(rp) - pc).days <= 365 for rp, pc in zip(C.results_first_submit, C.pc_date)]
    C["results_posted_any"] = C.has_results.astype(bool)
    pub = {"n_completed": int(len(C)), "n_published": int(C.event.sum())}
    pub["overall"], curve = km(C.months, C.event)
    pd.DataFrame(curve, columns=["months", "published", "lo", "hi"]).to_csv(SRC / "trial_publication_km_overall.csv", index=False)
    curves = {"overall": curve}
    groups = {"sponsor": np.where(C.sponsor_class.eq("INDUSTRY"), "Industry", "Non-industry"),
              "phase": np.where(C.phase_group.isin(["Phase 2", "Phase 3-4"]), "Phase 2-4", "Phase 1 or not applicable"),
              "stage": np.where(C.stage.isin(STAGES[2:6]), "Cirrhosis, HCC or transplant", "Pre-cirrhotic or unspecified"),
              "completion_period": np.where(C.pc_date.dt.year <= 2019, "Completed 2010-2019", "Completed 2020-2025")}
    pub["by_group"] = {}
    for gname, g in groups.items():
        pub["by_group"][gname] = {}
        for lev in sorted(set(g)):
            mk = g == lev
            est, cv = km(C.months[mk], C.event[mk])
            pub["by_group"][gname][lev] = {"n": int(mk.sum()), "km": est}
            curves[f"{gname}:{lev}"] = cv
    pub["results_submitted_within_12m"] = float(C.results_submitted_12m.mean())
    pub["results_submitted_within_12m_industry"] = float(C[C.sponsor_class.eq("INDUSTRY")].results_submitted_12m.mean())
    pub["results_posted_any"] = float(C.results_posted_any.mean())
    pub["results_posted_any_industry"] = float(C[C.sponsor_class.eq("INDUSTRY")].results_posted_any.mean())
    pub["published_or_results_posted"] = float(((C.event == 1) | C.results_posted_any).mean())
    C["industry"] = C.sponsor_class.eq("INDUSTRY").astype(int); C["advanced"] = C.stage.isin(STAGES[2:6]).astype(int)
    C["phase23"] = C.phase_group.isin(["Phase 2", "Phase 3-4"]).astype(int); C["recent"] = (C.pc_date.dt.year >= 2020).astype(int)
    try:
        from statsmodels.duration.hazard_regression import PHReg
        ph = PHReg(C.months.values, C[["industry", "phase23", "advanced", "recent"]].values.astype(float), status=C.event.values).fit()
        pub["cox_hr"] = {k: {"hr": float(np.exp(b)), "ci": [float(np.exp(b - 1.96 * s)), float(np.exp(b + 1.96 * s))]}
                         for k, b, s in zip(["industry", "phase2_4", "advanced_stage", "completed_2020_or_later"], ph.params, ph.bse)}
    except Exception as ex:
        pub["cox_error"] = str(ex)
    pd.DataFrame([{"curve": k, "months": a, "published": b, "lo": lo_, "hi": hi_} for k, v in curves.items() for a, b, lo_, hi_ in v]).to_csv(
        SRC / "trial_publication_km_curves.csv", index=False)
    C.drop(columns=["eligibility"]).to_csv(SRC / "trial_publication_completed.csv", index=False)
    out["publication"] = pub

    items = pd.read_parquet(TOPICS_EMB / "items.parquet", columns=["stream", "item_id", "year"])
    Tt = pd.read_parquet(TOPICS_ANA / f"{REF_RUN}_topics_items.parquet")
    cat = pd.read_csv(TOPICS_ANA / "topic_catalogue_named.csv")
    lit = np.where(Tt.stream.values == "literature")[0]
    LL = pd.read_parquet(CORPUS_PROC / "literature_included_final.parquet", columns=["record_id", "year", "evidence_type", "pub_format",
                                                                                     "title", "abstract"])
    LL = LL.set_index("record_id").loc[items.item_id.values[lit]].reset_index()
    LL["topic"] = Tt.topic.values[lit]
    LL = LL[LL.topic.isin(set(cat.topic[~cat.meta_topic])) & LL.year.between(2020, 2025)].copy()
    LL["ids"] = [sorted(set(x.upper() for x in REG_RX.findall(f"{a} {b}"))) for a, b in zip(LL.title.fillna(""), LL.abstract.fillna(""))]
    LL["registered"] = LL.ids.str.len() > 0
    LL["protocol_like"] = LL.title.fillna("").str.contains(r"protocol|rationale and design|design and rationale", case=False)
    LL["group"] = to_group(LL.evidence_type, LL.record_id)
    R_ = LL[~LL.protocol_like]
    byy = R_.groupby("year").registered.agg(["sum", "count"])
    lm = {"share_research_papers_by_year": (byy["sum"] / byy["count"]).to_dict(), "k_by_year": byy["sum"].to_dict()}
    R_ = R_.assign(yc=R_.year - 2020, registered_i=R_.registered.astype(int))
    mm = smf.glm("registered_i ~ yc", data=R_, family=sm.families.Binomial()).fit()
    lm["or_per_year_research_papers"] = {"or": float(np.exp(mm.params["yc"])), "ci": np.exp(mm.conf_int().loc["yc"]).tolist()}
    P = R_[R_.group.isin(["interventional", "observational", "basic"])]
    mm = smf.glm("registered_i ~ yc", data=P, family=sm.families.Binomial()).fit()
    lm["or_per_year_primary"] = {"or": float(np.exp(mm.params["yc"])), "ci": np.exp(mm.conf_int().loc["yc"]).tolist()}
    lm["share_primary_by_year"] = P.groupby("year").registered.mean().to_dict()
    seen, uniq = set(), {}
    for y, ids in zip(R_.sort_values("year").year, R_.sort_values("year").ids):
        new = [i for i in ids if i not in seen]; seen.update(new); uniq[int(y)] = uniq.get(int(y), 0) + len(new)
    lm["unique_registered_studies_first_reported_by_year"] = uniq
    regs = pd.Series([registry_of(i) for ids in R_.ids for i in ids])
    lm["registry_mix"] = regs.value_counts().to_dict()
    yr_reg = pd.DataFrame([(y, registry_of(i)) for y, ids in zip(R_.year, R_.ids) for i in ids], columns=["year", "registry"])
    lm["registry_mix_by_year"] = {int(y): g.registry.value_counts().to_dict() for y, g in yr_reg.groupby("year")}
    lm["agreement_with_interventional_label"] = {
        "registered_given_interventional": float(P[P.group == "interventional"].registered.mean()),
        "interventional_given_registered": float((P[P.registered].group == "interventional").mean())}
    pd.DataFrame({"year": list(byy.index), "registered": byy["sum"].values, "papers": byy["count"].values}).to_csv(
        SRC / "literature_registry_numbers_by_year.csv", index=False)
    out["literature_registry_marker"] = lm

    vdir = PKG / "validation"; vdir.mkdir(exist_ok=True)
    frozen = vdir / "registry_stage_verification_ids.txt"
    if frozen.exists():
        ids = frozen.read_text(encoding="utf-8").split()
        S = T.set_index("nct_id").loc[[i for i in ids if i in set(T.nct_id)]].reset_index()
    else:
        rng = np.random.default_rng(SEED)
        samp = []
        for s_, g in T.groupby("stage"):
            k = min(len(g), max(12, int(round(200 * len(g) / len(T)))))
            samp.append(g.sample(k, random_state=int(rng.integers(1e9))))
        S = pd.concat(samp).head(200)
        frozen.write_text("\n".join(S.nct_id), encoding="utf-8")
    S[["nct_id", "title", "conditions", "phases", "stage", "endpoint"]].assign(
        eligibility=S.eligibility.str.slice(0, 3000), primary_outcomes=S.primary_outcomes.apply(lambda x: " | ".join(x)[:1500]),
        verified_stage="", verified_endpoint="", reviewer_note="").to_csv(vdir / "registry_stage_verification_sample.csv", index=False)
    out["verification_sample_n"] = int(len(S))
    ai = vdir / "ai_registry_reading" / "registry_stage_ai_reading.csv"
    if ai.exists():
        from sklearn.metrics import cohen_kappa_score
        A_ = pd.read_csv(ai).merge(S[["nct_id", "stage", "endpoint"]], on="nct_id", suffixes=("_ai", "_rule"))
        adv4 = lambda s_: np.where(s_.isin(STAGES[2:6]), "advanced", np.where(s_.eq(STAGES[6]), "phase1", "early"))
        out["ai_reading_agreement"] = {
            "n": int(len(A_)),
            "stage_exact": float((A_.stage_ai == A_.stage_rule).mean()),
            "stage_kappa": float(cohen_kappa_score(A_.stage_ai, A_.stage_rule)),
            "three_class_agreement": float((adv4(A_.stage_ai) == adv4(A_.stage_rule)).mean()),
            "advanced_vs_not_agreement": float((A_.stage_ai.isin(STAGES[2:6]) == A_.stage_rule.isin(STAGES[2:6])).mean()),
            "advanced_vs_not_kappa": float(cohen_kappa_score(A_.stage_ai.isin(STAGES[2:6]), A_.stage_rule.isin(STAGES[2:6]))),
            "endpoint_exact": float((A_.endpoint_ai == A_.endpoint_rule).mean()),
            "endpoint_kappa": float(cohen_kappa_score(A_.endpoint_ai, A_.endpoint_rule)),
            "clinical_endpoint_agreement": float(((A_.endpoint_ai == "Clinical outcome") == (A_.endpoint_rule == "Clinical outcome")).mean()),
            "confusion_stage": pd.crosstab(A_.stage_rule, A_.stage_ai).to_dict()}
    update_results("trial_landscape", out)
    log("a10.landscape", n_targeted=out["masld_targeted_n"], adv_share=round(out["advanced_share_of_targeted"]["share"], 3),
        n_completed=pub["n_completed"], pub36=round(pub["overall"]["36"]["published"], 3))


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
