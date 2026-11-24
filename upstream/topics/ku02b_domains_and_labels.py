import os, sys, json
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from ku_common import *

REF = sys.argv[1] if len(sys.argv) > 1 else REF_RUN
K = 10

DOMAIN_NAMES = {0: "Correspondence, non-English and off-target", 1: "Clinical epidemiology and comorbidity",
                2: "Case-based and differential hepatology", 3: "Advanced liver disease and liver cancer",
                4: "Non-invasive diagnosis and imaging", 5: "Obesity, diabetes and cardiometabolic medicine",
                6: "Gut microbiome", 7: "Veterinary and livestock fatty liver", 8: "Experimental models and preclinical therapeutics",
                9: "Mechanistic reviews, omics and AI", 10: "Clinical trials and interventions"}
DOMAIN_SHORT = {0: "Meta", 1: "Clinical epidemiology", 2: "Case-based hepatology", 3: "Advanced disease", 4: "Diagnosis and imaging",
                5: "Cardiometabolic", 6: "Gut microbiome", 7: "Veterinary", 8: "Experimental models", 9: "Mechanisms, omics, AI",
                10: "Trials and interventions"}
LABELS = {
"0": "Mixed clinical hepatology",
"1": "Polycystic ovary syndrome",
"2": "Lipid-droplet fluorescent probes",
"3": "Bariatric and metabolic surgery",
"4": "COVID-19",
"5": "Spanish- and French-language papers",
"6": "German-language papers",
"7": "Thyroid hormone",
"8": "PNPLA3 and inherited risk",
"9": "Mendelian randomization",
"10": "Ayurvedic management",
"11": "Pancreatic steatosis",
"12": "Liver cancer mechanisms",
"13": "Inflammatory bowel disease",
"14": "Off-target records",
"15": "Thyroid function",
"16": "Liver cancer epidemiology and surveillance",
"17": "Bile acids",
"18": "Viral hepatitis with steatosis",
"19": "Liver cancer outcomes after treatment",
"20": "Autoimmune and drug-induced liver injury",
"21": "Portal hypertension",
"22": "Liver transplantation",
"23": "Cirrhosis and decompensation",
"24": "Acute fatty liver of pregnancy",
"25": "Case reports",
"26": "Ferroptosis and pyroptosis",
"27": "Type 2 diabetes associations",
"28": "Diabetes reviews",
"29": "Gut microbes in metabolic disease",
"30": "Gut microbiota alterations",
"31": "Microbiota reviews",
"32": "Gut-liver axis",
"33": "Obese children",
"34": "Cardiovascular mechanisms",
"35": "Paediatric disease",
"36": "Human organoids and tissue models",
"37": "FGF21",
"38": "Fructose",
"39": "Alcohol-related liver disease and MetALD",
"40": "Global burden estimates",
"41": "Vitamin D",
"42": "Awareness, literacy and care pathways",
"43": "Prevalence meta-analyses",
"44": "Machine learning and AI",
"45": "Clinical management and nomenclature",
"46": "Drug development reviews",
"47": "Diet-induced rodent models",
"48": "Air pollution and pollutants",
"49": "Lifestyle intervention reviews",
"50": "Natural product reviews",
"51": "Fibrosis outcome cohorts",
"52": "FXR agonists",
"53": "Triglyceride-glucose index",
"54": "Lipoprotein ratios",
"55": "Adiposity and prevalence",
"56": "Cardiovascular outcomes and mortality",
"57": "Sleep and physical activity",
"58": "Dietary intake",
"59": "Nutrition and fasting",
"60": "Replies and correspondence",
"61": "Inter-organ mouse models",
"62": "Non-invasive tests and biomarkers",
"63": "Fibrosis scores",
"64": "MRI-PDFF and ultrasound fat quantification",
"65": "Elastography",
"66": "Obesity and diabetes phenotypes",
"67": "Meta-analyses of trials",
"68": "Adipose tissue and mitochondria",
"69": "Cardiometabolic syndrome",
"70": "Obesity medicine",
"71": "Resmetirom",
"72": "Non-coding RNA",
"73": "Extracellular vesicles",
"74": "Hepatocyte lipid signalling",
"75": "Randomized intervention trials",
"76": "Phase 2-3 drug trials",
"77": "Laying hens",
"78": "Dairy cows",
"79": "GLP-1 and incretin agonists",
"80": "GLP-1 and SGLT2 drugs in steatotic liver disease",
"81": "Exercise training",
"82": "Hepatocyte signalling in mice",
"83": "Mitochondrial and lipotoxic mechanisms",
"84": "Transcriptomic hub genes",
"85": "Immune pathogenesis",
"86": "Immune cells in liver disease",
"87": "Maternal and developmental programming",
"88": "Environmental toxicants",
"89": "Herbal hepatoprotectives",
"90": "Phytochemicals",
"91": "Probiotics and microbiota in models",
"92": "Network pharmacology",
"93": "Plant extracts in rodents",
"94": "Lipid accumulation pathways"
}
LABELS = {int(k): v for k, v in LABELS.items()}
META = {60, 5, 6, 14}

if __name__ == "__main__":
    cat = pd.read_csv(os.path.join(ANA, f"{REF}_topic_catalogue.csv")).sort_values("topic").reset_index(drop=True)
    cent = np.load(os.path.join(ANA, f"{REF}_centroids.npy"))
    keep = ~cat.topic.isin(META).values
    Z = linkage(cent[keep], method="ward")
    np.save(os.path.join(ANA, f"{REF}_linkage_ward.npy"), Z)
    dom = dict(zip(cat.topic.values[keep], fcluster(Z, K, criterion="maxclust")))
    cat["domain_id"] = cat.topic.map(dom).fillna(0).astype(int)
    cat["domain_name"] = cat.domain_id.map(DOMAIN_NAMES); cat["domain_short"] = cat.domain_id.map(DOMAIN_SHORT)
    cat["topic_label"] = cat.topic.map(LABELS); cat["meta_topic"] = cat.topic.isin(META)
    assert cat.topic_label.notna().all(), "unlabelled topic"
    cat.to_csv(os.path.join(ANA, "topic_catalogue_named.csv"), index=False, encoding="utf-8-sig")
    log_event("labels.done", ref=REF, n_topics=len(cat), domains=cat.groupby("domain_short").n_assigned.sum().to_dict())
