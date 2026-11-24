import os, sys, re, json, time
import numpy as np, pandas as pd
from ku_common import *

TAG = sys.argv[1] if len(sys.argv) > 1 else "pubmedbert"
MCS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
RUN_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else SEED
FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
K_DOMAINS = 10
OUT = f"{TAG}_mcs{MCS}_s{RUN_SEED}" + (f"_f{int(FRAC*100)}" if FRAC < 1 else "")
seed_all(RUN_SEED)

DISEASE_STOP = {"nafld", "nafl", "nash", "masld", "mash", "mafld", "metald", "sld", "liver", "livers", "fatty", "disease",
    "diseases", "steatosis", "steatotic", "hepatic", "nonalcoholic", "non", "alcoholic", "steatohepatitis", "metabolic",
    "dysfunction", "associated", "dysfunction-associated", "sh", "patients", "patient", "study", "studies", "results", "methods", "conclusion",
    "conclusions", "background", "aim", "aims", "objective", "objectives", "using", "used", "use", "showed", "shown",
    "significant", "significantly", "compared", "group", "groups", "higher", "lower", "increased", "decreased",
    "including", "related", "risk", "analysis", "data", "based", "effect", "effects", "levels", "level", "role",
    "potential", "review", "new", "high", "low", "level", "total", "ci", "95", "vs", "p", "0", "1", "2", "3", "001", "05"}


def ctfidf(texts, labels, n_top=10):
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    stop = list(ENGLISH_STOP_WORDS | DISEASE_STOP)
    cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=10, max_df=0.5,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b", lowercase=True)
    X = cv.fit_transform(texts)
    terms = np.array(cv.get_feature_names_out())
    import scipy.sparse as sp
    ul = np.unique(labels)
    M = sp.csr_matrix((np.ones(len(labels)), (np.searchsorted(ul, labels), np.arange(len(labels)))), shape=(len(ul), len(labels)))
    C = (M @ X).astype(float)
    tf = sp.diags(1 / np.maximum(C.sum(1).A1, 1)) @ C
    A = C.sum() / len(ul)
    idf = np.log(1 + A / np.maximum(C.sum(0).A1, 1))
    S = (tf @ sp.diags(idf)).toarray()
    top = {int(t): list(terms[np.argsort(-S[i])[:n_top]]) for i, t in enumerate(ul)}
    return top, X, terms


if __name__ == "__main__":
    import umap, hdbscan
    from sklearn.decomposition import PCA
    t0 = time.time()
    items = pd.read_parquet(os.path.join(EMB, "items.parquet"))
    E = np.load(os.path.join(EMB, f"{TAG}.npy")).astype(np.float32)
    lit = np.where(items.stream.values == "literature")[0]
    rng = np.random.default_rng(RUN_SEED)
    fit_idx = np.sort(rng.choice(lit, int(len(lit) * FRAC), replace=False)) if FRAC < 1 else lit
    El = E[fit_idx]
    P = PCA(n_components=50, svd_solver="randomized", random_state=RUN_SEED).fit(El)
    Z = P.transform(El)
    U10 = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=10, metric="cosine", random_state=RUN_SEED, low_memory=True).fit_transform(Z)
    cl = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=10, cluster_selection_method="eom", core_dist_n_jobs=os.cpu_count()).fit(U10)
    lab = cl.labels_
    topics = np.unique(lab[lab >= 0])
    cent = np.stack([El[lab == t].mean(0) for t in topics]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    S_all = E @ cent.T
    assign = S_all.argmax(1); sim = S_all.max(1)
    df = items[["stream", "item_id", "year"]].copy()
    df["topic"] = topics[assign]; df["topic_sim"] = sim
    df["core"] = False
    core_map = pd.Series(lab, index=fit_idx)
    df.loc[core_map.index[core_map.values >= 0], "core"] = True
    df.loc[fit_idx, "hdbscan_label"] = lab
    res = {"tag": TAG, "mcs": MCS, "seed": RUN_SEED, "frac": FRAC, "n_fit": len(fit_idx), "n_topics": len(topics),
           "noise_frac": round(float((lab < 0).mean()), 4)}
    if FRAC == 1.0 and RUN_SEED == SEED:
        U2m = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric="cosine", random_state=RUN_SEED).fit(Z)
        U2 = np.full((len(items), 2), np.nan, dtype=np.float32)
        U2[fit_idx] = U2m.embedding_
        other = np.where(items.stream.values != "literature")[0]
        U2[other] = U2m.transform(P.transform(E[other]))
        np.save(os.path.join(ANA, f"{OUT}_umap2d.npy"), U2)
        texts = items.text.values[lit]
        if TAG.endswith("_neutral"):
            texts = pd.read_parquet(os.path.join(EMB, "items_text_neutral.parquet")).text_neutral.values[lit]
        top, X, terms = ctfidf(texts, df.topic.values[lit])
        L = pd.read_parquet(os.path.join(DATA, "literature_included_final.parquet"), columns=["record_id", "title", "cited_by"])
        cat = df.iloc[lit].groupby("topic").agg(n_assigned=("item_id", "size"), n_core=("core", "sum"),
                                                 mean_sim=("topic_sim", "mean")).reset_index()
        cat["top_terms"] = cat.topic.map(lambda t: "; ".join(top[int(t)]))
        reps = {}
        for k, t in enumerate(topics):
            mem = np.where((df.topic.values == t) & df.core.values & (items.stream.values == "literature"))[0]
            best = mem[np.argsort(-S_all[mem, k])[:3]]
            reps[int(t)] = " || ".join(items.text.values[best].astype(str).tolist())[:900]
        cat["representative"] = cat.topic.map(lambda t: reps[int(t)].split(". ")[0] if t in reps else "")
        cat["representative_titles"] = cat.topic.map(reps)
        from scipy.cluster.hierarchy import linkage, fcluster
        Zl = linkage(cent, method="ward")
        dom = fcluster(Zl, K_DOMAINS, criterion="maxclust")
        cat["domain_id"] = cat.topic.map(dict(zip(topics.astype(int), dom)))
        np.save(os.path.join(ANA, f"{OUT}_linkage.npy"), Zl)
        cat.to_csv(os.path.join(ANA, f"{OUT}_topic_catalogue.csv"), index=False, encoding="utf-8-sig")
        res["n_domains"] = K_DOMAINS
    np.save(os.path.join(ANA, f"{OUT}_centroids.npy"), cent)
    df.to_parquet(os.path.join(ANA, f"{OUT}_topics_items.parquet"), index=False)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(ANA, f"{OUT}_summary.json"), "w"), indent=1)
    log_event("universe.done", **res)
