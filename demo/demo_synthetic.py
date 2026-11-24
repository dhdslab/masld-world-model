from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MASLD_OUTPUT_DIR", str(HERE / "output"))
sys.path.insert(0, str(HERE.parent / "code"))
import w3_world_model as W

HORIZON = 3
FEATURES = ["semantic_cosine", "resource_allocation", "log_degree_product", "log_common_neighbours",
            "attention_low", "attention_high"]


def simulate(n, rng):
    k, dim = 8, 16
    centres = rng.normal(size=(k, dim))
    z = centres[rng.integers(k, size=n)] + 0.6 * rng.normal(size=(n, dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    attention = rng.lognormal(0.0, 1.0, size=n)
    i, j = np.triu_indices(n, 1)
    cos = (z[i] * z[j]).sum(1)
    lpp = np.log(attention[i] * attention[j])
    first = rng.random(len(i)) < 1 - np.exp(-np.exp(-4.8 + 0.9 * lpp + 2.5 * cos))
    return z, attention, i, j, cos, lpp, first


def graph_features(edges, i, j, n):
    A = np.zeros((n, n))
    A[i[edges], j[edges]] = A[j[edges], i[edges]] = 1.0
    deg = A.sum(1)
    cn = (A @ A)[i, j]
    ra = ((A / np.where(deg > 0, deg, 1.0)) @ A)[i, j]
    return cn, ra, deg[i] * deg[j]


def new_associations(edges, i, j, n, cos, lpp, rng):
    cn, _, _ = graph_features(edges, i, j, n)
    eta = -7.3 + 0.8 * lpp + 3.0 * cos + 0.6 * np.log1p(cn)
    return ~edges & (rng.random(len(i)) < 1 - np.exp(-HORIZON * np.exp(eta)))


def frame(edges, z_obs, n_obs, i, j, n):
    cn, ra, pa = graph_features(edges, i, j, n)
    cand = ~edges
    a, b = i[cand], j[cand]
    att = np.log1p(n_obs)
    X = np.column_stack([(z_obs[a] * z_obs[b]).sum(1), ra[cand], np.log1p(pa[cand]), np.log1p(cn[cand]),
                         np.minimum(att[a], att[b]), np.maximum(att[a], att[b])]).astype(np.float32)
    Z = np.hstack([z_obs[a] * z_obs[b], np.abs(z_obs[a] - z_obs[b])]).astype(np.float32)
    return cand, a, b, X, Z, pa[cand].astype(float), ra[cand].astype(float)


def case_control(y, rng, ratio=20):
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    take = rng.choice(neg, min(len(neg), ratio * len(pos)), replace=False)
    ix = np.concatenate([pos, take])
    w = np.where(y[ix], 1.0, len(neg) / len(take))
    return ix, (w / w.mean()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--draws", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    n = a.entities
    z, attention, i, j, cos, lpp, e0 = simulate(n, rng)
    z_obs = z + 0.3 * rng.normal(size=z.shape)
    z_obs /= np.linalg.norm(z_obs, axis=1, keepdims=True)
    y0 = new_associations(e0, i, j, n, cos, lpp, rng)
    e1 = e0 | y0
    y1 = new_associations(e1, i, j, n, cos, lpp, rng)
    n0, n1 = rng.poisson(10 * attention), rng.poisson(13 * attention)

    cand, _, _, X, Z, _, _ = frame(e0, z_obs, n0, i, j, n)
    y = y0[cand]
    ix, w = case_control(y, rng)
    models = {"mathematical world model": W.Hazard(False).fit(X[ix], Z[ix], y[ix].astype(np.float32), w, HORIZON),
              "neural world model": W.Hazard(True).fit(X[ix], Z[ix], y[ix].astype(np.float32), w, HORIZON)}

    cand, ta, tb, Xt, Zt, pa, ra = frame(e1, z_obs, n1, i, j, n)
    yt = y1[cand].astype(int)
    scores = {"degree product": pa, "resource allocation": ra,
              **{k: m.eta(Xt, Zt) for k, m in models.items()}}
    rows = []
    for k, s in scores.items():
        r = W.evaluate(yt, np.ones(len(yt)), s, pa=pa)
        rows.append({"model": k, "AUROC": r["auroc"], "popularity-adjusted AUROC": r["auroc_popularity_adjusted"],
                     "average precision": r["ap"]})
    F = pd.DataFrame({"a": ta, "b": tb, f"y{HORIZON}": yt, f"w{HORIZON}": 1.0})
    ci, diff, ok = W.node_bootstrap(F, HORIZON, scores, rng, draws=a.draws,
                                    contrasts=(("neural world model", "resource allocation"),
                                               ("mathematical world model", "resource allocation")))
    T = pd.DataFrame(rows).set_index("model")
    T["AUROC 95% CI"] = [f"{ci[k][0]:.3f}-{ci[k][1]:.3f}" for k in T.index]
    coef = models["mathematical world model"].coef(FEATURES)["standardized"]

    print(f"Synthetic field: {n} entities, {int(e1.sum()):,} reported associations at the test cutoff, "
          f"{len(yt):,} candidate pairs, {yt.sum():,} new associations within {HORIZON} years "
          f"(training cutoff: {int(y.sum()):,}).\n")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(T.to_string())
    print(f"\nPaired differences in AUROC (node bootstrap, {ok} draws, 95% interval):")
    for k, v in diff.items():
        print(f"  {k.replace('_minus_', ' minus ')}: {v[0]:.3f} to {v[1]:.3f}")
    print("\nStandardized coefficients of the mathematical world model (per standard deviation):")
    for k, v in coef.items():
        print(f"  {k:24s} {v:+.2f}")
    out = Path(os.environ["MASLD_OUTPUT_DIR"]) / "demo_results.json"
    out.write_text(json.dumps({"table": T.reset_index().to_dict("records"), "paired_differences": diff,
                               "coefficients": coef}, indent=1))
    try:
        shown = out.resolve().relative_to(HERE.parent).as_posix()
    except ValueError:
        shown = str(out)
    print(f"\nFinished in {time.time() - t0:.0f} s; results written to {shown}")


if __name__ == "__main__":
    main()
