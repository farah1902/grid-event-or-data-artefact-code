"""S1b control for the downstream forecasting test (external review C1, D111).

S2 differs from S1 in two ways at once: a better imputer (Eq. 22, the expected profile scaled to both edges, instead of
a straight line) and the typing (grid events kept, flagged by an indicator and left out of the training loss). S1b
applies S2's imputer to every flagged run, outages included, with no indicator (downstream.clean, strategy "S1b"):
S1 vs S1b measures the imputer, S1b vs S2 the typing.

Everything else is NB5's test unchanged (tools/forecast_all_datasets.py for the injected data): same flagged steps,
scoring area, forecasters (global LightGBM direct forecaster, persistence / seasonal naive), seed 0, horizons, subsets.
S2 is forecast again next to S1b and must reproduce the stored S2 results (real data: forecast_real.csv and
loss_real.pkl; injected: forecast_all/forecast_per_dataset.csv); S0, S1 and S3 are read from the stored files.
Evaluation only: nothing is selected or tuned on the test seeds (D63).

    python tools/s1b_control.py        (resumable, cached per dataset; ~3 min real data + ~2.5 min per injected dataset)
-> results/downstream_v1/s1b_control/{real_forecast.csv, real_dm.csv, loss_real_s1b.pkl, runs/, forecast_per_dataset.csv,
   dm_per_dataset.csv, forecast_summary.csv, dm_summary.csv, summary.json}
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import benchmark as B  # noqa: E402
import detector as D  # noqa: E402
import downstream as DS  # noqa: E402
import inject as INJ  # noqa: E402
import label_model as LM  # noqa: E402
import rules_v1 as R1  # noqa: E402
import review_robustness as RR  # noqa: E402   (DM bandwidths and block bootstraps, D110)

RATES, SEEDS = (0.01, 0.03, 0.07), range(10)
SEED = 0
XLSX = ROOT / "data" / "raw" / "Data Morocco.xlsx"
DSV = ROOT / "results" / "downstream_v1"
OUT = DSV / "s1b_control"
RUNS = ROOT / "results" / "benchmark_v1" / "runs"
GRID_T = ("G1", "G2", "G3", "G4", "U_SHOCK", "U_PARTIAL")
KEY = ["model", "horizon", "subset", "strategy"]
TOL = 1e-3                     # as tools/forecast_all_datasets.py (reproduction of NB5)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ real data (NB5 sections 1 and 3)
def real(res):
    f_fc, f_loss = OUT / "real_forecast.csv", OUT / "loss_real_s1b.pkl"
    if f_fc.exists() and f_loss.exists():
        return pd.read_csv(f_fc), pickle.load(open(f_loss, "rb"))
    P = D.build_panel(res["data"], res["exp"], res["masks"], "south")
    XH = DS.exp_panel(res["exp"], P)
    FLAG = DS.run_mask(P, res["zone_runs"])
    cat = pd.read_csv(ROOT / "results" / "main_detector" / "p_type_south.csv", parse_dates=["start", "end"])
    ART, GRID = DS.typed_masks(P, FLAG, cat[cat.source == "rule"][["start", "end", "zones", "label"]])
    south_items = res["items"][res["items"].system == "south"]
    NORMAL = ~FLAG & ~DS.span_mask(P, south_items) & ~DS.span_mask(P, cat[cat.source == "detector"]) & P["x"].notna()
    t = time.time()
    fc, loss = DS.forecast_eval(P, FLAG, {"S2": (ART, GRID), "S1b": (None, None)}, NORMAL, NORMAL, XH, seed=SEED, log=lambda *a, **k: None)
    log(f"real data: {time.time() - t:.0f} s")
    # S2 must reproduce NB5
    stored = pd.read_csv(DSV / "forecast_real.csv").set_index(KEY).sort_index()
    new = fc.set_index(KEY).sort_index()
    d = float((new.loc[[i for i in new.index if i[3] == "S2"], "MASE"] - stored.loc[[i for i in new.index if i[3] == "S2"], "MASE"]).abs().max())
    assert d < TOL, f"real-data S2 not reproduced (max |dMASE| = {d})"
    old_loss = pickle.load(open(DSV / "loss_real.pkl", "rb"))
    dl = max(float((loss[k] - old_loss[k]).abs().max()) for k in loss if k[0] == "S2")
    log(f"real data: S2 reproduced (max |dMASE| {d:.2e}, max |d loss| {dl:.2e})")
    fc = fc.assign(s2_max_abs_diff_mase=d)
    s1b = {k: v for k, v in loss.items() if k[0] == "S1b"}
    fc.to_csv(f_fc, index=False, lineterminator="\n")
    pickle.dump(s1b, open(f_loss, "wb"))
    return fc, s1b


def real_tests(s1b):
    """Paired tests on the real data: stored S0, S1, S2 losses against the new S1b losses."""
    loss = {**pickle.load(open(DSV / "loss_real.pkl", "rb")), **s1b}
    rows = []
    for (a, b) in (("S1", "S1b"), ("S1b", "S2"), ("S0", "S1b")):
        for (model, hname, sub) in sorted({k[1:] for k in loss}):
            d = (loss[(a, model, hname, sub)] - loss[(b, model, hname, sub)]).dropna()
            dv = d.to_numpy()
            stat, p = DS.dm_test(dv, lag=DS.H[hname])
            row = dict(model=model, horizon=hname, subset=sub, comparison=f"{a} vs {b}", n_steps=len(dv), mean_loss_diff=float(dv.mean()),
                       share_steps_second_better=float((dv > 0).mean()), DM=stat, dm_p_bandwidth_h=p,
                       dm_p_bandwidth_1day=RR.nw_p(dv, 144), dm_p_bandwidth_3days=RR.nw_p(dv, 432))
            day0 = d.index.min().normalize()
            for name, lab in (("stretch", RR.stretches(d.index)), ("day", (d.index.normalize() - day0).days),
                              ("week", (d.index.normalize() - day0).days // 7)):
                nb, lo, hi, pb, share = RR.block_boot(dv, lab)
                row.update({f"{name}_blocks": nb, f"{name}_p": pb})
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ injected data (tools/forecast_all_datasets.py)
def one(res, CLEAN, INJECTOR, rate, seed):
    P = D.build_panel(res["data"], res["exp"], res["masks"], "south")
    idx = P["x"].index
    tag = f"r{int(round(rate * 100)):02d}_s{seed:02d}"
    data_inj, truth = INJECTOR.sample(np.ones(len(idx), bool), rate, np.random.default_rng([int(round(rate * 1000)), seed]))
    stored = pd.read_csv(RUNS / f"{tag}_truth.csv", parse_dates=["start"])
    assert len(stored) == len(truth) and (stored.start.values == truth.start.values).all(), f"{tag}: differs from NB4"
    r = R1.build_from_data(res["raw"], data_inj)
    it = r["items"]; votes = r["votes"].set_index("item_id")
    q = LM.DawidSkene().fit(votes, it, R1.RULE_KIND).q_; q.index = it.item_id.values
    ev = it[(it.system == "south") & it.kind.isin(["dead", "fill"])].copy()
    ev["label"] = q.loc[ev.item_id].idxmax(axis=1).values
    Pi = D.build_panel(r["data"], r["exp"], r["masks"], "south")
    XHi = DS.exp_panel(r["exp"], Pi)
    FLi = DS.run_mask(Pi, r["zone_runs"])
    ART2, GRID2 = DS.typed_masks(Pi, FLi, ev[["start", "end", "zones", "label"]])
    INJ_ANY = DS.span_mask(Pi, truth, "end_affected")
    NORM_i = ~FLi & ~DS.span_mask(Pi, it[it.system == "south"]) & Pi["x"].notna()
    SCORE_i = NORM_i & CLEAN & ~INJ_ANY
    masks = {"S2": (ART2, GRID2), "S1b": (None, None)}
    fc, loss = DS.forecast_eval(Pi, FLi, masks, NORM_i, SCORE_i, XHi, seed=SEED, log=lambda *a, **k: None)
    dm = DS.dm_table(loss, pairs=(("S1b", "S2"),))
    return fc.assign(rate=rate, seed=seed), dm.assign(rate=rate, seed=seed), len(truth)


def summarise():
    stored = pd.read_csv(DSV / "forecast_all" / "forecast_per_dataset.csv")
    new = pd.concat([pd.read_csv(OUT / "runs" / f"r{int(round(r * 100)):02d}_s{s:02d}_forecast.csv") for r in RATES for s in SEEDS], ignore_index=True)
    dm = pd.concat([pd.read_csv(OUT / "runs" / f"r{int(round(r * 100)):02d}_s{s:02d}_dm.csv") for r in RATES for s in SEEDS], ignore_index=True)
    # S2 must reproduce the stored per-dataset results
    k = ["model", "horizon", "subset", "strategy", "rate", "seed"]
    a = new[new.strategy == "S2"].set_index(k).MASE.sort_index()
    b = stored[stored.strategy == "S2"].set_index(k).MASE.sort_index()
    rep = float((a - b.loc[a.index]).abs().max())
    assert rep < TOL, f"injected S2 not reproduced (max |dMASE| = {rep})"
    allfc = pd.concat([stored[stored.strategy.isin(["S0", "S1", "S3"])], new], ignore_index=True)
    allfc.to_csv(OUT / "forecast_per_dataset.csv", index=False, lineterminator="\n")
    dm.to_csv(OUT / "dm_per_dataset.csv", index=False, lineterminator="\n")
    piv = allfc.pivot_table(index=["model", "horizon", "subset", "rate", "seed"], columns="strategy", values="MASE").reset_index()

    def summ(g, label):
        out = dict(rate=label, n=len(g))
        for s in ("S0", "S1", "S1b", "S2", "S3"):
            out[f"MASE_{s}_mean"] = g[s].mean()
        for a_, b_ in (("S1", "S1b"), ("S1b", "S2"), ("S1", "S2"), ("S1b", "S3")):
            d = g[a_] - g[b_]                                    # > 0: the second is better
            out[f"{b_}_better_than_{a_}_n"] = int((d > 0).sum())
            out[f"wilcoxon_p_{a_}_vs_{b_}"] = float(wilcoxon(g[a_], g[b_]).pvalue) if (d != 0).any() else np.nan
        # share of the S1 -> S2 gain that the imputer alone (S1b) gives
        gain = (g["S1"] - g["S2"]).mean()
        out["imputer_share_of_gain"] = (g["S1"] - g["S1b"]).mean() / gain if gain != 0 else np.nan
        return out

    rows = []
    for (m, h, sub), g in piv.groupby(["model", "horizon", "subset"]):
        for r in RATES:
            rows.append(dict(model=m, horizon=h, subset=sub, **summ(g[np.isclose(g.rate, r)], f"{r:.0%}")))
        rows.append(dict(model=m, horizon=h, subset=sub, **summ(g, "all")))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "forecast_summary.csv", index=False, lineterminator="\n")
    dm["S2_sig_better"] = (dm.p < 0.05) & (dm.mean_loss_diff > 0)
    dm["S2_sig_worse"] = (dm.p < 0.05) & (dm.mean_loss_diff < 0)
    dms = dm.groupby(["model", "horizon", "subset", "comparison"]).agg(n=("p", "size"), sig_better=("S2_sig_better", "sum"),
                                                                     sig_worse=("S2_sig_worse", "sum"), DM_mean=("DM", "mean")).reset_index()
    dms.to_csv(OUT / "dm_summary.csv", index=False, lineterminator="\n")
    return tab, dms, rep


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    res = R1.build(XLSX)
    fc_real, s1b = real(res)
    tests = real_tests(s1b)
    tests.to_csv(OUT / "real_dm.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    log(fc_real.pivot_table(index=["model", "horizon", "subset"], columns="strategy", values="MASE").round(4).to_string())
    log(tests[["model", "horizon", "subset", "comparison", "mean_loss_diff", "dm_p_bandwidth_h", "dm_p_bandwidth_1day",
               "stretch_p", "day_p", "week_p"]].round(4).to_string(index=False))

    P = D.build_panel(res["data"], res["exp"], res["masks"], "south")
    CLEAN = B.clean_region(P, None)
    INJECTOR = INJ.Injector(res["data"], res["exp"], CLEAN, INJ.zone_map(res["data"], B.SOUTH, R1.CITY_SHORT))
    for rate in RATES:
        for seed in SEEDS:
            tag = f"r{int(round(rate * 100)):02d}_s{seed:02d}"
            f_fc, f_dm = OUT / "runs" / f"{tag}_forecast.csv", OUT / "runs" / f"{tag}_dm.csv"
            if f_fc.exists() and f_dm.exists():
                continue
            t = time.time()
            fc, dm, n = one(res, CLEAN, INJECTOR, rate, seed)
            fc.to_csv(f_fc, index=False, lineterminator="\n"); dm.to_csv(f_dm, index=False, lineterminator="\n")
            log(f"{tag}: {n} events, {time.time() - t:.0f} s")
    tab, dms, rep = summarise()
    log(tab[tab.rate == "all"].round(4).to_string(index=False))
    log(dms.to_string(index=False))
    json.dump(dict(datasets=30, rates=list(RATES), seeds=list(SEEDS), seed_forecaster=SEED,
                   s2_reproduced_real_max_abs_diff_mase=float(fc_real.s2_max_abs_diff_mase.iloc[0]),
                   s2_reproduced_injected_max_abs_diff_mase=rep,
                   strategies="S0, S1, S3 read from results/downstream_v1 (NB5, forecast_all); S1b and S2 forecast here",
                   minutes_this_call=round((time.time() - t0) / 60, 1)), open(OUT / "summary.json", "w"), indent=1)
    log("done in", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()
