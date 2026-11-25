"""Robustness checks asked for by the external review of 2026-11-24 (D110). Read-only on the stored results.

1. Real-data Diebold-Mariano tests (NB5, Table 8): the per-time-step losses of NB5 (results/downstream_v1/loss_real.pkl)
   are tested again with longer Newey-West bandwidths (1 and 3 days) and with paired bootstraps that resample whole
   blocks of time: post-anomaly stretches (maximal runs of consecutive scored time steps), calendar days and 7-day
   blocks. The statistic is the mean loss difference over the resampled time steps, as the DM numerator; the
   two-sided bootstrap p is 2 min(P(mean <= 0), P(mean >= 0)) over B resamples. Nothing is re-forecast.
2. Real-data outage counts (Section 6.1): the dead zone-hours of each rule item of the fused southern catalogue
   (results/main_detector/p_type_south.csv), counted exactly as NB5 counts them (dead state inside a qualifying zone
   run and inside the item's span and zones). Checks that naive minus typed counting equals the three multi-city zero
   fills (A1), and gives each one's share, the ratio with each one left out or re-typed as an outage, and the
   posterior-weighted typed count (each dead item weighted by its fused probability of a grid event).

    python tools/review_robustness.py          (~1 min: the rule engine rebuilds the per-step states)
-> results/downstream_v1/robustness/{dm_robust_real.csv, a1_items_real.csv, summary.json}
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import detector as D  # noqa: E402
import downstream as DS  # noqa: E402
import rules_v1 as R1  # noqa: E402

DSV = ROOT / "results" / "downstream_v1"
OUT = DSV / "robustness"
XLSX = ROOT / "data" / "raw" / "Data Morocco.xlsx"
B_BOOT, SEED = 20000, 0
AFTER = "after anomaly (2 days)"


# ------------------------------------------------------------------ 1. DM robustness
def nw_p(d, lag):
    """Two-sided DM p with a Newey-West (Bartlett) variance up to `lag` (as downstream.dm_test)."""
    return DS.dm_test(d, lag)[1]


def block_boot(d, labels, B=B_BOOT, seed=SEED):
    """Paired bootstrap over blocks: resample whole blocks with replacement; statistic = mean of d over the resampled
    time steps. Returns (number of blocks, 2.5 % and 97.5 % percentiles, two-sided p, share of blocks where the second
    strategy is better)."""
    s = pd.Series(np.asarray(d, float)).groupby(np.asarray(labels))
    sums, cnt = s.sum().to_numpy(), s.count().to_numpy()
    nb = len(sums)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, nb, size=(B, nb))
    m = sums[i].sum(1) / cnt[i].sum(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return nb, float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(min(p, 1.0)), float((sums > 0).mean())


def stretches(index):
    """Label maximal runs of consecutive 10-min time steps."""
    t = pd.DatetimeIndex(index).to_numpy()
    return np.cumsum(np.r_[True, np.diff(t) > np.timedelta64(10, "m")])


def dm_robust():
    loss = pickle.load(open(DSV / "loss_real.pkl", "rb"))
    stored = pd.read_csv(DSV / "dm_real.csv")
    rows = []
    for (model, hname, sub), g in stored.groupby(["model", "horizon", "subset"], sort=False):
        for r in g.itertuples():
            a, b = r.comparison.split(" vs ")
            d = (loss[(a, model, hname, sub)] - loss[(b, model, hname, sub)]).dropna()
            dv = d.to_numpy()
            p_h = nw_p(dv, DS.H[hname])
            assert np.isclose(p_h, r.p, rtol=1e-9, atol=1e-15), (model, hname, sub, r.comparison, p_h, r.p)   # reproduces NB5
            day0 = d.index.min().normalize()
            row = dict(model=model, horizon=hname, subset=sub, comparison=r.comparison, n_steps=len(dv),
                       mean_loss_diff=float(dv.mean()), dm_p_bandwidth_h=p_h,
                       dm_p_bandwidth_1day=nw_p(dv, 144), dm_p_bandwidth_3days=nw_p(dv, 432))
            for name, lab in (("stretch", stretches(d.index)), ("day", (d.index.normalize() - day0).days),
                              ("week", (d.index.normalize() - day0).days // 7)):
                nb, lo, hi, p, share = block_boot(dv, lab)
                row.update({f"{name}_blocks": nb, f"{name}_ci_low": lo, f"{name}_ci_high": hi, f"{name}_p": p,
                            f"{name}_share_blocks_second_better": share})
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. the three A1 items
def a1_items():
    res = R1.build(XLSX)
    P = D.build_panel(res["data"], res["exp"], res["masks"], "south")
    flag = DS.run_mask(P, res["zone_runs"])
    dead = P["states"]["dead"] & flag
    cat = pd.read_csv(ROOT / "results" / "main_detector" / "p_type_south.csv")
    rule = cat[cat.source == "rule"].copy()
    art, grid = DS.typed_masks(P, flag, rule[["start", "end", "zones", "label"]])
    naive = dead.to_numpy().sum() / 6
    typed = (dead & grid).to_numpy().sum() / 6
    a1_all = (dead & art).to_numpy().sum() / 6
    rule["dead_zone_hours"] = [(DS.span_mask(P, pd.DataFrame([r._asdict()])) & dead).to_numpy().sum() / 6
                               for r in rule[["start", "end", "zones"]].itertuples(index=False)]
    rule["p_grid"] = rule[[c for c in ("p_G1", "p_G2", "p_G3", "p_G4") if c in rule]].sum(1)
    deadi = rule[rule.dead_zone_hours > 0]
    # dead items do not overlap: their dead zone-hours add up to the naive count
    assert np.isclose(deadi.dead_zone_hours.sum(), naive), (deadi.dead_zone_hours.sum(), naive)
    a1 = deadi[deadi.label == "A1"].copy()
    assert len(a1) == 3 and np.isclose(a1.dead_zone_hours.sum(), naive - typed) and np.isclose(a1_all, naive - typed)
    a1["confidence"] = a1.p_A1
    a1["share_of_naive"] = a1.dead_zone_hours / naive
    a1["ratio_if_typed_outage"] = naive / (typed + a1.dead_zone_hours)
    a1["ratio_if_removed"] = (naive - a1.dead_zone_hours) / typed
    expected_typed = float((deadi.p_grid * deadi.dead_zone_hours).sum())
    cols = ["item_id", "start", "end", "zones", "dead_zone_hours", "confidence", "share_of_naive", "ratio_if_typed_outage",
            "ratio_if_removed"]
    summ = dict(naive_zone_hours=naive, typed_zone_hours=typed, ratio_naive_to_typed=naive / typed,
                a1_zone_hours=float(a1.dead_zone_hours.sum()), a1_share_of_naive=float(a1.dead_zone_hours.sum() / naive),
                naive_minus_typed_equals_a1=True, dead_items=int(len(deadi)),
                expected_typed_zone_hours=expected_typed, expected_ratio_naive_to_typed=naive / expected_typed,
                ratio_without_the_three_a1=(naive - a1.dead_zone_hours.sum()) / typed)
    return a1[cols].reset_index(drop=True), summ


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    dm = dm_robust()
    dm.to_csv(OUT / "dm_robust_real.csv", index=False, lineterminator="\n")
    a1, summ = a1_items()
    a1.to_csv(OUT / "a1_items_real.csv", index=False, lineterminator="\n")
    json.dump(dict(bootstrap_resamples=B_BOOT, bootstrap_seed=SEED,
                   blocks="stretch: maximal runs of consecutive scored 10-min steps; day: calendar days; week: 7-day blocks from the first scored day",
                   p_definition="two-sided: 2 min(P(mean <= 0), P(mean >= 0)) over the resamples; 0 means below 2/B",
                   dm_reproduced=True, outage_real=summ), open(OUT / "summary.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    show = dm[(dm.model == "LightGBM")][["horizon", "subset", "comparison", "dm_p_bandwidth_h", "dm_p_bandwidth_1day",
                                         "dm_p_bandwidth_3days", "stretch_blocks", "stretch_p", "day_p", "week_p"]]
    print(show.to_string(index=False))
    print(a1.to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
