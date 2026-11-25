"""Calibration of the attribution beyond the pooled ECE of Table 6 (external review of 2026-11-24, D110).

Same events as Table 6: the typed injected test events that the full method detects, in the 30 NB4 test datasets.
Same labellers as the paper: the label model q, the end model f (all features) and the fusion p (beta* per rate with
the D50 abstention, as NB4). Read-only on results/benchmark_v1/runs/*_events.csv; seconds.

Per dataset (then mean and s.d. over the 10 seeds of a rate, as Table 6):
  ece        arg-max ECE, 10 equal-width bins (Eq. 25; must reproduce Table 6)
  ece_mass   arg-max ECE, 10 equal-mass bins (adaptive binning)
  brier      multi-class Brier score (sum over the 8 classes of (p - onehot)^2)
  nll        negative log-likelihood of the true type, probabilities clipped at 1e-6
  p0_share   share of events whose true type gets probability below 1e-6
  ece_<c>    one-vs-rest ECE of class c (10 equal-width bins on p_c, all events)
Per true type, seeds pooled per rate: number of events, accuracy of the arg max, mean arg-max confidence and the gap.

    python tools/calibration_by_class.py
-> results/benchmark_v1/calibration_by_class/{per_dataset.csv, summary.csv, by_true_type.csv, summary.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import benchmark as B  # noqa: E402
import label_model as LM  # noqa: E402

BM = ROOT / "results" / "benchmark_v1"
OUT = BM / "calibration_by_class"
RATES = [0.01, 0.03, 0.07]
SEEDS = range(10)
EPS = 1e-6
LABELLERS = {"label model q": "label model q", "end model f": "end model f, all features", "fusion p": "fusion (β*, D50 abstention)"}


def beta_star():
    """beta* per rate as NB4 chose it (same function as tools/attribution_wilcoxon.py)."""
    bt = pd.read_csv(BM / "beta_tuning_train_period.csv")
    out = {}
    for rate in RATES:
        g = bt[np.isclose(bt.rate, rate) & (bt.abstain == True)]  # noqa: E712
        out[rate] = float(g.loc[(g.macro_f1 - 1e-9 * (g.beta - 0.5).abs()).idxmax(), "beta"])
    return out


def ece_equal_mass(conf, correct, n_bins=10):
    order = np.argsort(conf, kind="stable")
    return float(sum(len(b) / len(conf) * abs(correct[b].mean() - conf[b].mean())
                     for b in np.array_split(order, n_bins) if len(b)))


def metrics(y, probs):
    cl = np.array(B.CL)
    onehot = (cl[None, :] == y[:, None]).astype(float)
    pred = cl[probs.argmax(1)]
    conf = probs.max(1)
    correct = (pred == y).astype(float)
    p_true = (probs * onehot).sum(1)
    out = dict(n=len(y), ece=LM.ece(conf, correct), ece_mass=ece_equal_mass(conf, correct),
               brier=float(((probs - onehot) ** 2).sum(1).mean()), nll=float(-np.log(np.clip(p_true, EPS, 1)).mean()),
               p0_share=float((p_true < EPS).mean()))
    for k, c in enumerate(cl):
        if c in B.TYPED:
            out[f"ece_{c}"] = LM.ece(probs[:, k], onehot[:, k])          # one-vs-rest
    return out, pred, conf


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    beta = beta_star()
    rows, ev_rows = [], []
    for rate in RATES:
        for seed in SEEDS:
            ev = pd.read_csv(BM / "runs" / f"r{round(rate * 100):02d}_s{seed:02d}_events.csv")
            sub = ev[(ev.period == "test") & ev.detected & ev.type.isin(B.TYPED)]
            y = sub.type.to_numpy()
            probs = {"label model q": sub[[f"q_{c}" for c in B.CL]].to_numpy(float),
                     "end model f": sub[[f"f_{c}" for c in B.CL]].to_numpy(float),
                     "fusion p": B.fused(sub, beta[rate])}
            for name, pr in probs.items():
                m, pred, conf = metrics(y, pr)
                rows.append(dict(rate=f"{rate:.0%}", seed=seed, labeller=name, **m))
                ev_rows.append(pd.DataFrame(dict(rate=f"{rate:.0%}", seed=seed, labeller=name, type=y, pred=pred, conf=conf)))
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_dataset.csv", index=False, lineterminator="\n")

    # the arg-max ECE must reproduce Table 6 (and the Brier score of q stored there)
    t6 = pd.read_csv(BM / "table6_attribution.csv")
    for (rate, name), g in per.groupby(["rate", "labeller"]):
        r6 = t6[(t6.rate == rate) & (t6.labeler == LABELLERS[name])].iloc[0]
        assert abs(g.ece.mean() - float(r6.ECE.split("±")[0])) < 5e-4, (rate, name, g.ece.mean(), r6.ECE)
        if isinstance(r6.Brier, str) and "±" in r6.Brier:
            assert abs(g.brier.mean() - float(r6.Brier.split("±")[0])) < 5e-4, (rate, name, g.brier.mean(), r6.Brier)

    cols = [c for c in per.columns if c not in ("rate", "seed", "labeller")]
    summ = per.groupby(["labeller", "rate"], sort=False)[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / "summary.csv", index=False, lineterminator="\n")

    evs = pd.concat(ev_rows, ignore_index=True)
    bt = (evs.assign(correct=lambda d: d.type == d.pred).groupby(["labeller", "rate", "type"], sort=False)
          .agg(n=("correct", "size"), accuracy=("correct", "mean"), mean_confidence=("conf", "mean")).reset_index())
    bt["confidence_minus_accuracy"] = bt.mean_confidence - bt.accuracy
    bt.to_csv(OUT / "by_true_type.csv", index=False, lineterminator="\n")
    # the paper's example (Section 6.5): at 3 %, 44 of 160 injected A1 gaps get the fused label G1
    a1 = evs[(evs.labeller == "fusion p") & (evs.rate == "3%") & (evs.type == "A1")]
    assert len(a1) == 160 and (a1.pred == "G1").sum() == 44, (len(a1), (a1.pred == "G1").sum())

    json.dump(dict(beta_star={f"{r:.0%}": b for r, b in beta.items()}, events="test period, detected, true type in the seven typed classes (Table 6)",
                   labellers=LABELLERS, nll_clip=EPS, table6_ece_reproduced=True, a1_3pct_fused_g1="44 of 160 (Section 6.5) reproduced"),
              open(OUT / "summary.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    show = summ[[c for c in summ.columns if c.endswith("_mean")]].round(3)
    print(show.to_string())
    print(bt[bt.labeller == "fusion p"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
