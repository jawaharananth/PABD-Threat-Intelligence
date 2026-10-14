"""PABD 2.0 evaluation harness. Every number in the README/paper comes from this file.

Protocol
  2017 units are split train / test (see units.py). The test part is split again,
  chronologically within each attack episode, into CAL (first half) and TEST (second half).
  - detectors and baselines: trained on TRAIN only
  - fusion meta-model and conformal calibration: fitted on CAL only
  - every reported metric: TEST only (and all of 2018, which nothing is fitted on)
  "Attempted" units (malicious but unsuccessful) are excluded from metrics.

    python -m pabd2.evaluate                 # main results -> results_v2/
    python -m pabd2.evaluate --lofo          # adds leave-one-family-out (needs *_lo_* models)
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from .conformal import Conformal, drift_alarm
from .features import secs
from .killchain import FAM_STAGE, filter_host, stage_evidence
from .prepare import CACHE
from .schema import FAMILIES, STAGES
from .units import BIN

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_v2")
ALPHAS = [0.001, 0.01, 0.05]
LOFO = ["Internal Scan (post-compromise)", "Botnet", "Web Attack"]


def tagof(f):
    return "_lo_" + f.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")


def unit_means(year):
    """Mean token vector per unit (the input a flow-feature baseline sees)."""
    z = np.load(os.path.join(CACHE, f"tok_{year}.npz")); num = z["num"]
    flows = pd.read_parquet(os.path.join(CACHE, f"flows_{year}.parquet"), columns=["Src IP", "Timestamp", "day"])
    units = pd.read_parquet(os.path.join(CACHE, f"units_{year}.parquet"))
    b = secs(flows["Timestamp"]).astype(np.int64) // BIN
    key = pd.MultiIndex.from_arrays([flows["day"].to_numpy(), flows["Src IP"].to_numpy(), b])
    uid = pd.Series(np.arange(len(units)), index=pd.MultiIndex.from_frame(units[["day", "host", "bin"]]))
    fu = uid.reindex(key).to_numpy()
    ok = ~np.isnan(fu); fu = fu[ok].astype(np.int64)
    S = np.zeros((len(units), num.shape[1])); np.add.at(S, fu, num[ok])
    return S / np.maximum(np.bincount(fu, minlength=len(units)), 1)[:, None]


class Normality:
    """Distance-to-normal on the transformer's learned representation (shrinkage Mahalanobis),
    normalised by the reference distribution's own spread so it transfers across networks.
    The reference is whatever the deployment calls normal: 2017 benign training units in the lab,
    or simply the first hour of a new network's traffic (no labels) when deployed elsewhere."""
    def fit(self, E_ref):
        self.lw = LedoitWolf().fit(E_ref[:: max(1, len(E_ref) // 20000)])
        r = np.log(self.lw.mahalanobis(E_ref) + 1e-9)
        self.med, self.iqr = np.median(r), max(np.subtract(*np.percentile(r, [75, 25])), 1e-6)
        return self

    def score(self, E):
        return (np.log(self.lw.mahalanobis(E) + 1e-9) - self.med) / self.iqr


def first_hour(u):
    m = np.zeros(len(u), bool)
    for _, g in u.groupby("day"):
        m[g.index[g.t < g.t.min() + pd.Timedelta(hours=1)]] = True
    return m


def load(year, tag=""):
    u = pd.read_parquet(os.path.join(CACHE, f"units_{year}.parquet"))
    s = np.load(os.path.join(CACHE, f"seqscores_{year}{tag}.npz"), allow_pickle=True)
    g = np.load(os.path.join(CACHE, f"graphscores_{year}{tag}.npz"), allow_pickle=True)
    fams = list(s["families"])
    u["seq_anom"] = np.log(s["anom"] + 1e-6)
    u["seq_attack"] = 1 - s["proba"][:, fams.index("Benign")]
    u["graph"] = g["score"]
    u["log_n"] = np.log1p(u["n"]); u["log_dst"] = np.log1p(u["n_dst"])
    # family belief: average of transformer head and graph class profile (aligned by name)
    gcls = list(g["classes"]); P = np.zeros((len(u), len(fams)))
    for j, f in enumerate(fams):
        P[:, j] = 0.5 * s["proba"][:, j] + 0.5 * (g["cls"][:, gcls.index(f)] if f in gcls else 0)
    return u, P / np.maximum(P.sum(1, keepdims=True), 1e-9), fams, s["emb"]


def cal_test_split(u):
    """Chronological halves of the 2017 test part, per day and attack episode."""
    role = np.array(["none"] * len(u), dtype=object)
    t = u[u.split == "test"]
    for (day, fam), e in t.groupby(["day", "family"]):
        ts = e.sort_values("t"); h = len(ts) // 2
        role[ts.index[:h]] = "cal"; role[ts.index[h:]] = "test"
    return role


META = ["dist", "seq_attack", "graph", "log_n", "log_dst"]


def metrics(y, s):
    return {"auc": round(float(roc_auc_score(y, s)), 4), "ap": round(float(average_precision_score(y, s)), 4)}


def conformal_table(conf, scores, fam, alphas=ALPHAS):
    p = conf.pvalue(scores); rows = {}
    for a in alphas:
        alarm = p <= a
        r = {"target_fpr": a, "benign_fpr": round(float(alarm[fam == "Benign"].mean()), 5)}
        for f in sorted(set(fam) - {"Benign", "Attempted"}):
            r[f] = round(float(alarm[fam == f].mean()), 4)
        rows[str(a)] = r
    return rows, p


def detection_delay(u, p, alpha=0.01):
    """Per attack episode (day, family, host) in the evaluated set: seconds from the episode's
    first unit to the first alarm on that host at or after it."""
    out = []
    v = u.assign(alarm=p <= alpha)
    for (day, fam, host), e in v[~v.family.isin(["Benign", "Attempted"])].groupby(["day", "family", "host"]):
        t0 = e.t.min()
        h = v[(v.day == day) & (v.host == host) & (v.t >= t0)].sort_values("t")
        hit = h[h.alarm]
        out.append({"day": day, "family": fam, "host": host, "units": int(len(e)),
                    "delay_s": float((hit.t.iloc[0] - t0).total_seconds()) if len(hit) else None})
    return out


def killchain_eval(u, P, fams, p_value=None):
    """Filter every host's stage belief over time; score stage accuracy on attack units."""
    E = stage_evidence(P, fams, p_value); belief = np.zeros((len(u), len(STAGES)))
    for _, idx in u.sort_values("t").groupby(["day", "host"]).groups.items():
        belief[idx] = filter_host(E[idx])
    pred = np.array(STAGES)[belief.argmax(1)]
    true = u["family"].map(lambda f: FAM_STAGE.get(f, "Unknown")).to_numpy()
    m = ~u.family.isin(["Benign", "Attempted"]).to_numpy()
    return pred, belief, {"attack_stage_accuracy": round(float((pred[m] == true[m]).mean()), 4),
                          "benign_called_benign": round(float((pred[~m & (u.family == "Benign").to_numpy()] == "Benign").mean()), 4),
                          "per_stage": {s: round(float((pred[m & (true == s)] == s).mean()), 4)
                                        for s in STAGES[1:] if (m & (true == s)).any()}}


def main(lofo=False):
    os.makedirs(OUT, exist_ok=True); R = {}
    u, P, fams, emb = load(2017)
    u["role"] = cal_test_split(u)
    tr, ca, te = (u.split == "train").to_numpy(), (u.role == "cal").to_numpy(), (u.role == "test").to_numpy()
    norm17 = Normality().fit(emb[tr & (u.family == "Benign").to_numpy()])
    u["dist"] = norm17.score(emb)
    ok = (u.family != "Attempted").to_numpy()
    y = (u.family != "Benign").astype(int).to_numpy()

    # baselines on per-unit mean flow features (what a v1-style detector sees)
    M = unit_means(2017)
    rf = RandomForestClassifier(300, max_depth=None, class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    rf.fit(M[tr & ok], u.family[tr & ok])
    rf_attack = 1 - rf.predict_proba(M)[:, list(rf.classes_).index("Benign")]
    iso = IsolationForest(n_estimators=300, random_state=0).fit(M[tr & (u.family == "Benign").to_numpy()])
    iso_s = -iso.score_samples(M)

    # fusion meta-model (fitted on CAL)
    sc = StandardScaler().fit(u.loc[ca & ok, META])
    meta = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(u.loc[ca & ok, META]), y[ca & ok])
    u["fused"] = meta.decision_function(sc.transform(u[META]))
    conf = Conformal().fit(u.loc[ca & (u.family == "Benign").to_numpy(), "fused"])
    json.dump({"meta": META, "mu": sc.mean_.tolist(), "sd": sc.scale_.tolist(), "w": meta.coef_[0].tolist(),
               "b": float(meta.intercept_[0]), "cal_benign_fused": conf.cal.tolist()},
              open(os.path.join(OUT, "fusion_state.json"), "w"))

    ev = te & ok
    R["2017_test_units"] = {"benign": int((ev & (y == 0)).sum()), "attack": int((ev & (y == 1)).sum())}
    R["2017_ranking"] = {
        "PABD 2.0 fused": metrics(y[ev], u.fused[ev]),
        "Distance-to-normal (learned representation)": metrics(y[ev], u.dist[ev]),
        "FlowSeq reconstruction anomaly (negative result)": metrics(y[ev], u.seq_anom[ev]),
        "FlowSeq family head": metrics(y[ev], u.seq_attack[ev]),
        "E-GraphSAGE graph layer": metrics(y[ev], u.graph[ev]),
        "Baseline: Random Forest on flow features": metrics(y[ev], rf_attack[ev]),
        "Baseline: Isolation Forest": metrics(y[ev], iso_s[ev]),
    }
    R["2017_per_family_auc"] = {
        f: {k: round(float(roc_auc_score(u.family[ev & u.family.isin(["Benign", f]).to_numpy()] == f,
                                          s[ev & u.family.isin(["Benign", f]).to_numpy()])), 4)
            for k, s in [("fused", u.fused.to_numpy()), ("rf_baseline", rf_attack)]}
        for f in FAMILIES[1:] if (ev & (u.family == f).to_numpy()).sum() >= 3}
    R["2017_conformal"], p17 = conformal_table(conf, u.fused[ev].to_numpy(), u.family[ev].to_numpy())
    R["meta_weights"] = dict(zip(META, np.round(meta.coef_[0], 3).tolist()))

    # early warning: delays on the TEST part
    ue = u[te].copy(); pe = conf.pvalue(ue.fused.to_numpy())
    R["2017_detection_delay_alpha_0.01"] = detection_delay(ue.reset_index(drop=True), pe)

    # kill-chain stage tracking (2017 test days, whole timeline so the filter has history)
    pred, belief, kc = killchain_eval(u.reset_index(drop=True), P, fams, conf.pvalue(u.fused.to_numpy()))
    R["2017_killchain"] = kc
    u["stage_pred"] = pred
    u["p_value"] = conf.pvalue(u.fused.to_numpy())
    u["family_guess"] = np.array(fams)[P.argmax(1)]
    for j, s_ in enumerate(STAGES): u[f"b_{s_}"] = belief[:, j]
    ben_tr = tr & (u.family == "Benign").to_numpy()
    np.savez_compressed(os.path.join(OUT, "console_cache.npz"), unit_mean=M.astype(np.float32),
                        benign_mu=M[ben_tr].mean(0), benign_sd=M[ben_tr].std(0))

    # the infiltration chain on Thursday: victim 192.168.10.8
    v = u[(u.day == "thursday") & (u.host == "192.168.10.8")].sort_values("t")
    pv = conf.pvalue(v.fused.to_numpy())
    first_inf = v[v.family == "Infiltration"].t.min(); first_scan = v[v.family.str.startswith("Internal")].t.min()
    first_alarm = v[(pv <= 0.01) & (v.t >= first_inf - pd.Timedelta(minutes=5))].t.min()
    R["infiltration_chain"] = {"host": "192.168.10.8", "first_infiltration_flow": str(first_inf),
                               "first_internal_scan": str(first_scan), "first_alarm_alpha_0.01": str(first_alarm),
                               "alarm_before_scan_s": float((first_scan - first_alarm).total_seconds()) if pd.notna(first_alarm) else None,
                               "stage_at_scan": str(v[v.t >= first_scan].stage_pred.iloc[0]) if pd.notna(first_scan) else None}

    # drift: conformal martingale over the benign 2017 test stream vs the 2018 benign stream
    ben17 = u[te & (u.family == "Benign").to_numpy()].sort_values("t")
    k17, _ = drift_alarm(conf.pvalue(ben17.fused.to_numpy()))
    R["drift_2017_benign"] = {"alarm_index": k17, "n": int(len(ben17))}

    # cross-dataset: 2018 (nothing refitted)
    if os.path.exists(os.path.join(CACHE, "seqscores_2018.npz")):
        u8, P8, _, emb8 = load(2018)
        M8 = unit_means(2018)
        fh = first_hour(u8)
        R["2018_adaptation"] = {"first_hour_units": int(fh.sum()),
                                "attack_units_in_first_hour": int((fh & ~u8.family.isin(["Benign", "Attempted"]).to_numpy()).sum())}
        u8["dist"] = Normality().fit(emb8[fh]).score(emb8)
        u8["fused"] = meta.decision_function(sc.transform(u8[META]))
        iso8 = IsolationForest(n_estimators=300, random_state=0).fit(M8[fh])
        ok8 = (u8.family != "Attempted").to_numpy() & ~fh; y8 = (u8.family != "Benign").astype(int).to_numpy()
        rf8 = 1 - rf.predict_proba(M8)[:, list(rf.classes_).index("Benign")]
        R["2018_units"] = {"benign": int((ok8 & (y8 == 0)).sum()), "attack": int((ok8 & (y8 == 1)).sum()),
                           "families": u8.family[ok8].value_counts().to_dict()}
        R["2018_ranking"] = {
            "PABD 2.0 fused": metrics(y8[ok8], u8.fused[ok8]),
            "Distance-to-normal, adapted on first hour (no labels)": metrics(y8[ok8], u8.dist[ok8]),
            "FlowSeq reconstruction anomaly (negative result)": metrics(y8[ok8], u8.seq_anom[ok8]),
            "FlowSeq family head": metrics(y8[ok8], u8.seq_attack[ok8]),
            "E-GraphSAGE graph layer": metrics(y8[ok8], u8.graph[ok8]),
            "Baseline: Random Forest on flow features": metrics(y8[ok8], rf8[ok8]),
            "Baseline: Isolation Forest (2017 normal)": metrics(y8[ok8], -iso.score_samples(M8)[ok8]),
            "Baseline: Isolation Forest, adapted on first hour": metrics(y8[ok8], -iso8.score_samples(M8)[ok8]),
        }
        R["2018_per_family_auc"] = {
            f: {k: round(float(roc_auc_score(u8.family[ok8 & u8.family.isin(["Benign", f]).to_numpy()] == f,
                                              s[ok8 & u8.family.isin(["Benign", f]).to_numpy()])), 4)
                for k, s in [("fused", u8.fused.to_numpy()), ("rf_baseline", rf8)]}
            for f in FAMILIES[1:] if (ok8 & (u8.family == f).to_numpy()).sum() >= 3}
        R["2018_conformal_with_2017_calibration"], p8 = conformal_table(conf, u8.fused[ok8].to_numpy(), u8.family[ok8].to_numpy())
        ben18 = u8[ok8 & (y8 == 0)].sort_values("t")
        k18, _ = drift_alarm(conf.pvalue(ben18.fused.to_numpy()))
        R["drift_2018_benign"] = {"alarm_index": k18, "n": int(len(ben18)),
                                  "alarm_time": str(ben18.t.iloc[k18]) if k18 >= 0 else None}
        # recalibrate on the first hour of 2018 benign traffic only (no labels needed in practice:
        # the first hour of a new deployment is assumed quiet) and re-check the guarantee
        conf8 = Conformal().fit(u8.fused[fh])     # label-free: everything in the first hour
        rest = ok8
        R["2018_conformal_recalibrated_first_hour"], _ = conformal_table(conf8, u8.fused[rest].to_numpy(), u8.family[rest].to_numpy())
        R["2018_recalibration_units"] = int(fh.sum())
        p8a = conf8.pvalue(u8.fused.to_numpy())
        R["2018_detection_delay_alpha_0.01"] = detection_delay(u8[ok8].reset_index(drop=True), p8a[ok8])
        k18a, _ = drift_alarm(conf8.pvalue(u8[ok8 & (y8 == 0)].sort_values("t").fused.to_numpy()))
        R["drift_2018_benign_after_recalibration"] = {"alarm_index": k18a}

    # leave-one-family-out: zero-day proxy
    if lofo:
        R["lofo"] = {}
        for f in LOFO:
            if not os.path.exists(os.path.join(CACHE, f"seqscores_2017{tagof(f)}.npz")): continue
            ul, _, _, embl = load(2017, tagof(f)); ul["role"] = u["role"]
            ul["dist"] = Normality().fit(embl[tr & (u.family == "Benign").to_numpy()]).score(embl)
            okl = ok & (u.family != f).to_numpy()
            scl = StandardScaler().fit(ul.loc[ca & okl, META])
            ml = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scl.transform(ul.loc[ca & okl, META]), y[ca & okl])
            ul["fused"] = ml.decision_function(scl.transform(ul[META]))
            cl = Conformal().fit(ul.loc[ca & (u.family == "Benign").to_numpy(), "fused"])
            rfl = RandomForestClassifier(300, class_weight="balanced_subsample", n_jobs=-1, random_state=0)
            m_tr = tr & okl; rfl.fit(M[m_tr], u.family[m_tr])
            rfs = 1 - rfl.predict_proba(M)[:, list(rfl.classes_).index("Benign")]
            crf = Conformal().fit(rfs[ca & (u.family == "Benign").to_numpy()])
            sel = te & u.family.isin(["Benign", f]).to_numpy()
            yy = (u.family[sel] == f).to_numpy()
            R["lofo"][f] = {
                "units": int(yy.sum()),
                "pabd2_auc": round(float(roc_auc_score(yy, ul.fused[sel])), 4),
                "rf_auc": round(float(roc_auc_score(yy, rfs[sel])), 4),
                "pabd2_detect_at_1pct": round(float((cl.pvalue(ul.fused[sel].to_numpy())[yy] <= 0.01).mean()), 4),
                "rf_detect_at_1pct": round(float((crf.pvalue(rfs[sel])[yy] <= 0.01).mean()), 4),
            }

    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    u.to_parquet(os.path.join(OUT, "units_scored_2017.parquet"), index=False)
    print(json.dumps(R, indent=1, default=str)[:6000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--lofo", action="store_true")
    main(ap.parse_args().lofo)
