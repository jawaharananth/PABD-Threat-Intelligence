# PABD Threat Intelligence System

A real-time network threat detection dashboard, plus an offline research
pipeline evaluated on real CICIDS2017 traffic. First major ML project — the
goal was cybersecurity that goes beyond detecting attacks after they happen.

## The Problem

Most intrusion detection systems only alert once an attack is already
running. PABD tries to catch the early warning signs instead — the drift
in traffic behavior that happens before an attack fully executes.

## What's actually validated on real data vs. demo-only

This matters more than usual here, so it's stated up front rather than
buried: this project went through a full pass of replacing every unverified
claim with a measured one. Some things that sound similar are NOT the same:

- **Real-data classes** (real CICIDS2017 ground truth, real accuracy
  numbers below): Reconnaissance, Brute Force, DoS, DDoS, Bot, Web Attack.
- **Demo-only** (synthetic traffic simulation, no real-data ground truth):
  Lateral Movement. Selectable in the live dashboard's SIMULATE ATTACK
  button, but there is no real-CICIDS lateral-movement label in the days
  used, so no accuracy claim exists for it.
- **Small-n case study, not a trained class**: Infiltration. Only 36 raw
  flows / 26 windows exist in the source data. Detection rate on those
  windows was 22/26, but the false-positive rate on the *same day's benign
  windows* is 65% — meaning that "detection" is mostly indistinguishable
  from the detector's general noise level on that day, not a real signal.
  Reported as inconclusive, not a result.

## How It Works

Four components, evaluated separately and together:

- **Isolation Forest** — real-time anomaly scoring on 10 flow-level features
- **Random Forest** — classifies attack type once anomaly is flagged (73%
  accuracy, 7 real classes, held-out chronological split — see Results)
- **CUSUM** — tracks drift across all 10 features simultaneously, fires when
  N of them cross threshold at once
- **LSTM** — predicts time-to-attack-onset (real MAE ~374s, n=50 held-out —
  see Results; this is a genuinely hard prediction target, not a solved one)
- **Fusion layer** — a calibrated meta-model combining all of the above into
  one risk score, which beats every individual signal's ranking quality
  (AUC 0.964 vs. best individual signal's 0.930)

Live dashboard extras: MITRE ATT&CK tactic/technique tagging on every
detection, a live per-alert SHAP explanation ("why did THIS window trigger,"
not a static global ranking), a kill-chain progress bar, and a live network
graph of recent connections.

## Tech Used

Python, Dash + Plotly (dashboard), scikit-learn (Isolation Forest, Random
Forest, calibrated logistic fusion), TensorFlow/Keras (LSTM), SHAP
(explainability), pandas/numpy, pytest (test suite).

## Project Structure

```
PABD_System/
├── app.py                          # live dashboard
├── real_data_loader.py             # raw CICIDS CSV -> windowed features
├── consolidate_data.py             # windows -> network_data_real.csv (labels)
├── train_models.py                 # trains research (10-feat) + demo (6-feat) models
├── train_lstm_countdown_real.py    # real-data countdown LSTM
├── cusum_core.py                   # single shared CUSUM implementation (with reset fix)
├── cusum_adaptive.py               # fixed vs rolling-baseline CUSUM comparison
├── fusion_model.py                 # meta-model combining IsoForest+CUSUM+RF
├── adversarial_robustness.py       # black-box evasion test against RF
├── mitre_map.py                    # single source of truth: label ids -> MITRE ATT&CK
├── shap_analysis.py                # offline SHAP figures for the paper
├── tests/                          # pytest suite (22 tests)
├── Dockerfile / docker-compose.yml # not build-tested here (Docker unavailable in dev env)
└── requirements.txt
```

## How to Run

```
pip install -r requirements.txt
python real_data_loader.py          # only if rebuilding windows_*.csv from raw CICIDS CSVs
python consolidate_data.py
python train_models.py
python train_lstm_countdown_real.py
python app.py
```
Then open localhost:8050. Hit SIMULATE ATTACK to see it in action.

Run the test suite: `pytest tests/ -v`

## Results (real data, with sample sizes — see individual scripts for full detail)

- Random Forest: **73% accuracy**, 7 classes, 861 real windows, chronological
  70/30 split. Per-class performance varies a lot — Bot (the weakest, F1=0.39)
  is genuinely harder to catch than DDoS. DDoS showing 1.00/1.00 in one run
  is on only 2 test samples — not a statistically meaningful claim.
- CUSUM: **no single operating point gives both good recall and low false
  positives.** Best-F1 point (N=2 of 10 features) recalls 72% of attacks but
  fires on 73% of benign windows on a fully clean, unseen day (Monday). A
  stricter point (N=6) gets FPR to 0% but recall to 2.6%. This tradeoff is
  reported as a table, not hidden behind one cherry-picked number.
- LSTM countdown: **~374 second MAE (n=50 held-out)**, trained on real
  CICIDS attack-onset timelines. This replaced an earlier claim of "±90s"
  which was measured on synthetic training data only — the real number is
  worse, and that's the one to trust.
- Fusion (IsoForest + CUSUM + RF combined): **AUC 0.964**, beating every
  individual signal (best individual: RF alone at 0.930).
- Adversarial robustness: real attacks can be pushed toward looking benign,
  but not for free — median perturbation needed ranges from 37% of the way
  toward "average benign traffic" (Bot, easiest to disguise) to 74% (DDoS,
  hardest). Many of the features that would need to change are ones an
  attacker can't move without breaking the attack's actual function (see
  `adversarial_robustness.py`'s controllability table).

## What this project does not claim

- No real Snort/Suricata baseline comparison — only flow-level CSVs are
  available (no raw pcaps), so there is nothing to replay through an actual
  IDS engine. Any "beats traditional IDS" framing should cite this limit.
- Not validated on a second dataset (e.g. CICIDS2018) — would require a
  multi-GB download not done here.
- CICIDS2017 itself has documented labeling/flow-construction issues
  (see: "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017
  Case Study," IEEE). Acknowledged, not fixed — the dataset can't be rewritten.
- Docker setup is written correctly but not build-tested (Docker wasn't
  available in the development environment).
- Live dashboard (`app.py`) uses a separate 6-feature demo-scale model pair
  from the 10-feature research pipeline that produced the numbers above —
  the demo's synthetic traffic simulator only knows how to generate 6
  features. Don't read dashboard-displayed confidence numbers as the
  paper's validated accuracy; the research scripts' held-out results are.

## Future improvements

- Real Suricata/Snort comparison if raw pcaps become available
- Second-dataset validation (CICIDS2018)
- Data-driven causal/graph-based detection for lateral movement, since no
  real ground truth for it exists in the current data
- Push alerts to Slack/email, deploy for continuous monitoring
