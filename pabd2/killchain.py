"""Per-host kill-chain stage tracking and next-step forecast.

A hidden Markov model over stages Benign -> Reconnaissance -> Initial Access -> Command and
Control -> Impact. Each host-window contributes evidence: the calibrated family probabilities
from the detectors, collapsed onto stages. Forward filtering gives P(stage | evidence so far)
for every host in real time; the transition matrix encodes that attacks progress mostly forward
and that hosts return to benign between episodes. The forecast is one step of the chain applied
to the current belief: "what this host is likely to do next".

Transitions are a stated prior (forward drift, slow decay to benign), not fitted to the test
days -- CIC attacks are scheduled episodes, not organic campaigns, so fitting them would
overstate how predictable real intrusions are.
"""
import numpy as np
from .schema import FAMILY, MITRE, STAGES

FAM_STAGE = {"Benign": "Benign", "Port Scan": "Reconnaissance", "Internal Scan (post-compromise)": "Reconnaissance",
             "Brute Force": "Initial Access", "Web Attack": "Initial Access", "Infiltration": "Initial Access",
             "Botnet": "Command and Control", "DoS": "Impact", "DDoS": "Impact"}
NEXT_TECHNIQUE = {
    "Benign": "-",
    "Reconnaissance": "T1110 Brute Force / T1190 Exploit Public-Facing Application",
    "Initial Access": "T1046 Network Service Discovery (internal scanning)",
    "Command and Control": "T1498/T1499 Denial of Service, or T1041 Exfiltration",
    "Impact": "Sustained impact; containment",
}

# Family-specific next steps where the family tells us more than the stage does.
NEXT_BY_FAMILY = {
    "Internal Scan (post-compromise)": "T1021 Remote Services (lateral movement to the hosts it just mapped)",
    "Port Scan": "T1110 Brute Force / T1190 Exploit Public-Facing Application against open services",
    "Brute Force": "T1078 Valid Accounts (log in with cracked credentials), then internal discovery",
    "Web Attack": "T1505 Server Software Component (web shell), then T1046 internal discovery",
    "Infiltration": "T1046 Network Service Discovery from the compromised host",
    "Botnet": "T1041 Exfiltration over C2, or T1498 Network DoS on command",
}
S = len(STAGES)
T = np.array([
    # Benign  Recon  Access  C2     Impact
    [0.970, 0.015, 0.010, 0.003, 0.002],   # Benign
    [0.080, 0.820, 0.080, 0.010, 0.010],   # Reconnaissance
    [0.060, 0.060, 0.780, 0.070, 0.030],   # Initial Access
    [0.050, 0.020, 0.030, 0.850, 0.050],   # C2
    [0.080, 0.010, 0.010, 0.030, 0.870],   # Impact
])


def stage_evidence(proba, families, p_value=None, alpha_ref=0.05):
    """Stage likelihoods for each window.

    Whether a host is attacking is decided by the calibrated detector (its conformal p-value):
    attack mass = 1 - min(1, p / alpha_ref). Which stage it is in is decided by the family
    classifiers, renormalised over the non-benign families. Without p-values, the classifiers'
    own benign probability is used. A floor keeps every stage possible."""
    proba = np.asarray(proba, dtype=np.float64)
    b = families.index("Benign")
    attack = 1 - proba[:, b] if p_value is None else 1 - np.minimum(1, np.asarray(p_value) / alpha_ref)
    nb = proba.copy(); nb[:, b] = 0
    nb /= np.maximum(nb.sum(1, keepdims=True), 1e-12)
    E = np.full((len(proba), S), 1e-3)
    E[:, 0] += 1 - attack
    for j, f in enumerate(families):
        if j != b: E[:, STAGES.index(FAM_STAGE[f])] += attack * nb[:, j]
    return E / E.sum(1, keepdims=True)


def filter_host(E, prior=None):
    """Forward algorithm over one host's time-ordered windows. Returns beliefs (n, S)."""
    b = np.zeros(S); b[0] = 1.0 if prior is None else 0
    if prior is not None: b = prior
    out = np.zeros((len(E), S))
    for t in range(len(E)):
        b = (b @ T) * E[t]; b /= b.sum(); out[t] = b
    return out


def forecast(belief, family=None):
    """Next-window stage distribution and the most likely next MITRE technique."""
    nxt = belief @ T
    stage = STAGES[int(np.argmax(nxt[1:]) + 1)] if belief[0] < 0.5 else "Benign"
    return nxt, NEXT_BY_FAMILY.get(family, NEXT_TECHNIQUE[stage])
