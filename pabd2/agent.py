"""Investigation agent: turns one alert into an evidence bundle and a short incident report.

The agent's tools are plain functions over PABD's own outputs -- calibrated confidence, the
host's kill-chain belief and forecast, the features that deviate most from benign traffic, the
host's connections in the window (with first-time internal contacts), and the MITRE mapping.
A local model (Ollama, llama3.2 by default; free, nothing leaves the machine) writes the report
from that bundle only. If Ollama is unavailable, a deterministic template writes it instead.
Response actions are recommendations; a human approves them.
"""
import json, os, urllib.request
import numpy as np
import pandas as pd
from .features import is_internal
from .killchain import NEXT_BY_FAMILY, NEXT_TECHNIQUE
from .schema import FEATURES, MITRE, STAGES

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL = os.environ.get("PABD_AGENT_MODEL", "llama3.2")
FEAT_NAMES = FEATURES + ["time between connections", "first-time destination", "internal destination"]

ACTIONS = {
    "Reconnaissance": "Rate-limit or block the scanning source at the perimeter; watch for follow-on access attempts.",
    "Initial Access": "Lock the targeted accounts or service, force credential resets, and isolate the host if login succeeded.",
    "Command and Control": "Isolate the host from the network and block its outbound command-and-control destinations.",
    "Impact": "Engage DDoS/DoS mitigation upstream and shed the attacking source; protect the targeted service.",
}


def evidence(u_row, belief, p_value, unit_mean, benign_mu, benign_sd, flows_in_window, family_guess):
    z = (unit_mean - benign_mu) / np.where(benign_sd < 1e-6, 1, benign_sd)
    top = np.argsort(-np.abs(z))[:5]
    stage = STAGES[int(np.argmax(belief))]
    fw = flows_in_window
    new_internal = sorted(set(fw.loc[is_internal(fw["Dst IP"]), "Dst IP"]))
    tactic, tech = MITRE.get(family_guess, ("-", "-"))
    return {
        "host": u_row["host"], "time": str(u_row["t"]),
        "confidence": ">99.99%" if p_value < 1e-4 else f"{(1 - float(p_value)) * 100:.2f}%",
        "p_value": float(f"{p_value:.2g}"), "likely_activity": family_guess,
        "mitre": {"tactic": tactic, "technique": tech},
        "kill_chain_stage": stage, "stage_belief": {s: round(float(b), 3) for s, b in zip(STAGES, belief)},
        "forecast_next": NEXT_BY_FAMILY.get(family_guess, NEXT_TECHNIQUE[stage]),
        "connections_in_window": int(len(fw)), "distinct_destinations": int(fw["Dst IP"].nunique()),
        "internal_hosts_contacted": new_internal[:12],
        "top_deviations": [{"feature": FEAT_NAMES[i], "z": round(float(z[i]), 1)} for i in top],
        "recommended_action": ACTIONS.get(stage, "Monitor; no action needed."),
    }


PROMPT = """You are a SOC analyst. Write a concise incident report from the JSON evidence below.
Use ONLY facts in the evidence; do not invent hosts, users, malware names or numbers.
Format: a one-line headline, then 3-5 short bullet points (what happened, why PABD flagged it,
where it is in the kill chain and what is likely next, recommended action for a human to approve).

EVIDENCE:
{ev}
"""


def report(ev, timeout=60):
    try:
        req = urllib.request.Request(OLLAMA, data=json.dumps({"model": MODEL, "prompt": PROMPT.format(ev=json.dumps(ev, indent=1)),
                                                              "stream": False, "options": {"temperature": 0.2}}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["response"].strip(), "llm:" + MODEL
    except Exception:
        return template(ev), "template"


def template(ev):
    dev = ", ".join(f"{d['feature']} ({d['z']:+.0f} sd)" for d in ev["top_deviations"][:3])
    lines = [f"{ev['likely_activity']} from {ev['host']} at {ev['time']} ({ev['confidence']} confidence)",
             f"- PABD flagged {ev['connections_in_window']} connections to {ev['distinct_destinations']} destinations in 10 s.",
             f"- Biggest deviations from normal traffic: {dev}.",
             f"- Kill-chain stage: {ev['kill_chain_stage']} ({ev['mitre']['technique']}). Likely next: {ev['forecast_next']}.",
             f"- Recommended (needs approval): {ev['recommended_action']}"]
    if ev["internal_hosts_contacted"]:
        lines.insert(2, f"- Internal hosts contacted: {', '.join(ev['internal_hosts_contacted'][:6])}.")
    return "\n".join(lines)
