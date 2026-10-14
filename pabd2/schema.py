"""Shared schema for the corrected CIC-IDS-2017 / CSE-CIC-IDS-2018 flows (Liu, Engelen et al., CNS 2022).

Both years were re-extracted with the same fixed CICFlowMeter, so a model trained on one can be
scored on the other without any feature mapping. Identity columns (IPs, ports, timestamps, flow
ids) are kept for building host sequences and graphs but never fed to a model as features:
graph-NIDS audits show models otherwise learn attacker addresses instead of behaviour.
"""
import numpy as np

ID_COLS = ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Timestamp"]
DROP_COLS = ["id", "Flow ID", "Attempted Category"]

# Behavioural flow features. Ports and raw timestamps are deliberately excluded.
FEATURES = [
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
    "Total Length of Bwd Packet", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd RST Flags",
    "Bwd RST Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean", "Packet Length Std",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "Down/Up Ratio", "Average Packet Size", "FWD Init Win Bytes",
    "Bwd Init Win Bytes", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Max",
    "Idle Mean", "Idle Max", "Total TCP Flow Time",
]
# Coarse, generalisable port context instead of raw port numbers.
PORT_BUCKETS = {"web": {80, 443, 8080, 8443}, "ftp": {20, 21}, "ssh": {22}, "smb": {139, 445},
                "dns": {53}, "mail": {25, 110, 143, 465, 587, 993, 995}, "rdp": {3389}}
PORT_CATS = list(PORT_BUCKETS) + ["low_other", "high"]

# Attack families shared by both years -> canonical name, MITRE tactic, kill-chain stage.
# Stage order follows the Lockheed/MITRE progression used for the per-host stage tracker.
STAGES = ["Benign", "Reconnaissance", "Initial Access", "Command and Control", "Impact"]
FAMILY = {
    "BENIGN": ("Benign", "Benign"),
    "Portscan": ("Port Scan", "Reconnaissance"),
    "Infiltration - Portscan": ("Internal Scan (post-compromise)", "Reconnaissance"),
    "FTP-Patator": ("Brute Force", "Initial Access"), "SSH-Patator": ("Brute Force", "Initial Access"),
    "FTP-BruteForce": ("Brute Force", "Initial Access"), "SSH-BruteForce": ("Brute Force", "Initial Access"),
    "Web Attack - Brute Force": ("Web Attack", "Initial Access"), "Web Attack - XSS": ("Web Attack", "Initial Access"),
    "Web Attack - SQL Injection": ("Web Attack", "Initial Access"),
    "Web Attack - SQL": ("Web Attack", "Initial Access"), "Brute Force -Web": ("Web Attack", "Initial Access"),
    "Brute Force -XSS": ("Web Attack", "Initial Access"), "SQL Injection": ("Web Attack", "Initial Access"),
    "Infiltration": ("Infiltration", "Initial Access"),
    "Botnet": ("Botnet", "Command and Control"), "Bot": ("Botnet", "Command and Control"),
    "Botnet Ares": ("Botnet", "Command and Control"),
    "DoS Hulk": ("DoS", "Impact"), "DoS GoldenEye": ("DoS", "Impact"), "DoS Slowloris": ("DoS", "Impact"),
    "DoS Slowhttptest": ("DoS", "Impact"), "DoS-Hulk": ("DoS", "Impact"), "DoS-GoldenEye": ("DoS", "Impact"),
    "DoS-Slowloris": ("DoS", "Impact"), "DoS-SlowHTTPTest": ("DoS", "Impact"), "Heartbleed": ("DoS", "Impact"),
    "DDoS": ("DDoS", "Impact"), "DDoS-LOIC-HTTP": ("DDoS", "Impact"), "DDoS-HOIC": ("DDoS", "Impact"),
    "DDoS-LOIC-UDP": ("DDoS", "Impact"), "DDOS-LOIC-UDP": ("DDoS", "Impact"),
}
FAMILIES = ["Benign", "Port Scan", "Internal Scan (post-compromise)", "Brute Force", "Web Attack",
            "Infiltration", "Botnet", "DoS", "DDoS"]
MITRE = {
    "Port Scan": ("TA0043 Reconnaissance", "T1595 Active Scanning"),
    "Internal Scan (post-compromise)": ("TA0007 Discovery", "T1046 Network Service Discovery"),
    "Brute Force": ("TA0006 Credential Access", "T1110 Brute Force"),
    "Web Attack": ("TA0001 Initial Access", "T1190 Exploit Public-Facing Application"),
    "Infiltration": ("TA0001 Initial Access", "T1566 Phishing"),
    "Botnet": ("TA0011 Command and Control", "T1071 Application Layer Protocol"),
    "DoS": ("TA0040 Impact", "T1499 Endpoint Denial of Service"),
    "DDoS": ("TA0040 Impact", "T1498 Network Denial of Service"),
}


def canon(label: str):
    """Map a raw label to (family, stage, attempted). Attempted attacks (the dataset's
    '- Attempted' suffix: malicious intent, no successful effect) are kept as their family
    but flagged, so they can be excluded from the 'successful attack' ground truth."""
    lab = str(label).strip()
    attempted = lab.endswith("- Attempted")
    base = lab.replace(" - Attempted", "").strip()
    fam, stage = FAMILY.get(base, (None, None))
    if fam is None:
        return "Unknown", "Unknown", attempted
    return fam, stage, attempted


def port_cat(p: np.ndarray) -> np.ndarray:
    out = np.where(p >= 1024, "high", "low_other").astype(object)
    for k, s in PORT_BUCKETS.items():
        out[np.isin(p, list(s))] = k
    return out
