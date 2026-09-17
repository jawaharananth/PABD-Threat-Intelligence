"""
Single source of truth for MITRE ATT&CK tagging, shared by app.py (live
dashboard) and the training/reporting scripts (train_models.py, any results
table). Keeping this in one place instead of duplicating the mapping avoids
the two copies silently drifting apart -- which is exactly the kind of thing
that caused the ANAMES label-id bug fixed alongside this.

Technique IDs are the actual MITRE ATT&CK framework IDs, not invented labels.
Label ids match consolidate_data.py's LABEL_MAP.
"""

LABEL_IDS = {0: 'Normal', 1: 'Brute Force', 2: 'Reconnaissance', 3: 'DoS',
             4: 'DDoS', 5: 'Bot', 6: 'Web Attack'}

MITRE_MAP = {
    'Normal': {
        'tactic': None, 'technique_id': None, 'technique_name': None,
    },
    'Reconnaissance': {
        'tactic': 'Discovery', 'technique_id': 'T1046',
        'technique_name': 'Network Service Scanning',
    },
    'Brute Force': {
        'tactic': 'Credential Access', 'technique_id': 'T1110',
        'technique_name': 'Brute Force',
    },
    'DoS': {
        'tactic': 'Impact', 'technique_id': 'T1499',
        'technique_name': 'Endpoint Denial of Service',
    },
    'DDoS': {
        'tactic': 'Impact', 'technique_id': 'T1498',
        'technique_name': 'Network Denial of Service',
    },
    'Bot': {
        'tactic': 'Command and Control', 'technique_id': 'T1071',
        'technique_name': 'Application Layer Protocol',
    },
    'Web Attack': {
        'tactic': 'Initial Access', 'technique_id': 'T1190',
        'technique_name': 'Exploit Public-Facing Application',
    },
    # Real-data ground truth exists (36 raw flows / 26 windows) but is too
    # sparse to train a class on -- see consolidate_data.py. Kept here for
    # the case-study report only, not used by app.py's live classifier.
    'Infiltration': {
        'tactic': 'Initial Access', 'technique_id': 'T1203',
        'technique_name': 'Exploitation for Client Execution',
    },
    # Demo-only: driven by app.py's synthetic SIMULATE ATTACK button, no
    # real-data ground truth for this class in the days used (see README /
    # "what this project does not claim").
    'Lateral Movement': {
        'tactic': 'Lateral Movement', 'technique_id': 'T1021',
        'technique_name': 'Remote Services (demo-only, no real ground truth)',
    },
}


def tag(attack_type):
    """Return the MITRE tactic/technique dict for an attack_type string,
    or a clearly-marked unknown entry rather than silently defaulting."""
    return MITRE_MAP.get(attack_type, {
        'tactic': 'UNMAPPED', 'technique_id': None,
        'technique_name': f'No MITRE mapping defined for "{attack_type}"',
    })
