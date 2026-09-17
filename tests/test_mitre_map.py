"""
mitre_map.py is the single source of truth for label ids used by both
app.py's live classifier and the training pipeline. This is the exact spot
where a real bug shipped earlier (ANAMES had ids 1/2 swapped) -- these tests
exist specifically to catch that class of regression, not just to pad
coverage numbers.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mitre_map import LABEL_IDS, MITRE_MAP, tag


def test_label_ids_match_consolidate_data_label_map():
    # These ids must match consolidate_data.py's LABEL_MAP exactly. Hardcoded
    # here rather than imported, because the whole point is to catch drift
    # between the two files -- importing one to check itself proves nothing.
    expected = {
        0: "Normal", 1: "Brute Force", 2: "Reconnaissance",
        3: "DoS", 4: "DDoS", 5: "Bot", 6: "Web Attack",
    }
    assert LABEL_IDS == expected


def test_every_real_trained_class_has_a_mitre_entry():
    for label_id, name in LABEL_IDS.items():
        if name == "Normal":
            continue
        m = tag(name)
        assert m["tactic"] != "UNMAPPED", f"{name} (id={label_id}) has no MITRE mapping"
        assert m["technique_id"] is not None, f"{name} (id={label_id}) missing technique_id"


def test_unmapped_name_is_explicit_not_silent():
    # An unmapped attack_type must be clearly flagged, not silently fall back
    # to some default technique that would misattribute it.
    m = tag("Something Nobody Wrote A Mapping For")
    assert m["tactic"] == "UNMAPPED"
    assert m["technique_id"] is None


def test_technique_ids_are_real_mitre_format():
    # Sanity check the IDs actually look like ATT&CK technique IDs (T####),
    # not placeholder text.
    for name, m in MITRE_MAP.items():
        if m["technique_id"] is None:
            continue
        assert m["technique_id"].startswith("T")
        assert m["technique_id"][1:].isdigit()
