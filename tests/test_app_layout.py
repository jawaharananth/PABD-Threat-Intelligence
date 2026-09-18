"""
Regression guard for the dashboard wiring. app.py's layout tree and its
callback Output list were rewritten several times; a component id present
in one but not the other only fails at request time (HTTP 500 on every
tick), which is exactly how the dashboard looked "not working" once. This
catches it at test time instead.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest

app_mod = pytest.importorskip("app")


def _collect_ids(component, out):
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        out.add(cid)
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for c in children:
            _collect_ids(c, out)
    elif children is not None and hasattr(children, "children"):
        _collect_ids(children, out)


def test_every_callback_output_and_input_id_exists_in_layout():
    layout_ids = set()
    _collect_ids(app_mod.app.layout, layout_ids)
    referenced = set()
    for key in app_mod.app.callback_map:
        for part in key.strip(".").split("..."):
            referenced.add(part.split(".")[0])
    for key, cb in app_mod.app.callback_map.items():
        for inp in cb.get("inputs", []):
            referenced.add(inp["id"])
    missing = referenced - layout_ids
    assert not missing, f"callback references ids not in layout: {missing}"


def test_update_returns_one_value_per_declared_output():
    app_mod.S["running"] = True
    for _ in range(3):
        app_mod.sim_step()
    result = app_mod.update(1)
    main_key = next(k for k in app_mod.app.callback_map if "kpi-strip" in k)
    n_outputs = len(main_key.strip(".").split("..."))
    assert len(result) == n_outputs


def test_simulated_attack_actually_fires_pabd():
    # Guards the sigma-scaled attack profile: with absolute deltas against
    # the real-data baseline this never fired, which made the demo look dead.
    app_mod.S.update({"running": True, "attack_mode": True, "attack_type": "Brute Force",
                      "step": 0, "pabd_fired": False, "attack_classified": False,
                      "cusum": {f: 0.0 for f in app_mod.FEATURES}})
    for _ in range(60):
        app_mod.sim_step()
        if app_mod.S["pabd_fired"]:
            break
    assert app_mod.S["pabd_fired"], "simulated attack never crossed the CUSUM threshold in 60 steps"
