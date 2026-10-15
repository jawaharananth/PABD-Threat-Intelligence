"""PABD 2.0 security console: replays a recorded CIC-IDS-2017 day as a live stream.

    python -m pabd2.console [--day thursday] [--speed 60] [--port 8060]

Everything shown is computed by the pipeline (results_v2/units_scored_2017.parquet): calibrated
p-values, kill-chain beliefs, the fused score and the graph of who talked to whom.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from . import agent
from .conformal import drift_alarm
from .features import is_internal
from .prepare import CACHE
from .schema import STAGES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results_v2")
BG, PANEL, LINE, TXT, DIM, GOLD = "#07090c", "#0f141b", "#1d2530", "#e8edf2", "#8a96a3", "#e8c478"
STAGE_COL = {"Benign": "#3a4552", "Reconnaissance": "#e8c478", "Initial Access": "#ff9d5c",
             "Command and Control": "#ff6b8b", "Impact": "#ff4d4d"}
ALPHA = 0.01


def load(day):
    u = pd.read_parquet(os.path.join(RES, "units_scored_2017.parquet"))
    u = u[u.day == day].sort_values("t").reset_index()
    flows = pd.read_parquet(os.path.join(CACHE, "flows_2017.parquet"), columns=["Src IP", "Dst IP", "Timestamp", "day"])
    flows = flows[flows.day == day]
    cc = np.load(os.path.join(RES, "console_cache.npz"))
    return u, flows, cc


def card(children, **style):
    return html.Div(children, style={"background": PANEL, "border": f"1px solid {LINE}", "borderRadius": "14px",
                                     "padding": "14px 16px", **style})


def kpi(label, value_id):
    return card([html.Div(label, style={"color": GOLD, "fontSize": "11px", "letterSpacing": "3px", "fontWeight": 600}),
                 html.Div(id=value_id, style={"color": TXT, "fontSize": "30px", "fontWeight": 300, "marginTop": "4px"})],
                flex="1")


def layout_positions(flows, keep_external=10):
    """Stable positions for the whole day: internal hosts plus the busiest external peers;
    every other external address is collapsed into one 'Internet' node."""
    import networkx as nx
    ext = flows.loc[~is_internal(flows["Dst IP"]), "Dst IP"].value_counts().head(keep_external).index
    keep = set(flows.loc[is_internal(flows["Src IP"]), "Src IP"]) | set(flows.loc[is_internal(flows["Dst IP"]), "Dst IP"]) | set(ext)
    m = lambda s: s.where(s.isin(keep) | is_internal(s), "Internet")
    e = pd.DataFrame({"s": m(flows["Src IP"]), "d": m(flows["Dst IP"])}).value_counts().reset_index(name="n")
    G = nx.Graph()
    for r in e.itertuples(): G.add_edge(r.s, r.d, weight=np.log1p(r.n))
    pos = nx.spring_layout(G, seed=7, k=1.6 / np.sqrt(max(1, len(G))), iterations=200, weight="weight")
    return {h: tuple(p) for h, p in pos.items()}, keep


def build_app(day="thursday", speed=60, start=None):
    u, flows, cc = load(day)
    t0, t1 = u.t.min(), u.t.max()
    if start: t0 = pd.Timestamp(f"{u.t.min().date()} {start}")
    POS, KEEP = layout_positions(flows)
    ben = u[u.family == "Benign"]
    app = Dash(__name__, title="PABD 2.0")
    app.layout = html.Div(style={"background": BG, "minHeight": "100vh", "padding": "22px 26px",
                                 "fontFamily": "Segoe UI, system-ui, sans-serif", "color": TXT}, children=[
        html.Div([html.Div([html.Span("PABD", style={"fontWeight": 600, "fontSize": "26px"}),
                            html.Span(" 2.0", style={"color": GOLD, "fontWeight": 300, "fontSize": "26px"}),
                            html.Div("Predictive attack behaviour detection  ·  calibrated · graph-aware · self-supervised",
                                     style={"color": DIM, "fontSize": "13px", "marginTop": "2px"})]),
                  html.Div([html.Div(id="clock", style={"fontSize": "22px", "fontWeight": 300, "textAlign": "right"}),
                            html.Div(f"Replaying CIC-IDS-2017 (corrected), {day.title()}  ·  {speed}× real time",
                                     style={"color": DIM, "fontSize": "12px"})])],
                 style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px"}),
        html.Div([kpi("HOSTS SEEN", "k-hosts"), kpi("ALERTS  (α = 1 %)", "k-alerts"),
                  kpi("FALSE-ALARM BUDGET", "k-fpr"), kpi("DRIFT MONITOR", "k-drift")],
                 style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),
        html.Div([
            card([html.Div("NETWORK  ·  LAST 60 SECONDS", style={"color": GOLD, "fontSize": "11px", "letterSpacing": "3px", "fontWeight": 600}),
                  dcc.Graph(id="g-net", config={"displayModeBar": False}, style={"height": "430px"})], flex="1.35"),
            card([html.Div("ALERTS", style={"color": GOLD, "fontSize": "11px", "letterSpacing": "3px", "fontWeight": 600}),
                  html.Div(id="feed", style={"height": "430px", "overflowY": "auto", "marginTop": "6px"})], flex="1"),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),
        html.Div([
            card([html.Div("KILL CHAIN  ·  STAGE BELIEF PER HOST", style={"color": GOLD, "fontSize": "11px", "letterSpacing": "3px", "fontWeight": 600}),
                  dcc.Graph(id="g-chain", config={"displayModeBar": False}, style={"height": "250px"})], flex="1.35"),
            card([html.Div("INCIDENT REPORT", style={"color": GOLD, "fontSize": "11px", "letterSpacing": "3px", "fontWeight": 600}),
                  dcc.Markdown(id="report", style={ "fontSize": "14px", "lineHeight": "1.55",
                                               "height": "250px", "overflowY": "auto", "marginTop": "8px", "color": "#d6dde4"},
                           children="Select an alert to investigate.")], flex="1"),
        ], style={"display": "flex", "gap": "12px"}),
        dcc.Interval(id="tick", interval=1000), dcc.Store(id="now", data=0), dcc.Store(id="sel"),
        dcc.Location(id="url"),
    ])

    def at(sec):
        return t0 + pd.Timedelta(seconds=sec)

    from urllib.parse import parse_qs

    @app.callback(Output("now", "data"), Output("tick", "disabled"), Input("tick", "n_intervals"),
                  Input("url", "search"), State("now", "data"))
    def advance(_, search, s):
        q = parse_qs((search or "").lstrip("?"))
        if "t" in q:   # frame mode: a fixed replay time, no ticking (for film capture)
            return (pd.Timestamp(f"{u.t.min().date()} {q['t'][0]}") - t0).total_seconds(), True
        s = (s or 0) + speed
        return (0 if at(s) > t1 else s), False

    @app.callback(Output("clock", "children"), Output("k-hosts", "children"), Output("k-alerts", "children"),
                  Output("k-fpr", "children"), Output("k-drift", "children"), Output("g-net", "figure"),
                  Output("g-chain", "figure"), Output("feed", "children"), Input("now", "data"))
    def render(s):
        now = at(s); seen = u[u.t <= now]
        alerts = seen[seen.p_value <= ALPHA]
        b = seen[seen.family == "Benign"]
        fpr = (b.p_value <= ALPHA).mean() if len(b) else 0
        k, stat = drift_alarm(b.sort_values("t").p_value.to_numpy()) if len(b) > 5 else (-1, np.array([0]))
        drift = "stable" if k < 0 else "shift detected"
        # network graph of the last 60 s
        w = flows[(flows.Timestamp > now - pd.Timedelta(seconds=60)) & (flows.Timestamp <= now)]
        mapn = lambda sr: sr.where(sr.isin(KEEP) | is_internal(sr), "Internet")
        e = pd.DataFrame({"s": mapn(w["Src IP"]), "d": mapn(w["Dst IP"])}).value_counts().reset_index(name="n")
        stage_now = seen[seen.t > now - pd.Timedelta(minutes=10)].sort_values("t").groupby("host").stage_pred.last().to_dict()
        fig = go.Figure()
        # faint full-day skeleton so the map never looks empty
        for r in e.itertuples():
            if r.s not in POS or r.d not in POS: continue
            (x0, y0), (x1, y1) = POS[r.s], POS[r.d]
            st_ = stage_now.get(r.s, "Benign")
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(width=0.8 + 2.4 * min(1, np.log1p(r.n) / 7), color=STAGE_COL[st_]),
                                     opacity=0.9 if st_ != "Benign" else 0.45))
        hosts = [h for h in POS]
        active = set(e.s) | set(e.d)
        xs = [POS[h][0] for h in hosts]; ys = [POS[h][1] for h in hosts]
        stg = [stage_now.get(h, "Benign") for h in hosts]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", showlegend=False,
                                 text=[(h if (s_ != "Benign" or h == "Internet") else "") for h, s_ in zip(hosts, stg)],
                                 textposition="top center", textfont=dict(color=TXT, size=12),
                                 marker=dict(size=[22 if s_ != "Benign" else (14 if h == "Internet" else (9 if h in active else 6)) for h, s_ in zip(hosts, stg)],
                                             color=[STAGE_COL[s_] if s_ != "Benign" else ("#5b6b7c" if h in active else "#26303b") for h, s_ in zip(hosts, stg)],
                                             line=dict(color=BG, width=1.5)),
                                 hovertext=[f"{h} · {s_}" for h, s_ in zip(hosts, stg)], hoverinfo="text"))
        xr = [min(xs) - 0.15, max(xs) + 0.15]; yr = [min(ys) - 0.15, max(ys) + 0.15]
        fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, margin=dict(l=4, r=4, t=4, b=4),
                          xaxis=dict(visible=False, range=xr), yaxis=dict(visible=False, range=yr))
        # kill-chain belief strips for the most suspicious hosts
        top = seen.groupby("host").fused.max().sort_values(ascending=False).head(6).index.tolist()
        ch = go.Figure()
        for i, h in enumerate(top[::-1]):
            hh = seen[seen.host == h].tail(240)
            st = hh[[f"b_{s_}" for s_ in STAGES]].to_numpy().argmax(1)
            ch.add_trace(go.Scatter(x=hh.t, y=[i] * len(hh), mode="markers", marker=dict(symbol="square", size=9,
                         color=[STAGE_COL[STAGES[k]] for k in st]), hovertext=[STAGES[k] for k in st], hoverinfo="text", showlegend=False))
        ch.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, margin=dict(l=110, r=8, t=4, b=24),
                         yaxis=dict(tickvals=list(range(len(top))), ticktext=top[::-1], color=DIM, showgrid=False, zeroline=False),
                         xaxis=dict(color=DIM, showgrid=False, zeroline=False))
        feed = []
        for r in alerts.sort_values("t", ascending=False).head(25).itertuples():
            col = STAGE_COL.get(r.stage_pred, GOLD)
            feed.append(html.Div(id={"type": "alert", "i": int(r.index)}, n_clicks=0, style={
                "borderLeft": f"3px solid {col}", "padding": "8px 10px", "margin": "6px 0", "cursor": "pointer",
                "background": "#121922", "borderRadius": "8px"}, children=[
                html.Div([html.Span(r.family_guess if r.family_guess != "Benign" else "Anomalous behaviour", style={"fontWeight": 600}),
                          html.Span(f"   {(1 - r.p_value) * 100:.2f}%", style={"color": GOLD, "float": "right"})]),
                html.Div(f"{r.host}  ·  {r.t.strftime('%H:%M:%S')}  ·  {r.stage_pred}", style={"color": DIM, "fontSize": "12px"})]))
        return (now.strftime("%H:%M:%S"), f"{seen.host.nunique()}", f"{len(alerts)}",
                f"{fpr * 100:.2f}% of 1%", drift, fig, ch, feed)

    from dash import ALL

    @app.callback(Output("report", "children"), Input("url", "search"))
    def preset_report(search):
        q = parse_qs((search or "").lstrip("?"))
        f = os.path.join(RES, f"report_{q['report'][0]}.txt") if "report" in q else None
        return open(f, encoding="utf-8").read() if f and os.path.exists(f) else no_update

    @app.callback(Output("report", "children", allow_duplicate=True), Input({"type": "alert", "i": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def investigate(clicks):
        if not ctx.triggered_id or not any(clicks): return no_update
        i = ctx.triggered_id["i"]; r = u[u["index"] == i].iloc[0]
        fw = flows[(flows["Src IP"] == r.host) & (flows.Timestamp >= r.t) & (flows.Timestamp < r.t + pd.Timedelta(seconds=10))]
        bel = np.array([r[f"b_{s_}"] for s_ in STAGES])
        ev = agent.evidence(r, bel, r.p_value, cc["unit_mean"][i], cc["benign_mu"], cc["benign_sd"], fw, r.family_guess)
        text, src = agent.report(ev)
        return text + f"\n\n— generated by {src} from PABD evidence; actions need human approval"

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="thursday"); ap.add_argument("--speed", type=int, default=60)
    ap.add_argument("--port", type=int, default=8060); ap.add_argument("--start", default=None, help="HH:MM:SS")
    a = ap.parse_args()
    build_app(a.day, a.speed, a.start).run(host="127.0.0.1", port=a.port, debug=False)
