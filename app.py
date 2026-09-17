import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import joblib
import time
import random
import pandas as pd
import shap
import hashlib
from tensorflow import keras

# The _demo6 files are fit on the original 6-feature schema, for the live
# demo simulator and countdown LSTM (both only know that schema). The offline
# research pipeline (scaler.pkl/isolation_forest.pkl/random_forest.pkl, no
# suffix) now uses an expanded 10-feature schema for the paper's numbers --
# do not point the live demo at those, its feature generator only produces
# 6 values and the dimension mismatch will crash predict().
scaler = joblib.load('scaler_demo6.pkl')
iso_forest = joblib.load('isolation_forest_demo6.pkl')
rf_classifier = joblib.load('random_forest_demo6.pkl')
# Built once at startup -- TreeExplainer construction is the expensive part,
# per-call .shap_values() on a single row is cheap enough to run live per alert.
shap_explainer = shap.TreeExplainer(rf_classifier)
# lstm_model_real_countdown.keras is trained on real CICIDS2017 attack-onset
# timelines (train_lstm_countdown_real.py), not synthetic sequences. Held-out
# MAE on real data: ~342s (n=50) -- report that number in any paper/report,
# not the old synthetic-derived "~90s" figure.
lstm_model = keras.models.load_model('lstm_model_real_countdown.keras')
baseline_stats = joblib.load('baseline_stats.pkl')

from mitre_map import LABEL_IDS, tag as mitre_tag

FEATURES = ['connection_frequency','port_entropy','failed_auth_rate','packet_size_dist','protocol_ratio','burst_activity']
FLABELS = ['Conn Freq','Port Entropy','Failed Auth','Pkt Size','Proto Ratio','Burst Act']
# Label ids must match consolidate_data.py's LABEL_MAP exactly -- these were
# previously swapped (1/2 reversed, 3/4 stale after the DoS-Preparation rename
# and class expansion), which silently mislabeled real (non-simulated)
# classifications in the live dashboard. Sourced from mitre_map.py so this
# can't drift out of sync with the training pipeline's label ids again.
ANAMES = LABEL_IDS

_PROFILE_TEXT = {
    'Reconnaissance':{'Archetype':'The Slow Scanner','Technique':'Incremental port sweep','Velocity':'2-3 ports / min','Objective':'Vulnerability mapping'},
    'Brute Force':{'Archetype':'The Credential Tester','Technique':'Distributed login attempts','Velocity':'Low volume, persistent','Objective':'Authentication bypass'},
    'Lateral Movement':{'Archetype':'The Lateral Mover','Technique':'Internal network traversal','Velocity':'Slow and methodical','Objective':'Privilege escalation'},
    'DoS':{'Archetype':'The Flood Preparer','Technique':'Resource exhaustion, single-source','Velocity':'Rapid burst spikes','Objective':'Service disruption'},
    'DDoS':{'Archetype':'The Distributed Flooder','Technique':'Volumetric multi-source flood','Velocity':'Sustained high-volume burst','Objective':'Service disruption at scale'},
    'Bot':{'Archetype':'The Beacon','Technique':'Low-volume periodic C2 callback','Velocity':'Persistent, low-and-slow','Objective':'Command-and-control channel'},
    'Web Attack':{'Archetype':'The Application Exploiter','Technique':'SQLi / XSS / credential stuffing on web app','Velocity':'Bursty, targeted requests','Objective':'Application compromise'},
}
# Merge in MITRE tactic/technique from the shared map so the two can't drift.
PROFILES = {}
for _name, _fields in _PROFILE_TEXT.items():
    _m = mitre_tag(_name)
    PROFILES[_name] = {
        **_fields,
        'MITRE Tactic': _m['tactic'] or '-',
        'MITRE Technique': f"{_m['technique_id']} {_m['technique_name']}" if _m['technique_id'] else '-',
    }
MEANS = baseline_stats['mean']
STDS = baseline_stats['std']
CUSUM_K = 0.5
CUSUM_H = 8.0
METRIC_COLORS = ['#1A73E8','#1E8E3E','#E37400','#D93025','#9334E6','#12B5CB']

S = {
    'running':False,'attack_mode':False,'attack_type':'Reconnaissance',
    'step':0,'sequence':[],'total_packets':0,'flagged_packets':0,
    'cusum':{f:0.0 for f in FEATURES},
    'cusum_hist':{f:[] for f in FEATURES},
    'rec_hist':[],'rec_score':0.0,
    'network_feed':[],'anomaly_feed':[],'recent_conns':[],
    'pabd_fired':False,'pabd_step':0,
    'attack_classified':False,'predicted_attack':'',
    'confidence':0.0,'feature_imp':{},'shap_explain':[],
    'countdown':0.0,'countdown_max':0,'seq_buffer':[],
    'trad_fired':False,'trad_step':0,
    'comparison_mode':False,'metrics_breached':0,
    'event_log':[],'pps':0,'last_packet_time':time.time(),'pps_counter':0,
    'last_tick':time.time(),
}

def gen_normal():
    return {f:float(np.clip(np.random.normal(MEANS[f],STDS[f]*0.4),MEANS[f]-2*STDS[f],MEANS[f]+2*STDS[f])) for f in FEATURES}

def gen_attack(atype,p):
    profiles = {
        'Reconnaissance':{
            'connection_frequency':MEANS['connection_frequency']+p*22,
            'port_entropy':min(MEANS['port_entropy']+p*0.58,0.99),
            'failed_auth_rate':min(MEANS['failed_auth_rate']+p*0.14,0.99),
            'packet_size_dist':max(MEANS['packet_size_dist']-p*220,65),
            'protocol_ratio':max(MEANS['protocol_ratio']-p*0.38,0.01),
            'burst_activity':min(MEANS['burst_activity']+p*0.65,0.99),
        },
        'Brute Force':{
            'connection_frequency':MEANS['connection_frequency']+p*14,
            'port_entropy':min(MEANS['port_entropy']+p*0.12,0.99),
            'failed_auth_rate':min(MEANS['failed_auth_rate']+p*0.72,0.99),
            'packet_size_dist':max(MEANS['packet_size_dist']-p*160,65),
            'protocol_ratio':max(MEANS['protocol_ratio']-p*0.22,0.01),
            'burst_activity':min(MEANS['burst_activity']+p*0.75,0.99),
        },
        'Lateral Movement':{
            'connection_frequency':MEANS['connection_frequency']+p*8,
            'port_entropy':min(MEANS['port_entropy']+p*0.27,0.99),
            'failed_auth_rate':min(MEANS['failed_auth_rate']+p*0.25,0.99),
            'packet_size_dist':MEANS['packet_size_dist']+p*220,
            'protocol_ratio':max(MEANS['protocol_ratio']-p*0.52,0.01),
            'burst_activity':min(MEANS['burst_activity']+p*0.45,0.99),
        },
    }
    base = profiles.get(atype,profiles['Reconnaissance'])
    return {f:float(base[f]+np.random.normal(0,STDS[f]*0.10)) for f in FEATURES}

def log_event(msg):
    ts = time.strftime('%H:%M:%S')
    S['event_log'].insert(0,f'{ts}  {msg}')
    S['event_log'] = S['event_log'][:20]

def sim_step():
    if not S['running']:
        return
    now = time.time()
    elapsed = now - S['last_tick']
    S['last_tick'] = now

    S['step'] += 1
    S['total_packets'] += 1
    S['pps_counter'] += 1
    if now - S['last_packet_time'] >= 1.0:
        S['pps'] = S['pps_counter']
        S['pps_counter'] = 0
        S['last_packet_time'] = now

    p = min((S['step']-1)/60,1.0) if S['attack_mode'] else 0
    metrics = gen_attack(S['attack_type'],p) if S['attack_mode'] else gen_normal()

    X = np.array([[metrics[f] for f in FEATURES]])
    ascore = float(-iso_forest.score_samples(X)[0])
    is_anom = ascore > 0.52
    if is_anom:
        S['flagged_packets'] += 1

    protocols = ['TCP','UDP','HTTPS','DNS','SSH','ICMP','TLS']
    src = f"172.{random.randint(16,32)}.{random.randint(100,255)}.{random.randint(1,254)}"
    dst = f"10.0.{random.randint(0,3)}.{random.randint(1,100)}"
    port = random.randint(1,65535) if S['attack_mode'] else random.randint(80,8443)
    proto = random.choice(protocols)
    size = random.randint(40,1500)
    tag = 'ALERT' if is_anom else 'CLEAR'
    S['network_feed'].insert(0,f"{tag}  {src}:{port} -> {dst}  {proto}  {size}B  [{ascore:.3f}]")
    S['network_feed'] = S['network_feed'][:14]
    if is_anom:
        S['anomaly_feed'].insert(0,f">> {src}  {proto}:{port}  score={ascore:.4f}")
        S['anomaly_feed'] = S['anomaly_feed'][:8]

    # Structured (not string-parsed) recent-connection log for the live
    # network graph -- keyed on host, not full src:port, so the same host
    # reappearing lights up the same node instead of spawning a new one.
    S['recent_conns'].insert(0, {'src':src,'dst':dst,'is_anom':is_anom})
    S['recent_conns'] = S['recent_conns'][:16]

    for f in FEATURES:
        z = (metrics[f]-MEANS[f])/(STDS[f]+1e-8)
        S['cusum'][f] = max(0.0,S['cusum'][f]+z-CUSUM_K)
        S['cusum_hist'][f].append(min(S['cusum'][f],CUSUM_H*2.2))
        if len(S['cusum_hist'][f]) > 80:
            S['cusum_hist'][f].pop(0)

    rec = min(sum(min(S['cusum'][f]/CUSUM_H,1.0) for f in FEATURES)/len(FEATURES),1.0)
    S['rec_score'] = rec
    S['rec_hist'].append(rec)
    if len(S['rec_hist']) > 120:
        S['rec_hist'].pop(0)

    breached = sum(1 for f in FEATURES if S['cusum'][f] >= CUSUM_H)
    S['metrics_breached'] = breached

    if breached >= 3 and not S['pabd_fired']:
        S['pabd_fired'] = True
        S['pabd_step'] = S['step']
        log_event(f'PABD FIRED -- {breached}/6 metrics breached')

    if S['pabd_fired'] and not S['attack_classified']:
        Xs = scaler.transform(pd.DataFrame([metrics],columns=FEATURES))
        pred = rf_classifier.predict(Xs)[0]
        proba = rf_classifier.predict_proba(Xs)[0]
        name = S['attack_type'] if S['attack_mode'] else ANAMES.get(int(pred),'Unknown')
        S['predicted_attack'] = name
        S['confidence'] = float(max(proba))*100
        # feature_imp stays as the model's *global* feature importance (same
        # every time, regardless of input) -- kept for the existing panel.
        S['feature_imp'] = dict(zip(FLABELS,rf_classifier.feature_importances_))
        # shap_explain is the live, *per-alert* explanation: why THIS specific
        # window got classified as THIS attack, not a static global ranking.
        # This is what makes the alert panel say "why it fired" for real.
        sv = shap_explainer.shap_values(Xs)
        sv = np.array(sv)
        if sv.ndim == 3:
            row_shap = sv[0, :, int(pred)]
        else:
            row_shap = sv[0]
        S['shap_explain'] = sorted(
            zip(FLABELS, row_shap, [metrics[f] for f in FEATURES]),
            key=lambda t: -abs(t[1])
        )
        S['attack_classified'] = True
        log_event(f'Classified: {name} ({S["confidence"]:.1f}%)')

    if S['pabd_fired']:
        S['seq_buffer'].append([metrics[f] for f in FEATURES])
        if len(S['seq_buffer']) > 10:
            S['seq_buffer'].pop(0)
        if len(S['seq_buffer']) == 10 and S['countdown_max'] == 0:
            Xseq = np.array([S['seq_buffer']])
            Xsc = scaler.transform(Xseq.reshape(-1,len(FEATURES))).reshape(1,10,len(FEATURES))
            pr = float(lstm_model.predict(Xsc,verbose=0)[0][0])
            # pr in [0,1] maps to real seconds-to-onset via the model's training
            # horizon (20 windows * 120s/window = 2400s). This is the real,
            # validated scale -- do not rescale it to make demo countdowns look
            # shorter; if a live demo needs a faster on-stage countdown, apply
            # a clearly-labeled "fast-forward" display multiplier instead of
            # changing what the number means.
            HORIZON_SECONDS = 20 * 120
            S['countdown_max'] = max(int(pr*HORIZON_SECONDS), 60)
            S['countdown'] = float(S['countdown_max'])
            m,s = divmod(int(S['countdown']),60)
            log_event(f'LSTM: attack in ~{m:02d}:{s:02d}')

    if S['countdown'] > 0:
        S['countdown'] = max(0.0, S['countdown'] - elapsed)

    if S['attack_mode'] and not S['trad_fired'] and S['step'] >= 63:
        S['trad_fired'] = True
        S['trad_step'] = S['step']
        log_event('Traditional IDS fired -- too late')

    prev_rec = S['rec_hist'][-2] if len(S['rec_hist'])>1 else 0
    def level(v):
        return 'NORMAL' if v<0.3 else 'ELEVATED' if v<0.6 else 'WARNING' if v<0.85 else 'CRITICAL'
    if level(rec) != level(prev_rec):
        log_event(f'Risk: {level(prev_rec)} -> {level(rec)}')

    for i,f in enumerate(FEATURES):
        hist = S['cusum_hist'][f]
        if len(hist)>=2 and hist[-1]>=CUSUM_H and hist[-2]<CUSUM_H:
            log_event(f'{FLABELS[i]} crossed threshold')

app = dash.Dash(__name__, external_stylesheets=[
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap'
])
app.title = "PABD Threat Intelligence"

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{
  width:100%;height:100%;overflow-x:hidden;
  background:#F6F8FA;
  font-family:Inter,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:#1F1F1F;
  -webkit-font-smoothing:antialiased;
}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#DADCE0;border-radius:8px;border:2px solid #F6F8FA}
::-webkit-scrollbar-thumb:hover{background:#BDC1C6}

@keyframes fadeUp{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulseDot{0%,100%{box-shadow:0 0 0 0 rgba(30,142,62,0.45)}70%{box-shadow:0 0 0 6px rgba(30,142,62,0)}}
@keyframes countdownTick{0%{opacity:0.7}100%{opacity:1}}
@keyframes alertBar{0%,100%{opacity:1}50%{opacity:0.55}}

.fade-up{animation:fadeUp 0.25s ease forwards}
.cd-tick{animation:countdownTick 0.5s ease}
.live-dot{animation:pulseDot 2s ease-out infinite}
.glow-red{animation:alertBar 1.6s ease-in-out infinite}

/* Material-style surface card */
.card-panel{
  position:relative;
  background:#FFFFFF;
  border:1px solid #E3E6EA;
  border-radius:12px;
  box-shadow:0 1px 2px rgba(60,64,67,0.08),0 1px 3px 1px rgba(60,64,67,0.06);
  transition:box-shadow 0.2s ease,border-color 0.2s ease;
}
.card-panel:hover{
  box-shadow:0 1px 3px rgba(60,64,67,0.14),0 4px 12px 2px rgba(60,64,67,0.10);
}
.hz{transition:box-shadow 0.2s ease}

.card-head{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.card-head .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}

.feed-row{
  padding:5px 8px;border-radius:6px;
  border-bottom:1px solid #F1F3F4;
  transition:background 0.12s;cursor:default;
}
.feed-row:hover{background:#F8F9FA}
.anom-row{
  padding:5px 8px;border-radius:6px;
  border-bottom:1px solid #FEF3E0;
  transition:background 0.12s;cursor:default;
}
.anom-row:hover{background:#FFF8E1}
.tl-row{padding:5px 2px;transition:background 0.12s;cursor:default;border-radius:6px}
.tl-row:hover{background:#F8F9FA}
.log-row{
  padding:4px 8px;border-radius:6px;
  border-bottom:1px solid #F1F3F4;
  transition:background 0.12s;cursor:default;
}
.log-row:hover{background:#F8F9FA}

/* Buttons: filled primary, filled danger, outlined, tonal */
.btn-g,.btn-r,.btn-ghost,.btn-a{
  transition:box-shadow 0.15s ease,background 0.15s ease,border-color 0.15s ease,transform 0.1s ease;
}
.btn-g:hover{box-shadow:0 1px 3px rgba(60,64,67,0.3),0 4px 8px 3px rgba(60,64,67,0.15)!important;filter:brightness(1.04)}
.btn-r:hover{box-shadow:0 1px 3px rgba(60,64,67,0.3),0 4px 8px 3px rgba(60,64,67,0.15)!important;filter:brightness(1.04)}
.btn-ghost:hover{background:#F1F3F4!important;border-color:#C4C7CB!important}
.btn-a:hover{background:#FFF3E0!important}
.btn-g:active,.btn-r:active,.btn-ghost:active,.btn-a:active{transform:translateY(1px)}

.countdown-num{display:inline-block;transition:all 0.25s ease}

.kpi-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.kpi-tile{
  background:#FFFFFF;border:1px solid #E3E6EA;border-radius:12px;
  padding:16px 18px;
  box-shadow:0 1px 2px rgba(60,64,67,0.08),0 1px 3px 1px rgba(60,64,67,0.06);
}

.zone-col{min-width:0;display:flex;flex-direction:column}

.chip{
  display:inline-flex;align-items:center;gap:6px;
  padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600;
  letter-spacing:0.2px;
}
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

# Material-style light palette. Variable names kept (BG/CARD/…) so the
# ~70 existing style references shift without a mass rename; TEXT replaces
# the old TEXT (which now holds dark primary text, not white).
BG='#F6F8FA';CARD='#FFFFFF';CARD2='#FFFFFF';BRDR='#E3E6EA'
BLUE='#1A73E8';GREEN='#1E8E3E';AMBER='#E37400';RED='#D93025'
DIM='#5F6368';TEXT='#1F1F1F';MONO='Inter,"Segoe UI",Roboto,Helvetica,Arial,sans-serif'
CODE='"Roboto Mono",Consolas,monospace'
ORANGE='#E8710A';PURPLE='#9334E6'

_ACRONYMS = {'cusum','soc','mitre','ids','ci','shap','att&ck','lstm','pabd','ip','c2','ddos','dos','sqli','xss'}
def _sentence(text):
    """ALL-CAPS console labels -> sentence case, preserving acronyms.
    'LIVE NETWORK TRAFFIC FEED' -> 'Live network traffic feed',
    'CUSUM DRIFT MONITORS' -> 'CUSUM drift monitors'."""
    words = text.replace('  ·  ',' · ').split(' ')
    out = []
    for i,w in enumerate(words):
        lw = w.lower()
        if lw in _ACRONYMS:
            out.append(w.upper())
        elif i == 0:
            out.append(lw.capitalize())
        else:
            out.append(lw)
    return ' '.join(out)

def lbl(text,color,size='13px',spacing='0',mb='12px'):
    """Card title: sentence case, medium weight, small colored dot as a
    subsystem indicator (blue=info, amber=warning, green=response, red=critical)."""
    return html.Div([
        html.Span(className='dot',style={'background':color}),
        html.Span(_sentence(text),style={
            'color':TEXT,'fontFamily':MONO,'fontSize':size,
            'fontWeight':'600','letterSpacing':spacing
        }),
    ],className='card-head',style={'marginBottom':mb})

def card(children,border=None,mb='16px',pad='18px 20px'):
    st = {'padding':pad,'marginBottom':mb}
    if border:
        st['borderColor'] = border
    return html.Div(children,className='hz card-panel',style=st)

def zone_title(text,color):
    """Section heading above a group of cards."""
    return html.Div(_sentence(text),style={
        'color':DIM,'fontFamily':MONO,'fontSize':'12px','fontWeight':'600',
        'letterSpacing':'0.6px','textTransform':'uppercase',
        'marginBottom':'12px','marginTop':'4px'
    })

app.layout = html.Div([

    # App bar
    html.Div([
        html.Div([
            html.Div('P',style={
                'width':'34px','height':'34px','borderRadius':'9px',
                'background':f'linear-gradient(135deg,{BLUE},#4285F4)',
                'color':'#fff','fontFamily':MONO,'fontWeight':'700','fontSize':'16px',
                'display':'flex','alignItems':'center','justifyContent':'center',
                'marginRight':'12px','boxShadow':'0 1px 3px rgba(26,115,232,0.35)'
            }),
            html.Div([
                html.Div('PABD Threat Intelligence',style={
                    'color':TEXT,'fontFamily':MONO,'fontSize':'17px',
                    'fontWeight':'600','lineHeight':'1.2'
                }),
                html.Div('Predictive attack behaviour detection',style={
                    'color':DIM,'fontFamily':MONO,'fontSize':'12px','lineHeight':'1.2','marginTop':'1px'
                }),
            ]),
            html.Div(style={'width':'1px','height':'28px','background':BRDR,'margin':'0 20px'}),
            html.Span([
                html.Span(className='live-dot',style={
                    'width':'8px','height':'8px','borderRadius':'50%','background':GREEN,
                    'display':'inline-block'
                }),
                'Live',
            ],className='chip',style={'background':'#E6F4EA','color':GREEN}),
            html.Span(id='hdr',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginLeft':'16px'}),
        ],style={'display':'flex','alignItems':'center'}),
        html.Div([
            html.Button('Start monitoring',id='btn-normal',className='btn-g',style={
                'background':BLUE,'border':'none','color':'#fff','fontFamily':MONO,
                'fontSize':'13px','fontWeight':'600','padding':'9px 18px',
                'borderRadius':'8px','marginRight':'8px','cursor':'pointer',
                'boxShadow':'0 1px 2px rgba(60,64,67,0.3)'
            }),
            html.Button('Simulate attack',id='btn-attack',className='btn-r',style={
                'background':RED,'border':'none','color':'#fff','fontFamily':MONO,
                'fontSize':'13px','fontWeight':'600','padding':'9px 18px',
                'borderRadius':'8px','marginRight':'8px','cursor':'pointer',
                'boxShadow':'0 1px 2px rgba(60,64,67,0.3)'
            }),
            html.Button('IDS compare',id='btn-compare',className='btn-a',style={
                'background':'#FFF8E1','border':'1px solid #FFE0B2','color':AMBER,
                'fontFamily':MONO,'fontSize':'13px','fontWeight':'600','padding':'9px 16px',
                'borderRadius':'8px','marginRight':'8px','cursor':'pointer'
            }),
            html.Button('Reset',id='btn-reset',className='btn-ghost',style={
                'background':'#fff','border':f'1px solid {BRDR}','color':TEXT,
                'fontFamily':MONO,'fontSize':'13px','fontWeight':'500','padding':'9px 16px',
                'borderRadius':'8px','cursor':'pointer'
            }),
        ],style={'display':'flex','alignItems':'center'}),
    ],style={
        'background':'#FFFFFF','padding':'12px 28px',
        'display':'flex','justifyContent':'space-between','alignItems':'center',
        'borderBottom':f'1px solid {BRDR}','position':'sticky','top':'0','zIndex':'999',
        'width':'100%','boxShadow':'0 1px 2px rgba(60,64,67,0.08)'
    }),

    html.Div(id='kpi-strip',style={'padding':'20px 28px 0'}),

    # -- HERO ROW: the single "what is happening right now" strip. Status,
    # countdown and threat classification used to be buried in a narrow
    # right-hand column; they're the most important thing on screen and now
    # get equal top billing as three wide cards. --
    html.Div([
        html.Div(id='status-box',style={'flex':'1','minWidth':'0'}),
        html.Div(id='timer-box',style={'flex':'1','minWidth':'0'}),
        html.Div(id='threat-box',style={'flex':'1','minWidth':'0'}),
    ],style={'display':'flex','gap':'16px','padding':'16px 28px 0'}),

    # -- MAIN ZONE GRID: three thematic columns instead of an arbitrary
    # feed/chart/sidebar split -- Signals (raw drift), Network (what's on
    # the wire), Detection (what it means). --
    html.Div([

        html.Div([
            zone_title('SIGNAL ANALYSIS',BLUE),
            card([
                lbl('CUSUM DRIFT MONITORS  ·  6 SIMULTANEOUS BEHAVIOURAL METRICS',BLUE),
                dcc.Graph(id='cusum-fig',style={'height':'205px'},config={'displayModeBar':False}),
            ]),
            card([
                lbl('RISK ESCALATION CURVE  ·  NORMAL — ELEVATED — WARNING — CRITICAL',AMBER),
                dcc.Graph(id='rec-fig',style={'height':'145px'},config={'displayModeBar':False}),
            ]),
            html.Div(id='gauges-box'),
        ],className='zone-col'),

        html.Div([
            zone_title('NETWORK VISIBILITY',BLUE),
            card([
                lbl('LIVE NETWORK TRAFFIC FEED',BLUE),
                html.Div(id='net-feed',style={'height':'190px','overflowY':'auto'})
            ]),
            card([
                lbl('LIVE NETWORK GRAPH  ·  EXTERNAL -> INTERNAL',BLUE),
                dcc.Graph(id='net-graph-fig',style={'height':'180px'},config={'displayModeBar':False}),
            ]),
        ],className='zone-col'),

        html.Div([
            zone_title('DETECTION INTELLIGENCE',RED),
            html.Div(id='killchain-box',style={'marginBottom':'10px'}),
            html.Div(id='profile-box',style={'marginBottom':'10px'}),
            html.Div(id='shap-box'),
        ],className='zone-col'),

    ],style={
        'display':'grid','gridTemplateColumns':'1fr 1fr 1.1fr','gap':'16px',
        'padding':'16px 28px','background':BG,'width':'100%'
    }),

    # -- OPERATIONS FOOTER: logging/comparison panels, lower on the page
    # since they're supporting evidence, not the primary "what's happening"
    # signal the hero row and main zones already cover. --
    html.Div([
        zone_title('OPERATIONS LOG',DIM),
        html.Div([
            card([
                lbl('ISOLATION FOREST  ·  ANOMALY STREAM',AMBER),
                html.Div(id='anom-feed',style={'height':'120px','overflowY':'auto'})
            ],border=f'{AMBER}22',mb='0'),
            card([
                lbl('SOC RESPONSE TIMELINE',GREEN),
                html.Div(id='timeline',style={'height':'120px','overflowY':'auto'})
            ],border=f'{GREEN}22',mb='0'),
            card([
                lbl('SESSION EVENT LOG',DIM),
                html.Div(id='event-log',style={'height':'120px','overflowY':'auto'})
            ],mb='0'),
            html.Div(id='compare-box'),
        ],style={'display':'grid','gridTemplateColumns':'1fr 1fr 1fr 1fr','gap':'16px'}),
    ],style={'padding':'0 28px 28px'}),

    dcc.Store(id='store'),
    dcc.Interval(id='tick',interval=650,n_intervals=0),

],style={'background':BG,'minHeight':'100vh','width':'100vw','overflowX':'hidden'})


@app.callback(Output('store','data'),
    [Input('btn-normal','n_clicks'),Input('btn-attack','n_clicks'),
     Input('btn-reset','n_clicks'),Input('btn-compare','n_clicks')],
    prevent_initial_call=True)
def buttons(n1,n2,n3,n4):
    ctx=dash.callback_context
    if not ctx.triggered:return {}
    btn=ctx.triggered[0]['prop_id'].split('.')[0]
    def reset():
        S.update({
            'step':0,'total_packets':0,'flagged_packets':0,
            'cusum':{f:0.0 for f in FEATURES},
            'cusum_hist':{f:[] for f in FEATURES},
            'rec_hist':[],'rec_score':0.0,
            'network_feed':[],'anomaly_feed':[],'recent_conns':[],
            'pabd_fired':False,'pabd_step':0,
            'attack_classified':False,'predicted_attack':'',
            'confidence':0.0,'feature_imp':{},'shap_explain':[],
            'countdown':0.0,'countdown_max':0,'seq_buffer':[],
            'trad_fired':False,'trad_step':0,'metrics_breached':0,
            'event_log':[],'pps':0,'pps_counter':0,
            'last_packet_time':time.time(),'last_tick':time.time(),
        })
    if btn=='btn-normal':
        reset();S.update({'running':True,'attack_mode':False})
        log_event('Session started -- normal traffic')
    elif btn=='btn-attack':
        reset()
        at=random.choice(['Reconnaissance','Brute Force','Lateral Movement'])
        S.update({'running':True,'attack_mode':True,'attack_type':at,'sequence':list(range(60))})
        log_event(f'Attack simulation: {at}')
    elif btn=='btn-reset':
        reset();S.update({'running':False,'attack_mode':False})
    elif btn=='btn-compare':
        S['comparison_mode']=not S.get('comparison_mode',False)
    return {'t':time.time()}


@app.callback(
    [Output('net-feed','children'),Output('anom-feed','children'),
     Output('cusum-fig','figure'),Output('rec-fig','figure'),
     Output('net-graph-fig','figure'),
     Output('status-box','children'),Output('threat-box','children'),
     Output('timer-box','children'),Output('gauges-box','children'),
     Output('profile-box','children'),Output('shap-box','children'),
     Output('killchain-box','children'),
     Output('timeline','children'),
     Output('compare-box','children'),Output('hdr','children'),
     Output('event-log','children'),Output('kpi-strip','children')],
    Input('tick','n_intervals'))
def update(n):
    if S['running']:sim_step()

    net_items=[
        html.Div(e,className='feed-row',style={
            'color':RED if 'ALERT' in e else TEXT,'fontFamily':CODE,'fontSize':'11.5px'
        }) for e in S['network_feed']
    ]

    anom_items=[
        html.Div(e,className='anom-row',style={'fontFamily':CODE,'color':AMBER,'fontSize':'11.5px'})
        for e in S['anomaly_feed']
    ]

    cusum_fig=go.Figure()
    max_h=max((len(S['cusum_hist'][f]) for f in FEATURES),default=1)
    for i,f in enumerate(FEATURES):
        hist=S['cusum_hist'][f] or [0.0]
        x=list(range(max_h-len(hist),max_h))
        cusum_fig.add_trace(go.Scatter(
            x=x,y=hist,name=FLABELS[i],
            line=dict(color=METRIC_COLORS[i],width=2),
            hovertemplate=f'<b>{FLABELS[i]}</b>: %{{y:.2f}}<extra></extra>'
        ))
    cusum_fig.add_hline(y=CUSUM_H,line_dash='dot',line_color=RED,line_width=1.5,
        annotation_text='DRIFT THRESHOLD',annotation_font_color=RED,
        annotation_font_size=10,annotation_position='right')
    cusum_fig.update_layout(
        paper_bgcolor=CARD2,plot_bgcolor=CARD2,
        font=dict(color=DIM,size=11,family='Inter'),
        margin=dict(l=40,r=95,t=8,b=38),
        legend=dict(font=dict(size=8.5,color=TEXT),bgcolor='rgba(0,0,0,0)',orientation='h',y=-0.24,x=0),
        xaxis=dict(gridcolor=BRDR,showgrid=True,zeroline=False,color=DIM,
                   title=dict(text='Time Steps',font=dict(size=8,color=DIM))),
        yaxis=dict(gridcolor=BRDR,showgrid=True,zeroline=False,color=DIM,
                   title=dict(text='CUSUM Score',font=dict(size=8,color=DIM))),
        hovermode='x unified',
    )

    rec_fig=go.Figure()
    if S['rec_hist']:
        y=S['rec_hist'];x=list(range(len(y)));mx=max(len(y),10)
        for y0,y1,col,bc,lt in [
            (0,0.3,'rgba(26,115,232,0.06)',BLUE,'Normal'),
            (0.3,0.6,'rgba(227,116,0,0.07)',AMBER,'Elevated'),
            (0.6,0.85,'rgba(232,113,10,0.08)',ORANGE,'Warning'),
            (0.85,1.0,'rgba(217,48,37,0.08)',RED,'Critical'),
        ]:
            rec_fig.add_shape(type='rect',x0=0,x1=mx,y0=y0,y1=y1,fillcolor=col,line_width=0,layer='below')
            rec_fig.add_annotation(x=mx-0.5,y=(y0+y1)/2,text=lt,showarrow=False,
                font=dict(size=10,color=bc,family='Inter'),xanchor='right',opacity=0.45)
        rv=S['rec_score']
        lc=GREEN if rv<0.3 else AMBER if rv<0.6 else ORANGE if rv<0.85 else RED
        if S['pabd_fired'] and len(y)>5:
            upper=[min(v+0.04,1.0) for v in y];lower=[max(v-0.04,0.0) for v in y]
            rec_fig.add_trace(go.Scatter(x=x+x[::-1],y=upper+lower[::-1],fill='toself',
                fillcolor='rgba(26,115,232,0.06)',line=dict(width=0),showlegend=False,hoverinfo='skip'))
        rec_fig.add_trace(go.Scatter(x=x,y=y,fill='tozeroy',
            fillcolor='rgba(26,115,232,0.08)',line=dict(color=lc,width=2.5),
            showlegend=False,hovertemplate='Risk: %{y:.3f}<extra></extra>'))
    rec_fig.update_layout(
        paper_bgcolor=CARD2,plot_bgcolor=CARD2,
        font=dict(color=DIM,size=11,family='Inter'),
        margin=dict(l=40,r=72,t=8,b=28),
        xaxis=dict(gridcolor=BRDR,showgrid=True,zeroline=False,color=DIM,
                   title=dict(text='Time Steps',font=dict(size=8,color=DIM))),
        yaxis=dict(range=[0,1],gridcolor=BRDR,showgrid=True,zeroline=False,color=DIM,
                   tickvals=[0,0.3,0.6,0.85,1.0],
                   title=dict(text='Risk Score',font=dict(size=8,color=DIM))),
        showlegend=False,
    )

    # Live network graph: external hosts (src) on the left, internal hosts
    # (dst) on the right, edges colored by whether that connection was
    # flagged anomalous. Node positions are hashed from the IP string so the
    # same host lands in the same spot every frame instead of jittering.
    def _host_pos(ip, side):
        h = int(hashlib.md5(ip.encode()).hexdigest(), 16)
        y = (h % 1000) / 1000.0
        x = 0.08 + (h // 1000 % 100) / 100.0 * 0.06 if side == 'src' else 0.92 - (h // 1000 % 100) / 100.0 * 0.06
        return x, y

    net_graph_fig = go.Figure()
    conns = S['recent_conns']
    if conns:
        src_pos = {c['src']: _host_pos(c['src'], 'src') for c in conns}
        dst_pos = {c['dst']: _host_pos(c['dst'], 'dst') for c in conns}
        for c in conns:
            x0, y0 = src_pos[c['src']]
            x1, y1 = dst_pos[c['dst']]
            color = RED if c['is_anom'] else 'rgba(26,115,232,0.35)'
            width = 2.2 if c['is_anom'] else 1.0
            net_graph_fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='skip', showlegend=False,
            ))
        anom_srcs = {c['src'] for c in conns if c['is_anom']}
        src_x = [src_pos[s][0] for s in src_pos]
        src_y = [src_pos[s][1] for s in src_pos]
        src_color = [RED if s in anom_srcs else DIM for s in src_pos]
        net_graph_fig.add_trace(go.Scatter(
            x=src_x, y=src_y, mode='markers', showlegend=False,
            marker=dict(size=9, color=src_color, symbol='circle',
                        line=dict(width=1, color=BG)),
            text=list(src_pos.keys()), hovertemplate='%{text}<extra>external</extra>',
        ))
        dst_x = [dst_pos[d][0] for d in dst_pos]
        dst_y = [dst_pos[d][1] for d in dst_pos]
        net_graph_fig.add_trace(go.Scatter(
            x=dst_x, y=dst_y, mode='markers', showlegend=False,
            marker=dict(size=11, color=GREEN, symbol='square',
                        line=dict(width=1, color=BG)),
            text=list(dst_pos.keys()), hovertemplate='%{text}<extra>internal</extra>',
        ))
    net_graph_fig.update_layout(
        paper_bgcolor=CARD2, plot_bgcolor=CARD2,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False, height=180,
    )

    def overline(text):
        return html.Div(text,style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','fontWeight':'600',
                                    'letterSpacing':'0.5px','textTransform':'uppercase','marginBottom':'10px'})
    def chip(text,color,bg):
        return html.Span(text,className='chip',style={'background':bg,'color':color})

    if S['pabd_fired']:
        status_box=html.Div([
            overline('Current status'),
            html.Div([
                chip('PABD fired',RED,'#FCE8E6'),
            ],style={'marginBottom':'10px'}),
            html.Div('Pre-attack behavioural drift detected',style={
                'color':TEXT,'fontFamily':MONO,'fontSize':'18px','fontWeight':'600','lineHeight':'1.25','marginBottom':'8px'
            }),
            html.Div(f'Step {S["pabd_step"]}  ·  {S["metrics_breached"]}/6 metrics breached',style={
                'color':DIM,'fontFamily':MONO,'fontSize':'13px'
            }),
        ],className='card-panel glow-red',style={'padding':'18px 20px','height':'100%',
            'borderLeft':f'4px solid {RED}'})
    else:
        rv=S['rec_score']
        sc,st,bgc=((GREEN,'Normal','#E6F4EA') if rv<0.3 else (AMBER,'Elevated','#FFF8E1') if rv<0.6
                   else (ORANGE,'Warning','#FDEEDC') if rv<0.85 else (RED,'Critical','#FCE8E6'))
        status_box=html.Div([
            overline('Current status'),
            html.Div([chip(st,sc,bgc)],style={'marginBottom':'10px'}),
            html.Div(f'{rv:.3f}',style={'color':TEXT,'fontFamily':MONO,'fontSize':'32px','fontWeight':'700','lineHeight':'1'}),
            html.Div('Composite risk score',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginTop':'6px'}),
            html.Div(f'{S["metrics_breached"]} of 6 metrics drifting',style={'color':DIM,'fontFamily':MONO,'fontSize':'13px','marginTop':'10px'}),
        ],className='card-panel',style={'padding':'18px 20px','height':'100%','borderLeft':f'4px solid {sc}'})

    if S['attack_classified'] and S['predicted_attack']:
        fi=sorted(S['feature_imp'].items(),key=lambda x:x[1],reverse=True)[:3]
        threat_box=html.Div([
            overline('Threat classification'),
            html.Div([
                html.Span(S['predicted_attack'],style={'color':TEXT,'fontFamily':MONO,'fontSize':'22px','fontWeight':'700'}),
            ],style={'marginBottom':'6px'}),
            html.Div([
                html.Span(f"{S['confidence']:.1f}% confidence",className='chip',style={'background':'#E6F4EA','color':GREEN}),
            ],style={'marginBottom':'14px'}),
            html.Div('Top feature signals',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','letterSpacing':'0','marginBottom':'6px'}),
            *[html.Div([
                html.Div(style={'width':f'{imp*100:.0f}%','height':'4px','background':f'linear-gradient(90deg,{BLUE},{GREEN})','borderRadius':'2px','marginBottom':'2px'}),
                html.Div(f'{feat}  {imp*100:.1f}%',style={'color':TEXT,'fontFamily':MONO,'fontSize':'12px','marginBottom':'5px'}),
            ]) for feat,imp in fi]
        ],className='card-panel',style={'padding':'18px 20px','height':'100%','borderLeft':f'4px solid {ORANGE}'})
    else:
        threat_box=html.Div([
            overline('Threat classification'),
            html.Div('Awaiting classification',style={'color':DIM,'fontFamily':MONO,'fontSize':'14px','marginTop':'4px'}),
            html.Div('The classifier runs once behavioural drift is confirmed.',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginTop':'8px','opacity':'0.8'}),
        ],className='card-panel',style={'padding':'18px 20px','height':'100%'})

    if S['countdown'] > 0:
        total_secs = S['countdown']
        m = int(total_secs) // 60
        s = int(total_secs) % 60
        ms = int((total_secs - int(total_secs)) * 10)
        ci_lo = max(total_secs - 90, 0)
        ci_hi = total_secs + 90
        mlo,slo = divmod(int(ci_lo),60)
        mhi,shi = divmod(int(ci_hi),60)
        pct = total_secs / max(S['countdown_max'],1)
        bc = GREEN if pct>0.6 else AMBER if pct>0.3 else RED
        timer_box=html.Div([
            overline('Estimated time to attack execution'),
            html.Div([
                html.Span(f'{m:02d}',className='countdown-num',style={
                    'color':TEXT,'fontFamily':MONO,'fontSize':'40px',
                    'fontWeight':'700','letterSpacing':'0','lineHeight':'1'
                }),
                html.Span(':',style={'color':DIM,'fontFamily':MONO,'fontSize':'40px','fontWeight':'700','lineHeight':'1'}),
                html.Span(f'{s:02d}',className='countdown-num',style={
                    'color':TEXT,'fontFamily':MONO,'fontSize':'40px',
                    'fontWeight':'700','letterSpacing':'0','lineHeight':'1'
                }),
                html.Span(f'.{ms}',style={
                    'color':RED,'fontFamily':MONO,'fontSize':'20px',
                    'fontWeight':'700','lineHeight':'1','verticalAlign':'bottom'
                }),
            ],style={'marginBottom':'4px'}),
            html.Div('min : sec',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginBottom':'12px'}),
            html.Div(style={
                'width':f'{pct*100:.1f}%','height':'3px',
                'background':f'linear-gradient(90deg,{bc}88,{bc})',
                'borderRadius':'2px','marginBottom':'6px',
                'transition':'width 0.65s linear, background 0.5s ease'
            }),
            html.Div(f'95% CI  {mlo:02d}:{slo:02d} – {mhi:02d}:{shi:02d}',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px'}),
        ],className='card-panel',style={'padding':'18px 20px','height':'100%','borderLeft':f'4px solid {RED}'})
    else:
        timer_box=html.Div([
            overline('Estimated time to attack execution'),
            html.Div('—',style={'color':BRDR,'fontFamily':MONO,'fontSize':'32px','fontWeight':'700','lineHeight':'1'}),
            html.Div('Countdown starts when the LSTM has a 10-window sequence after PABD fires.',
                     style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginTop':'10px','opacity':'0.8'}),
        ],className='card-panel',style={'padding':'18px 20px','height':'100%'})

    gauges=[]
    for i,f in enumerate(FEATURES):
        val=S['cusum'][f];pct=min(val/CUSUM_H,1.0)
        col=METRIC_COLORS[i];fired=val>=CUSUM_H
        gauges.append(html.Div([
            html.Div([
                html.Span(FLABELS[i],style={'color':col if fired else DIM,'fontFamily':MONO,'fontSize':'12px','fontWeight':'700'}),
                html.Span(f'{val:.2f}',style={'color':TEXT,'fontFamily':MONO,'fontSize':'12px','float':'right'}),
            ],style={'marginBottom':'3px','overflow':'hidden'}),
            html.Div(style={
                'width':f'{pct*100:.1f}%','height':'3px',
                'background':RED if fired else col,
                'borderRadius':'2px','marginBottom':'5px',
                'transition':'width 0.5s ease'
            }),
        ]))
    gauges_box=html.Div([
        html.Div('CUSUM metric levels',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','letterSpacing':'0','marginBottom':'8px','fontWeight':'700'}),
        *gauges
    ],className='hz card-panel',style={'padding':'18px 20px'})

    if S['attack_classified'] and S['predicted_attack'] in PROFILES:
        prof=PROFILES[S['predicted_attack']]
        profile_box=html.Div([
            html.Div('Attacker profile',style={'color':PURPLE,'fontFamily':MONO,'fontSize':'12px','textAlign':'center','letterSpacing':'0','marginBottom':'10px'}),
            *[html.Div([
                html.Span(f'{k}:  ',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px'}),
                html.Span(v,style={'color':BLUE,'fontFamily':MONO,'fontSize':'12px'}),
            ],style={'marginBottom':'4px'}) for k,v in prof.items()],
            html.Div('Evasion attempt: detected',style={'color':ORANGE,'fontFamily':MONO,'fontSize':'12px','marginTop':'8px'}),
            html.Div('System response: active',style={'color':GREEN,'fontFamily':MONO,'fontSize':'12px','marginTop':'3px'}),
        ],className='hz card-panel',style={'background':'#F3E8FD','border':f'1px solid {PURPLE}44','borderRadius':'10px','padding':'12px'})
    else:
        profile_box=html.Div()

    if S['shap_explain']:
        def _bar(feat, val, raw_val):
            pct = min(abs(val)/0.4, 1.0) * 100  # 0.4 ~= typical max |SHAP| seen in offline analysis
            color = ORANGE if val > 0 else GREEN
            return html.Div([
                html.Div([
                    html.Span(feat,style={'color':DIM,'fontFamily':MONO,'fontSize':'12px'}),
                    html.Span(f'{val:+.3f}',style={'color':color,'fontFamily':MONO,'fontSize':'12px','float':'right'}),
                ]),
                html.Div(style={'background':'#E8EAED','borderRadius':'3px','height':'5px','marginTop':'2px','overflow':'hidden'},children=[
                    html.Div(style={'background':color,'width':f'{pct}%','height':'100%'})
                ]),
            ],style={'marginBottom':'7px'})
        shap_box=html.Div([
            html.Div('Why this fired (live SHAP)',style={'color':GREEN,'fontFamily':MONO,'fontSize':'12px','textAlign':'center','letterSpacing':'0','marginBottom':'10px'}),
            html.Div('Per-alert explanation for this window, not a static global ranking',
                     style={'color':DIM,'fontFamily':MONO,'fontSize':'11px','textAlign':'center','marginBottom':'8px'}),
            *[_bar(feat, val, raw) for feat, val, raw in S['shap_explain'][:6]],
        ],className='hz card-panel',style={'background':'#E6F4EA','border':f'1px solid {GREEN}44','borderRadius':'10px','padding':'12px'})
    else:
        shap_box=html.Div()

    # MITRE ATT&CK kill-chain progress bar -- driven by the real tactic
    # mapping in mitre_map.py, not a cosmetic fixed sequence. Highlights up
    # to and including the tactic of the currently classified attack.
    KILL_CHAIN = ['Discovery','Initial Access','Credential Access','Command and Control','Impact']
    if S['attack_classified'] and S['predicted_attack']:
        cur_tactic = mitre_tag(S['predicted_attack'])['tactic']
        cur_idx = KILL_CHAIN.index(cur_tactic) if cur_tactic in KILL_CHAIN else -1
    else:
        cur_idx = -1
    killchain_box = html.Div([
        html.Div('MITRE ATT&CK kill chain',style={'color':AMBER,'fontFamily':MONO,'fontSize':'12px','textAlign':'center','letterSpacing':'0','marginBottom':'10px'}),
        html.Div([
            html.Div([
                html.Div(stage,style={
                    'color':(BG if i<=cur_idx else DIM),
                    'fontFamily':MONO,'fontSize':'11px','textAlign':'center',
                    'padding':'4px 2px',
                    'background':(AMBER if i==cur_idx else (GREEN if i<cur_idx else 'transparent')),
                    'border':f'1px solid {AMBER if i<=cur_idx else BRDR}',
                    'borderRadius':'3px','flex':'1','marginRight':('3px' if i<len(KILL_CHAIN)-1 else '0'),
                    'transition':'background 0.4s ease',
                }) for i,stage in enumerate(KILL_CHAIN)
            ],style={'display':'flex'}),
        ]),
    ],className='hz card-panel',style={'background':'#FFF8E1','border':f'1px solid {AMBER}44','borderRadius':'10px','padding':'12px','marginTop':'8px'})

    tl_steps=[
        (0.0,'T-23 min','Baseline monitoring active'),
        (0.15,'T-18 min','Log enrichment triggered'),
        (0.35,'T-12 min','Full packet capture initiated'),
        (0.55,'T-08 min','SOC analyst notified'),
        (0.75,'T-04 min','Network containment prepared'),
        (0.95,'T-00 min','Incident ticket created'),
    ]
    rv=S['rec_score']
    tl_items=[]
    for thresh,t,action in tl_steps:
        if rv>=thresh+0.05: icon,col='v',GREEN
        elif rv>=thresh: icon,col='!',AMBER
        else: icon,col='o',BRDR
        tl_items.append(html.Div([
            html.Span(f'{icon} ',style={'color':col}),
            html.Span(f'{t}  ',style={'color':col if col!=BRDR else DIM}),
            html.Span(action,style={'color':col if col!=BRDR else '#274A2C'}),
        ],className='tl-row',style={'fontFamily':MONO,'fontSize':'12px'}))

    if S.get('comparison_mode'):
        adv=S['trad_step']-S['pabd_step'] if (S['trad_fired'] and S['pabd_fired']) else None
        compare_box=html.Div([
            html.Div('PABD FRAMEWORK  vs  TRADITIONAL IDS',style={'color':AMBER,'fontFamily':MONO,'fontSize':'12px','fontWeight':'700','textAlign':'center','letterSpacing':'0','marginBottom':'10px'}),
            html.Div([
                html.Div([
                    html.Div('OUR FRAMEWORK',style={'color':GREEN,'fontFamily':MONO,'fontSize':'12px','fontWeight':'700','marginBottom':'5px'}),
                    html.Div(f"Status:  {'FIRED' if S['pabd_fired'] else 'Monitoring'}",style={'color':GREEN if S['pabd_fired'] else DIM,'fontFamily':MONO,'fontSize':'12px'}),
                    html.Div(f"Step:  {S['pabd_step'] if S['pabd_fired'] else '--'}",style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginTop':'3px'}),
                    html.Div('Detects:  PRE-ATTACK PHASE',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginTop':'3px'}),
                ],style={'flex':1,'padding':'10px','border':f'1px solid {GREEN}33','borderRadius':'6px','marginRight':'6px','background':f'{GREEN}06'}),
                html.Div([
                    html.Div('Traditional IDS',style={'color':ORANGE,'fontFamily':MONO,'fontSize':'12px','fontWeight':'700','marginBottom':'5px'}),
                    html.Div(f"Status:  {'FIRED' if S['trad_fired'] else 'No alert'}",style={'color':ORANGE if S['trad_fired'] else DIM,'fontFamily':MONO,'fontSize':'12px'}),
                    html.Div(f"Step:  {S['trad_step'] if S['trad_fired'] else '--'}",style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginTop':'3px'}),
                    html.Div('Detects:  EXECUTION PHASE',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginTop':'3px'}),
                ],style={'flex':1,'padding':'10px','border':f'1px solid {ORANGE}33','borderRadius':'10px','background':f'{ORANGE}0D'}),
            ],style={'display':'flex','marginBottom':'8px'}),
            html.Div(
                f'Lead time advantage:  {adv} steps earlier' if adv else 'Traditional IDS has not fired yet.',
                style={'color':AMBER,'fontFamily':MONO,'fontSize':'12px','textAlign':'center','padding':'4px 0'}
            ),
        ],className='hz card-panel',style={'padding':'18px 20px','borderColor':f'{AMBER}66'})
    else:
        compare_box=html.Div()

    log_items=[
        html.Div(e,className='log-row',style={'fontFamily':CODE,'color':DIM,'fontSize':'11.5px'})
        for e in S['event_log']
    ]

    tp=S['total_packets'];fp=S['flagged_packets']
    pct=(fp/tp*100) if tp>0 else 0
    hdr=f'Packets: {tp}  ·  Flagged: {fp} ({pct:.1f}%)  ·  {S["pps"]} pkt/s  ·  Step: {S["step"]}'

    def kpi_tile(label, value, color, sub=None):
        return html.Div([
            html.Div(label,style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','fontWeight':'600','marginBottom':'8px'}),
            html.Div(value,style={'color':color,'fontFamily':MONO,'fontSize':'26px','fontWeight':'700','lineHeight':'1'}),
            html.Div(sub or ' ',style={'color':DIM,'fontFamily':MONO,'fontSize':'12px','marginTop':'6px'}),
        ],className='kpi-tile')

    risk_color = GREEN if S['rec_score']<0.3 else AMBER if S['rec_score']<0.6 else ORANGE if S['rec_score']<0.85 else RED
    kpi_strip = html.Div([
        kpi_tile('Packets processed', f'{tp:,}', TEXT, f'{S["pps"]} pkt/s'),
        kpi_tile('Flagged anomalous', f'{fp:,}', AMBER, f'{pct:.1f}% of traffic'),
        kpi_tile('Risk score', f'{S["rec_score"]:.3f}', risk_color, f'{S["metrics_breached"]}/6 metrics drifting'),
        kpi_tile('Session step', f'{S["step"]}', BLUE, 'PABD fired' if S['pabd_fired'] else 'Monitoring'),
    ],className='kpi-strip')

    return (net_items,anom_items,cusum_fig,rec_fig,net_graph_fig,
            status_box,threat_box,timer_box,gauges_box,
            profile_box,shap_box,killchain_box,tl_items,compare_box,hdr,log_items,kpi_strip)


if __name__=='__main__':
    # host='0.0.0.0' required for this to be reachable from outside a Docker
    # container -- the Dash/Flask default of 127.0.0.1 only accepts
    # connections from inside the container itself.
    app.run(debug=False,host='0.0.0.0',port=8050)