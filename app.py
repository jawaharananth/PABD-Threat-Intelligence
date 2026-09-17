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
METRIC_COLORS = ['#00FFC2','#39FF6A','#FF7A1A','#FFD600','#FF4081','#B388FF']

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
    'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap'
])
app.title = "PABD Threat Intelligence Center"

app.index_string = '''<!DOCTYPE html>
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
  background:
    radial-gradient(ellipse 1200px 800px at 15% -10%, #0A2010AA 0%, transparent 60%),
    radial-gradient(ellipse 1000px 700px at 100% 10%, #08170CAA 0%, transparent 55%),
    repeating-linear-gradient(0deg, transparent 0px, transparent 23px, #39FF6A08 23px, #39FF6A08 24px),
    repeating-linear-gradient(90deg, transparent 0px, transparent 23px, #39FF6A05 23px, #39FF6A05 24px),
    #050B06;
  font-family:"IBM Plex Mono",monospace;
}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#081208}
::-webkit-scrollbar-thumb{background:#2A5A34;border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:#39FF6A66}

@keyframes glowRed{0%,100%{box-shadow:0 0 8px #FF3B3B40}50%{box-shadow:0 0 22px #FF3B3BCC}}
@keyframes breathe{0%,100%{opacity:1}50%{opacity:0.3}}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@keyframes countdownTick{0%{opacity:0.6;transform:scale(0.96)}100%{opacity:1;transform:scale(1)}}
@keyframes scanline{0%{transform:translateY(-100%)}100%{transform:translateY(100vh)}}
@keyframes borderPulse{0%,100%{opacity:0.5}50%{opacity:1}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}

.glow-red{animation:glowRed 1.4s ease-in-out infinite}
.live-dot{animation:breathe 2s ease-in-out infinite}
.fade-up{animation:fadeUp 0.4s ease forwards}
.cd-tick{animation:countdownTick 0.6s cubic-bezier(0.34,1.56,0.64,1)}

.scanline-overlay{
  position:fixed;top:0;left:0;width:100%;height:140px;pointer-events:none;
  background:linear-gradient(180deg, transparent 0%, #39FF6A0D 50%, transparent 100%);
  animation:scanline 7s linear infinite;z-index:9998;
}

/* Card system: subtle glass panel with an accent top-edge and inset glow */
.card-panel{
  position:relative;z-index:1;
  background:linear-gradient(165deg, #0A160BEE 0%, #071009EE 100%);
  border:1px solid #1E4527;
  border-radius:10px;
  box-shadow:0 1px 0 #39FF6A0A inset, 0 12px 24px -12px #000000AA;
  transition:transform 0.3s cubic-bezier(0.34,1.56,0.64,1),
             box-shadow 0.3s ease, border-color 0.3s ease;
  will-change:transform;
  backdrop-filter:blur(2px);
}
.card-panel::before{
  content:'';position:absolute;top:0;left:14px;right:14px;height:1px;
  background:linear-gradient(90deg, transparent, #39FF6A88, transparent);
}
.card-panel:hover{
  transform:translateY(-2px);
  border-color:#39FF6A55;
  box-shadow:0 1px 0 #39FF6A1A inset, 0 20px 36px -16px #000000CC, 0 0 0 1px #39FF6A22;
  z-index:50;
}

.card-head{
  display:flex;align-items:center;gap:7px;margin-bottom:9px;
}
.card-head .dot{
  width:5px;height:5px;border-radius:50%;flex-shrink:0;
  box-shadow:0 0 6px currentColor;
}

.hz{
  transition:transform 0.25s cubic-bezier(0.34,1.56,0.64,1),box-shadow 0.25s ease;
  position:relative;z-index:1;will-change:transform
}
.hz:hover{
  transform:scale(1.05);
  box-shadow:0 8px 30px #00FFC222,0 0 0 1px #00FFC233;
  z-index:100
}

.feed-row{
  padding:3px 6px 3px 10px;
  border-bottom:1px solid #123018;
  border-left:2px solid transparent;
  transition:border-left-color 0.15s,background 0.15s,padding-left 0.15s;
  cursor:default
}
.feed-row:hover{border-left-color:#00FFC2;background:#0E241080;padding-left:12px}

.anom-row{
  padding:3px 6px 3px 10px;
  border-bottom:1px solid #0E1F10;
  border-left:2px solid transparent;
  transition:border-left-color 0.15s,background 0.15s,padding-left 0.15s;
  cursor:default
}
.anom-row:hover{border-left-color:#FFB300;background:#1A130080;padding-left:12px}

.tl-row{
  padding:4px 3px;
  transition:padding-left 0.15s,opacity 0.15s;
  cursor:default
}
.tl-row:hover{padding-left:8px;opacity:0.9}

.log-row{
  padding:3px 6px;
  border-bottom:1px solid #0A160B;
  transition:background 0.15s,padding-left 0.15s;
  cursor:default
}
.log-row:hover{background:#0E241080;padding-left:4px}

.btn-premium{
  position:relative;overflow:hidden;
  transition:box-shadow 0.25s ease,transform 0.15s ease,filter 0.25s ease;
}
.btn-g{transition:box-shadow 0.25s ease,transform 0.15s ease,filter 0.25s ease}
.btn-g:hover{box-shadow:0 0 22px #39FF6A66,0 4px 12px #00000066!important;transform:translateY(-2px);filter:brightness(1.1)}
.btn-r{transition:box-shadow 0.25s ease,transform 0.15s ease,filter 0.25s ease}
.btn-r:hover{box-shadow:0 0 22px #FF3B3B66,0 4px 12px #00000066!important;transform:translateY(-2px);filter:brightness(1.1)}
.btn-ghost{transition:border-color 0.2s,color 0.2s,background 0.2s}
.btn-ghost:hover{border-color:#39FF6A!important;color:#B6FFC2!important;background:#39FF6A0F!important}
.btn-a{transition:box-shadow 0.2s,border-color 0.2s,background 0.2s}
.btn-a:hover{box-shadow:0 0 14px #FFB30040!important;border-color:#FFB300!important;background:#FFB3000F!important}

.countdown-num{
  transition:all 0.3s ease;
  display:inline-block
}

.brand-mark{
  animation:borderPulse 2.5s ease-in-out infinite;
}

.kpi-strip{
  display:flex;gap:10px;
}
.kpi-tile{
  flex:1;background:linear-gradient(165deg,#0A160BEE,#071009EE);
  border:1px solid #1E4527;border-radius:8px;padding:10px 12px;
  position:relative;overflow:hidden;
}
.kpi-tile::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg, transparent, #39FF6A, transparent);
  opacity:0.6;
}

.zone-col{min-width:0;display:flex;flex-direction:column}

@media (max-width: 1100px){
  .zone-col{grid-column:span 1}
}
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''

BG='#050B06';CARD='#081208';CARD2='#0A160B';BRDR='#1E4527'
BLUE='#00FFC2';GREEN='#39FF6A';AMBER='#FFB300';RED='#FF3B3B'
DIM='#3A6B44';WHITE='#B6FFC2';MONO='"IBM Plex Mono",monospace'

def lbl(text,color,size='9px',spacing='1.5px',mb='9px'):
    """Card header label with a small glowing accent dot -- the dot color
    doubles as a quick visual index of which subsystem a panel belongs to
    (blue/teal=info, amber=warning-tier, green=response, red=critical)."""
    return html.Div([
        html.Span(className='dot',style={'background':color,'color':color}),
        html.Span(text,style={
            'color':color,'fontFamily':MONO,'fontSize':size,
            'letterSpacing':spacing,'fontWeight':'700'
        }),
    ],className='card-head',style={'marginBottom':mb})

def card(children,border=None,mb='10px',pad='14px'):
    return html.Div(children,className='hz card-panel',style={
        'borderColor':border or BRDR,
        'padding':pad,'marginBottom':mb
    })

def zone_title(text,color):
    """Section divider above a group of related cards -- gives the new
    grid-based layout named zones (Signal Analysis, Network Visibility,
    Detection Intelligence, Operations Log) instead of an unlabeled column."""
    return html.Div([
        html.Div(style={'width':'14px','height':'1px','background':color}),
        html.Div(text,style={
            'color':color,'fontFamily':MONO,'fontSize':'9.5px','fontWeight':'700',
            'letterSpacing':'3px','margin':'0 10px'
        }),
        html.Div(style={'flex':'1','height':'1px','background':f'linear-gradient(90deg,{color}55,transparent)'}),
    ],style={'display':'flex','alignItems':'center','marginBottom':'12px','marginTop':'2px'})

app.layout = html.Div([

    html.Div(className='scanline-overlay'),

    html.Div([
        html.Div([
            html.Span('◈',className='brand-mark',style={
                'color':GREEN,'fontSize':'22px','verticalAlign':'middle',
                'marginRight':'10px','textShadow':f'0 0 12px {GREEN}88'
            }),
            html.Div([
                html.Div('PABD',style={
                    'color':WHITE,'fontFamily':MONO,'fontSize':'15px',
                    'fontWeight':'700','letterSpacing':'5px','lineHeight':'1.1',
                    'textShadow':f'0 0 18px {GREEN}44'
                }),
                html.Div('THREAT INTELLIGENCE CENTER',style={
                    'color':DIM,'fontFamily':MONO,'fontSize':'8px',
                    'letterSpacing':'2.5px','lineHeight':'1.1','marginTop':'2px'
                }),
            ]),
            html.Div(style={'width':'1px','height':'26px','background':BRDR,'margin':'0 18px'}),
            html.Span('●  LIVE',className='live-dot',style={
                'color':GREEN,'fontFamily':MONO,'fontSize':'9.5px','letterSpacing':'2px',
                'background':f'{GREEN}14','padding':'4px 10px','borderRadius':'20px',
                'border':f'1px solid {GREEN}33'
            }),
            html.Span(id='hdr',style={'color':DIM,'fontFamily':MONO,'fontSize':'9.5px','marginLeft':'16px'}),
        ],style={'display':'flex','alignItems':'center'}),
        html.Div([
            html.Button('▶  NORMAL MODE',id='btn-normal',className='btn-g',style={
                'background':f'linear-gradient(135deg,{GREEN},#28D157)','border':'none','color':'#04140A','fontFamily':MONO,
                'fontSize':'10px','fontWeight':'700','padding':'9px 16px',
                'borderRadius':'10px','marginRight':'7px','cursor':'pointer','letterSpacing':'1px',
                'boxShadow':f'0 2px 10px {GREEN}33'
            }),
            html.Button('⚡  SIMULATE ATTACK',id='btn-attack',className='btn-r',style={
                'background':f'linear-gradient(135deg,{RED},#C41E1E)','border':'none','color':'#fff','fontFamily':MONO,
                'fontSize':'10px','fontWeight':'700','padding':'9px 16px',
                'borderRadius':'10px','marginRight':'7px','cursor':'pointer','letterSpacing':'1px',
                'boxShadow':f'0 2px 10px {RED}33'
            }),
            html.Button('↺  RESET',id='btn-reset',className='btn-ghost',style={
                'background':'transparent','border':f'1px solid {BRDR}','color':DIM,
                'fontFamily':MONO,'fontSize':'10px','padding':'9px 14px',
                'borderRadius':'10px','marginRight':'7px','cursor':'pointer'
            }),
            html.Button('⚖  IDS COMPARE',id='btn-compare',className='btn-a',style={
                'background':'transparent','border':f'1px solid {AMBER}30','color':AMBER,
                'fontFamily':MONO,'fontSize':'10px','padding':'9px 14px',
                'borderRadius':'10px','cursor':'pointer'
            }),
        ],style={'display':'flex','alignItems':'center'}),
    ],style={
        'background':f'linear-gradient(90deg,{BG},{CARD} 60%,{BG})','padding':'12px 22px',
        'display':'flex','justifyContent':'space-between','alignItems':'center',
        'borderBottom':f'1px solid {BRDR}','position':'sticky','top':'0','zIndex':'999',
        'width':'100%','boxShadow':'0 4px 20px #00000066'
    }),

    html.Div(id='kpi-strip',style={'padding':'12px 20px 0'}),

    # -- HERO ROW: the single "what is happening right now" strip. Status,
    # countdown and threat classification used to be buried in a narrow
    # right-hand column; they're the most important thing on screen and now
    # get equal top billing as three wide cards. --
    html.Div([
        html.Div(id='status-box',style={'flex':'1','minWidth':'0'}),
        html.Div(id='timer-box',style={'flex':'1','minWidth':'0'}),
        html.Div(id='threat-box',style={'flex':'1','minWidth':'0'}),
    ],style={'display':'flex','gap':'12px','padding':'12px 20px 0'}),

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
                html.Div(id='net-feed',style={'height':'175px','overflowY':'auto','fontFamily':MONO,'fontSize':'9px'})
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
        'display':'grid','gridTemplateColumns':'1fr 1fr 1.1fr','gap':'14px',
        'padding':'14px 20px','background':BG,'width':'100%'
    }),

    # -- OPERATIONS FOOTER: logging/comparison panels, lower on the page
    # since they're supporting evidence, not the primary "what's happening"
    # signal the hero row and main zones already cover. --
    html.Div([
        zone_title('OPERATIONS LOG',DIM),
        html.Div([
            card([
                lbl('ISOLATION FOREST  ·  ANOMALY STREAM',AMBER),
                html.Div(id='anom-feed',style={'height':'110px','overflowY':'auto','fontFamily':MONO,'fontSize':'9px','color':AMBER})
            ],border=f'{AMBER}22',mb='0'),
            card([
                lbl('SOC RESPONSE TIMELINE',GREEN),
                html.Div(id='timeline',style={'fontFamily':MONO,'fontSize':'9.5px','height':'110px','overflowY':'auto'})
            ],border=f'{GREEN}22',mb='0'),
            card([
                lbl('SESSION EVENT LOG',DIM),
                html.Div(id='event-log',style={'height':'110px','overflowY':'auto','fontFamily':MONO,'fontSize':'8.5px'})
            ],mb='0'),
            html.Div(id='compare-box'),
        ],style={'display':'grid','gridTemplateColumns':'1fr 1fr 1fr 1fr','gap':'14px'}),
    ],style={'padding':'0 20px 20px'}),

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
            'color':RED if 'ALERT' in e else DIM,'fontFamily':MONO
        }) for e in S['network_feed']
    ]

    anom_items=[
        html.Div(e,className='anom-row',style={'fontFamily':MONO,'color':AMBER})
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
        annotation_font_size=9,annotation_position='right')
    cusum_fig.update_layout(
        paper_bgcolor=CARD2,plot_bgcolor=CARD2,
        font=dict(color=DIM,size=9,family='IBM Plex Mono'),
        margin=dict(l=40,r=95,t=8,b=38),
        legend=dict(font=dict(size=8.5,color=WHITE),bgcolor='rgba(0,0,0,0)',orientation='h',y=-0.24,x=0),
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
            (0,0.3,'rgba(0,200,255,0.04)',BLUE,'NORMAL'),
            (0.3,0.6,'rgba(255,179,0,0.05)',AMBER,'ELEVATED'),
            (0.6,0.85,'rgba(255,109,0,0.06)','#FF7A1A','WARNING'),
            (0.85,1.0,'rgba(255,61,87,0.08)',RED,'CRITICAL'),
        ]:
            rec_fig.add_shape(type='rect',x0=0,x1=mx,y0=y0,y1=y1,fillcolor=col,line_width=0,layer='below')
            rec_fig.add_annotation(x=mx-0.5,y=(y0+y1)/2,text=lt,showarrow=False,
                font=dict(size=8,color=bc,family='IBM Plex Mono'),xanchor='right',opacity=0.45)
        rv=S['rec_score']
        lc=GREEN if rv<0.3 else AMBER if rv<0.6 else '#FF7A1A' if rv<0.85 else RED
        if S['pabd_fired'] and len(y)>5:
            upper=[min(v+0.04,1.0) for v in y];lower=[max(v-0.04,0.0) for v in y]
            rec_fig.add_trace(go.Scatter(x=x+x[::-1],y=upper+lower[::-1],fill='toself',
                fillcolor='rgba(0,200,255,0.04)',line=dict(width=0),showlegend=False,hoverinfo='skip'))
        rec_fig.add_trace(go.Scatter(x=x,y=y,fill='tozeroy',
            fillcolor='rgba(0,200,255,0.05)',line=dict(color=lc,width=2.5),
            showlegend=False,hovertemplate='Risk: %{y:.3f}<extra></extra>'))
    rec_fig.update_layout(
        paper_bgcolor=CARD2,plot_bgcolor=CARD2,
        font=dict(color=DIM,size=9,family='IBM Plex Mono'),
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
            color = RED if c['is_anom'] else 'rgba(0,200,255,0.35)'
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

    if S['pabd_fired']:
        status_box=html.Div([
            html.Div('⚡  P A B D   F I R E D',style={
                'color':RED,'fontFamily':MONO,'fontSize':'17px','fontWeight':'700',
                'textAlign':'center','letterSpacing':'4px','padding':'10px 0 4px'
            }),
            html.Div('PRE-ATTACK BEHAVIOURAL DRIFT DETECTED',style={
                'color':'#FF3B3B66','fontFamily':MONO,'fontSize':'8px',
                'textAlign':'center','letterSpacing':'2px','paddingBottom':'4px'
            }),
            html.Div(f'Step {S["pabd_step"]}  ·  {S["metrics_breached"]}/6 metrics breached',style={
                'color':DIM,'fontFamily':MONO,'fontSize':'9px','textAlign':'center'
            }),
        ],className='glow-red hz',style={
            'background':'rgba(255,61,87,0.06)','border':f'1px solid {RED}44',
            'borderRadius':'10px','padding':'8px'
        })
    else:
        rv=S['rec_score']
        sc,st=((GREEN,'NORMAL') if rv<0.3 else (AMBER,'ELEVATED') if rv<0.6 else ('#FF7A1A','WARNING') if rv<0.85 else (RED,'CRITICAL'))
        status_box=html.Div([
            html.Div(f'● {st}',style={'color':sc,'fontFamily':MONO,'fontSize':'20px','fontWeight':'700','textAlign':'center','padding':'8px 0 4px'}),
            html.Div(f'Risk Score  {rv:.4f}',style={'color':DIM,'fontFamily':MONO,'fontSize':'10px','textAlign':'center'}),
            html.Div(f'Metrics drifting: {S["metrics_breached"]} / 6',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','textAlign':'center','marginTop':'3px'}),
        ],className='hz',style={'border':f'1px solid {sc}33','borderRadius':'10px','padding':'8px','background':f'{sc}08'})

    if S['attack_classified'] and S['predicted_attack']:
        fi=sorted(S['feature_imp'].items(),key=lambda x:x[1],reverse=True)[:3]
        threat_box=html.Div([
            html.Div('--  THREAT  CLASSIFIED  --',style={'color':RED,'fontFamily':MONO,'fontSize':'9px','textAlign':'center','letterSpacing':'2px','marginBottom':'10px'}),
            html.Div([
                html.Span('Attack Type:  ',style={'color':DIM,'fontFamily':MONO,'fontSize':'10px'}),
                html.Span(S['predicted_attack'],style={'color':'#FF9A2E','fontFamily':MONO,'fontSize':'13px','fontWeight':'700'}),
            ],style={'marginBottom':'5px'}),
            html.Div([
                html.Span('Confidence:  ',style={'color':DIM,'fontFamily':MONO,'fontSize':'10px'}),
                html.Span(f"{S['confidence']:.1f}%",style={'color':GREEN,'fontFamily':MONO,'fontSize':'13px','fontWeight':'700'}),
            ],style={'marginBottom':'10px'}),
            html.Div('TOP FEATURE SIGNALS:',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','letterSpacing':'1px','marginBottom':'6px'}),
            *[html.Div([
                html.Div(style={'width':f'{imp*100:.0f}%','height':'4px','background':f'linear-gradient(90deg,{BLUE},{GREEN})','borderRadius':'2px','marginBottom':'2px'}),
                html.Div(f'{feat}  {imp*100:.1f}%',style={'color':WHITE,'fontFamily':MONO,'fontSize':'9.5px','marginBottom':'5px'}),
            ]) for feat,imp in fi]
        ],className='hz',style={'background':'rgba(255,61,87,0.04)','border':f'1px solid {RED}33','borderRadius':'10px','padding':'12px'})
    else:
        threat_box=html.Div('Awaiting classification...',style={
            'color':BRDR,'fontFamily':MONO,'fontSize':'10px','textAlign':'center',
            'padding':'18px','border':f'1px solid {BRDR}','borderRadius':'10px'
        })

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
            html.Div('TIME TO ATTACK EXECUTION',style={
                'color':DIM,'fontFamily':MONO,'fontSize':'9px',
                'textAlign':'center','letterSpacing':'2px','marginBottom':'8px'
            }),
            html.Div([
                html.Span(f'{m:02d}',className='countdown-num',style={
                    'color':RED,'fontFamily':MONO,'fontSize':'44px',
                    'fontWeight':'700','letterSpacing':'2px','lineHeight':'1'
                }),
                html.Span(':',style={'color':RED,'fontFamily':MONO,'fontSize':'44px','fontWeight':'700','lineHeight':'1'}),
                html.Span(f'{s:02d}',className='countdown-num',style={
                    'color':RED,'fontFamily':MONO,'fontSize':'44px',
                    'fontWeight':'700','letterSpacing':'2px','lineHeight':'1'
                }),
                html.Span(f'.{ms}',style={
                    'color':'#FF3B3B88','fontFamily':MONO,'fontSize':'22px',
                    'fontWeight':'700','lineHeight':'1','verticalAlign':'bottom'
                }),
            ],style={'textAlign':'center','marginBottom':'4px'}),
            html.Div('min  :  sec  . tenth',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','textAlign':'center','marginBottom':'10px'}),
            html.Div(style={
                'width':f'{pct*100:.1f}%','height':'3px',
                'background':f'linear-gradient(90deg,{bc}88,{bc})',
                'borderRadius':'2px','marginBottom':'6px',
                'transition':'width 0.65s linear, background 0.5s ease'
            }),
            html.Div(f'95% CI: {mlo:02d}:{slo:02d} — {mhi:02d}:{shi:02d}',style={'color':AMBER,'fontFamily':MONO,'fontSize':'9.5px','textAlign':'center'}),
        ],className='hz',style={'background':'rgba(255,61,87,0.04)','border':f'1px solid {RED}33','borderRadius':'10px','padding':'12px'})
    else:
        timer_box=html.Div()

    gauges=[]
    for i,f in enumerate(FEATURES):
        val=S['cusum'][f];pct=min(val/CUSUM_H,1.0)
        col=METRIC_COLORS[i];fired=val>=CUSUM_H
        gauges.append(html.Div([
            html.Div([
                html.Span(FLABELS[i],style={'color':col if fired else DIM,'fontFamily':MONO,'fontSize':'8.5px','fontWeight':'700'}),
                html.Span(f'{val:.2f}',style={'color':WHITE,'fontFamily':MONO,'fontSize':'8.5px','float':'right'}),
            ],style={'marginBottom':'3px','overflow':'hidden'}),
            html.Div(style={
                'width':f'{pct*100:.1f}%','height':'3px',
                'background':RED if fired else col,
                'borderRadius':'2px','marginBottom':'5px',
                'transition':'width 0.5s ease'
            }),
        ]))
    gauges_box=html.Div([
        html.Div('CUSUM METRIC LEVELS',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','letterSpacing':'1.5px','marginBottom':'8px','fontWeight':'700'}),
        *gauges
    ],className='hz',style={'background':CARD,'border':f'1px solid {BRDR}','borderRadius':'10px','padding':'12px'})

    if S['attack_classified'] and S['predicted_attack'] in PROFILES:
        prof=PROFILES[S['predicted_attack']]
        profile_box=html.Div([
            html.Div('--  ATTACKER  PROFILE  --',style={'color':'#B388FF','fontFamily':MONO,'fontSize':'9px','textAlign':'center','letterSpacing':'2px','marginBottom':'10px'}),
            *[html.Div([
                html.Span(f'{k}:  ',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px'}),
                html.Span(v,style={'color':BLUE,'fontFamily':MONO,'fontSize':'9.5px'}),
            ],style={'marginBottom':'4px'}) for k,v in prof.items()],
            html.Div('Evasion Attempt:  DETECTED',style={'color':'#FF7A1A','fontFamily':MONO,'fontSize':'9.5px','marginTop':'8px'}),
            html.Div('System Response:  ACTIVE',style={'color':GREEN,'fontFamily':MONO,'fontSize':'9.5px','marginTop':'3px'}),
        ],className='hz',style={'background':'rgba(179,136,255,0.04)','border':'1px solid #B388FF33','borderRadius':'10px','padding':'12px'})
    else:
        profile_box=html.Div()

    if S['shap_explain']:
        def _bar(feat, val, raw_val):
            pct = min(abs(val)/0.4, 1.0) * 100  # 0.4 ~= typical max |SHAP| seen in offline analysis
            color = '#FF7A1A' if val > 0 else '#39FF6A'
            return html.Div([
                html.Div([
                    html.Span(feat,style={'color':DIM,'fontFamily':MONO,'fontSize':'9px'}),
                    html.Span(f'{val:+.3f}',style={'color':color,'fontFamily':MONO,'fontSize':'9px','float':'right'}),
                ]),
                html.Div(style={'background':'#0F2012','borderRadius':'3px','height':'5px','marginTop':'2px','overflow':'hidden'},children=[
                    html.Div(style={'background':color,'width':f'{pct}%','height':'100%'})
                ]),
            ],style={'marginBottom':'7px'})
        shap_box=html.Div([
            html.Div('--  WHY THIS FIRED (live SHAP)  --',style={'color':'#39FF6A','fontFamily':MONO,'fontSize':'9px','textAlign':'center','letterSpacing':'2px','marginBottom':'10px'}),
            html.Div('Per-alert explanation for this window -- not a static global ranking.',
                     style={'color':DIM,'fontFamily':MONO,'fontSize':'7.5px','textAlign':'center','marginBottom':'8px'}),
            *[_bar(feat, val, raw) for feat, val, raw in S['shap_explain'][:6]],
        ],className='hz',style={'background':'rgba(0,230,118,0.04)','border':'1px solid #39FF6A33','borderRadius':'10px','padding':'12px'})
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
        html.Div('--  MITRE ATT&CK KILL CHAIN  --',style={'color':AMBER,'fontFamily':MONO,'fontSize':'9px','textAlign':'center','letterSpacing':'2px','marginBottom':'10px'}),
        html.Div([
            html.Div([
                html.Div(stage,style={
                    'color':(BG if i<=cur_idx else DIM),
                    'fontFamily':MONO,'fontSize':'7.5px','textAlign':'center',
                    'padding':'4px 2px',
                    'background':(AMBER if i==cur_idx else (GREEN if i<cur_idx else 'transparent')),
                    'border':f'1px solid {AMBER if i<=cur_idx else BRDR}',
                    'borderRadius':'3px','flex':'1','marginRight':('3px' if i<len(KILL_CHAIN)-1 else '0'),
                    'transition':'background 0.4s ease',
                }) for i,stage in enumerate(KILL_CHAIN)
            ],style={'display':'flex'}),
        ]),
    ],className='hz',style={'background':'rgba(255,179,0,0.04)','border':f'1px solid {AMBER}33','borderRadius':'10px','padding':'12px','marginTop':'8px'})

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
        ],className='tl-row',style={'fontFamily':MONO,'fontSize':'9.5px'}))

    if S.get('comparison_mode'):
        adv=S['trad_step']-S['pabd_step'] if (S['trad_fired'] and S['pabd_fired']) else None
        compare_box=html.Div([
            html.Div('PABD FRAMEWORK  vs  TRADITIONAL IDS',style={'color':AMBER,'fontFamily':MONO,'fontSize':'9px','fontWeight':'700','textAlign':'center','letterSpacing':'2px','marginBottom':'10px'}),
            html.Div([
                html.Div([
                    html.Div('OUR FRAMEWORK',style={'color':GREEN,'fontFamily':MONO,'fontSize':'9px','fontWeight':'700','marginBottom':'5px'}),
                    html.Div(f"Status:  {'FIRED' if S['pabd_fired'] else 'Monitoring'}",style={'color':GREEN if S['pabd_fired'] else DIM,'fontFamily':MONO,'fontSize':'9.5px'}),
                    html.Div(f"Step:  {S['pabd_step'] if S['pabd_fired'] else '--'}",style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','marginTop':'3px'}),
                    html.Div('Detects:  PRE-ATTACK PHASE',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','marginTop':'3px'}),
                ],style={'flex':1,'padding':'10px','border':f'1px solid {GREEN}33','borderRadius':'6px','marginRight':'6px','background':f'{GREEN}06'}),
                html.Div([
                    html.Div('TRADITIONAL IDS',style={'color':'#FF7A1A','fontFamily':MONO,'fontSize':'9px','fontWeight':'700','marginBottom':'5px'}),
                    html.Div(f"Status:  {'FIRED' if S['trad_fired'] else 'No alert'}",style={'color':'#FF7A1A' if S['trad_fired'] else DIM,'fontFamily':MONO,'fontSize':'9.5px'}),
                    html.Div(f"Step:  {S['trad_step'] if S['trad_fired'] else '--'}",style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','marginTop':'3px'}),
                    html.Div('Detects:  EXECUTION PHASE',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','marginTop':'3px'}),
                ],style={'flex':1,'padding':'10px','border':'1px solid #FF7A1A33','borderRadius':'6px','background':'#FF7A1A06'}),
            ],style={'display':'flex','marginBottom':'8px'}),
            html.Div(
                f'Lead time advantage:  {adv} steps earlier' if adv else 'Traditional IDS has not fired yet.',
                style={'color':AMBER,'fontFamily':MONO,'fontSize':'9.5px','textAlign':'center','padding':'4px 0'}
            ),
        ],className='hz',style={'background':CARD,'border':f'1px solid {AMBER}33','borderRadius':'10px','padding':'12px','marginTop':'8px'})
    else:
        compare_box=html.Div()

    log_items=[
        html.Div(e,className='log-row',style={'fontFamily':MONO,'color':DIM,'fontSize':'8.5px'})
        for e in S['event_log']
    ]

    tp=S['total_packets'];fp=S['flagged_packets']
    pct=(fp/tp*100) if tp>0 else 0
    hdr=f'Packets: {tp}  ·  Flagged: {fp} ({pct:.1f}%)  ·  {S["pps"]} pkt/s  ·  Step: {S["step"]}'

    def kpi_tile(label, value, color, sub=None):
        return html.Div([
            html.Div(label,style={'color':DIM,'fontFamily':MONO,'fontSize':'8px','letterSpacing':'1.5px','marginBottom':'5px'}),
            html.Div(value,style={'color':color,'fontFamily':MONO,'fontSize':'19px','fontWeight':'700','lineHeight':'1'}),
            html.Div(sub or ' ',style={'color':DIM,'fontFamily':MONO,'fontSize':'8px','marginTop':'4px'}),
        ],className='kpi-tile')

    risk_color = GREEN if S['rec_score']<0.3 else AMBER if S['rec_score']<0.6 else '#FF7A1A' if S['rec_score']<0.85 else RED
    kpi_strip = html.Div([
        kpi_tile('PACKETS PROCESSED', f'{tp:,}', WHITE, f'{S["pps"]} pkt/s'),
        kpi_tile('FLAGGED ANOMALOUS', f'{fp:,}', AMBER, f'{pct:.1f}% of traffic'),
        kpi_tile('RISK SCORE', f'{S["rec_score"]:.3f}', risk_color, f'{S["metrics_breached"]}/6 metrics drifting'),
        kpi_tile('SESSION STEP', f'{S["step"]}', BLUE, 'PABD FIRED' if S['pabd_fired'] else 'monitoring'),
    ],className='kpi-strip')

    return (net_items,anom_items,cusum_fig,rec_fig,net_graph_fig,
            status_box,threat_box,timer_box,gauges_box,
            profile_box,shap_box,killchain_box,tl_items,compare_box,hdr,log_items,kpi_strip)


if __name__=='__main__':
    # host='0.0.0.0' required for this to be reachable from outside a Docker
    # container -- the Dash/Flask default of 127.0.0.1 only accepts
    # connections from inside the container itself.
    app.run(debug=False,host='0.0.0.0',port=8050)