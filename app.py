import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import joblib
import time
import random
import pandas as pd
from tensorflow import keras

scaler = joblib.load('scaler.pkl')
iso_forest = joblib.load('isolation_forest.pkl')
rf_classifier = joblib.load('random_forest.pkl')
lstm_model = keras.models.load_model('lstm_model.keras')
baseline_stats = joblib.load('baseline_stats.pkl')

FEATURES = ['connection_frequency','port_entropy','failed_auth_rate','packet_size_dist','protocol_ratio','burst_activity']
FLABELS = ['Conn Freq','Port Entropy','Failed Auth','Pkt Size','Proto Ratio','Burst Act']
ANAMES = {0:'Normal',1:'Reconnaissance',2:'Brute Force',3:'Lateral Movement',4:'DoS Preparation'}
PROFILES = {
    'Reconnaissance':{'Archetype':'The Slow Scanner','Technique':'Incremental port sweep','Velocity':'2-3 ports / min','Objective':'Vulnerability mapping'},
    'Brute Force':{'Archetype':'The Credential Tester','Technique':'Distributed login attempts','Velocity':'Low volume, persistent','Objective':'Authentication bypass'},
    'Lateral Movement':{'Archetype':'The Lateral Mover','Technique':'Internal network traversal','Velocity':'Slow and methodical','Objective':'Privilege escalation'},
    'DoS Preparation':{'Archetype':'The Flood Preparer','Technique':'Resource exhaustion staging','Velocity':'Rapid burst spikes','Objective':'Service disruption'},
}
MEANS = baseline_stats['mean']
STDS = baseline_stats['std']
CUSUM_K = 0.5
CUSUM_H = 8.0
METRIC_COLORS = ['#00C8FF','#00E676','#FF6D00','#FFD600','#FF4081','#B388FF']

S = {
    'running':False,'attack_mode':False,'attack_type':'Reconnaissance',
    'step':0,'sequence':[],'total_packets':0,'flagged_packets':0,
    'cusum':{f:0.0 for f in FEATURES},
    'cusum_hist':{f:[] for f in FEATURES},
    'rec_hist':[],'rec_score':0.0,
    'network_feed':[],'anomaly_feed':[],
    'pabd_fired':False,'pabd_step':0,
    'attack_classified':False,'predicted_attack':'',
    'confidence':0.0,'feature_imp':{},
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
        S['feature_imp'] = dict(zip(FLABELS,rf_classifier.feature_importances_))
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
            S['countdown_max'] = max(int(pr*60*3),90)
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
html,body{width:100%;height:100%;overflow-x:hidden;background:#07090F;font-family:"IBM Plex Mono",monospace}
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-track{background:#0B0F1A}
::-webkit-scrollbar-thumb{background:#1A2640;border-radius:2px}

@keyframes glowRed{0%,100%{box-shadow:0 0 8px #FF3D5740}50%{box-shadow:0 0 20px #FF3D57AA}}
@keyframes breathe{0%,100%{opacity:1}50%{opacity:0.35}}
@keyframes fadeUp{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
@keyframes countdownTick{0%{opacity:0.7}100%{opacity:1}}

.glow-red{animation:glowRed 1.5s ease-in-out infinite}
.live-dot{animation:breathe 2.2s ease-in-out infinite}
.fade-up{animation:fadeUp 0.35s ease forwards}
.cd-tick{animation:countdownTick 0.65s ease}

.hz{
  transition:transform 0.25s cubic-bezier(0.34,1.56,0.64,1),box-shadow 0.25s ease;
  position:relative;z-index:1;will-change:transform
}
.hz:hover{
  transform:scale(1.06);
  box-shadow:0 6px 28px #00C8FF1A,0 0 0 1px #00C8FF1A;
  z-index:100
}

.feed-row{
  padding:2px 4px 2px 8px;
  border-bottom:1px solid #0F1A2A;
  border-left:2px solid transparent;
  transition:border-left-color 0.15s,background 0.15s;
  cursor:default
}
.feed-row:hover{border-left-color:#00C8FF44;background:#0D1525}

.anom-row{
  padding:2px 4px 2px 8px;
  border-bottom:1px solid #180E00;
  border-left:2px solid transparent;
  transition:border-left-color 0.15s,background 0.15s;
  cursor:default
}
.anom-row:hover{border-left-color:#FFB30055;background:#120C00}

.tl-row{
  padding:3px 2px;
  transition:padding-left 0.15s,opacity 0.15s;
  cursor:default
}
.tl-row:hover{padding-left:6px;opacity:0.85}

.log-row{
  padding:2px 4px;
  border-bottom:1px solid #0C1220;
  transition:background 0.15s;
  cursor:default
}
.log-row:hover{background:#0D1525}

.btn-g{transition:box-shadow 0.2s,transform 0.15s}
.btn-g:hover{box-shadow:0 0 14px #00E67640!important;transform:translateY(-1px)}
.btn-r{transition:box-shadow 0.2s,transform 0.15s}
.btn-r:hover{box-shadow:0 0 14px #FF3D5740!important;transform:translateY(-1px)}
.btn-ghost{transition:border-color 0.2s,color 0.2s}
.btn-ghost:hover{border-color:#3A5070!important;color:#8AB0D0!important}
.btn-a{transition:box-shadow 0.2s,border-color 0.2s}
.btn-a:hover{box-shadow:0 0 10px #FFB30030!important;border-color:#FFB30055!important}

.countdown-num{
  transition:all 0.3s ease;
  display:inline-block
}
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''

BG='#07090F';CARD='#0B0F1A';CARD2='#0D1220';BRDR='#1A2640'
BLUE='#00C8FF';GREEN='#00E676';AMBER='#FFB300';RED='#FF3D57'
DIM='#3A5070';WHITE='#C8D8F0';MONO='"IBM Plex Mono",monospace'

def lbl(text,color,size='9px',spacing='1.5px',mb='7px'):
    return html.Div(text,style={
        'color':color,'fontFamily':MONO,'fontSize':size,
        'letterSpacing':spacing,'marginBottom':mb,'fontWeight':'700'
    })

def card(children,border=None,mb='8px',pad='12px'):
    return html.Div(children,className='hz',style={
        'background':CARD,'border':f'1px solid {border or BRDR}',
        'borderRadius':'6px','padding':pad,'marginBottom':mb
    })

app.layout = html.Div([

    html.Div([
        html.Div([
            html.Span('◈ ',style={'color':BLUE,'fontSize':'19px','verticalAlign':'middle'}),
            html.Span('PABD  THREAT  INTELLIGENCE  CENTER',style={
                'color':WHITE,'fontFamily':MONO,'fontSize':'13px',
                'fontWeight':'700','letterSpacing':'4px','verticalAlign':'middle'
            }),
            html.Span('  ·  LIVE',className='live-dot',style={
                'color':GREEN,'fontFamily':MONO,'fontSize':'10px','marginLeft':'12px','letterSpacing':'2px'
            }),
            html.Span(id='hdr',style={'color':DIM,'fontFamily':MONO,'fontSize':'10px','marginLeft':'22px'}),
        ],style={'display':'flex','alignItems':'center'}),
        html.Div([
            html.Button('▶  NORMAL MODE',id='btn-normal',className='btn-g',style={
                'background':GREEN,'border':'none','color':'#000','fontFamily':MONO,
                'fontSize':'10px','fontWeight':'700','padding':'8px 15px',
                'borderRadius':'4px','marginRight':'6px','cursor':'pointer','letterSpacing':'1px'
            }),
            html.Button('⚡  SIMULATE ATTACK',id='btn-attack',className='btn-r',style={
                'background':RED,'border':'none','color':'#fff','fontFamily':MONO,
                'fontSize':'10px','fontWeight':'700','padding':'8px 15px',
                'borderRadius':'4px','marginRight':'6px','cursor':'pointer','letterSpacing':'1px'
            }),
            html.Button('↺  RESET',id='btn-reset',className='btn-ghost',style={
                'background':'transparent','border':f'1px solid {BRDR}','color':DIM,
                'fontFamily':MONO,'fontSize':'10px','padding':'8px 13px',
                'borderRadius':'4px','marginRight':'6px','cursor':'pointer'
            }),
            html.Button('⚖  IDS COMPARE',id='btn-compare',className='btn-a',style={
                'background':'transparent','border':'1px solid #FFB30030','color':AMBER,
                'fontFamily':MONO,'fontSize':'10px','padding':'8px 13px',
                'borderRadius':'4px','cursor':'pointer'
            }),
        ],style={'display':'flex','alignItems':'center'}),
    ],style={
        'background':f'linear-gradient(90deg,{BG},{CARD})','padding':'10px 20px',
        'display':'flex','justifyContent':'space-between','alignItems':'center',
        'borderBottom':f'1px solid {BRDR}','position':'sticky','top':'0','zIndex':'999',
        'width':'100%'
    }),

    html.Div([

        html.Div([
            card([
                lbl('LIVE NETWORK TRAFFIC FEED',BLUE),
                html.Div(id='net-feed',style={'height':'175px','overflowY':'auto','fontFamily':MONO,'fontSize':'9px'})
            ]),
            card([
                lbl('ISOLATION FOREST  ·  ANOMALY STREAM',AMBER),
                html.Div(id='anom-feed',style={'height':'110px','overflowY':'auto','fontFamily':MONO,'fontSize':'9px','color':AMBER})
            ],border='#FFB30022'),
            card([
                lbl('SOC RESPONSE TIMELINE',GREEN),
                html.Div(id='timeline',style={'fontFamily':MONO,'fontSize':'9.5px'})
            ],border='#00E67622'),
            card([
                lbl('SESSION EVENT LOG',DIM),
                html.Div(id='event-log',style={'height':'95px','overflowY':'auto','fontFamily':MONO,'fontSize':'8.5px'})
            ]),
        ],style={'width':'23%','padding':'8px 5px 8px 8px','flexShrink':'0'}),

        html.Div([
            card([
                lbl('CUSUM DRIFT MONITORS  ·  6 SIMULTANEOUS BEHAVIOURAL METRICS',BLUE),
                dcc.Graph(id='cusum-fig',style={'height':'210px'},config={'displayModeBar':False}),
            ]),
            card([
                lbl('RISK ESCALATION CURVE  ·  NORMAL — ELEVATED — WARNING — CRITICAL',AMBER),
                dcc.Graph(id='rec-fig',style={'height':'155px'},config={'displayModeBar':False}),
            ]),
            html.Div(id='compare-box'),
        ],style={'flex':'1','padding':'8px 5px','minWidth':'0'}),

        html.Div([
            html.Div(id='status-box',style={'marginBottom':'8px'}),
            html.Div(id='threat-box',style={'marginBottom':'8px'}),
            html.Div(id='timer-box',style={'marginBottom':'8px'}),
            html.Div(id='gauges-box',style={'marginBottom':'8px'}),
            html.Div(id='profile-box'),
        ],style={'width':'28%','padding':'8px 8px 8px 5px','flexShrink':'0'}),

    ],style={
        'display':'flex','background':BG,
        'minHeight':'calc(100vh - 50px)',
        'width':'100%','overflow':'hidden'
    }),

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
            'network_feed':[],'anomaly_feed':[],
            'pabd_fired':False,'pabd_step':0,
            'attack_classified':False,'predicted_attack':'',
            'confidence':0.0,'feature_imp':{},
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
     Output('status-box','children'),Output('threat-box','children'),
     Output('timer-box','children'),Output('gauges-box','children'),
     Output('profile-box','children'),Output('timeline','children'),
     Output('compare-box','children'),Output('hdr','children'),
     Output('event-log','children')],
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
            (0.6,0.85,'rgba(255,109,0,0.06)','#FF6D00','WARNING'),
            (0.85,1.0,'rgba(255,61,87,0.08)',RED,'CRITICAL'),
        ]:
            rec_fig.add_shape(type='rect',x0=0,x1=mx,y0=y0,y1=y1,fillcolor=col,line_width=0,layer='below')
            rec_fig.add_annotation(x=mx-0.5,y=(y0+y1)/2,text=lt,showarrow=False,
                font=dict(size=8,color=bc,family='IBM Plex Mono'),xanchor='right',opacity=0.45)
        rv=S['rec_score']
        lc=GREEN if rv<0.3 else AMBER if rv<0.6 else '#FF6D00' if rv<0.85 else RED
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

    if S['pabd_fired']:
        status_box=html.Div([
            html.Div('⚡  P A B D   F I R E D',style={
                'color':RED,'fontFamily':MONO,'fontSize':'17px','fontWeight':'700',
                'textAlign':'center','letterSpacing':'4px','padding':'10px 0 4px'
            }),
            html.Div('PRE-ATTACK BEHAVIOURAL DRIFT DETECTED',style={
                'color':'#FF3D5766','fontFamily':MONO,'fontSize':'8px',
                'textAlign':'center','letterSpacing':'2px','paddingBottom':'4px'
            }),
            html.Div(f'Step {S["pabd_step"]}  ·  {S["metrics_breached"]}/6 metrics breached',style={
                'color':DIM,'fontFamily':MONO,'fontSize':'9px','textAlign':'center'
            }),
        ],className='glow-red hz',style={
            'background':'rgba(255,61,87,0.06)','border':f'1px solid {RED}44',
            'borderRadius':'6px','padding':'8px'
        })
    else:
        rv=S['rec_score']
        sc,st=((GREEN,'NORMAL') if rv<0.3 else (AMBER,'ELEVATED') if rv<0.6 else ('#FF6D00','WARNING') if rv<0.85 else (RED,'CRITICAL'))
        status_box=html.Div([
            html.Div(f'● {st}',style={'color':sc,'fontFamily':MONO,'fontSize':'20px','fontWeight':'700','textAlign':'center','padding':'8px 0 4px'}),
            html.Div(f'Risk Score  {rv:.4f}',style={'color':DIM,'fontFamily':MONO,'fontSize':'10px','textAlign':'center'}),
            html.Div(f'Metrics drifting: {S["metrics_breached"]} / 6',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','textAlign':'center','marginTop':'3px'}),
        ],className='hz',style={'border':f'1px solid {sc}33','borderRadius':'6px','padding':'8px','background':f'{sc}08'})

    if S['attack_classified'] and S['predicted_attack']:
        fi=sorted(S['feature_imp'].items(),key=lambda x:x[1],reverse=True)[:3]
        threat_box=html.Div([
            html.Div('--  THREAT  CLASSIFIED  --',style={'color':RED,'fontFamily':MONO,'fontSize':'9px','textAlign':'center','letterSpacing':'2px','marginBottom':'10px'}),
            html.Div([
                html.Span('Attack Type:  ',style={'color':DIM,'fontFamily':MONO,'fontSize':'10px'}),
                html.Span(S['predicted_attack'],style={'color':'#FF8C00','fontFamily':MONO,'fontSize':'13px','fontWeight':'700'}),
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
        ],className='hz',style={'background':'rgba(255,61,87,0.04)','border':f'1px solid {RED}33','borderRadius':'6px','padding':'12px'})
    else:
        threat_box=html.Div('Awaiting classification...',style={
            'color':BRDR,'fontFamily':MONO,'fontSize':'10px','textAlign':'center',
            'padding':'18px','border':f'1px solid {BRDR}','borderRadius':'6px'
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
                    'color':'#FF3D5788','fontFamily':MONO,'fontSize':'22px',
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
        ],className='hz',style={'background':'rgba(255,61,87,0.04)','border':f'1px solid {RED}33','borderRadius':'6px','padding':'12px'})
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
    ],className='hz',style={'background':CARD,'border':f'1px solid {BRDR}','borderRadius':'6px','padding':'12px'})

    if S['attack_classified'] and S['predicted_attack'] in PROFILES:
        prof=PROFILES[S['predicted_attack']]
        profile_box=html.Div([
            html.Div('--  ATTACKER  PROFILE  --',style={'color':'#B388FF','fontFamily':MONO,'fontSize':'9px','textAlign':'center','letterSpacing':'2px','marginBottom':'10px'}),
            *[html.Div([
                html.Span(f'{k}:  ',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px'}),
                html.Span(v,style={'color':BLUE,'fontFamily':MONO,'fontSize':'9.5px'}),
            ],style={'marginBottom':'4px'}) for k,v in prof.items()],
            html.Div('Evasion Attempt:  DETECTED',style={'color':'#FF6D00','fontFamily':MONO,'fontSize':'9.5px','marginTop':'8px'}),
            html.Div('System Response:  ACTIVE',style={'color':GREEN,'fontFamily':MONO,'fontSize':'9.5px','marginTop':'3px'}),
        ],className='hz',style={'background':'rgba(179,136,255,0.04)','border':'1px solid #B388FF33','borderRadius':'6px','padding':'12px'})
    else:
        profile_box=html.Div()

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
            html.Span(action,style={'color':col if col!=BRDR else '#253545'}),
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
                ],style={'flex':1,'padding':'10px','border':f'1px solid {GREEN}33','borderRadius':'4px','marginRight':'6px','background':f'{GREEN}06'}),
                html.Div([
                    html.Div('TRADITIONAL IDS',style={'color':'#FF6D00','fontFamily':MONO,'fontSize':'9px','fontWeight':'700','marginBottom':'5px'}),
                    html.Div(f"Status:  {'FIRED' if S['trad_fired'] else 'No alert'}",style={'color':'#FF6D00' if S['trad_fired'] else DIM,'fontFamily':MONO,'fontSize':'9.5px'}),
                    html.Div(f"Step:  {S['trad_step'] if S['trad_fired'] else '--'}",style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','marginTop':'3px'}),
                    html.Div('Detects:  EXECUTION PHASE',style={'color':DIM,'fontFamily':MONO,'fontSize':'9px','marginTop':'3px'}),
                ],style={'flex':1,'padding':'10px','border':'1px solid #FF6D0033','borderRadius':'4px','background':'#FF6D0006'}),
            ],style={'display':'flex','marginBottom':'8px'}),
            html.Div(
                f'Lead time advantage:  {adv} steps earlier' if adv else 'Traditional IDS has not fired yet.',
                style={'color':AMBER,'fontFamily':MONO,'fontSize':'9.5px','textAlign':'center','padding':'4px 0'}
            ),
        ],className='hz',style={'background':CARD,'border':f'1px solid {AMBER}33','borderRadius':'6px','padding':'12px','marginTop':'8px'})
    else:
        compare_box=html.Div()

    log_items=[
        html.Div(e,className='log-row',style={'fontFamily':MONO,'color':DIM,'fontSize':'8.5px'})
        for e in S['event_log']
    ]

    tp=S['total_packets'];fp=S['flagged_packets']
    pct=(fp/tp*100) if tp>0 else 0
    hdr=f'Packets: {tp}  ·  Flagged: {fp} ({pct:.1f}%)  ·  {S["pps"]} pkt/s  ·  Step: {S["step"]}'

    return (net_items,anom_items,cusum_fig,rec_fig,
            status_box,threat_box,timer_box,gauges_box,
            profile_box,tl_items,compare_box,hdr,log_items)


if __name__=='__main__':
    app.run(debug=False,port=8050)