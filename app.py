# ============================================================================
#  K R A K E N  //  BLAST SHIELD  -  cascade-failure digital twin
#  pip install streamlit networkx pandas pyyaml      ->      streamlit run app.py
# ============================================================================
import json
import math
import random
import time

import networkx as nx
import pandas as pd

# ================================ ENGINE ====================================
TICK_S = 5        # seconds per simulation tick
T = 24            # ticks per run
FAULT_T = 2       # tick at which the fault lands
RETRIES = 2       # client retries per failed call (the source of retry storms)
TIMEOUT_MS = 800

DEFAULT_EDGES = [  # (caller, dependency, share of caller's traffic that hits it)
    ("API Gateway", "Auth Service", 1.0),
    ("API Gateway", "Payment Service", 0.4),
    ("Auth Service", "Users DB", 1.0),
    ("Payment Service", "Stripe API", 1.0),
    ("Payment Service", "Orders DB", 1.0),
]
DEFAULT_LAT = {"API Gateway": 11, "Auth Service": 14, "Payment Service": 19,
               "Users DB": 2, "Stripe API": 82, "Orders DB": 4}

NONE = "None (let it crash)"
SHIELD = "KRAKEN Shield (all three)"
STRATS = {  # name -> (circuit breaker, autoscale, hot failover)
    NONE: (False, False, False),
    "Circuit breaking": (True, False, False),
    "Auto-scaling": (False, True, False),
    "Hot failover": (False, False, True),
    SHIELD: (True, True, True),
}


def make_topo(edges, lat=None, fallback=0.6):
    G = nx.DiGraph()
    for a, b, w in edges:
        G.add_edge(a, b, w=float(w))
    if G.number_of_nodes() == 0:
        raise ValueError("Topology has no services")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Topology has a dependency cycle - please break it")
    entries = [n for n in G if G.in_degree(n) == 0]
    base = {n: 0.0 for n in G}
    for n in entries:
        base[n] = 1000.0
    for n in nx.topological_sort(G):
        for d in G.successors(n):
            base[d] += base[n] * G[n][d]["w"]
    cap = {n: max(base[n] * 1.6, 100.0) for n in G}
    blat = {n: (lat or {}).get(n, 20) for n in G}
    return dict(G=G, entries=entries, base=base, cap=cap, lat=blat, fallback=fallback)


def simulate(topo, fault=None, strat=(False, False, False), ticks=T):
    """Tick-synchronous cascade simulation. fault = (node, 'crash'|'brownout'|'surge')."""
    G, entries, base = topo["G"], topo["entries"], topo["base"]
    cap0, blat, fb = topo["cap"], topo["lat"], topo.get("fallback", 0.6)
    breaker, autoscale, failover = strat
    nodes = list(G.nodes)
    avail = {n: 1.0 for n in nodes}
    load = dict(base)
    lat = dict(blat)
    util_prev = {n: base[n] / cap0[n] for n in nodes}
    capm = {n: 1.0 for n in nodes}
    open_edges, low = set(), {}
    hard_streak = {n: 0 for n in nodes}
    over_streak = {n: 0 for n in nodes}
    scale_at, promoted = {}, set()
    hist, events = [], []

    for t in range(ticks):
        if fault and t == FAULT_T:
            events.append((t, f"FAULT     {fault[1].upper()} on {fault[0]}"))
        # 1. own health (fault, failover)
        own = {}
        for n in nodes:
            o = 1.0
            if fault and t >= FAULT_T and n == fault[0]:
                if fault[1] == "crash":
                    o = 0.0
                elif fault[1] == "brownout":
                    o = 0.4
            hard_streak[n] = hard_streak[n] + 1 if o < 0.5 else 0
            if failover and fault and n == fault[0] and n not in promoted and hard_streak[n] >= 3:
                promoted.add(n)
                events.append((t, f"FAILOVER  {n}: standby promoted"))
            if n in promoted:
                o = 0.97
            own[n] = o
        # 2. offered load (previous-tick availability => retry amplification)
        newload = {n: 0.0 for n in nodes}
        surge = 3.0 if (fault and fault[1] == "surge" and t >= FAULT_T) else 1.0
        for n in entries:
            newload[n] += base[n] * surge
        for c, d, e in G.edges(data=True):
            shed = min(1.0, 1.0 / max(util_prev[c], 1e-9))
            calls = load[c] * shed * e["w"]
            if (c, d) in open_edges:
                newload[d] += calls * 0.03          # half-open probes only
            else:
                newload[d] += calls * (1 + RETRIES * (1 - avail[d]))
        # 3. capacity, autoscaling lag, overload shedding
        util = {}
        for n in nodes:
            if autoscale and n in scale_at and t >= scale_at[n]:
                capm[n] = min(capm[n] * 2, 4.0)
                del scale_at[n]
                events.append((t, f"SCALE-UP  {n}: capacity x{capm[n]:.0f}"))
            util[n] = newload[n] / (cap0[n] * capm[n])
            over_streak[n] = over_streak[n] + 1 if util[n] > 1 else 0
            if autoscale and over_streak[n] >= 2 and n not in scale_at and capm[n] < 4:
                scale_at[n] = t + 3
                events.append((t, f"AUTOSCALE {n}: overload, provisioning (3 tick lag)"))
            if util[n] > 1:
                own[n] *= util[n] ** -1.3
        # 4. availability + latency, one hop of propagation per tick
        newav, newlat = {}, {}
        for n in nodes:
            a = own[n]
            l = blat[n] * (1 + 0.5 * min(util[n], 3) ** 2)
            for d in G.successors(n):
                w = G[n][d]["w"]
                if (n, d) in open_edges:
                    a *= 1 - w * (1 - fb)           # fail fast + fallback (cache/stale)
                    l += w * 5
                else:
                    succ = 1 - (1 - avail[d]) ** (1 + RETRIES)
                    a *= 1 - w * (1 - succ)
                    l += w * (avail[d] * lat[d] + (1 - avail[d]) * min(TIMEOUT_MS * (1 + RETRIES), 2400))
            newav[n], newlat[n] = a, l
        # 5. circuit-breaker state machine
        if breaker:
            for c, d in G.edges:
                if (c, d) in open_edges:
                    if newav[d] >= 0.9:
                        open_edges.discard((c, d))
                        low[(c, d)] = 0
                        events.append((t, f"BREAKER   {c} -> {d}: CLOSED (dependency recovered)"))
                else:
                    low[(c, d)] = low.get((c, d), 0) + 1 if newav[d] < 0.5 else 0
                    if low[(c, d)] >= 2:
                        open_edges.add((c, d))
                        events.append((t, f"BREAKER   {c} -> {d}: OPEN (dependency at {newav[d]:.0%})"))
        avail, lat, load, util_prev = newav, newlat, newload, util
        hist.append(dict(
            t=t, avail=dict(avail), lat=dict(lat), load=dict(load), util=dict(util),
            open=set(open_edges),
            success=sum(avail[e] for e in entries) / len(entries),
            latency=sum(lat[e] for e in entries) / len(entries)))
    return dict(hist=hist, events=cascade_log(topo, hist, events))


def cascade_log(topo, hist, events):
    G, out = topo["G"], list(events)
    for t in range(FAULT_T, len(hist)):
        for n in G.nodes:
            a, prev = hist[t]["avail"][n], hist[t - 1]["avail"][n]
            if a < 0.9 <= prev:
                culprits = [d for d in G.successors(n) if hist[t - 1]["avail"][d] < 0.9]
                if culprits:
                    cause = "cascading from " + ", ".join(culprits)
                elif hist[t]["util"][n] > 1:
                    cause = f"saturation (util {hist[t]['util'][n]:.0%})"
                else:
                    cause = "origin (own health failing)"
                out.append((t, f"DEGRADED  {n}: {a:.0%} <- {cause}"))
    out.sort(key=lambda x: x[0])
    return out


def summarize(hist):
    succ = [h["success"] for h in hist]
    post = succ[FAULT_T:]
    rec = None
    for t in range(FAULT_T, len(succ)):
        if all(s >= 0.95 for s in succ[t:]):
            rec = (t - FAULT_T) * TICK_S
            break
    return dict(
        loss=1 - sum(post) / len(post), worst=min(post), recovery=rec,
        peak_lat=max(h["latency"] for h in hist[FAULT_T:]),
        max_degraded=max(sum(1 for v in h["avail"].values() if v < 0.9) for h in hist))


def hypotheses(topo):
    hs = [(n, k) for n in topo["G"].nodes for k in ("crash", "brownout")]
    hs.append((topo["entries"][0], "surge"))
    return hs


def infer(topo, strat, observed, t_obs):
    """Digital-twin abduction: simulate every root-cause hypothesis, score vs observation."""
    scored = []
    for n, kind in hypotheses(topo):
        pred = simulate(topo, (n, kind), strat, t_obs + 1)["hist"][t_obs]["avail"]
        rmse = math.sqrt(sum((pred[k] - observed[k]) ** 2 for k in pred) / len(pred))
        scored.append((n, kind, rmse))
    ws = [math.exp(-r / 0.06) for _, _, r in scored]
    z = sum(ws)
    return sorted(((n, k, w / z) for (n, k, _), w in zip(scored, ws)), key=lambda x: -x[2])


def label_h(n, kind):
    return "traffic surge" if kind == "surge" else f"{n} - {kind}"


# ===== UI =====
import streamlit as st  # noqa: E402

st.set_page_config(page_title="KRAKEN // BLAST SHIELD", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;700;900&family=Geist+Mono:wght@400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #000000 !important; font-family: 'Geist', sans-serif !important; color: #f8fafc; }
    code, pre, [data-testid="stCodeBlock"] { font-family: 'Geist Mono', monospace !important; }
    div[data-testid="stMetric"] { background-color: #09090b !important; border: 1px solid #18181b !important;
        padding: 12px 16px !important; border-radius: 6px !important; }
    [data-testid="stSidebar"] { background-color: #020205 !important; border-right: 1px solid #18181b !important; }
    .micro-tag { font-family: 'Geist Mono', monospace; font-size: 0.75rem; letter-spacing: 0.15em;
        text-transform: uppercase; color: #71717a; margin: 0 0 6px 0; }
    .verdict { background: linear-gradient(180deg, #021a0c 0%, #000 100%); border: 1px solid #166534;
        border-radius: 8px; padding: 14px 18px; margin: 8px 0 14px 0; position: relative; }
    .verdict::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: #22c55e; box-shadow: 0 0 10px #22c55e; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<p class='micro-tag'>SYSTEM PREFERENCES</p>", unsafe_allow_html=True)
ui_px = st.sidebar.slider("Interface scale (px)", 12, 20, 15)
st.markdown(f"<style>html {{ font-size: {ui_px}px; }}</style>", unsafe_allow_html=True)

st.markdown("""
<div style="margin: -20px 0 28px 0; border-bottom: 1px solid #18181b; padding-bottom: 26px;">
  <h1 style="font-weight:900; font-size: 4.2rem; letter-spacing: 0.18em; line-height: 1.1; margin: 0; color:#fff; text-transform: uppercase;">
    K R A K E N <span style="font-weight:300; letter-spacing:0.02em; color:#6366f1;">// BLAST SHIELD</span></h1>
  <p style="color:#a1a1aa; margin-top: 12px; font-size: 1.15rem; letter-spacing: 0.02em;">
    A digital twin of your infrastructure: simulate cascades, infer the root cause from symptoms, and stress-test containment before production does.</p>
</div>
""", unsafe_allow_html=True)


def parse_upload(f):
    raw = f.getvalue().decode("utf-8", "ignore")
    edges = []
    if f.name.lower().endswith(".json"):
        for a, b, *w in json.loads(raw)["edges"]:
            edges.append((a, b, float(w[0]) if w else 1.0))
    else:
        import yaml
        for svc, cfg in (yaml.safe_load(raw).get("services") or {}).items():
            dep = (cfg or {}).get("depends_on", [])
            for d in (list(dep) if isinstance(dep, (list, dict)) else []):
                edges.append((svc, d, 1.0))
    if not edges:
        raise ValueError("no dependencies found")
    return edges


with st.sidebar.expander("BRING YOUR OWN TOPOLOGY"):
    up = st.file_uploader("docker-compose.yml or topology.json", type=["yml", "yaml", "json"])
    st.caption('JSON format: {"edges": [["web","api",1.0], ["api","db"]]}')

edges, lat = DEFAULT_EDGES, DEFAULT_LAT
if up is not None:
    try:
        edges, lat = parse_upload(up), None
    except Exception as ex:  # noqa: BLE001
        st.sidebar.error(f"Could not parse topology: {ex}")

st.sidebar.markdown("<div style='margin: 14px 0; border-top: 1px solid #18181b;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='micro-tag'>CHAOS GENERATION</p>", unsafe_allow_html=True)
fb_pct = st.sidebar.slider("Fallback hit-rate when a breaker is open (%)", 0, 100, 60,
                           help="Share of calls still served from cache/stale data once a circuit breaker opens.")
try:
    topo = make_topo(edges, lat, fb_pct / 100)
except ValueError as ex:
    st.sidebar.error(str(ex))
    topo = make_topo(DEFAULT_EDGES, DEFAULT_LAT, fb_pct / 100)

nodes = list(topo["G"].nodes)
fault_kind = st.sidebar.selectbox("Fault type", ["crash", "brownout", "surge"],
                                  format_func={"crash": "Hard crash", "brownout": "Brownout (60% errors)",
                                               "surge": "Traffic surge x3 (hits entry)"}.get)
target = st.sidebar.selectbox("Target service", nodes, disabled=(fault_kind == "surge"))
if fault_kind == "surge":
    target = topo["entries"][0]
st.sidebar.markdown("<div style='margin: 14px 0; border-top: 1px solid #18181b;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='micro-tag'>AUTONOMOUS STRATEGY LAYER</p>", unsafe_allow_html=True)
strat_name = st.sidebar.selectbox("Orchestration rule", list(STRATS), index=list(STRATS).index(SHIELD))

fault = (target, fault_kind)
run = simulate(topo, fault, STRATS[strat_name])
run0 = simulate(topo, fault, STRATS[NONE])
S, S0 = summarize(run["hist"]), summarize(run0["hist"])
secs = [(t - FAULT_T) * TICK_S for t in range(T)]


def gv(container, dot):
    try:
        container.graphviz_chart(dot, width="stretch")
    except TypeError:
        container.graphviz_chart(dot, use_container_width=True)


def node_style(a):
    if a >= 0.95:
        return "#052e16", "#22c55e"
    if a >= 0.75:
        return "#422006", "#f59e0b"
    if a >= 0.30:
        return "#431407", "#f97316"
    return "#450a0a", "#ef4444"


def make_dot(snap):
    G = topo["G"]
    L = ['digraph G {', 'bgcolor="transparent"; rankdir=LR; nodesep=0.5; ranksep=1.0;',
         'node [shape=box style="rounded,filled" fontname="Helvetica" fontcolor="white" penwidth=2 margin="0.22,0.14"];',
         'edge [fontname="Helvetica" fontsize=10];']
    for n in G.nodes:
        a = snap["avail"][n]
        fill, line = node_style(a)
        extra = f"\\n{snap['lat'][n]:.0f} ms" if snap.get("lat") else ""
        L.append(f'"{n}" [label="{n}\\n{a:.0%}{extra}" fillcolor="{fill}" color="{line}"];')
    for c, d, e in G.edges(data=True):
        opened = (c, d) in snap.get("open", set())
        amp = snap["load"][d] / topo["base"][d] if snap.get("load") else 1.0
        col = "#ef4444" if opened else ("#f59e0b" if amp > 1.3 else "#52525b")
        lab = "BREAKER OPEN" if opened else (f"{amp:.1f}x load" if amp > 1.3 else "")
        sty = "dashed" if opened else "solid"
        L.append(f'"{c}" -> "{d}" [color="{col}" fontcolor="{col}" label="{lab}" style="{sty}" penwidth={1 + min(amp, 3):.1f}];')
    L.append("}")
    return "\n".join(L)


# ---- headline metrics (all computed, none hardcoded) ----
m1, m2, m3, m4 = st.columns(4)
m1.metric("TRAFFIC LOST (avg after fault)", f"{S['loss']:.1%}",
          None if strat_name == NONE else f"{(S0['loss'] - S['loss']) * 100:+.1f} pts vs doing nothing")
m2.metric("TIME TO RECOVER", "never" if S["recovery"] is None else f"{S['recovery']} s",
          None if S0["recovery"] is None or S["recovery"] is None or strat_name == NONE
          else f"{S0['recovery'] - S['recovery']:+d} s vs doing nothing", delta_color="inverse")
m3.metric("PEAK ENTRY LATENCY", f"{S['peak_lat']:.0f} ms")
m4.metric("SERVICES DEGRADED (max)", f"{S['max_degraded']} / {len(nodes)}")
st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["CASCADE LAB", "BLIND ROOT-CAUSE", "STRATEGY ARENA", "BLAST RADIUS"])

# ------------------------------ TAB 1 ---------------------------------------
with tab1:
    cl, cr = st.columns([1.25, 1], gap="medium")
    with cl:
        t_sel = st.slider("Timeline tick (1 tick = 5 s)", 0, T - 1, FAULT_T + 4)
        play = st.button("Replay cascade")
        ph = st.empty()

        def draw(t):
            h = run["hist"][t]
            st.markdown(f"<p class='micro-tag'>T{(t - FAULT_T) * TICK_S:+d}s | traffic success {h['success']:.0%} | "
                        f"entry latency {h['latency']:.0f} ms</p>", unsafe_allow_html=True)
            gv(st, make_dot(h))

        if play:
            for t in range(T):
                with ph.container():
                    draw(t)
                time.sleep(0.35)
        else:
            with ph.container():
                draw(t_sel)
        st.caption("Amber edges = retry storm (load amplification). Dashed red = circuit breaker open.")
    with cr:
        st.markdown("<p class='micro-tag'>USER-VISIBLE SUCCESS RATE (%)</p>", unsafe_allow_html=True)
        df = pd.DataFrame({strat_name: [h["success"] * 100 for h in run["hist"]],
                           NONE: [h["success"] * 100 for h in run0["hist"]]}, index=secs)
        st.line_chart(df)
        st.markdown("<p class='micro-tag'>INCIDENT LOG (GENERATED FROM THE SIMULATION)</p>", unsafe_allow_html=True)
        st.code("\n".join(f"T{(t - FAULT_T) * TICK_S:+4d}s  {m}" for t, m in run["events"]) or "no events",
                language=None)

# ------------------------------ TAB 2 ---------------------------------------
with tab2:
    st.markdown("<p class='micro-tag'>DIGITAL-TWIN ROOT-CAUSE INFERENCE</p>", unsafe_allow_html=True)
    st.caption("A fault is injected in secret. You only see noisy monitoring symptoms. KRAKEN simulates every possible "
               "root cause on its twin and ranks which one reproduces what you observe.")
    r = st.session_state.setdefault("rca", dict(truth=None, seed=0, infer=False, revealed=False,
                                                scored=False, hits=0, rounds=0))
    if r["truth"] and r["truth"][0] not in nodes:
        r.update(truth=None, infer=False, revealed=False)
    k = st.slider("Observe N ticks after the fault", 2, 14, 6)
    t_obs = FAULT_T + k
    b1, b2, b3 = st.columns(3)
    if b1.button("Inject hidden fault"):
        r.update(truth=random.choice(hypotheses(topo)), seed=random.randint(0, 10 ** 6),
                 infer=False, revealed=False, scored=False)
    if b2.button("Run inference"):
        r["infer"] = True
    if b3.button("Reveal ground truth"):
        r["revealed"] = True

    if r["truth"] is None:
        st.info("Press **Inject hidden fault** to start a round.")
    else:
        rng = random.Random(r["seed"])
        true_av = simulate(topo, r["truth"], STRATS[strat_name], t_obs + 1)["hist"][t_obs]["avail"]
        observed = {n: min(1.0, max(0.0, a + rng.gauss(0, 0.03))) for n, a in true_av.items()}
        g1, g2 = st.columns([1.2, 1], gap="medium")
        with g1:
            st.markdown("<p class='micro-tag'>WHAT MONITORING SEES</p>", unsafe_allow_html=True)
            gv(st, make_dot(dict(avail=observed)))
        with g2:
            if r["infer"] or r["revealed"]:
                post = infer(topo, STRATS[strat_name], observed, t_obs)
                st.markdown("<p class='micro-tag'>POSTERIOR OVER ROOT CAUSES</p>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{"Hypothesis": label_h(n, kd), "Probability %": round(p * 100, 1)}
                                           for n, kd, p in post[:5]]), hide_index=True)
                agg = {}
                for n, kd, p in post:
                    agg[n] = agg.get(n, 0) + p * 100
                st.bar_chart(pd.Series(agg, name="P(root-cause service) %"))
            else:
                st.info("Press **Run inference** to let KRAKEN diagnose it.")
        if r["revealed"]:
            if not r["scored"]:
                r["scored"] = True
                r["rounds"] += 1
                r["hits"] += int((post[0][0], post[0][1]) == tuple(r["truth"]))
            rank = [(n, kd) for n, kd, _ in post].index(tuple(r["truth"])) + 1
            msg = (f"Ground truth: **{label_h(*r['truth'])}**. KRAKEN ranked it #{rank} "
                   f"(top guess: {label_h(post[0][0], post[0][1])}).")
            (st.success if rank == 1 else st.warning)(msg)
        if r["rounds"]:
            st.caption(f"Session accuracy: {r['hits']}/{r['rounds']} rounds diagnosed correctly at rank 1.")

# ------------------------------ TAB 3 ---------------------------------------
with tab3:
    st.markdown("<p class='micro-tag'>STRATEGY ARENA - SAME FAULT, EVERY STRATEGY</p>", unsafe_allow_html=True)
    rows, curves = [], {}
    for name, s in STRATS.items():
        res = simulate(topo, fault, s)
        sm = summarize(res["hist"])
        curves[name] = [h["success"] * 100 for h in res["hist"]]
        rows.append({"Strategy": name, "Traffic lost %": round(sm["loss"] * 100, 1),
                     "Worst success %": round(sm["worst"] * 100, 1),
                     "Recovery": "never" if sm["recovery"] is None else f"{sm['recovery']} s",
                     "Peak latency ms": round(sm["peak_lat"])})
    best = min(rows, key=lambda x: x["Traffic lost %"])
    none_loss = next(x for x in rows if x["Strategy"] == NONE)["Traffic lost %"]
    st.markdown(f"<div class='verdict'><p class='micro-tag' style='color:#22c55e;'>[ KRAKEN RECOMMENDS ]</p>"
                f"<b>{best['Strategy']}</b> for <b>{fault_kind}</b> on <b>{target}</b>: "
                f"{best['Traffic lost %']}% traffic lost vs {none_loss}% when doing nothing.</div>",
                unsafe_allow_html=True)
    st.line_chart(pd.DataFrame(curves, index=secs))
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("<p class='micro-tag'>RESILIENCE MATRIX - % TRAFFIC LOST WHEN EACH SERVICE CRASHES</p>",
                unsafe_allow_html=True)
    mat = {}
    for name, s in STRATS.items():
        mat[name] = {n: round(summarize(simulate(topo, (n, "crash"), s)["hist"])["loss"] * 100, 1) for n in nodes}
    st.dataframe(pd.DataFrame(mat).T)

# ------------------------------ TAB 4 ---------------------------------------
with tab4:
    st.markdown("<p class='micro-tag'>BLAST RADIUS & SINGLE POINTS OF FAILURE</p>", unsafe_allow_html=True)
    raw_key = "Lost if crashed (unprotected) %"
    prot_key = f"Lost with {strat_name} %"
    rows = []
    for n in nodes:
        raw = summarize(simulate(topo, (n, "crash"), STRATS[NONE])["hist"])["loss"] * 100
        prot = summarize(simulate(topo, (n, "crash"), STRATS[strat_name])["hist"])["loss"] * 100
        rows.append({"Service": n, "Upstream services": len(nx.ancestors(topo["G"], n)),
                     raw_key: round(raw, 1), prot_key: round(prot, 1),
                     "SPOF": "YES" if raw >= 25 else "-"})
    rows.sort(key=lambda x: -x[raw_key])
    dfb = pd.DataFrame(rows)
    st.dataframe(dfb, hide_index=True)
    st.bar_chart(dfb.set_index("Service")[[raw_key, prot_key]])
    w = rows[0]
    st.markdown(f"<div class='verdict'><p class='micro-tag' style='color:#22c55e;'>[ HIGHEST RISK ]</p>"
                f"<b>{w['Service']}</b> is the largest blast radius: losing it drops "
                f"{w[raw_key]}% of traffic unprotected and {w[prot_key]}% with <b>{strat_name}</b>.</div>",
                unsafe_allow_html=True)
    

# ============================================================
# KRAKEN // BLAST SHIELD
# INTELLIGENT RECOVERY STRATEGY RECOMMENDER
# Connected to the existing cascade simulation
# ============================================================

st.markdown("---")
st.markdown("## 🛡️ LIVE RECOVERY STRATEGY CENTER")
st.caption(
    "Recommendations generated from the current digital-twin "
    "simulation, fault configuration, and resilience metrics."
)

# Evaluate every available strategy against the SAME fault.
recovery_results = {}

for strategy_name, strategy_config in STRATS.items():
    simulation = simulate(topo, fault, strategy_config)
    metrics = summarize(simulation["hist"])

    recovery_results[strategy_name] = {
        "metrics": metrics,
        "events": simulation["events"],
        "config": strategy_config,
    }

# Rank strategies by traffic loss, worst success, recovery time,
# and peak latency. These metrics come from the existing simulator.
def recovery_rank(item):
    name, result = item
    metrics = result["metrics"]

    return (
        metrics["loss"],
        -metrics["worst"],
        metrics["recovery"]
        if metrics["recovery"] is not None
        else float("inf"),
        metrics["peak_lat"],
    )

recommended_name, recommended = min(
    recovery_results.items(),
    key=recovery_rank,
)

recommended_metrics = recommended["metrics"]
current_metrics = S

# Map the chosen strategy to its protective mechanisms.
strategy_actions = {
    NONE: [
        (
            "1. Investigate the failure",
            "Inspect the affected service, dependency health, "
            "latency, and retry amplification."
        ),
        (
            "2. Reduce further impact",
            "Evaluate an approved protective strategy before "
            "the dependency failure spreads."
        ),
        (
            "3. Validate recovery",
            "Confirm traffic success and service health "
            "before considering the incident resolved."
        ),
    ],
    "Circuit breaking": [
        (
            "1. Isolate the failing dependency",
            "Use circuit breakers to stop repeated calls to "
            "an unhealthy dependency."
        ),
        (
            "2. Protect upstream services",
            "Use the configured fallback behavior where "
            "appropriate and monitor request failures."
        ),
        (
            "3. Verify dependency recovery",
            "Confirm dependency health before allowing "
            "normal traffic to resume."
        ),
    ],
    "Auto-scaling": [
        (
            "1. Identify overloaded services",
            "Inspect utilization, offered load, and "
            "latency in the dependency graph."
        ),
        (
            "2. Increase capacity",
            "Use approved autoscaling policies to add "
            "capacity when sustained overload is detected."
        ),
        (
            "3. Verify stability",
            "Check whether additional capacity improves "
            "traffic success and reduces latency."
        ),
    ],
    "Hot failover": [
        (
            "1. Confirm service failure",
            "Validate the affected service's health and "
            "the availability of its standby."
        ),
        (
            "2. Promote the standby",
            "Use an approved failover procedure when "
            "the primary service meets failure criteria."
        ),
        (
            "3. Validate the replacement",
            "Verify availability and downstream dependencies "
            "after the standby is promoted."
        ),
    ],
    SHIELD: [
        (
            "1. Break the failure cascade",
            "Use circuit breaking to reduce retry amplification "
            "from unhealthy dependencies."
        ),
        (
            "2. Scale overloaded services",
            "Use autoscaling to add capacity when the simulated "
            "load exceeds service capacity."
        ),
        (
            "3. Fail over unhealthy services",
            "Promote a standby when the configured failover "
            "criteria are met."
        ),
        (
            "4. Verify end-to-end recovery",
            "Check traffic success, latency, and degraded "
            "services throughout the dependency graph."
        ),
    ],
}

# Headline incident context.
fault_label = {
    "crash": "Hard crash",
    "brownout": "Partial service failure",
    "surge": "Traffic surge",
}.get(fault_kind, fault_kind)

st.markdown(
    f"**Detected simulation scenario:** {fault_label}  \n"
    f"**Affected service:** {target}"
)

# Current strategy vs recommended strategy.
c1, c2, c3 = st.columns(3)

c1.metric(
    "RECOMMENDED STRATEGY",
    recommended_name,
)

c2.metric(
    "PREDICTED TRAFFIC LOST",
    f"{recommended_metrics['loss']:.1%}",
    delta=(
        f"{(S['loss'] - recommended_metrics['loss']) * 100:+.1f} "
        "percentage points vs selected"
    ),
    delta_color="inverse",
)

c3.metric(
    "PREDICTED RECOVERY",
    (
        "Not within simulation"
        if recommended_metrics["recovery"] is None
        else f"{recommended_metrics['recovery']} s"
    ),
)

# Explain why this strategy was selected.
st.markdown("### 🧠 Why KRAKEN recommends this strategy")

st.markdown(
    f"""
    <div class="verdict">
      <p class="micro-tag" style="color:#22c55e;">
        [ DIGITAL TWIN RECOMMENDATION ]
      </p>
      <b>{recommended_name}</b> produced the lowest average
      post-fault traffic loss among the configured strategies
      for this simulated scenario.

      <br><br>
      Predicted traffic lost:
      <b>{recommended_metrics['loss']:.1%}</b>

      <br>
      Worst traffic success:
      <b>{recommended_metrics['worst']:.1%}</b>

      <br>
      Peak entry latency:
      <b>{recommended_metrics['peak_lat']:.0f} ms</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# Actionable recovery checklist.
st.markdown("### 📋 Recommended recovery action plan")

for i, (title, description) in enumerate(
    strategy_actions[recommended_name]
):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(description)
        st.checkbox(
            "Mark as completed",
            key=f"kraken_recovery_{recommended_name}_{i}",
        )

# Compare all strategies so the recommendation is explainable.
st.markdown("### 📊 Strategy comparison")

comparison_rows = []

for name, result in recovery_results.items():
    metrics = result["metrics"]

    comparison_rows.append({
        "Strategy": name,
        "Traffic lost (%)": round(metrics["loss"] * 100, 1),
        "Worst success (%)": round(metrics["worst"] * 100, 1),
        "Recovery time": (
            "Not recovered"
            if metrics["recovery"] is None
            else f"{metrics['recovery']} s"
        ),
        "Peak latency (ms)": round(metrics["peak_lat"]),
        "Recommended": "YES" if name == recommended_name else "",
    })

st.dataframe(
    pd.DataFrame(comparison_rows),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Simulation-based recommendation, not live production telemetry. "
    "Results update when the Streamlit app reruns. Validate recovery "
    "actions with an authorized operator before applying them to "
    "real infrastructure."
)


