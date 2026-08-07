"""
app.py — Streamlit Dashboard for the AI Cyber Discovery Engine.

The human interface. Judges, professors, and recruiters see this.

Layout:
    Sidebar          — Run controls, stats, source selector
    Tab 1: Overview  — Threat summary cards, severity chart, timeline
    Tab 2: Alerts    — Full alert list with AI explanations
    Tab 3: Attack Graph — Interactive NetworkX/Plotly entity graph
    Tab 4: MITRE Map — ATT&CK kill-chain heatmap

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

# Engine imports
from engine.ingest import load_and_normalize
from engine.enrich import enrich_events
from engine.analyze import run_analysis
from engine.store import init_db, save_results, load_results, clear_db, get_stats
from engine.models import Severity, ThreatAlert, AnalysisResult

# ── Page Configuration ────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Cyber Discovery Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)

# ── Custom CSS ────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
        border-right: 1px solid #1e3a5f;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1f35 0%, #112240 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 100, 255, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 100, 255, 0.15);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Alert cards */
    .alert-card {
        background: linear-gradient(135deg, #0d1f35 0%, #0f2744 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }

    .alert-card:hover {
        transform: translateX(4px);
    }

    .alert-card.critical { border-left-color: #ef4444; }
    .alert-card.high     { border-left-color: #f97316; }
    .alert-card.medium   { border-left-color: #eab308; }
    .alert-card.low      { border-left-color: #22c55e; }

    .severity-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .sev-critical { background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }
    .sev-high     { background: rgba(249,115,22,0.2); color: #f97316; border: 1px solid rgba(249,115,22,0.4); }
    .sev-medium   { background: rgba(234,179,8,0.2);  color: #eab308; border: 1px solid rgba(234,179,8,0.4); }
    .sev-low      { background: rgba(34,197,94,0.2);  color: #22c55e; border: 1px solid rgba(34,197,94,0.4); }
    .sev-info     { background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid rgba(99,102,241,0.4); }

    .risk-score-display {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Tabs */
    [data-testid="stTab"] {
        background: transparent !important;
        color: #64748b !important;
        border-bottom: 2px solid transparent !important;
        font-weight: 500 !important;
    }

    [aria-selected="true"][data-testid="stTab"] {
        color: #3b82f6 !important;
        border-bottom-color: #3b82f6 !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #0d1f35 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
    }

    /* Code blocks for explanation */
    pre {
        background: #050e1a !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
        color: #94d2bd !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.82rem !important;
        line-height: 1.6 !important;
        padding: 16px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 10px 20px;
        transition: opacity 0.2s ease, transform 0.2s ease;
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    h1 { color: #e2e8f0; font-weight: 700; }
    h2 { color: #cbd5e1; font-weight: 600; }
    h3 { color: #94a3b8; font-weight: 600; }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────

SEVERITY_ORDER = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
SEVERITY_COLORS = {
    "critical": "#ef4444",
    "high":     "#f97316",
    "medium":   "#eab308",
    "low":      "#22c55e",
    "info":     "#818cf8",
}
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "xaxis": {"gridcolor": "#1e3a5f", "linecolor": "#1e3a5f"},
        "yaxis": {"gridcolor": "#1e3a5f", "linecolor": "#1e3a5f"},
    }
}


def _severity_badge(severity: str) -> str:
    return f'<span class="severity-badge sev-{severity}">{severity.upper()}</span>'


def _risk_color(score: float) -> str:
    if score >= 80: return "#ef4444"
    if score >= 60: return "#f97316"
    if score >= 40: return "#eab308"
    return "#22c55e"


# ── Sidebar ───────────────────────────────────────────────────────────

def render_sidebar() -> tuple[bool, str]:
    """Render sidebar controls. Returns (run_clicked, log_source)."""
    with st.sidebar:
        # Logo / title
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px;">
            <div style="font-size:2.5rem;">🛡️</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:8px;">
                AI Cyber Discovery Engine
            </div>
            <div style="font-size:0.72rem; color:#475569; margin-top:4px; letter-spacing:0.1em;">
                CAPSTONE PROJECT v0.1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown("### ⚙️ Analysis Controls")
        log_source = st.selectbox(
            "Log Source",
            ["All Sources", "auth_logs.json", "firewall_logs.json", "network_logs.json"],
            help="Select a specific log file or analyse all sources together",
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        run_clicked = st.button("🔍 Run Analysis", use_container_width=True)

        if st.button("🗑️ Clear Results", use_container_width=True):
            clear_db()
            st.success("Results cleared.")
            st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Quick stats
        try:
            stats = get_stats()
            st.markdown("### 📊 Quick Stats")
            st.metric("Total Alerts", stats["total_alerts"])
            st.metric("High-Risk Alerts", stats["high_risk_alerts"])
            if stats["last_run"]:
                ts = stats["last_run"][:19].replace("T", " ")
                st.markdown(f"**Last Run:** `{ts}`")
                st.markdown(f"**Events Processed:** `{stats['total_events_processed']}`")
        except Exception:
            pass

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#334155; font-size:0.7rem; text-align:center;">
            AI Cyber Discovery Engine<br>
            Capstone Project · 2025<br>
            Powered by NetworkX · scikit-learn
        </div>
        """, unsafe_allow_html=True)

    return run_clicked, log_source


# ── Tab 1: Overview ───────────────────────────────────────────────────

def render_overview(result: AnalysisResult) -> None:
    alerts = result.alerts

    st.markdown("## 🌐 Threat Overview")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events Analyzed", result.total_events)
    c2.metric("Threats Detected", len(alerts))
    c3.metric("High Risk", result.high_risk_count, delta=f"+{result.high_risk_count}" if result.high_risk_count else None, delta_color="inverse")
    top_score = f"{alerts[0].risk_score.total:.1f}" if alerts else "—"
    c4.metric("Highest Risk Score", top_score)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not alerts:
        st.info("No alerts above threshold. Run the analysis to generate results.")
        return

    col_left, col_right = st.columns([1, 1])

    # Severity donut chart
    with col_left:
        st.markdown("### Severity Distribution")
        sev_counts = defaultdict(int)
        for a in alerts:
            sev_counts[a.severity.value] += 1

        labels = list(sev_counts.keys())
        values = list(sev_counts.values())
        colors = [SEVERITY_COLORS.get(l, "#64748b") for l in labels]

        fig = go.Figure(go.Pie(
            labels=[l.upper() for l in labels],
            values=values,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
            textfont=dict(color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Inter"),
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8")),
            margin=dict(t=20, b=20, l=0, r=0),
            height=300,
            annotations=[dict(
                text=f"<b>{len(alerts)}</b><br><span style='font-size:12px'>ALERTS</span>",
                x=0.5, y=0.5, font_size=18, font_color="#e2e8f0", showarrow=False
            )],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk score bar chart
    with col_right:
        st.markdown("### Risk Scores by Alert")
        df = pd.DataFrame({
            "Threat": [a.threat_type[:35] + "…" if len(a.threat_type) > 35 else a.threat_type
                       for a in alerts],
            "Risk Score": [a.risk_score.total for a in alerts],
            "Severity": [a.severity.value for a in alerts],
        })
        fig2 = px.bar(
            df, x="Risk Score", y="Threat", orientation="h",
            color="Severity",
            color_discrete_map=SEVERITY_COLORS,
            text="Risk Score",
        )
        fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Inter"),
            xaxis=dict(range=[0, 105], title="Risk Score (0–100)", gridcolor="#1e3a5f", linecolor="#1e3a5f"),
            yaxis=dict(title="", gridcolor="#1e3a5f", linecolor="#1e3a5f"),
            showlegend=False,
            margin=dict(t=20, b=20, l=0, r=0),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # MITRE Kill-chain progression
    st.markdown("### 🗡️ Kill-Chain Progression")
    all_tactics: list[tuple[int, str, str]] = []
    for alert in alerts:
        for tech in alert.mitre_techniques:
            all_tactics.append((tech.tactic_order, tech.tactic, tech.technique_id))

    if all_tactics:
        unique_tactics = sorted(set((o, t) for o, t, _ in all_tactics), key=lambda x: x[0])
        cols = st.columns(len(unique_tactics))
        for i, (order, tactic) in enumerate(unique_tactics):
            with cols[i]:
                icon = ["🔍", "🛠️", "🔑", "📡", "⬆️", "↔️", "📤", "💥"][min(order - 1, 7)]
                st.markdown(f"""
                <div style="text-align:center; background: linear-gradient(135deg, #0d1f35, #112240);
                     border-radius:10px; padding:14px 8px; border:1px solid #1e3a5f;">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div style="font-size:0.65rem; color:#64748b; font-weight:700;
                         letter-spacing:0.08em; margin-top:6px;">PHASE {order}</div>
                    <div style="font-size:0.75rem; color:#cbd5e1; font-weight:600;
                         margin-top:4px;">{tactic}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Tab 2: Alerts ─────────────────────────────────────────────────────

def render_alerts(result: AnalysisResult) -> None:
    alerts = result.alerts
    st.markdown("## 🚨 Threat Alerts")

    if not alerts:
        st.info("No alerts detected. Run the analysis pipeline first.")
        return

    # Filter controls
    col1, col2 = st.columns([1, 3])
    with col1:
        min_score = st.slider("Min Risk Score", 0, 100, 0, 5)
    with col2:
        sev_filter = st.multiselect(
            "Severity Filter",
            options=["critical", "high", "medium", "low", "info"],
            default=["critical", "high", "medium"],
        )

    filtered = [
        a for a in alerts
        if a.risk_score.total >= min_score and a.severity.value in sev_filter
    ]

    st.markdown(f"*Showing {len(filtered)} of {len(alerts)} alerts*")

    for alert in filtered:
        sev = alert.severity.value
        score = alert.risk_score.total

        with st.container():
            st.markdown(f"""
            <div class="alert-card {sev}">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div style="font-size:1.05rem; font-weight:700; color:#e2e8f0; margin-bottom:8px;">
                            {alert.title}
                        </div>
                        {_severity_badge(sev)}
                    </div>
                    <div style="text-align:right;">
                        <div class="risk-score-display">{score:.1f}</div>
                        <div style="font-size:0.7rem; color:#475569; font-weight:600;">RISK SCORE</div>
                    </div>
                </div>
                <div style="margin-top:12px; display:flex; gap:16px; flex-wrap:wrap;">
                    <span style="font-size:0.75rem; color:#64748b;">
                        📡 <b style="color:#94a3b8;">{len(alert.source_ips)}</b> source IP(s)
                    </span>
                    <span style="font-size:0.75rem; color:#64748b;">
                        🎯 <b style="color:#94a3b8;">{len(alert.dest_ips)}</b> destination(s)
                    </span>
                    <span style="font-size:0.75rem; color:#64748b;">
                        📋 <b style="color:#94a3b8;">{len(alert.event_ids)}</b> correlated events
                    </span>
                    <span style="font-size:0.75rem; color:#64748b;">
                        🧬 <b style="color:#94a3b8;">{len(alert.mitre_techniques)}</b> MITRE technique(s)
                    </span>
                    <span style="font-size:0.75rem; color:#64748b;">
                        ⚡ Anomaly: <b style="color:#94a3b8;">{alert.anomaly_score:.2f}</b>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🤖 View AI Explanation & Risk Breakdown"):
                if alert.mitre_techniques:
                    st.markdown("**MITRE ATT&CK Techniques:**")
                    tech_cols = st.columns(min(len(alert.mitre_techniques), 4))
                    for i, tech in enumerate(alert.mitre_techniques):
                        with tech_cols[i % 4]:
                            st.markdown(f"""
                            <div style="background:#050e1a; border:1px solid #1e3a5f;
                                 border-radius:8px; padding:10px; text-align:center;">
                                <div style="color:#3b82f6; font-weight:700; font-size:0.85rem;">
                                    {tech.technique_id}
                                </div>
                                <div style="color:#94a3b8; font-size:0.72rem; margin-top:4px;">
                                    {tech.technique_name}
                                </div>
                                <div style="color:#475569; font-size:0.65rem; margin-top:2px;">
                                    {tech.tactic}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("**AI Reasoning:**")
                st.code(alert.explanation, language="")

                # Risk score breakdown chart
                st.markdown("**Risk Score Breakdown:**")
                rs = alert.risk_score
                components = {
                    "Severity": rs.severity_component,
                    "Anomaly": rs.anomaly_component,
                    "MITRE Stage": rs.mitre_stage_component,
                    "Frequency": rs.frequency_component,
                    "Asset Criticality": rs.asset_criticality_component,
                }
                fig = go.Figure(go.Bar(
                    x=list(components.values()),
                    y=list(components.keys()),
                    orientation="h",
                    marker=dict(
                        color=list(components.values()),
                        colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#8b5cf6"]],
                    ),
                    text=[f"{v:.1f}" for v in components.values()],
                    textposition="outside",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1", family="Inter"),
                    xaxis=dict(range=[0, 30], title="Score Contribution", gridcolor="#1e3a5f", linecolor="#1e3a5f"),
                    yaxis=dict(title="", gridcolor="#1e3a5f", linecolor="#1e3a5f"),
                    height=220,
                    margin=dict(t=10, b=10, l=0, r=40),
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("🛡️ Mitigation Recommendations"):
                for i, step in enumerate(alert.mitigation, 1):
                    st.markdown(f"""
                    <div style="display:flex; gap:12px; margin-bottom:10px;
                         background:#050e1a; border-radius:8px; padding:12px;
                         border:1px solid #1e3a5f;">
                        <div style="background:linear-gradient(135deg,#1d4ed8,#7c3aed);
                             border-radius:50%; width:26px; height:26px; display:flex;
                             align-items:center; justify-content:center; flex-shrink:0;
                             font-size:0.75rem; font-weight:700; color:white;">{i}</div>
                        <div style="color:#cbd5e1; font-size:0.85rem; line-height:1.5;">{step}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("")


# ── Tab 3: Attack Graph ───────────────────────────────────────────────

def render_attack_graph(result: AnalysisResult) -> None:
    st.markdown("## 🕸️ Attack Entity Graph")

    if not result.alerts:
        st.info("No alerts to visualize. Run the analysis pipeline first.")
        return

    st.markdown(
        "*Each node is an entity (IP). Edges represent events between entities. "
        "Edge weight reflects event frequency. Hover for details.*"
    )

    # Build graph from all alerts
    G = nx.Graph()
    edge_data: dict[tuple, dict] = {}

    for alert in result.alerts:
        src_ips = alert.source_ips
        dst_ips = alert.dest_ips
        all_ips = list(set(src_ips + dst_ips))

        for ip in all_ips:
            if not G.has_node(ip):
                G.add_node(ip, alerts=[], role="unknown")
            G.nodes[ip]["alerts"].append(alert.title)

        for src in src_ips:
            for dst in dst_ips:
                if src != dst:
                    key = (min(src, dst), max(src, dst))
                    if key not in edge_data:
                        edge_data[key] = {"weight": 0, "threats": []}
                    edge_data[key]["weight"] += alert.risk_score.total
                    edge_data[key]["threats"].append(alert.threat_type)

    for (src, dst), data in edge_data.items():
        G.add_edge(src, dst, weight=data["weight"], threats=data["threats"])

    if len(G.nodes) == 0:
        st.warning("No entity relationships found in current alerts.")
        return

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Build Plotly trace
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#1e3a5f"),
        hoverinfo="none",
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = list(G.nodes())
    node_text = [
        f"<b>{n}</b><br>Alerts: " + "<br>".join(G.nodes[n].get("alerts", [])[:3])
        for n in G.nodes()
    ]
    node_colors = [
        "#ef4444" if any("Brute" in a or "Ransomware" in a or "Exfil" in a
                         for a in G.nodes[n].get("alerts", []))
        else "#3b82f6"
        for n in G.nodes()
    ]
    node_sizes = [18 + G.degree(n) * 6 for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(color="#94a3b8", size=10),
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(color="#0a0e1a", width=2),
            symbol="circle",
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter"),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, linecolor="rgba(0,0,0,0)"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, linecolor="rgba(0,0,0,0)"),
        height=500,
        margin=dict(t=20, b=20, l=20, r=20),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display:flex; gap:20px; justify-content:center; margin-top:8px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:14px; height:14px; border-radius:50%; background:#ef4444;"></div>
            <span style="color:#94a3b8; font-size:0.78rem;">High-risk entity</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:14px; height:14px; border-radius:50%; background:#3b82f6;"></div>
            <span style="color:#94a3b8; font-size:0.78rem;">Standard entity</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:2px; height:14px; background:#1e3a5f;"></div>
            <span style="color:#94a3b8; font-size:0.78rem;">Event relationship</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Tab 4: MITRE ATT&CK Heatmap ──────────────────────────────────────

def render_mitre_heatmap(result: AnalysisResult) -> None:
    st.markdown("## 🗺️ MITRE ATT&CK Coverage")

    if not result.alerts:
        st.info("No alerts to map. Run the analysis pipeline first.")
        return

    # Build technique frequency matrix
    tech_data: dict[str, dict] = {}
    for alert in result.alerts:
        for tech in alert.mitre_techniques:
            key = tech.technique_id
            if key not in tech_data:
                tech_data[key] = {
                    "id": tech.technique_id,
                    "name": tech.technique_name,
                    "tactic": tech.tactic,
                    "order": tech.tactic_order,
                    "count": 0,
                    "max_risk": 0.0,
                }
            tech_data[key]["count"] += 1
            tech_data[key]["max_risk"] = max(tech_data[key]["max_risk"], alert.risk_score.total)

    if not tech_data:
        st.warning("No MITRE techniques mapped in current alerts.")
        return

    df = pd.DataFrame(list(tech_data.values())).sort_values("order")

    # Kill-chain stage cards
    st.markdown("### ATT&CK Kill-Chain Coverage")
    tactics_ordered = sorted(df[["tactic", "order"]].drop_duplicates().values.tolist(), key=lambda x: x[1])

    cols = st.columns(len(tactics_ordered))
    for i, (tactic, order) in enumerate(tactics_ordered):
        tactic_df = df[df["tactic"] == tactic]
        with cols[i]:
            techniques_html = "".join(
                f'<div style="font-size:0.7rem; color:#64748b; margin-top:4px;">'
                f'{row["id"]}</div>'
                for _, row in tactic_df.iterrows()
            )
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0d1f35,#112240);
                 border:1px solid #1e3a5f; border-radius:10px; padding:14px 10px;
                 text-align:center; min-height:120px;">
                <div style="font-size:0.65rem; color:#3b82f6; font-weight:700;
                     letter-spacing:0.1em; text-transform:uppercase;">Phase {order}</div>
                <div style="font-size:0.78rem; color:#e2e8f0; font-weight:600;
                     margin-top:6px;">{tactic}</div>
                <div style="font-size:0.7rem; color:#ef4444; font-weight:700;
                     margin-top:4px;">{len(tactic_df)} technique(s)</div>
                {techniques_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Techniques table with risk scores
    st.markdown("### Technique Risk Map")
    fig = px.bar(
        df, x="max_risk", y="name", color="max_risk", orientation="h",
        color_continuous_scale=["#1e3a5f", "#3b82f6", "#8b5cf6", "#ef4444"],
        range_color=[0, 100],
        text="id",
        labels={"max_risk": "Max Risk Score", "name": "Technique"},
    )
    fig.update_traces(textposition="outside", textfont=dict(color="#94a3b8", size=11))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter"),
        xaxis=dict(range=[0, 115], gridcolor="#1e3a5f", linecolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f"),
        coloraxis_showscale=False,
        height=max(300, len(df) * 50),
        margin=dict(t=20, b=20, l=0, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Main App ──────────────────────────────────────────────────────────

def main() -> None:
    # Initialize DB
    init_db()

    # Sidebar
    run_clicked, log_source = render_sidebar()

    # Handle run action
    if run_clicked:
        log_dir = Path("data/sample_logs")

        with st.spinner("🔍 Running AI analysis pipeline..."):
            try:
                # Stage 1+2: Ingest and normalize
                st.toast("📥 Loading and normalizing logs...", icon="📥")
                events = load_and_normalize(log_dir)

                # Stage 3: Enrich
                st.toast("🔬 Enriching events with threat intelligence...", icon="🔬")
                enriched = enrich_events(events)

                # Stage 4: AI Reasoning
                st.toast("🤖 Running AI reasoning engine...", icon="🤖")
                result = run_analysis(enriched)

                # Stage 5: Store
                save_results(result)
                st.toast(f"✅ Analysis complete — {len(result.alerts)} threats detected!", icon="✅")
                st.rerun()

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)

    # Load results from DB
    result = load_results()

    # Header
    st.markdown("""
    <div style="padding: 20px 0 10px;">
        <h1 style="margin:0; font-size:2rem;">
            🛡️ AI Cyber Discovery Engine
        </h1>
        <p style="color:#475569; margin-top:6px; font-size:0.9rem;">
            AI-powered threat correlation, risk scoring, and explainable security intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if result is None:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px;
             background:linear-gradient(135deg,#0d1f35,#112240);
             border-radius:16px; border:1px solid #1e3a5f; margin:20px 0;">
            <div style="font-size:3rem; margin-bottom:16px;">🔍</div>
            <div style="font-size:1.3rem; font-weight:600; color:#e2e8f0; margin-bottom:8px;">
                No Analysis Results Yet
            </div>
            <div style="color:#64748b; font-size:0.9rem; max-width:400px; margin:0 auto;">
                Click <strong style="color:#3b82f6;">Run Analysis</strong> in the sidebar to
                start the AI pipeline and discover threats in the sample logs.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Overview",
        "🚨 Alerts & Explanations",
        "🕸️ Attack Graph",
        "🗺️ MITRE ATT&CK Map",
    ])

    with tab1:
        render_overview(result)

    with tab2:
        render_alerts(result)

    with tab3:
        render_attack_graph(result)

    with tab4:
        render_mitre_heatmap(result)


if __name__ == "__main__":
    main()
