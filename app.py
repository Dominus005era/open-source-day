"""WikiInsight-AI: Production Streamlit Web Dashboard.

Interactive research & data analytics dashboard on Wikimedia Foundation Wikipedia Structured Contents.
Built for the WikiClub Tech Envoy Portfolio & Open Source Day initiative.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.quality_metrics import (
    calculate_citation_density,
    calculate_structural_health,
    classify_quality_tier,
    audit_article_quality,
    evaluate_dataframe_quality,
)
from src.knowledge_graph import (
    build_knowledge_graph,
    compute_graph_metrics,
    identify_knowledge_hubs,
    extract_visual_subgraph,
)
from src.wikiclub_recommender import (
    generate_wikiclub_sprints,
)

# Wikimedia Brand Colors
WIKI_BLUE = "#006699"
WIKI_RED = "#990000"
WIKI_GREEN = "#339966"
WIKI_DARK = "#202122"
WIKI_LIGHT_BG = "#F8F9FA"

# Page Configuration
st.set_page_config(
    page_title="WikiInsight-AI | Wikimedia Structured Analytics",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, {WIKI_BLUE} 0%, #004466 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .main-header h1 {{
        color: white;
        margin: 0;
        font-size: 2.2rem;
    }}
    .main-header p {{
        color: #E0F2FE;
        margin-top: 0.5rem;
        font-size: 1.05rem;
    }}
    .metric-card {{
        background-color: white;
        border-left: 5px solid {WIKI_BLUE};
        padding: 1rem;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .badge-envoy {{
        background-color: {WIKI_GREEN};
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 48px;
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare_data():
    """Loads and caches structured Wikipedia benchmark dataset."""
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_structured_wiki.csv")
    loader = DataLoader(data_path=data_file)
    df = loader.load(max_records=500, fallback_synthetic=True)
    if "structural_health" not in df.columns:
        df = evaluate_dataframe_quality(df)
    return df


@st.cache_resource
def load_and_prepare_graph(df):
    """Builds and caches NetworkX knowledge graph and topological metrics."""
    G = build_knowledge_graph(df)
    metrics = compute_graph_metrics(G)
    hubs = identify_knowledge_hubs(G, metrics=metrics, top_n=25)
    return G, metrics, hubs


@st.cache_data
def load_sprints(df, _G, _metrics):
    """Generates and caches WikiClub sprint queues."""
    return generate_wikiclub_sprints(df, G=_G, graph_metrics=_metrics)


# Load Pipeline Data
df = load_and_prepare_data()
G, metrics, hubs_df = load_and_prepare_graph(df)
sprints = load_sprints(df, G, metrics)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/80/Wikipedia-logo-v2.svg", width=90)
    st.title("WikiInsight-AI")
    st.markdown("**WikiClub Tech Envoy Portfolio**\n*Open Source Day Initiative*")
    st.markdown("---")

    st.subheader("📊 Dataset Filter")
    # Domain categories
    all_categories = set()
    for cats in df["categories"]:
        if isinstance(cats, list):
            all_categories.update(cats)
    sorted_cats = sorted(list(all_categories))

    selected_category = st.selectbox("Category Scope", ["All Domains"] + sorted_cats)

    # Quality Tier Filter
    all_tiers = list(df["quality_tier"].unique())
    selected_tiers = st.multiselect("Quality Tiers", all_tiers, default=all_tiers)

    st.markdown("---")
    st.markdown("### 🏆 Envoy Quick Stats")
    st.markdown(f"- **Audited Articles**: `{len(df)}`")
    st.markdown(f"- **Directed Hyperlinks**: `{G.number_of_edges()}`")
    st.markdown(f"- **Critical #1Lib1Ref Tasks**: `{len(sprints['citation_deficit'])}`")
    st.markdown(f"- **Infobox Backlog**: `{len(sprints['missing_infobox'])}`")
    st.markdown(f"- **Stub Expansion Tasks**: `{len(sprints['stub_expansion'])}`")


# Apply Filters
filtered_df = df.copy()
if selected_category != "All Domains":
    filtered_df = filtered_df[filtered_df["categories"].apply(lambda cats: selected_category in cats if isinstance(cats, list) else False)]
if selected_tiers:
    filtered_df = filtered_df[filtered_df["quality_tier"].isin(selected_tiers)]

# Main Header
st.markdown("""
<div class="main-header">
    <div style="display:flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>🌐 WikiInsight-AI</h1>
            <p>Wikimedia Foundation Structured Contents: Research, Citation Rigor & Community Sprint Engine</p>
        </div>
        <div>
            <span class="badge-envoy">WikiClub Tech Envoy 2026</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Articles Audited", f"{len(filtered_df):,}", f"Scope: {selected_category}")
with kpi2:
    avg_health = filtered_df["structural_health"].mean() if not filtered_df.empty else 0.0
    st.metric("Mean Structural Health", f"{avg_health:.1f} / 100", "0-100 MoS Scale")
with kpi3:
    avg_density = filtered_df["citation_density"].mean() if not filtered_df.empty else 0.0
    st.metric("Mean Citation Density", f"{avg_density:.2f}", "per 1,000 words")
with kpi4:
    top_hub_name = hubs_df.iloc[0]["name"] if not hubs_df.empty else "N/A"
    st.metric("Top PageRank Authority Hub", top_hub_name, f"PR: {hubs_df.iloc[0]['pagerank_authority']:.4f}")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Quality & Citation Analytics",
    "🕸️ Knowledge Graph & PageRank",
    "🎯 WikiClub Sprint Recommender",
    "✍️ Live Article Scorer",
])

# ---------------------------------------------------------
# TAB 1: QUALITY & CITATION ANALYTICS
# ---------------------------------------------------------
with tab1:
    st.subheader("Encyclopedic Quality & Citation Rigor Distribution")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Quality Tier Donut Chart
        tier_counts = filtered_df["quality_tier"].value_counts().reset_index()
        tier_counts.columns = ["Quality Tier", "Count"]

        color_map = {
            "Good / Featured Article": WIKI_GREEN,
            "B Class": WIKI_BLUE,
            "C Class": "#EE8019",
            "Start Class": "#3366CC",
            "Stub Class": WIKI_RED,
        }

        fig_pie = px.pie(
            tier_counts,
            names="Quality Tier",
            values="Count",
            title="Distribution of Wikimedia Quality Tiers",
            color="Quality Tier",
            color_discrete_map=color_map,
            hole=0.45,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Structural Health Histogram
        fig_hist = px.histogram(
            filtered_df,
            x="structural_health",
            nbins=25,
            color="quality_tier",
            color_discrete_map=color_map,
            title="Structural Health Score Distribution (0-100)",
            labels={"structural_health": "Structural Health Score", "count": "Number of Articles"},
        )
        fig_hist.add_vline(x=60.0, line_dash="dash", line_color=WIKI_GREEN, annotation_text="Healthy MoS Threshold")
        fig_hist.update_layout(bargap=0.08, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Citation Density vs Word Count Scatter Plot
    st.subheader("Citation Density vs. Article Depth")
    st.markdown(
        "*Quadrant Analysis: High-word-count articles with low citation density represent critical citation deficit vulnerabilities under WP:V.*"
    )

    fig_scatter = px.scatter(
        filtered_df,
        x="word_count",
        y="citation_density",
        color="quality_tier",
        size="structural_health",
        color_discrete_map=color_map,
        hover_name="name",
        hover_data=["citations_count", "structural_health", "sections_count", "has_infobox"],
        labels={"word_count": "Article Word Count", "citation_density": "Citation Density (Citations / 1k Words)"},
        title="Word Count vs. Citation Density (Size = Structural Health)",
        height=500,
    )
    # Threshold lines
    fig_scatter.add_hline(y=2.5, line_dash="dot", line_color=WIKI_RED, annotation_text="Critical Citation Deficit (<2.5 / 1k)")
    fig_scatter.add_vline(x=1000, line_dash="dot", line_color="#888", annotation_text="Substantial Length Threshold")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Data Table View
    with st.expander("🔍 Explore Filtered Article Data Table"):
        st.dataframe(
            filtered_df[["name", "quality_tier", "word_count", "citations_count", "citation_density", "structural_health", "has_infobox"]],
            use_container_width=True,
            height=300,
        )


# ---------------------------------------------------------
# TAB 2: KNOWLEDGE GRAPH & PAGERANK HUBS
# ---------------------------------------------------------
with tab2:
    st.subheader("Inter-Article Relational Network & Authority Centrality")
    st.markdown(
        "Modeling directed hyperlinks between Wikipedia entries using NetworkX. "
        "Articles with high PageRank Authority and In-Degree act as foundational **Knowledge Hubs** across the encyclopedic graph."
    )

    g_col1, g_col2 = st.columns([1.2, 0.8])

    with g_col1:
        # Plotly 2D Subgraph Visualization
        st.markdown("#### 🌐 Knowledge Graph Topology (Top Hubs Subgraph)")
        nodes_data, edges_data = extract_visual_subgraph(G, max_nodes=50, metrics=metrics)

        # Build Edge Traces
        edge_x = []
        edge_y = []
        node_lookup = {n["id"]: n for n in nodes_data}
        for edge in edges_data:
            if edge["source"] in node_lookup and edge["target"] in node_lookup:
                src_n = node_lookup[edge["source"]]
                tgt_n = node_lookup[edge["target"]]
                edge_x.extend([src_n["x"], tgt_n["x"], None])
                edge_y.extend([src_n["y"], tgt_n["y"], None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.0, color="#CBD5E1"),
            hoverinfo="none",
            mode="lines"
        )

        # Build Node Traces
        node_x = [n["x"] for n in nodes_data]
        node_y = [n["y"] for n in nodes_data]
        node_text = [
            f"<b>{n['label']}</b><br>PageRank: {n['pagerank']}<br>In-Degree: {n['in_degree']}<br>Tier: {n['tier']}<br>Health: {n['health']:.1f}"
            for n in nodes_data
        ]
        node_sizes = [max(12, int(n["pagerank"] * 1200)) for n in nodes_data]
        node_colors = [n["health"] for n in nodes_data]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=[n["label"] if n["pagerank"] > 0.015 else "" for n in nodes_data],
            textposition="top center",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=12,
                    title=dict(text="Health Score", side="right"),
                    xanchor="left",
                ),
                line_width=1.5,
                line_color="#1E293B",
            )
        )

        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=520,
            )
        )
        st.plotly_chart(fig_net, use_container_width=True)

    with g_col2:
        st.markdown("#### 🏆 Top Authority Hub Articles")
        st.markdown("Ranked by stationary random-walk PageRank authority ($\\alpha = 0.85$):")
        st.dataframe(
            hubs_df[["rank", "name", "pagerank_authority", "in_degree", "structural_health", "quality_tier"]].head(10),
            use_container_width=True,
            height=320,
        )

        # Centrality Correlation Bar Chart
        st.markdown("#### In-Degree vs PageRank Correlation")
        fig_bar = px.bar(
            hubs_df.head(8),
            x="name",
            y="in_degree",
            color="pagerank_authority",
            color_continuous_scale="Blues",
            labels={"in_degree": "Incoming Hyperlinks", "name": "Topic"},
            height=250,
        )
        fig_bar.update_layout(margin=dict(t=20, b=50, l=20, r=20), xaxis_tickangle=-35)
        st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------
# TAB 3: WIKICLUB SPRINT RECOMMENDER
# ---------------------------------------------------------
with tab3:
    st.subheader("🎯 Automated Action Queues for University WikiClubs & Edit-a-Thons")
    st.markdown(
        "WikiInsight-AI automatically converts data quality audits into targeted work queues for university edit-a-thons, "
        "ensuring student editors prioritize high-visibility articles with critical deficits."
    )

    sprint_choice = st.radio(
        "Select Campaign Sprint Queue:",
        ["Unified Priority Queue", "📚 #1Lib1Ref Citation Deficit", "📋 TemplateData Missing Infobox", "🌱 Stub Expansion"],
        horizontal=True,
    )

    if sprint_choice == "📚 #1Lib1Ref Citation Deficit":
        active_queue = sprints["citation_deficit"]
        desc = "High word count articles lacking adequate secondary sources. Prime targets for librarian & university citation drives."
    elif sprint_choice == "📋 TemplateData Missing Infobox":
        active_queue = sprints["missing_infobox"]
        desc = "Well-established articles lacking structured data infoboxes. Direct targets for structured data workshops."
    elif sprint_choice == "🌱 Stub Expansion":
        active_queue = sprints["stub_expansion"]
        desc = "Brief stub entries with incoming traffic. Ideal starter tasks for beginner WikiClub editors."
    else:
        active_queue = sprints["unified_sprint"]
        desc = "Master prioritized backlog ranked by PageRank impact and deficit severity."

    st.info(f"**Queue Mission**: {desc} (Total Tasks Available: **{len(active_queue)}**)")

    # Filter by Priority
    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        if "priority" in active_queue.columns and not active_queue.empty:
            avail_priorities = list(active_queue["priority"].unique())
            sel_pri = st.multiselect("Filter Priority", avail_priorities, default=avail_priorities)
            display_queue = active_queue[active_queue["priority"].isin(sel_pri)]
        else:
            display_queue = active_queue

    with p_col2:
        # Download Button
        csv_bytes = display_queue.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sprint Task CSV for WikiClub",
            data=csv_bytes,
            file_name=f"wikiclub_{sprint_choice.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

    # Display Queue
    cols_show = [c for c in ["task_id", "priority", "name", "campaign_queue", "word_count", "citations_count", "structural_health", "recommended_action"] if c in display_queue.columns]
    st.dataframe(
        display_queue[cols_show],
        use_container_width=True,
        height=400,
    )

    # Envoy Action Guide
    st.markdown("---")
    st.markdown("### 🎓 Tech Envoy Edit-a-Thon Protocol")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**1. Assign by Expertise**")
        st.caption("Distribute high-priority tasks to editors matching academic domains (STEM, Humanities, Arts).")
    with g2:
        st.markdown("**2. Apply Verifiability (WP:V)**")
        st.caption("Cross-reference citations using peer-reviewed journals, university library databases, or open repositories.")
    with g3:
        st.markdown("**3. Re-Audit & Track Velocity**")
        st.caption("Run WikiInsight-AI post-sprint audit to measure structural health point gains achieved during the session.")


# ---------------------------------------------------------
# TAB 4: LIVE ARTICLE QUALITY SCORER
# ---------------------------------------------------------
with tab4:
    st.subheader("✍️ Live Article Diagnostic Scorer")
    st.markdown(
        "Paste drafted prose or configure article parameters to instantly calculate **Citation Density**, "
        "**Structural Health (0-100)**, and predict the resulting **Wikimedia Quality Tier**."
    )

    c_left, c_right = st.columns([1.1, 0.9])

    with c_left:
        st.markdown("#### Input Article Details")
        live_title = st.text_input("Article Title", value="Quantum Machine Learning")
        
        mode = st.radio("Input Method:", ["Parameters", "Paste Article Text"], horizontal=True)

        if mode == "Paste Article Text":
            live_text = st.text_area(
                "Paste Wikipedia Prose:",
                value="Quantum machine learning explores the synergies between quantum computing and classical machine learning algorithms. By leveraging quantum phenomena such as superposition and entanglement, researchers aim to develop quantum algorithms that can process complex datasets exponentially faster than classical counterparts. Key applications include quantum neural networks, quantum support vector machines, and variational quantum eigensolvers for chemistry and materials science.",
                height=180,
            )
            detected_words = len(live_text.split())
            st.caption(f"Detected Word Count: **{detected_words} words**")
            live_words = detected_words
        else:
            live_words = st.slider("Word Count", min_value=50, max_value=5000, value=1850, step=50)

        param_c1, param_c2 = st.columns(2)
        with param_c1:
            live_citations = st.number_input("Verified Citations Count", min_value=0, max_value=200, value=16)
            live_sections = st.number_input("Section Headings Count", min_value=1, max_value=30, value=6)
        with param_c2:
            live_has_ib = st.checkbox("Has Standard Infobox", value=True)
            live_ib_fields = st.number_input("Infobox Parameters Count", min_value=0, max_value=40, value=10, disabled=not live_has_ib)

        live_categories = st.multiselect(
            "Assigned Categories",
            ["Quantum Computing", "Machine Learning", "Physics", "Computer Science", "Emerging Technology"],
            default=["Quantum Computing", "Machine Learning"]
        )

    with c_right:
        st.markdown("#### Real-Time Diagnostic Audit")

        article_payload = {
            "name": live_title,
            "word_count": live_words,
            "citations_count": live_citations,
            "sections_count": live_sections,
            "has_infobox": live_has_ib,
            "infobox_fields_count": live_ib_fields if live_has_ib else 0,
            "categories": live_categories,
        }

        audit_res = audit_article_quality(article_payload)

        # Health Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=audit_res["structural_health"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{audit_res['quality_tier']}</b>", "font": {"size": 22, "color": WIKI_BLUE}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": WIKI_BLUE},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 25], "color": "#FEE2E2"},
                    {"range": [25, 45], "color": "#FEF3C7"},
                    {"range": [45, 65], "color": "#E0E7FF"},
                    {"range": [65, 85], "color": "#CFFAFE"},
                    {"range": [85, 100], "color": "#DCFCE7"},
                ],
                "threshold": {
                    "line": {"color": WIKI_GREEN, "width": 4},
                    "thickness": 0.75,
                    "value": 85,
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Citation Density KPI
        dens_col1, dens_col2 = st.columns(2)
        with dens_col1:
            st.metric("Citation Density", f"{audit_res['citation_density']}", "per 1k words")
        with dens_col2:
            verdict = "Healthy" if audit_res['citation_density'] >= 8.0 else ("Adequate" if audit_res['citation_density'] >= 4.0 else "Deficit")
            st.metric("Verifiability Status", verdict)

        # Subscore Radar / Bar
        st.markdown("##### Subscore Pillar Breakdown")
        sub_df = pd.DataFrame(list(audit_res["subscores"].items()), columns=["Pillar", "Points"])
        sub_df["MaxPoints"] = [30.0, 30.0, 20.0, 10.0, 10.0]
        sub_df["Label"] = sub_df["Pillar"].apply(lambda p: p.replace("_", " ").title())

        fig_sub = px.bar(
            sub_df,
            x="Points",
            y="Label",
            orientation="h",
            text="Points",
            range_x=[0, 30],
            color="Points",
            color_continuous_scale="Teal",
            height=200,
        )
        fig_sub.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
        st.plotly_chart(fig_sub, use_container_width=True)

        # Actionable Recommendations
        st.markdown("##### 🚀 Recommended Actions to Level Up:")
        for rec in audit_res["suggestions"]:
            st.markdown(f"- {rec}")
