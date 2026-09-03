"""Knowledge Graph Modeling Engine for Wikimedia Structured Contents.

Uses NetworkX to build inter-article directed relational graphs, calculate In-Degree Centrality,
PageRank Authority, and identify central knowledge hubs.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_knowledge_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Constructs a NetworkX directed graph (DiGraph) from Wikipedia article hyperlink relations.

    Nodes represent Wikipedia articles with their structural metadata (word count, health, tier).
    Edges represent directed hyperlinks (Article A -> Article B).

    Args:
        df: DataFrame containing article records with 'name' and 'outgoing_links'.

    Returns:
        nx.DiGraph: The modeled knowledge graph.
    """
    G = nx.DiGraph()

    # Pre-index existing article titles for fast link validation
    known_titles = set(df["name"].dropna().astype(str).tolist())

    # Add nodes with attributes
    for _, row in df.iterrows():
        title = str(row["name"])
        G.add_node(
            title,
            identifier=str(row.get("identifier", "")),
            word_count=int(row.get("word_count", 0)),
            citations_count=int(row.get("citations_count", 0)),
            structural_health=float(row.get("structural_health", 0.0)),
            quality_tier=str(row.get("quality_tier", "Unrated")),
            categories=row.get("categories", []),
        )

    # Add directed edges
    for _, row in df.iterrows():
        source = str(row["name"])
        outgoing = row.get("outgoing_links", [])

        if isinstance(outgoing, str):
            outgoing = [link.strip() for link in outgoing.split(",") if link.strip()]
        elif not isinstance(outgoing, list):
            outgoing = []

        for target in outgoing:
            target_str = str(target).strip()
            # Only connect to known nodes in current corpus to maintain bounded topology
            if target_str in known_titles and target_str != source:
                G.add_edge(source, target_str)

    logger.info("Built Knowledge Graph: %d nodes, %d directed edges.", G.number_of_nodes(), G.number_of_edges())
    return G


def compute_graph_metrics(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """Computes comprehensive topological centrality metrics for all articles.

    Metrics:
    - in_degree_centrality: incoming link popularity
    - out_degree_centrality: outgoing link density
    - pagerank: stationary probability distribution of random surfer (alpha=0.85)
    - betweenness_centrality: knowledge bridge score

    Returns:
        Dict[str, Dict[str, float]]: Mapping metric name -> dict of {node: score}
    """
    if G.number_of_nodes() == 0:
        return {
            "in_degree": {},
            "out_degree": {},
            "pagerank": {},
            "betweenness": {},
        }

    # In-Degree and Out-Degree
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)

    # PageRank with standard Google damping factor alpha=0.85
    try:
        pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)
    except Exception as e:
        logger.warning("PageRank standard convergence failed, falling back to uniform: %s", e)
        n = max(1, G.number_of_nodes())
        pagerank_scores = {node: 1.0 / n for node in G.nodes()}

    # Approximate betweenness centrality for speed on large networks
    k_sample = min(100, G.number_of_nodes())
    betweenness_scores = nx.betweenness_centrality(G, k=k_sample, seed=42)

    return {
        "raw_in_degree": in_degrees,
        "raw_out_degree": out_degrees,
        "in_degree_centrality": in_degree_centrality,
        "out_degree_centrality": out_degree_centrality,
        "pagerank": pagerank_scores,
        "betweenness_centrality": betweenness_scores,
    }


def identify_knowledge_hubs(
    G: nx.DiGraph,
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """Identifies top Wikipedia Knowledge Hub topics combining PageRank authority and in-degree.

    Args:
        G: The NetworkX directed knowledge graph.
        metrics: Precomputed metrics dict (computed if None).
        top_n: Number of top hubs to extract.

    Returns:
        pd.DataFrame: Table of top knowledge hubs ranked by PageRank Authority.
    """
    if metrics is None:
        metrics = compute_graph_metrics(G)

    pr_dict = metrics.get("pagerank", {})
    in_deg_dict = metrics.get("raw_in_degree", {})
    out_deg_dict = metrics.get("raw_out_degree", {})
    btw_dict = metrics.get("betweenness_centrality", {})

    hubs = []
    for node in G.nodes():
        node_attrs = G.nodes[node]
        pr = pr_dict.get(node, 0.0)
        in_deg = in_deg_dict.get(node, 0)
        out_deg = out_deg_dict.get(node, 0)
        btw = btw_dict.get(node, 0.0)

        # Composite Authority Index: (0.7 * PR_rank + 0.3 * InDegree_rank)
        hubs.append({
            "name": node,
            "pagerank_authority": round(float(pr), 6),
            "in_degree": int(in_deg),
            "out_degree": int(out_deg),
            "betweenness_centrality": round(float(btw), 6),
            "structural_health": node_attrs.get("structural_health", 0.0),
            "quality_tier": node_attrs.get("quality_tier", "Unrated"),
            "word_count": node_attrs.get("word_count", 0),
        })

    df_hubs = pd.DataFrame(hubs)
    if not df_hubs.empty:
        df_hubs = df_hubs.sort_values(by=["pagerank_authority", "in_degree"], ascending=False).reset_index(drop=True)
        df_hubs["rank"] = range(1, len(df_hubs) + 1)
        return df_hubs.head(top_n)

    return pd.DataFrame()


def export_graph_metrics(
    G: nx.DiGraph,
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
) -> Dict[str, Any]:
    """Exports structured graph metrics to a JSON benchmark artifact.

    Args:
        G: NetworkX directed graph.
        metrics: Computed graph metrics.
        output_path: Path to save graph_metrics.json.

    Returns:
        Dict[str, Any]: Summary dictionary.
    """
    hubs_df = identify_knowledge_hubs(G, metrics=metrics, top_n=20)
    top_hubs = hubs_df.to_dict(orient="records") if not hubs_df.empty else []

    # Format node details
    node_metrics = {}
    for node in G.nodes():
        node_metrics[node] = {
            "pagerank": round(metrics["pagerank"].get(node, 0.0), 6),
            "in_degree": metrics["raw_in_degree"].get(node, 0),
            "out_degree": metrics["raw_out_degree"].get(node, 0),
            "betweenness": round(metrics["betweenness_centrality"].get(node, 0.0), 6),
            "tier": G.nodes[node].get("quality_tier", "Unrated"),
            "health": G.nodes[node].get("structural_health", 0.0),
        }

    summary = {
        "network_statistics": {
            "total_articles (nodes)": G.number_of_nodes(),
            "total_hyperlinks (edges)": G.number_of_edges(),
            "graph_density": round(nx.density(G), 6) if G.number_of_nodes() > 0 else 0.0,
            "is_strongly_connected": nx.is_strongly_connected(G) if G.number_of_nodes() > 0 else False,
            "is_weakly_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
        },
        "top_knowledge_hubs": top_hubs,
        "node_metrics_sample": dict(list(node_metrics.items())[:50]),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Exported graph metrics summary to %s", output_path)
    return summary


def extract_visual_subgraph(
    G: nx.DiGraph,
    max_nodes: int = 60,
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extracts a high-impact subgraph for interactive 2D/3D visualization.

    Selects top nodes by PageRank and their direct neighbors.

    Returns:
        Tuple[List[node_dict], List[edge_dict]] suitable for Plotly or web rendering.
    """
    if metrics is None:
        metrics = compute_graph_metrics(G)

    pr = metrics.get("pagerank", {})
    sorted_nodes = sorted(G.nodes(), key=lambda n: pr.get(n, 0.0), reverse=True)
    selected_nodes = set(sorted_nodes[:max_nodes])

    subG = G.subgraph(selected_nodes)
    pos = nx.spring_layout(subG, k=0.35, iterations=50, seed=42)

    nodes_data = []
    for node in subG.nodes():
        coords = pos.get(node, (0.0, 0.0))
        nodes_data.append({
            "id": node,
            "label": node,
            "x": float(coords[0]),
            "y": float(coords[1]),
            "pagerank": round(pr.get(node, 0.0), 5),
            "in_degree": metrics["raw_in_degree"].get(node, 0),
            "tier": subG.nodes[node].get("quality_tier", "Unrated"),
            "health": subG.nodes[node].get("structural_health", 0.0),
            "word_count": subG.nodes[node].get("word_count", 0),
        })

    edges_data = []
    for u, v in subG.edges():
        edges_data.append({
            "source": u,
            "target": v,
        })

    return nodes_data, edges_data
