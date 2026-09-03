"""WikiClub Sprint Recommender Engine for University WikiClubs & Tech Envoys.

Automated sprint queue generator supporting:
1. #1Lib1Ref Citation Deficit Queue (high words, low references)
2. TemplateData Infobox Queue (missing structured infoboxes)
3. Stub Expansion Queue (high-interest stubs needing development)
"""

from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_citation_deficit_queue(
    df: pd.DataFrame,
    min_words: int = 500,
    max_citations: int = 3,
    max_density: float = 2.5,
) -> pd.DataFrame:
    """Filters articles suffering from critical citation deficits (#1Lib1Ref campaign).

    These articles contain substantial encyclopedic prose but lack adequate source citations,
    representing credibility vulnerabilities under Wikipedia's Verifiability policy (WP:V).
    """
    mask = (
        (df["word_count"] >= min_words)
        & ((df["citations_count"] <= max_citations) | (df["citation_density"] <= max_density))
    )
    deficit_df = df[mask].copy()

    if deficit_df.empty:
        return pd.DataFrame()

    # Calculate deficit priority score: higher word count + lower density = higher urgency
    pr_col = "pagerank_authority" if "pagerank_authority" in deficit_df.columns else "in_degree"
    pr_vals = deficit_df[pr_col].fillna(0).values if pr_col in deficit_df.columns else np.zeros(len(deficit_df))

    # Normalize components
    w_norm = deficit_df["word_count"].rank(pct=True).values
    pr_norm = pd.Series(pr_vals).rank(pct=True).values

    priority_score = (0.55 * pr_norm) + (0.45 * w_norm)
    deficit_df["priority_score"] = np.round(priority_score * 100, 1)

    deficit_df["campaign_queue"] = "#1Lib1Ref Citation Deficit"
    deficit_df["priority"] = deficit_df["priority_score"].apply(
        lambda s: "P1 - Critical" if s >= 75 else ("P2 - High" if s >= 45 else "P3 - Medium")
    )
    deficit_df["recommended_action"] = (
        "Add inline reliable citations (#1Lib1Ref). Target at least "
        + ((deficit_df["word_count"] / 1000.0 * 8).round().astype(int).clip(lower=4)).astype(str)
        + " secondary sources."
    )

    return deficit_df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)


def filter_infobox_queue(
    df: pd.DataFrame,
    min_words: int = 450,
) -> pd.DataFrame:
    """Filters notable articles lacking standardized TemplateData infoboxes.

    These articles fail Wikidata / structured data guidelines and lack quick-fact summaries.
    """
    mask = (~df["has_infobox"]) & (df["word_count"] >= min_words)
    infobox_df = df[mask].copy()

    if infobox_df.empty:
        return pd.DataFrame()

    in_deg_col = "in_degree" if "in_degree" in infobox_df.columns else "word_count"
    deg_rank = infobox_df[in_deg_col].rank(pct=True).values
    w_rank = infobox_df["word_count"].rank(pct=True).values

    priority_score = (0.5 * deg_rank) + (0.5 * w_rank)
    infobox_df["priority_score"] = np.round(priority_score * 100, 1)

    infobox_df["campaign_queue"] = "TemplateData Infobox"
    infobox_df["priority"] = infobox_df["priority_score"].apply(
        lambda s: "P1 - Critical" if s >= 75 else ("P2 - High" if s >= 40 else "P3 - Medium")
    )
    infobox_df["recommended_action"] = (
        "Integrate TemplateData Infobox with at least 8 key biographical/domain parameters."
    )

    return infobox_df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)


def filter_stub_queue(
    df: pd.DataFrame,
    max_words: int = 350,
) -> pd.DataFrame:
    """Filters Stub Class articles that have readership or incoming links.

    These are high-potential entry points for new WikiClub student editors.
    """
    mask = (df["quality_tier"] == "Stub Class") | (df["word_count"] <= max_words)
    stub_df = df[mask].copy()

    if stub_df.empty:
        return pd.DataFrame()

    # Prioritize stubs with existing hyperlinks or authority
    in_deg_col = "in_degree" if "in_degree" in stub_df.columns else "word_count"
    deg_rank = stub_df[in_deg_col].rank(pct=True).values

    priority_score = deg_rank
    stub_df["priority_score"] = np.round(priority_score * 100, 1)

    stub_df["campaign_queue"] = "Stub Expansion"
    stub_df["priority"] = stub_df["priority_score"].apply(
        lambda s: "P1 - Critical" if s >= 70 else ("P2 - High" if s >= 40 else "P3 - Medium")
    )
    stub_df["recommended_action"] = (
        "Expand lead section beyond stub threshold (>500 words) and add at least 2 structured subheadings."
    )

    return stub_df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)


def generate_wikiclub_sprints(
    df: pd.DataFrame,
    G: Optional[nx.DiGraph] = None,
    graph_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generates complete suite of actionable sprint queues for University WikiClubs.

    Args:
        df: Enriched DataFrame with quality metrics.
        G: Optional NetworkX knowledge graph.
        graph_metrics: Optional precomputed graph metrics.

    Returns:
        Dict[str, pd.DataFrame]:
            - "citation_deficit": #1Lib1Ref queue
            - "missing_infobox": TemplateData queue
            - "stub_expansion": Stub expansion queue
            - "unified_sprint": Combined prioritized master sprint task table
    """
    working_df = df.copy()

    # Attach graph metrics if available
    if graph_metrics:
        pr_map = graph_metrics.get("pagerank", {})
        in_deg_map = graph_metrics.get("raw_in_degree", {})
        working_df["pagerank_authority"] = working_df["name"].map(pr_map).fillna(0.0)
        working_df["in_degree"] = working_df["name"].map(in_deg_map).fillna(0).astype(int)
    elif "pagerank_authority" not in working_df.columns:
        working_df["pagerank_authority"] = 0.0
        working_df["in_degree"] = 0

    # Generate individual sprint queues
    q_citation = filter_citation_deficit_queue(working_df)
    q_infobox = filter_infobox_queue(working_df)
    q_stub = filter_stub_queue(working_df)

    # Combine into unified queue
    cols_to_keep = [
        "campaign_queue",
        "priority",
        "priority_score",
        "name",
        "quality_tier",
        "word_count",
        "citations_count",
        "citation_density",
        "structural_health",
        "has_infobox",
        "pagerank_authority",
        "in_degree",
        "recommended_action",
    ]

    combined_frames = []
    for queue_df in [q_citation, q_infobox, q_stub]:
        if not queue_df.empty:
            available_cols = [c for c in cols_to_keep if c in queue_df.columns]
            combined_frames.append(queue_df[available_cols])

    if combined_frames:
        unified_sprint = pd.concat(combined_frames, ignore_index=True)
        # Deduplicate if an article qualifies for multiple queues, keep highest priority
        unified_sprint = unified_sprint.sort_values(by="priority_score", ascending=False).drop_duplicates(
            subset=["name", "campaign_queue"]
        ).reset_index(drop=True)
        unified_sprint["task_id"] = [f"SPRINT-{i+1:04d}" for i in range(len(unified_sprint))]
        # Reorder with task_id first
        task_cols = ["task_id"] + [c for c in unified_sprint.columns if c != "task_id"]
        unified_sprint = unified_sprint[task_cols]
    else:
        unified_sprint = pd.DataFrame(columns=["task_id"] + cols_to_keep)

    logger.info(
        "Generated WikiClub Sprints: %d #1Lib1Ref, %d Infobox, %d Stub, %d Unified.",
        len(q_citation),
        len(q_infobox),
        len(q_stub),
        len(unified_sprint),
    )

    return {
        "citation_deficit": q_citation,
        "missing_infobox": q_infobox,
        "stub_expansion": q_stub,
        "unified_sprint": unified_sprint,
    }


def export_sprint_tasks(sprint_df: pd.DataFrame, output_path: str) -> None:
    """Exports sprint task queue to an actionable CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sprint_df.to_csv(output_path, index=False)
    logger.info("Exported %d sprint tasks to %s", len(sprint_df), output_path)
