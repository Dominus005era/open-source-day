"""WikiInsight-AI Production CLI Tool.

Command-line interface for Wikipedia Structured Contents quality auditing,
graph-based PageRank computation, and university WikiClub sprint generation.

Built for the WikiClub Tech Envoy Portfolio & Open Source Day initiative.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Optional

# Ensure UTF-8 output across Windows environments to handle symbols and characters gracefully
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add current folder to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data_loader import DataLoader, generate_synthetic_benchmark
from src.quality_metrics import (
    audit_article_quality,
    calculate_citation_density,
    calculate_structural_health,
    classify_quality_tier,
    evaluate_dataframe_quality,
)
from src.knowledge_graph import (
    build_knowledge_graph,
    compute_graph_metrics,
    export_graph_metrics,
    identify_knowledge_hubs,
)
from src.wikiclub_recommender import (
    export_sprint_tasks,
    generate_wikiclub_sprints,
)


def banner():
    """Prints WikiInsight-AI Tech Envoy banner."""
    print("=" * 75)
    print("  WikiInsight-AI | Wikimedia Structured Contents Intelligence Engine")
    print("  WikiClub Tech Envoy Portfolio & Open Source Day Initiative")
    print("=" * 75)


def cmd_audit(args: argparse.Namespace):
    """Executes dataset audit, calculates quality metrics, and exports benchmark dataset."""
    banner()
    print(f"\n[*] Initiating Dataset Audit (Records requested: {args.records})...")
    
    loader = DataLoader(data_path=args.data)
    df_raw = loader.load(max_records=args.records, fallback_synthetic=True)
    print(f"[+] Loaded {len(df_raw)} records successfully.")
    
    print("[*] Computing Citation Density, Structural Health (0-100), and Quality Tiers...")
    df_evaluated = evaluate_dataframe_quality(df_raw)

    # Summary distribution
    tier_counts = df_evaluated["quality_tier"].value_counts().to_dict()
    avg_health = df_evaluated["structural_health"].mean()
    avg_density = df_evaluated["citation_density"].mean()

    print("\n--- AUDIT SUMMARY ---")
    print(f"Total Articles Audited : {len(df_evaluated)}")
    print(f"Mean Structural Health : {avg_health:.2f} / 100")
    print(f"Mean Citation Density  : {avg_density:.2f} citations / 1k words")
    print("\nQuality Tier Distribution:")
    for tier, count in tier_counts.items():
        pct = (count / len(df_evaluated)) * 100
        print(f"  - {tier:<26}: {count:>4} ({pct:5.1f}%)")

    output_path = args.output
    DataLoader.save_benchmark(df_evaluated, output_path)
    print(f"\n[OK] Evaluated dataset exported to: {output_path}")


def cmd_graph(args: argparse.Namespace):
    """Builds knowledge graph, computes PageRank authority, and exports metrics."""
    banner()
    print(f"\n[*] Constructing Knowledge Graph from dataset: {args.data}...")

    loader = DataLoader(data_path=args.data)
    df = loader.load(max_records=args.records, fallback_synthetic=True)
    if "structural_health" not in df.columns:
        df = evaluate_dataframe_quality(df)

    G = build_knowledge_graph(df)
    print(f"[+] Modeled DiGraph: {G.number_of_nodes()} article nodes, {G.number_of_edges()} directed hyperlinks.")

    print("[*] Computing PageRank Authority (alpha=0.85) & Centrality Metrics...")
    metrics = compute_graph_metrics(G)

    hubs_df = identify_knowledge_hubs(G, metrics=metrics, top_n=10)
    print("\n--- TOP 10 KNOWLEDGE HUB ARTICLES (PAGERANK AUTHORITY) ---")
    for _, row in hubs_df.iterrows():
        print(
            f"  #{row['rank']:<2} {row['name']:<32} | PR: {row['pagerank_authority']:.5f} | "
            f"In-Degree: {row['in_degree']:>3} | Health: {row['structural_health']:>5.1f} | Tier: {row['quality_tier']}"
        )

    export_path = args.export
    export_graph_metrics(G, metrics, export_path)
    print(f"\n[OK] Graph metrics exported to: {export_path}")


def cmd_sprint(args: argparse.Namespace):
    """Generates WikiClub sprint task queues (#1Lib1Ref, Infobox, Stub Expansion)."""
    banner()
    print(f"\n[*] Generating WikiClub Sprints from dataset: {args.data}...")

    loader = DataLoader(data_path=args.data)
    df = loader.load(max_records=args.records, fallback_synthetic=True)
    if "structural_health" not in df.columns:
        df = evaluate_dataframe_quality(df)

    G = build_knowledge_graph(df)
    metrics = compute_graph_metrics(G)

    sprints = generate_wikiclub_sprints(df, G=G, graph_metrics=metrics)
    unified_df = sprints["unified_sprint"]

    print("\n--- WIKICLUB ACTIONABLE CAMPAIGN QUEUES ---")
    print(f"  1. #1Lib1Ref Citation Deficit Queue : {len(sprints['citation_deficit']):>4} tasks")
    print(f"  2. TemplateData Infobox Queue       : {len(sprints['missing_infobox']):>4} tasks")
    print(f"  3. Stub Expansion Queue             : {len(sprints['stub_expansion']):>4} tasks")
    print(f"  Total Unified Actionable Tasks      : {len(unified_df):>4} tasks")

    output_path = args.output
    export_sprint_tasks(unified_df, output_path)
    print(f"\n[OK] Prioritized sprint queue exported to: {output_path}")

    # Display Top 5 high-priority tasks
    if not unified_df.empty:
        print("\n--- TOP CRITICAL SPRINT TASKS ---")
        for _, task in unified_df.head(5).iterrows():
            print(f"  [{task['task_id']}] {task['priority']} | {task['campaign_queue']}")
            print(f"       Article: {task['name']}")
            print(f"       Action : {task['recommended_action']}\n")


def cmd_score(args: argparse.Namespace):
    """Scores a single article interactively or via CLI parameters."""
    banner()
    title = args.title or "Untitled Draft"
    
    # Calculate word count from text if provided
    if args.text:
        words = len(args.text.split())
    else:
        words = max(1, args.words)

    citations = max(0, args.citations)
    sections = max(1, args.sections)
    has_ib = bool(args.infobox)
    ib_fields = max(0, args.infobox_fields) if has_ib else 0

    article_dict = {
        "name": title,
        "word_count": words,
        "citations_count": citations,
        "sections_count": sections,
        "has_infobox": has_ib,
        "infobox_fields_count": ib_fields,
        "categories": ["Wikipedia Article"],
    }

    audit = audit_article_quality(article_dict)

    print(f"\n--- AUDIT REPORT FOR: {audit['name']} ---")
    print(f"  Word Count        : {audit['word_count']}")
    print(f"  Citations Count   : {audit['citations_count']}")
    print(f"  Citation Density  : {audit['citation_density']} citations / 1,000 words")
    print(f"  Structural Health : {audit['structural_health']} / 100.0")
    print(f"  Quality Tier      : {audit['quality_tier']}")
    print("\nSubscore Breakdown:")
    for metric, score in audit["subscores"].items():
        print(f"  - {metric.replace('_', ' ').title():<26}: {score:>5.1f} pts")

    print("\nActionable Recommendations:")
    for i, sug in enumerate(audit["suggestions"], 1):
        print(f"  {i}. {sug}")


def main():
    parser = argparse.ArgumentParser(
        description="WikiInsight-AI: Research & Analytics Engine for Wikimedia Structured Contents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # Command: audit
    p_audit = subparsers.add_parser("audit", help="Audit dataset and calculate quality metrics.")
    p_audit.add_argument("--data", type=str, default=None, help="Path to input snapshot (.csv/.jsonl/.parquet)")
    p_audit.add_argument("--records", type=int, default=500, help="Number of records to process")
    p_audit.add_argument("--output", type=str, default="data/sample_structured_wiki.csv", help="Output path for audited CSV")
    p_audit.set_defaults(func=cmd_audit)

    # Command: graph
    p_graph = subparsers.add_parser("graph", help="Compute knowledge graph and PageRank metrics.")
    p_graph.add_argument("--data", type=str, default="data/sample_structured_wiki.csv", help="Input dataset path")
    p_graph.add_argument("--records", type=int, default=500, help="Max records to include")
    p_graph.add_argument("--export", type=str, default="data/graph_metrics.json", help="Path to export graph metrics JSON")
    p_graph.set_defaults(func=cmd_graph)

    # Command: sprint
    p_sprint = subparsers.add_parser("sprint", help="Generate university WikiClub sprint action queues.")
    p_sprint.add_argument("--data", type=str, default="data/sample_structured_wiki.csv", help="Input dataset path")
    p_sprint.add_argument("--records", type=int, default=500, help="Max records to include")
    p_sprint.add_argument("--output", type=str, default="data/wikiclub_sprint_tasks.csv", help="Output path for sprint CSV")
    p_sprint.set_defaults(func=cmd_sprint)

    # Command: score
    p_score = subparsers.add_parser("score", help="Score an individual article draft.")
    p_score.add_argument("--title", type=str, default="Draft Article", help="Article Title")
    p_score.add_argument("--words", type=int, default=1200, help="Total word count")
    p_score.add_argument("--citations", type=int, default=8, help="Number of citations")
    p_score.add_argument("--sections", type=int, default=4, help="Number of sections")
    p_score.add_argument("--infobox", action="store_true", help="Set flag if infobox is present")
    p_score.add_argument("--infobox-fields", type=int, default=6, help="Number of infobox fields")
    p_score.add_argument("--text", type=str, default=None, help="Optional raw text to compute word count from")
    p_score.set_defaults(func=cmd_score)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
