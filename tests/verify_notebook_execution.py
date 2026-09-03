"""Verification runner for the analysis pipeline and notebook execution."""
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.quality_metrics import evaluate_dataframe_quality
from src.knowledge_graph import build_knowledge_graph, compute_graph_metrics, identify_knowledge_hubs
from src.wikiclub_recommender import generate_wikiclub_sprints

def main():
    print("Testing notebook pipeline end-to-end...")
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_structured_wiki.csv")
    loader = DataLoader(csv_path)
    df = loader.load(max_records=500)
    print(f"[1] Loaded dataset: {len(df)} records")

    df = evaluate_dataframe_quality(df)
    print(f"[2] Computed quality metrics: mean health = {df['structural_health'].mean():.2f}")

    G = build_knowledge_graph(df)
    print(f"[3] Built knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    metrics = compute_graph_metrics(G)
    hubs = identify_knowledge_hubs(G, metrics=metrics, top_n=10)
    print(f"[4] Computed PageRank metrics: top hub = {hubs.iloc[0]['name']} (PR: {hubs.iloc[0]['pagerank_authority']})")

    sprints = generate_wikiclub_sprints(df, G=G, graph_metrics=metrics)
    unified = sprints["unified_sprint"]
    print(f"[5] Generated WikiClub Sprints: {len(unified)} actionable tasks")

    print("\n[ALL PIPELINE STEPS VERIFIED SUCCESSFULLY!]")

if __name__ == "__main__":
    main()
