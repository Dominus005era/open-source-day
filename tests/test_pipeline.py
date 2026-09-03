"""Automated Unit Test Suite for WikiInsight-AI.

Tests:
- DataLoader: Streaming ingestion, normalization, synthetic benchmark generation.
- Quality Metrics: Citation density formula, structural health score bounds, tier assignments.
- Knowledge Graph: NetworkX DiGraph creation, PageRank convergence, knowledge hub ranking.
- WikiClub Recommender: Sprint queues (#1Lib1Ref, Infobox, Stub), priority sorting, export formatting.
"""

import json
import os
import sys
import unittest
import pandas as pd
import networkx as nx

# Add src to python path for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.data_loader import (
    DataLoader,
    generate_synthetic_benchmark,
    stream_jsonl,
    _normalize_record,
)
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
    export_graph_metrics,
)
from src.wikiclub_recommender import (
    generate_wikiclub_sprints,
    filter_citation_deficit_queue,
    filter_infobox_queue,
    filter_stub_queue,
)


class TestDataLoader(unittest.TestCase):
    """Test suite for memory-optimized data loading and synthetic generation."""

    def test_synthetic_benchmark_generation(self):
        df = generate_synthetic_benchmark(num_records=100, random_seed=42)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)

        expected_cols = [
            "identifier", "name", "in_language", "url", "abstract",
            "word_count", "sections_count", "citations_count",
            "has_infobox", "infobox_fields_count", "categories", "outgoing_links"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Check positive bounds
        self.assertTrue((df["word_count"] > 0).all())
        self.assertTrue((df["citations_count"] >= 0).all())
        self.assertTrue((df["sections_count"] >= 1).all())

    def test_normalize_record_edge_cases(self):
        # Empty raw dictionary
        empty_raw = {}
        normalized = _normalize_record(empty_raw)
        self.assertEqual(normalized["name"], "Untitled Article")
        self.assertEqual(normalized["word_count"], 100)
        self.assertEqual(normalized["citations_count"], 0)
        self.assertFalse(normalized["has_infobox"])

        # Dict with custom text
        custom_raw = {
            "title": "Quantum Physics",
            "text": "This is a five word sentence.",
            "references": [1, 2, 3],
            "infobox": {"field1": "val1", "field2": "val2"},
        }
        norm_custom = _normalize_record(custom_raw)
        self.assertEqual(norm_custom["name"], "Quantum Physics")
        self.assertEqual(norm_custom["word_count"], 6)
        self.assertEqual(norm_custom["citations_count"], 3)
        self.assertTrue(norm_custom["has_infobox"])
        self.assertEqual(norm_custom["infobox_fields_count"], 2)


class TestQualityMetrics(unittest.TestCase):
    """Test suite for Citation Density, Structural Health, and Quality Tiers."""

    def test_citation_density(self):
        # Standard: 10 citations in 1000 words -> 10.0
        self.assertEqual(calculate_citation_density(10, 1000), 10.0)
        # 5 citations in 250 words -> 20.0
        self.assertEqual(calculate_citation_density(5, 250), 20.0)
        # Edge case: zero words (prevent division by zero)
        self.assertEqual(calculate_citation_density(5, 0), 0.0)
        # Edge case: zero citations
        self.assertEqual(calculate_citation_density(0, 1000), 0.0)

    def test_structural_health_bounds(self):
        # High quality article
        score_high, sub_high = calculate_structural_health(
            word_count=3500,
            citations_count=45,
            sections_count=12,
            has_infobox=True,
            infobox_fields_count=15,
            categories_count=5,
        )
        self.assertGreaterEqual(score_high, 85.0)
        self.assertLessEqual(score_high, 100.0)

        # Minimal stub
        score_low, sub_low = calculate_structural_health(
            word_count=50,
            citations_count=0,
            sections_count=1,
            has_infobox=False,
            infobox_fields_count=0,
            categories_count=0,
        )
        self.assertLessEqual(score_low, 25.0)
        self.assertGreaterEqual(score_low, 0.0)

        # Check subscores presence
        for k in ["text_depth", "citation_rigor", "infobox_completeness", "section_modularization", "categorical_breadth"]:
            self.assertIn(k, sub_high)

    def test_classify_quality_tier(self):
        # Good/Featured Article
        tier_fa = classify_quality_tier(
            word_count=3200,
            structural_health=90.0,
            citations_count=40,
            citation_density=12.5,
            has_infobox=True,
            sections_count=8,
            categories_count=4,
        )
        self.assertEqual(tier_fa, "Good / Featured Article")

        # Stub Class
        tier_stub = classify_quality_tier(
            word_count=150,
            structural_health=15.0,
            citations_count=1,
            citation_density=6.6,
        )
        self.assertEqual(tier_stub, "Stub Class")

    def test_evaluate_dataframe_quality(self):
        df = generate_synthetic_benchmark(num_records=50, random_seed=42)
        evaluated_df = evaluate_dataframe_quality(df)

        self.assertIn("citation_density", evaluated_df.columns)
        self.assertIn("structural_health", evaluated_df.columns)
        self.assertIn("quality_tier", evaluated_df.columns)
        self.assertEqual(len(evaluated_df), 50)

    def test_audit_article_quality(self):
        article = {
            "name": "Audit Test Subject",
            "word_count": 2000,
            "citations_count": 1,  # Critical deficit
            "sections_count": 5,
            "has_infobox": False,  # Missing infobox
            "infobox_fields_count": 0,
        }
        audit = audit_article_quality(article)
        self.assertEqual(audit["name"], "Audit Test Subject")
        # Should flag #1Lib1Ref and TemplateData suggestions
        suggestion_text = " ".join(audit["suggestions"])
        self.assertIn("#1Lib1Ref", suggestion_text)
        self.assertIn("TemplateData", suggestion_text)


class TestKnowledgeGraph(unittest.TestCase):
    """Test suite for NetworkX graph modeling, PageRank, and Knowledge Hubs."""

    def setUp(self):
        df = generate_synthetic_benchmark(num_records=60, random_seed=42)
        self.evaluated_df = evaluate_dataframe_quality(df)
        self.G = build_knowledge_graph(self.evaluated_df)

    def test_graph_structure(self):
        self.assertIsInstance(self.G, nx.DiGraph)
        self.assertGreater(self.G.number_of_nodes(), 0)
        self.assertGreater(self.G.number_of_edges(), 0)

        # Node attributes
        sample_node = list(self.G.nodes())[0]
        attrs = self.G.nodes[sample_node]
        self.assertIn("word_count", attrs)
        self.assertIn("structural_health", attrs)
        self.assertIn("quality_tier", attrs)

    def test_compute_graph_metrics(self):
        metrics = compute_graph_metrics(self.G)
        self.assertIn("raw_in_degree", metrics)
        self.assertIn("raw_out_degree", metrics)
        self.assertIn("pagerank", metrics)
        self.assertIn("betweenness_centrality", metrics)

        # PageRank sum should be approximately 1.0 across all nodes
        pr_sum = sum(metrics["pagerank"].values())
        self.assertAlmostEqual(pr_sum, 1.0, places=2)

    def test_identify_knowledge_hubs(self):
        hubs_df = identify_knowledge_hubs(self.G, top_n=10)
        self.assertFalse(hubs_df.empty)
        self.assertLessEqual(len(hubs_df), 10)
        self.assertIn("pagerank_authority", hubs_df.columns)
        self.assertIn("in_degree", hubs_df.columns)

        # Verify sorted descending by authority
        pr_values = hubs_df["pagerank_authority"].tolist()
        self.assertEqual(pr_values, sorted(pr_values, reverse=True))

    def test_export_graph_metrics(self):
        metrics = compute_graph_metrics(self.G)
        temp_export = os.path.join(project_dir, "data", "temp_test_graph_metrics.json")
        try:
            summary = export_graph_metrics(self.G, metrics, temp_export)
            self.assertTrue(os.path.exists(temp_export))
            with open(temp_export, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("network_statistics", loaded)
            self.assertIn("top_knowledge_hubs", loaded)
        finally:
            if os.path.exists(temp_export):
                os.remove(temp_export)


class TestWikiClubRecommender(unittest.TestCase):
    """Test suite for WikiClub sprint queues and prioritization."""

    def setUp(self):
        df = generate_synthetic_benchmark(num_records=70, random_seed=42)
        self.evaluated_df = evaluate_dataframe_quality(df)
        self.G = build_knowledge_graph(self.evaluated_df)
        self.metrics = compute_graph_metrics(self.G)

    def test_generate_wikiclub_sprints(self):
        sprints = generate_wikiclub_sprints(self.evaluated_df, self.G, self.metrics)

        self.assertIn("citation_deficit", sprints)
        self.assertIn("missing_infobox", sprints)
        self.assertIn("stub_expansion", sprints)
        self.assertIn("unified_sprint", sprints)

        unified = sprints["unified_sprint"]
        self.assertFalse(unified.empty)
        self.assertIn("task_id", unified.columns)
        self.assertIn("campaign_queue", unified.columns)
        self.assertIn("priority", unified.columns)
        self.assertIn("recommended_action", unified.columns)

        # Check task id formatting
        self.assertTrue(unified["task_id"].iloc[0].startswith("SPRINT-"))


if __name__ == "__main__":
    unittest.main()
