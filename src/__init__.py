"""WikiInsight-AI: Research & Data Analytics Engine for Wikimedia Structured Contents.

Developed for the WikiClub Tech Envoy Portfolio & Open Source Day initiative.
"""

from .data_loader import (
    DataLoader,
    generate_synthetic_benchmark,
    stream_jsonl,
    stream_parquet,
)
from .quality_metrics import (
    calculate_citation_density,
    calculate_structural_health,
    classify_quality_tier,
    audit_article_quality,
    evaluate_dataframe_quality,
)
from .knowledge_graph import (
    build_knowledge_graph,
    compute_graph_metrics,
    identify_knowledge_hubs,
    export_graph_metrics,
)
from .wikiclub_recommender import (
    generate_wikiclub_sprints,
    filter_citation_deficit_queue,
    filter_infobox_queue,
    filter_stub_queue,
)

__version__ = "1.0.0"
__author__ = "WikiClub Tech Envoy Team"

__all__ = [
    "DataLoader",
    "generate_synthetic_benchmark",
    "stream_jsonl",
    "stream_parquet",
    "calculate_citation_density",
    "calculate_structural_health",
    "classify_quality_tier",
    "audit_article_quality",
    "evaluate_dataframe_quality",
    "build_knowledge_graph",
    "compute_graph_metrics",
    "identify_knowledge_hubs",
    "export_graph_metrics",
    "generate_wikiclub_sprints",
    "filter_citation_deficit_queue",
    "filter_infobox_queue",
    "filter_stub_queue",
]
