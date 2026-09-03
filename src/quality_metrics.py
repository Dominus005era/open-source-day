"""Quality Metrics Engine for Wikimedia Structured Contents.

Computes:
1. Citation Density: (Citations / Word Count) * 1000
2. Normalized Structural Health Score (0-100 scale)
3. Standard Wikimedia Quality Tier Classification (Stub, Start, C-Class, B-Class, Good/Featured Article)
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


def calculate_citation_density(citations_count: int, word_count: int) -> float:
    """Calculates Wikipedia citation density formula: (Citations / Word Count) * 1000.

    Measures citations per 1,000 words. A standard well-referenced Wikipedia article
    typically maintains 8 to 15 citations per 1,000 words.

    Args:
        citations_count: Total count of verified references/citations in the article.
        word_count: Total word count of the article text.

    Returns:
        float: Citation density rounded to 2 decimal places.
    """
    if word_count <= 0:
        return 0.0
    citations = max(0, citations_count)
    density = (citations / word_count) * 1000.0
    return round(float(density), 2)


def calculate_structural_health(
    word_count: int,
    citations_count: int,
    sections_count: int = 1,
    has_infobox: bool = False,
    infobox_fields_count: int = 0,
    categories_count: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Computes the Normalized Structural Health Score on a 0-100 scale.

    The score aggregates 5 fundamental pillars of Wikipedia Manual of Style:
    - Text Depth (0-30 pts): Logarithmic scaling up to 3,000 words.
    - Citation Rigor (0-30 pts): Balanced between citation count and density.
    - Infobox Completeness (0-20 pts): Standard TemplateData presence and field richness.
    - Section Modularization (0-10 pts): Article structural organization.
    - Categorical Breadth (0-10 pts): Ontological discovery links.

    Returns:
        Tuple[float, Dict[str, float]]: (Overall score 0-100, subscore breakdown dictionary)
    """
    w = max(0, word_count)
    c = max(0, citations_count)
    s = max(0, sections_count)
    cat = max(0, categories_count)
    ib_fields = max(0, infobox_fields_count)

    # 1. Text Depth (30 points maximum)
    if w <= 10:
        score_words = 0.0
    else:
        log_w = math.log10(w)
        log_min = 1.0   # 10 words -> 0 pts
        log_max = 3.477 # ~3000 words -> 30 pts
        norm_w = (log_w - log_min) / (log_max - log_min)
        score_words = min(30.0, max(0.0, norm_w * 30.0))

    # 2. Citation Rigor (30 points maximum)
    # 15 pts for absolute citation volume (target: 25+ citations)
    # 15 pts for density (target: 10.0+ citations per 1,000 words)
    score_cite_count = min(15.0, (c / 25.0) * 15.0)
    density = calculate_citation_density(c, w)
    score_cite_density = min(15.0, (density / 10.0) * 15.0)
    score_citations = score_cite_count + score_cite_density

    # 3. Infobox Completeness (20 points maximum)
    if has_infobox:
        score_ib_base = 10.0
        score_ib_fields = min(10.0, (ib_fields / 8.0) * 10.0)
        score_infobox = score_ib_base + score_ib_fields
    else:
        score_infobox = 0.0

    # 4. Section Modularization (10 points maximum, target: 5+ sections)
    score_sections = min(10.0, (s / 5.0) * 10.0)

    # 5. Categorical Breadth (10 points maximum, target: 3+ categories)
    score_categories = min(10.0, (cat / 3.0) * 10.0)

    total_score = score_words + score_citations + score_infobox + score_sections + score_categories
    total_score = round(min(100.0, max(0.0, total_score)), 2)

    subscores = {
        "text_depth": round(score_words, 2),
        "citation_rigor": round(score_citations, 2),
        "infobox_completeness": round(score_infobox, 2),
        "section_modularization": round(score_sections, 2),
        "categorical_breadth": round(score_categories, 2),
    }

    return total_score, subscores


def classify_quality_tier(
    word_count: int,
    structural_health: float,
    citations_count: int,
    citation_density: float,
    has_infobox: bool = False,
    sections_count: int = 1,
    categories_count: int = 0,
) -> str:
    """Classifies an article into standard Wikimedia Quality Tiers.

    Tiers:
    - Good / Featured Article: Comprehensive encyclopedic article with stellar sourcing and layout.
    - B Class: High quality, well referenced with structured infobox, minor gaps.
    - C Class: Substantial content, basic sourcing, needs stylistic and reference polish.
    - Start Class: Basic outline with rudimentary sourcing, under-developed.
    - Stub Class: Minimal content, missing sections or references.

    Returns:
        str: One of ['Good / Featured Article', 'B Class', 'C Class', 'Start Class', 'Stub Class']
    """
    # Good / Featured Article conditions
    if (
        word_count >= 2400
        and structural_health >= 82.0
        and citations_count >= 25
        and citation_density >= 7.5
        and has_infobox
        and sections_count >= 4
    ):
        return "Good / Featured Article"

    # B Class conditions
    if (
        word_count >= 1400
        and structural_health >= 62.0
        and citations_count >= 10
        and has_infobox
    ):
        return "B Class"

    # C Class conditions
    if (
        word_count >= 650
        and structural_health >= 42.0
        and citations_count >= 3
    ):
        return "C Class"

    # Start Class conditions
    if (
        word_count >= 280
        and structural_health >= 22.0
    ):
        return "Start Class"

    # Otherwise Stub Class
    return "Stub Class"


def audit_article_quality(article: Dict[str, Any]) -> Dict[str, Any]:
    """Generates an exhaustive diagnostic quality audit for an individual article.

    Args:
        article: Dictionary containing article attributes.

    Returns:
        Dict[str, Any] with calculated metrics, tier, and actionable suggestions.
    """
    word_count = int(article.get("word_count", 0))
    citations_count = int(article.get("citations_count", 0))
    sections_count = int(article.get("sections_count", 1))
    has_infobox = bool(article.get("has_infobox", False))
    infobox_fields = int(article.get("infobox_fields_count", 0))

    categories = article.get("categories", [])
    cat_count = len(categories) if isinstance(categories, (list, set)) else 0

    citation_density = calculate_citation_density(citations_count, word_count)
    health_score, subscores = calculate_structural_health(
        word_count=word_count,
        citations_count=citations_count,
        sections_count=sections_count,
        has_infobox=has_infobox,
        infobox_fields_count=infobox_fields,
        categories_count=cat_count,
    )

    quality_tier = classify_quality_tier(
        word_count=word_count,
        structural_health=health_score,
        citations_count=citations_count,
        citation_density=citation_density,
        has_infobox=has_infobox,
        sections_count=sections_count,
        categories_count=cat_count,
    )

    # Formulate prioritized editorial action suggestions
    suggestions: List[str] = []
    if citations_count < 3 or citation_density < 3.0:
        needed_refs = max(3, int(math.ceil((word_count / 1000.0) * 8.0)))
        suggestions.append(f"#1Lib1Ref Priority: Add inline citations (target: ~{needed_refs} verified sources).")

    if not has_infobox:
        suggestions.append("TemplateData Priority: Integrate standard Wikidata-aligned infobox.")
    elif infobox_fields < 5:
        suggestions.append(f"Infobox Enhancement: Expand parameters (currently {infobox_fields}, aim for 8+).")

    if word_count < 300:
        suggestions.append("Stub Expansion: Expand lead paragraph and key contextual background.")
    elif sections_count < 4:
        suggestions.append("Structural Modularization: Break content into distinct subheadings.")

    if cat_count < 2:
        suggestions.append("Categorical Linking: Tag with relevant parent and regional categories.")

    if not suggestions:
        suggestions.append("Exemplary encyclopedic standard. Ready for Peer Review or Featured Article nomination.")

    return {
        "name": article.get("name", "Untitled"),
        "word_count": word_count,
        "citations_count": citations_count,
        "citation_density": citation_density,
        "structural_health": health_score,
        "quality_tier": quality_tier,
        "subscores": subscores,
        "suggestions": suggestions,
    }


def evaluate_dataframe_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized quality evaluation across an entire DataFrame of articles.

    Adds:
    - citation_density
    - structural_health
    - quality_tier
    - text_depth_score
    - citation_rigor_score
    - infobox_score

    Returns:
        pd.DataFrame with new evaluation columns.
    """
    df_out = df.copy()

    # Pre-extract attributes
    words = df_out["word_count"].fillna(0).astype(int).values
    citations = df_out["citations_count"].fillna(0).astype(int).values
    sections = df_out["sections_count"].fillna(1).astype(int).values
    has_ib = df_out["has_infobox"].fillna(False).astype(bool).values
    ib_fields = df_out["infobox_fields_count"].fillna(0).astype(int).values

    cat_counts = []
    for val in df_out.get("categories", []):
        if isinstance(val, (list, set)):
            cat_counts.append(len(val))
        elif isinstance(val, str):
            cat_counts.append(len([x for x in val.split(",") if x.strip()]))
        else:
            cat_counts.append(0)
    cat_counts = np.array(cat_counts, dtype=int)

    densities = []
    health_scores = []
    tiers = []

    for i in range(len(df_out)):
        cd = calculate_citation_density(int(citations[i]), int(words[i]))
        densities.append(cd)

        score, _ = calculate_structural_health(
            word_count=int(words[i]),
            citations_count=int(citations[i]),
            sections_count=int(sections[i]),
            has_infobox=bool(has_ib[i]),
            infobox_fields_count=int(ib_fields[i]),
            categories_count=int(cat_counts[i]),
        )
        health_scores.append(score)

        tier = classify_quality_tier(
            word_count=int(words[i]),
            structural_health=score,
            citations_count=int(citations[i]),
            citation_density=cd,
            has_infobox=bool(has_ib[i]),
            sections_count=int(sections[i]),
            categories_count=int(cat_counts[i]),
        )
        tiers.append(tier)

    df_out["citation_density"] = densities
    df_out["structural_health"] = health_scores
    df_out["quality_tier"] = tiers

    return df_out
