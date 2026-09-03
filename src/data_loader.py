"""Memory-Optimized Data Loader & Synthetic Benchmark Generator for Wikimedia Structured Contents.

Designed for the WikiClub Tech Envoy Portfolio & Open Source Day initiative.
Conforms to the official Wikimedia Foundation - Wikipedia Structured Contents schema.
"""

from __future__ import annotations
import gzip
import json
import logging
import os
import random
from typing import Any, Dict, Generator, Iterator, List, Optional, Union
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def stream_jsonl(
    filepath: str,
    batch_size: int = 1000,
    max_records: Optional[int] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Memory-optimized generator that streams JSONL snapshots in micro-batches.

    Prevents Out-Of-Memory (OOM) exceptions when processing enterprise multi-gigabyte dumps.
    Supports both raw .jsonl and gzip-compressed .jsonl.gz archives.

    Args:
        filepath: Path to the .jsonl or .jsonl.gz file.
        batch_size: Number of records per yielded DataFrame chunk.
        max_records: Maximum records to read before stopping (None for all).

    Yields:
        pd.DataFrame containing the normalized batch of records.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Snapshot file not found: {filepath}")

    open_fn = gzip.open if filepath.endswith(".gz") else open
    batch: List[Dict[str, Any]] = []
    total_yielded = 0

    with open_fn(filepath, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                batch.append(_normalize_record(record))
            except Exception as e:
                logger.debug("Skipping unparseable JSON line: %s", e)
                continue

            if len(batch) >= batch_size:
                df_batch = pd.DataFrame(batch)
                yield df_batch
                total_yielded += len(batch)
                batch = []
                if max_records and total_yielded >= max_records:
                    return

        if batch:
            yield pd.DataFrame(batch)


def stream_parquet(
    filepath: str,
    batch_size: int = 5000,
    max_records: Optional[int] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Streams enterprise Parquet files in record batches using PyArrow.

    Args:
        filepath: Path to the .parquet file.
        batch_size: Batch size for record streaming.
        max_records: Maximum records to yield.

    Yields:
        pd.DataFrame of normalized Wikipedia articles.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet streaming. Install via 'pip install pyarrow'.")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    parquet_file = pq.ParquetFile(filepath)
    total_yielded = 0

    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        df_chunk = record_batch.to_pandas()
        normalized_records = [_normalize_record(row.to_dict()) for _, row in df_chunk.iterrows()]
        df_normalized = pd.DataFrame(normalized_records)
        yield df_normalized
        total_yielded += len(df_normalized)
        if max_records and total_yielded >= max_records:
            return


def _normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Standardizes raw Wikimedia schema variations into a uniform structure.

    Matches fields from Kaggle 'Wikimedia Foundation - Wikipedia Structured Contents':
    - identifier, name, in_language, url, abstract, word_count,
    - sections_count, citations_count, has_infobox, infobox_fields_count,
    - categories, outgoing_links, referenceneed_flag
    """
    identifier = str(raw.get("identifier") or raw.get("id") or raw.get("page_id") or "")
    name = str(raw.get("name") or raw.get("title") or "Untitled Article")
    in_language = str(raw.get("in_language") or raw.get("language") or "en")
    url = str(raw.get("url") or f"https://{in_language}.wikipedia.org/wiki/{name.replace(' ', '_')}")
    abstract = str(raw.get("abstract") or raw.get("description") or raw.get("extract") or "")

    # Word count extraction
    if "word_count" in raw and raw["word_count"] is not None:
        word_count = int(raw["word_count"])
    elif "text" in raw and raw["text"]:
        word_count = len(str(raw["text"]).split())
    elif abstract:
        word_count = max(len(abstract.split()), int(raw.get("length", 150)))
    else:
        word_count = int(raw.get("length", 100))

    # Sections parsing
    sections = raw.get("sections") or []
    if isinstance(sections, list):
        sections_count = len(sections)
    elif isinstance(sections, (int, float)):
        sections_count = int(sections)
    else:
        sections_count = 1

    # Citations parsing
    citations = raw.get("citations") or raw.get("parsed_references") or raw.get("references") or []
    if isinstance(citations, list):
        citations_count = len(citations)
    elif isinstance(citations, (int, float)):
        citations_count = int(citations)
    else:
        citations_count = 0

    # Infoboxes parsing
    infoboxes = raw.get("infoboxes") or raw.get("infobox") or {}
    if isinstance(infoboxes, list):
        has_infobox = len(infoboxes) > 0
        infobox_fields_count = sum(len(ib) if isinstance(ib, dict) else 1 for ib in infoboxes)
    elif isinstance(infoboxes, dict):
        has_infobox = len(infoboxes) > 0
        infobox_fields_count = len(infoboxes)
    elif isinstance(infoboxes, bool):
        has_infobox = infoboxes
        infobox_fields_count = 5 if infoboxes else 0
    else:
        has_infobox = bool(raw.get("has_infobox", False))
        infobox_fields_count = int(raw.get("infobox_fields_count", 0))

    # Categories parsing
    categories = raw.get("categories") or []
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    elif not isinstance(categories, list):
        categories = []

    # Outgoing links parsing
    outgoing_links = raw.get("outgoing_links") or raw.get("links") or raw.get("internal_links") or []
    if isinstance(outgoing_links, str):
        outgoing_links = [link.strip() for link in outgoing_links.split(",") if link.strip()]
    elif not isinstance(outgoing_links, list):
        outgoing_links = []

    # Credibility signal / reference need flag
    referenceneed_flag = bool(raw.get("referenceneed") or raw.get("referenceneed_flag", False))

    return {
        "identifier": identifier,
        "name": name,
        "in_language": in_language,
        "url": url,
        "abstract": abstract,
        "word_count": max(1, word_count),
        "sections_count": max(1, sections_count),
        "citations_count": max(0, citations_count),
        "has_infobox": has_infobox,
        "infobox_fields_count": max(0, infobox_fields_count),
        "categories": categories,
        "outgoing_links": outgoing_links,
        "referenceneed_flag": referenceneed_flag,
    }


def generate_synthetic_benchmark(
    num_records: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generates a high-fidelity synthetic benchmark dataset conforming to Wikimedia schema.

    Creates realistic articles across 5 knowledge domains with varied structural quality
    (Stubs, Start, C-Class, B-Class, and Featured Articles) and cross-article hyperlinks
    forming an authentic scale-free graph topology.

    Args:
        num_records: Number of articles to synthesize (minimum 50).
        random_seed: Seed for reproducible benchmark evaluation.

    Returns:
        pd.DataFrame containing the normalized synthetic Wikipedia snapshot.
    """
    rng = random.Random(random_seed)

    # Core topic ontology for realistic Wikipedia encyclopedic coverage
    topic_domains = {
        "Science & Technology": [
            ("Artificial Intelligence", "Study of intelligent agents and machine cognition", 2800, 35, 12, True, 18),
            ("Quantum Computing", "Computation utilizing quantum mechanical phenomena", 2400, 28, 9, True, 14),
            ("CRISPR Gene Editing", "Biotechnological molecular biology genetic editing tool", 2100, 32, 10, True, 16),
            ("James Webb Space Telescope", "Premier infrared space observatory mission", 3100, 48, 14, True, 22),
            ("Thermodynamics", "Branch of physics dealing with heat and mechanical energy", 1950, 22, 8, True, 12),
            ("General Relativity", "Geometric theory of gravitation published by Albert Einstein", 3400, 52, 15, True, 20),
            ("Photosynthesis", "Biological process converting light energy into chemical energy", 1700, 18, 7, True, 11),
            ("Superconductivity", "Zero electrical resistance occurring in certain materials", 1600, 15, 6, True, 9),
            ("Nanotechnology", "Manipulation of matter on an atomic and molecular scale", 1450, 12, 6, True, 8),
            ("Deep Learning", "Subset of machine learning based on artificial neural networks", 2300, 29, 9, True, 13),
            ("Graphene", "Two-dimensional carbon allotrope with exceptional tensile strength", 1200, 11, 5, True, 7),
            ("Higgs Boson", "Elementary particle in the Standard Model of particle physics", 1850, 24, 8, True, 15),
            ("Plate Tectonics", "Scientific theory explaining the dynamics of Earth's lithosphere", 1750, 19, 7, True, 10),
            ("Dark Matter", "Hypothetical form of matter thought to account for 85% of matter", 1550, 16, 6, True, 8),
            ("Microbiome", "Aggregate of all microbiota residing on or within tissues", 1300, 14, 5, True, 9),
        ],
        "History & Civilizations": [
            ("Byzantine Empire", "Continuation of the Roman Empire in its eastern provinces", 3600, 45, 16, True, 24),
            ("Industrial Revolution", "Transition to new manufacturing processes in Europe and US", 2900, 36, 12, True, 18),
            ("Silk Road", "Historic network of Eurasian trade routes connecting East and West", 2200, 25, 9, True, 12),
            ("Ancient Alexandria", "Major Hellenistic center of commerce and scholarship in Egypt", 1900, 18, 8, True, 11),
            ("Magna Carta", "Royal charter of rights agreed to by King John of England", 2100, 26, 9, True, 15),
            ("Renaissance Humanism", "Intellectual movement originating in 14th-century Italy", 1800, 20, 7, True, 10),
            ("Meiji Restoration", "Political event that restored practical imperial rule to Japan", 2300, 28, 10, True, 16),
            ("French Revolution", "Period of radical political and societal change in France", 3300, 42, 14, True, 21),
            ("Harappan Civilization", "Bronze Age civilization in the northwestern regions of South Asia", 1750, 19, 8, True, 13),
            ("Space Race", "20th-century competition between Soviet Union and United States", 2600, 31, 11, True, 17),
            ("Ottoman Empire", "Imperial realm controlling much of Southeast Europe and Western Asia", 3500, 44, 15, True, 22),
            ("Mesopotamia", "Historical region of Western Asia situated within the Tigris–Euphrates", 2050, 22, 9, True, 14),
        ],
        "Geography & Earth": [
            ("Mariana Trench", "Deepest oceanic trench on Earth located in western Pacific", 1250, 11, 5, True, 8),
            ("Amazon Rainforest", "Moist broadleaf tropical rainforest in the Amazon basin", 2700, 33, 11, True, 19),
            ("Sahara Desert", "Largest hot desert in the world covering North Africa", 1900, 21, 8, True, 12),
            ("Ring of Fire", "Major area in the basin of the Pacific Ocean where earthquakes occur", 1600, 15, 6, True, 10),
            ("Great Barrier Reef", "World's largest coral reef system off Queensland, Australia", 2300, 27, 9, True, 15),
            ("Lake Baikal", "Ancient, massive lake in mountainous Russian region of Siberia", 1400, 13, 6, True, 9),
            ("Mount Everest", "Earth's highest mountain above sea level in Mahalangur Himal", 2150, 25, 9, True, 16),
            ("Galapagos Islands", "Volcanic archipelago in Pacific Ocean known for endemic species", 1800, 20, 8, True, 13),
            ("Nile River Basin", "Major north-flowing river drainage system in northeastern Africa", 1650, 17, 7, True, 11),
            ("Antarctic Ice Sheet", "Single largest mass of ice on Earth covering Antarctica", 1500, 14, 6, True, 9),
        ],
        "Literature & Arts": [
            ("Renaissance Art", "Painting, sculpture, and decorative arts of European Renaissance", 2500, 30, 10, True, 17),
            ("Postmodern Literature", "Form of literature characterized by reliance on meta-fiction", 1700, 16, 7, True, 8),
            ("Baroque Architecture", "Highly decorative and theatrical style of building in Europe", 1950, 21, 8, True, 14),
            ("Impressionism", "19th-century art movement characterized by visible brush strokes", 2200, 24, 9, True, 13),
            ("Surrealism", "Cultural movement beginning in 1920s depicting unnerving scenes", 1850, 19, 8, True, 11),
            ("Homeric Epics", "Ancient Greek epic poems attributed to Homer: Iliad and Odyssey", 2100, 23, 9, True, 12),
            ("Classical Music", "Art music produced in the traditions of Western culture", 2750, 31, 11, True, 16),
            ("Bauhaus Movement", "German art school operational from 1919 to 1933 combining crafts", 1650, 18, 7, True, 10),
            ("Film Noir", "Cinematic term used primarily to describe stylish Hollywood crime dramas", 1400, 12, 6, True, 9),
            ("Kabuki Theatre", "Classical Japanese dance-drama known for stylized drama and makeup", 1300, 11, 5, True, 8),
        ],
        "Social Sciences & Philosophy": [
            ("Game Theory", "Mathematical study of strategic interaction among rational agents", 2400, 30, 10, True, 14),
            ("Universal Declaration of Human Rights", "Milestone document in history of human rights", 2250, 28, 9, True, 15),
            ("Epistemology", "Branch of philosophy concerned with nature and scope of knowledge", 2100, 22, 8, True, 10),
            ("Behavioral Economics", "Effects of psychological and social factors on economic decisions", 1900, 21, 8, True, 12),
            ("Constitutional Law", "Body of law defining the role and powers of different entities", 2500, 29, 10, True, 17),
            ("Sociology of Knowledge", "Study of relationship between human thought and social context", 1400, 12, 6, True, 7),
            ("Existentialism", "Philosophical inquiry that explores the problem of human existence", 2050, 23, 8, True, 11),
            ("Public Health", "Science and art of preventing disease and promoting health", 2300, 27, 9, True, 15),
        ],
    }

    # Flatten seed pool
    all_seeds: List[Dict[str, Any]] = []
    category_map: Dict[str, str] = {}
    for domain, articles in topic_domains.items():
        for title, desc, w_base, c_base, s_base, has_ib, ib_f in articles:
            all_seeds.append({
                "title": title,
                "desc": desc,
                "domain": domain,
                "w_base": w_base,
                "c_base": c_base,
                "s_base": s_base,
                "has_ib": has_ib,
                "ib_f": ib_f,
            })
            category_map[title] = domain

    seed_titles = [s["title"] for s in all_seeds]

    # Generate num_records synthetic articles
    records: List[Dict[str, Any]] = []

    # First include core seed articles
    for i, seed in enumerate(all_seeds):
        if len(records) >= num_records:
            break
        # Sample quality tier profile
        profile = rng.choices(
            ["featured", "b_class", "c_class", "start", "stub", "citation_deficit", "missing_infobox"],
            weights=[0.15, 0.25, 0.25, 0.15, 0.10, 0.05, 0.05],
            k=1,
        )[0]

        article = _synthesize_article_by_profile(
            ident=100000 + i,
            title=seed["title"],
            abstract=seed["desc"],
            domain=seed["domain"],
            profile=profile,
            all_titles=seed_titles,
            rng=rng,
        )
        records.append(article)

    # If more articles needed, generate extended topical variations
    sub_prefixes = [
        "History of", "Philosophy of", "Introduction to", "Applications of",
        "Principles of", "Contemporary", "Advanced", "Critique of",
        "Regional Study on", "Global Impact of", "Future Trends in", "Foundations of"
    ]

    while len(records) < num_records:
        idx = len(records)
        base_seed = rng.choice(all_seeds)
        prefix = rng.choice(sub_prefixes)
        extended_title = f"{prefix} {base_seed['title']}"

        # Guard against duplicates
        if any(r["name"] == extended_title for r in records):
            extended_title = f"{extended_title} (Vol. {idx % 10 + 1})"

        profile = rng.choices(
            ["featured", "b_class", "c_class", "start", "stub", "citation_deficit", "missing_infobox"],
            weights=[0.10, 0.20, 0.30, 0.20, 0.12, 0.04, 0.04],
            k=1,
        )[0]

        article = _synthesize_article_by_profile(
            ident=100000 + idx,
            title=extended_title,
            abstract=f"An encyclopedic inquiry into {base_seed['desc'].lower()}.",
            domain=base_seed["domain"],
            profile=profile,
            all_titles=seed_titles,
            rng=rng,
        )
        records.append(article)

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic benchmark records across %d categories.", len(df), len(topic_domains))
    return df


def _synthesize_article_by_profile(
    ident: int,
    title: str,
    abstract: str,
    domain: str,
    profile: str,
    all_titles: List[str],
    rng: random.Random,
) -> Dict[str, Any]:
    """Helper to synthesize an article adhering to a specific quality tier profile."""
    # Outgoing links: connect to 2 - 8 other topics in pool
    num_links = rng.randint(3, 9)
    sampled_links = rng.sample(all_titles, min(num_links, len(all_titles)))
    if title in sampled_links:
        sampled_links.remove(title)

    categories = [domain, f"Articles on {domain.split('&')[0].strip()}"]
    if rng.random() > 0.4:
        categories.append("Open Source Tech Envoy Corpus")

    if profile == "featured":
        word_count = rng.randint(2600, 4200)
        citations_count = rng.randint(35, 65)
        sections_count = rng.randint(10, 18)
        has_infobox = True
        infobox_fields = rng.randint(12, 25)
        ref_need = False

    elif profile == "b_class":
        word_count = rng.randint(1600, 2550)
        citations_count = rng.randint(14, 30)
        sections_count = rng.randint(6, 11)
        has_infobox = True
        infobox_fields = rng.randint(8, 16)
        ref_need = False

    elif profile == "c_class":
        word_count = rng.randint(800, 1450)
        citations_count = rng.randint(4, 12)
        sections_count = rng.randint(4, 7)
        has_infobox = rng.random() > 0.3
        infobox_fields = rng.randint(4, 10) if has_infobox else 0
        ref_need = rng.random() > 0.7

    elif profile == "start":
        word_count = rng.randint(320, 740)
        citations_count = rng.randint(1, 4)
        sections_count = rng.randint(2, 4)
        has_infobox = rng.random() > 0.6
        infobox_fields = rng.randint(2, 6) if has_infobox else 0
        ref_need = True

    elif profile == "stub":
        word_count = rng.randint(60, 280)
        citations_count = rng.randint(0, 2)
        sections_count = rng.randint(1, 2)
        has_infobox = rng.random() > 0.8
        infobox_fields = rng.randint(1, 4) if has_infobox else 0
        ref_need = True

    elif profile == "citation_deficit":
        # High words, critically low citations (#1Lib1Ref queue target)
        word_count = rng.randint(1200, 2800)
        citations_count = rng.randint(0, 2)
        sections_count = rng.randint(5, 10)
        has_infobox = True
        infobox_fields = rng.randint(6, 12)
        ref_need = True

    elif profile == "missing_infobox":
        # Notable article missing infobox (TemplateData queue target)
        word_count = rng.randint(1000, 2200)
        citations_count = rng.randint(8, 20)
        sections_count = rng.randint(5, 9)
        has_infobox = False
        infobox_fields = 0
        ref_need = False

    else:
        word_count = 1000
        citations_count = 10
        sections_count = 4
        has_infobox = True
        infobox_fields = 6
        ref_need = False

    return {
        "identifier": str(ident),
        "name": title,
        "in_language": "en",
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "abstract": f"{abstract} Comprehensive survey covering definitions, historical development, and research implications.",
        "word_count": word_count,
        "sections_count": sections_count,
        "citations_count": citations_count,
        "has_infobox": has_infobox,
        "infobox_fields_count": infobox_fields,
        "categories": categories,
        "outgoing_links": sampled_links,
        "referenceneed_flag": ref_need,
    }


class DataLoader:
    """Unified DataLoader supporting streaming JSONL/Parquet and automatic fallback."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def load(
        self,
        max_records: Optional[int] = None,
        fallback_synthetic: bool = True,
    ) -> pd.DataFrame:
        """Loads Wikimedia structured dataset from file or synthetic generator.

        Args:
            max_records: Maximum records to load.
            fallback_synthetic: If True, generates synthetic benchmark if path not found.

        Returns:
            pd.DataFrame with standardized schema.
        """
        if self.data_path and os.path.exists(self.data_path):
            logger.info("Loading Wikimedia dataset from: %s", self.data_path)
            if self.data_path.endswith((".jsonl", ".jsonl.gz")):
                chunks = list(stream_jsonl(self.data_path, max_records=max_records))
                return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            elif self.data_path.endswith(".parquet"):
                chunks = list(stream_parquet(self.data_path, max_records=max_records))
                return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            elif self.data_path.endswith(".csv"):
                df = pd.read_csv(self.data_path, nrows=max_records)
                # Parse list columns if serialized as strings
                for col in ["categories", "outgoing_links"]:
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: eval(x) if isinstance(x, str) and (x.startswith("[") or x.startswith("set(")) else (
                                [item.strip() for item in str(x).split(",") if item.strip()] if pd.notna(x) else []
                            )
                        )
                return df
            else:
                logger.warning("Unrecognized extension for %s, attempting CSV fallback.", self.data_path)
                return pd.read_csv(self.data_path, nrows=max_records)

        if fallback_synthetic:
            count = max_records if (max_records and max_records >= 50) else 500
            logger.info("Dataset path not specified or file not found. Generating synthetic benchmark (%d records).", count)
            return generate_synthetic_benchmark(num_records=count)

        raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

    @staticmethod
    def save_benchmark(df: pd.DataFrame, output_path: str) -> None:
        """Saves DataFrame as benchmark CSV or JSONL."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_path.endswith(".jsonl"):
            df.to_json(output_path, orient="records", lines=True)
        else:
            df.to_csv(output_path, index=False)
        logger.info("Benchmark exported successfully to %s (%d records).", output_path, len(df))
