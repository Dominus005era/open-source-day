# 🌐 WikiInsight-AI: Production Research & Analytics Engine

> **Official Open Source Portfolio for the WikiClub Tech Envoy & Open Source Day Initiative**  
> Built on the enterprise-scale dataset: [Wikimedia Foundation - Wikipedia Structured Contents](https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data)

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Wikipedia%20Structured%20Contents-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data)
[![Wikimedia Foundation](https://img.shields.io/badge/Wikimedia-Foundation-006699?logo=wikipedia&logoColor=white)](https://wikimediafoundation.org)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-339966.svg)](https://opensource.org/licenses/MIT)
[![Open Source Day](https://img.shields.io/badge/Initiative-Open%20Source%20Day%202026-EE8019.svg)](#)

---

## 📖 Executive Summary & Envoy Manifesto

**WikiInsight-AI** is a production-grade algorithmic research and decision-support platform designed to empower Wikipedia editors, university **WikiClubs**, librarians, and open-knowledge envoys. 

While Wikipedia serves billions of readers worldwide, content quality and verifiability remain unevenly distributed across the encyclopedia. Thousands of high-traffic entries suffer from critical **Citation Deficits**, lack structured **TemplateData Infoboxes**, or languish as unexpanded **Stubs**.

**WikiInsight-AI** solves this by operationalizing Wikipedia's Manual of Style (MoS) into quantitative metrics, modeling inter-article authority networks with **NetworkX PageRank**, and generating automated, prioritized sprint queues for community edit-a-thons and `#1Lib1Ref` campaigns.

---

## 🏛️ System Architecture

```
wiki contribution/
│
├── .gitignore                         # Git exclusion rules
├── README.md                          # Comprehensive Envoy documentation & user guide
├── requirements.txt                   # Production package dependencies
├── cli.py                             # Unified CLI tool with Windows UTF-8 stdout support
├── app.py                             # Multi-tab interactive Streamlit web dashboard
│
├── src/                               # Core Python Analytical Engine
│   ├── __init__.py                    # Public API exports & package metadata
│   ├── data_loader.py                 # Memory-optimized streaming reader & synthetic benchmark generator
│   ├── quality_metrics.py             # Citation Density, Structural Health (0-100), & Quality Tiers
│   ├── knowledge_graph.py             # NetworkX DiGraph, In-Degree, PageRank Authority, & Hubs
│   └── wikiclub_recommender.py        # Automated sprint queues (#1Lib1Ref, Infobox, Stub Expansion)
│
├── notebooks/
│   ├── wikipedia_structured_contents_analysis.ipynb   # Publication-grade Kaggle research notebook
│   └── build_notebook.py                              # Programmatic notebook compiler
│
├── tests/
│   ├── test_pipeline.py               # Comprehensive unit test suite (100% pass)
│   └── verify_notebook_execution.py   # End-to-end pipeline verification runner
│
└── data/                              # Pre-computed benchmark artifacts
    ├── sample_structured_wiki.csv     # 500+ evaluated benchmark articles with full schema
    ├── graph_metrics.json             # PageRank scores, degree centralities, & knowledge hub rankings
    └── wikiclub_sprint_tasks.csv      # Prioritized edit-a-thon tasks for WikiClubs
```

---

## 🔬 Methodological Formulations

### 1. Citation Density ($\rho_{\text{cite}}$)
Measures source rigor under Wikipedia's core policy of **Verifiability (WP:V)**:
$$\rho_{\text{cite}} = \left( \frac{\text{Citations Count}}{\max(\text{Word Count}, 1)} \right) \times 1000$$
- **Healthy Standard**: $8.0$ to $15.0$ citations per 1,000 words.
- **Citation Deficit Zone**: Word count $> 600$ with $\rho_{\text{cite}} < 2.5$ (Target for `#1Lib1Ref`).

### 2. Normalized Structural Health Score ($S_{\text{health}} \in [0, 100]$)
An objective index aggregating 5 Manual of Style pillars:
$$S_{\text{health}} = S_{\text{words}} + S_{\text{citations}} + S_{\text{infobox}} + S_{\text{sections}} + S_{\text{categories}}$$
- **Text Depth ($S_{\text{words}} \in [0, 30]$)**: Logarithmically scaled up to 3,000 words.
- **Citation Rigor ($S_{\text{citations}} \in [0, 30]$)**: 15 pts for absolute volume ($C \ge 25$) + 15 pts for density ($\rho_{\text{cite}} \ge 10.0$).
- **Infobox Completeness ($S_{\text{infobox}} \in [0, 20]$)**: 10 base pts for presence + 10 pts for field completeness ($\ge 8$ parameters).
- **Section Modularization ($S_{\text{sections}} \in [0, 10]$)**: Hierarchical subheadings (target $\ge 5$ sections).
- **Categorical Breadth ($S_{\text{categories}} \in [0, 10]$)**: Ontological links (target $\ge 3$ categories).

### 3. Wikimedia Quality Tier Classification
Classifies articles into official Wikipedia assessment scales:
- **Good / Featured Article**: $S_{\text{health}} \ge 82$, $W \ge 2400$, $\rho_{\text{cite}} \ge 7.5$, structured infobox, $\ge 4$ sections.
- **B Class**: $S_{\text{health}} \ge 62$, $W \ge 1400$, Citations $\ge 10$, structured infobox.
- **C Class**: $S_{\text{health}} \ge 42$, $W \ge 650$, Citations $\ge 3$.
- **Start Class**: $S_{\text{health}} \ge 22$, $W \ge 280$.
- **Stub Class**: Underdeveloped entries requiring foundational expansion.

### 4. PageRank Authority Centrality ($\mathbf{p}$)
Models inter-article directed hyperlink topology $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ using stationary Markov chain random surfer probability with damping factor $\alpha = 0.85$:
$$\mathbf{p} = \left( \frac{1 - \alpha}{N} \right) \mathbf{1} + \alpha \mathbf{M} \mathbf{p}$$
Articles exhibiting top PageRank authority and high in-degree centrality represent **Knowledge Hubs** whose quality directly anchors the reader experience.

---

## 💻 Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 1. Clone & Set Up Directory
```bash
git clone <repository_url>
cd "wiki contribution"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Production CLI Tool (`cli.py`)

The CLI includes built-in UTF-8 stdout encoding for Windows and cross-platform terminal compatibility.

### 1. Run Complete Quality Audit
Audits articles, computes Citation Density, Structural Health, and Quality Tiers, and exports the evaluated dataset:
```bash
python cli.py audit --records 500 --output data/sample_structured_wiki.csv
```

### 2. Construct Knowledge Graph & PageRank Hubs
Computes In-Degree Centrality, Out-Degree Centrality, and PageRank Authority, exporting the top Knowledge Hubs to JSON:
```bash
python cli.py graph --data data/sample_structured_wiki.csv --export data/graph_metrics.json
```

### 3. Generate WikiClub Campaign Sprint Tasks
Triages articles into targeted campaign tracks and outputs a prioritized task queue:
```bash
python cli.py sprint --data data/sample_structured_wiki.csv --output data/wikiclub_sprint_tasks.csv
```

### 4. Audit an Individual Article Draft
Score drafted prose or check an article's readiness directly from the command line:
```bash
python cli.py score --title "Artificial Intelligence in Healthcare" --words 1850 --citations 16 --sections 6 --infobox --infobox-fields 10
```

---

## 🌐 Interactive Streamlit Web Dashboard (`app.py`)

Launch the web application:
```bash
streamlit run app.py
```

### Dashboard Features:
1. **Quality & Citation Analytics Tab**:
   - Donut charts of Wikimedia Quality Tier breakdown.
   - Interactive Plotly scatter plot: Word Count vs. Citation Density highlighting the **Citation Deficit Zone**.
   - Structural Health distribution histograms with policy threshold indicators.
2. **Knowledge Graph & PageRank Hubs Tab**:
   - 2D Network topology visualization of core hub connections.
   - Top 10 Authority Knowledge Hubs table.
   - Centrality correlation bar charts.
3. **WikiClub Sprint Recommender Tab**:
   - Filterable campaign queues: `#1Lib1Ref Citation Deficit`, `TemplateData Infobox`, `Stub Expansion`, or `Unified Priority`.
   - Priority filters (P1 Critical, P2 High, P3 Medium).
   - One-click **CSV Download Button** for university edit-a-thon coordinators.
   - Step-by-step Tech Envoy Edit-a-Thon Protocol guide.
4. **Live Article Quality Scorer Tab**:
   - Paste draft text or adjust sliders to instantly calculate Citation Density and Structural Health (0–100).
   - Animated visual gauge chart with color-coded quality tier bands.
   - Pillar-by-pillar subscore breakdown and actionable suggestions checklist.

---

## 📓 Kaggle Notebook Deployment Guide

The notebook located at [`notebooks/wikipedia_structured_contents_analysis.ipynb`](notebooks/wikipedia_structured_contents_analysis.ipynb) is self-contained and formatted for direct publication on Kaggle.

### How to Upload to Kaggle:
1. Navigate to **[Kaggle.com](https://www.kaggle.com)** and log in to your account.
2. Click **Create** $\rightarrow$ **New Notebook** (or go to [kaggle.com/code](https://www.kaggle.com/code)).
3. In the top navigation bar of the notebook editor, click **File** $\rightarrow$ **Upload Notebook**.
4. Browse and select `wikipedia_structured_contents_analysis.ipynb` from the `notebooks/` directory.
5. In the right-hand panel under **Data**, click **+ Add Data**.
6. In the search box, paste: `Wikimedia Foundation - Wikipedia Structured Contents` (or search for `wikimedia-foundation/wikipedia-structured-contents`).
7. Click the **+ Add** button next to the dataset.
8. Set the environment accelerator (CPU is sufficient).
9. Click **Run All** to execute all cells.
10. Click **Share** in the upper right, set visibility to **Public**, and tag with `wikimedia`, `nlp`, `network-analysis`, and `data-visualization`.

---

## 🧪 Automated Unit Test Suite

Run the full unit test suite with `pytest`:
```bash
pytest tests/test_pipeline.py -v
```
Or run using Python's built-in `unittest`:
```bash
python -m unittest tests/test_pipeline.py -v
```

### Test Coverage Summary:
- ✅ **TestDataLoader**: Micro-batch streaming reader, schema normalization, and reproducible synthetic generator.
- ✅ **TestQualityMetrics**: Citation density formula, division-by-zero bounds, structural health $[0, 100]$ score bounds, and tier classification.
- ✅ **TestKnowledgeGraph**: Directed NetworkX graph construction, PageRank convergence ($\sum p_i \approx 1.0$), and hub rankings.
- ✅ **TestWikiClubRecommender**: Priority score weighting, queue partitioning, and CSV serialization.

---

## 🤝 Open Source Contribution & License

Contributions are welcome! Please submit issues, feature requests, or pull requests following standard Git workflows.

This project is licensed under the **MIT License** - see the LICENSE file for details.  
Built with pride for the **WikiClub Tech Envoy Portfolio & Open Source Day 2026**.
