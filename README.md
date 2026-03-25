# Space Exploration Knowledge Graph

**DIA4 — Knowledge Engineering Project**  
Domain: Space Exploration (NASA, ESA, missions, telescopes, astronauts)

---

## Project Overview

End-to-end pipeline from raw web pages to a queryable knowledge graph with reasoning and RAG:

| Step | Module | Description |
|------|--------|-------------|
| TD1 | `src/crawl/` + `src/ie/` | Web crawling + NER/relation extraction |
| TD2 | `src/kg/` | RDF KB construction, entity alignment, SPARQL expansion |
| TD5 | `src/reason/` + `src/kge/` | SWRL reasoning + Knowledge Graph Embeddings |
| TD6 | `src/rag/` | RAG: NL→SPARQL with self-repair loop |

---

## Repository Structure

```
project-root/
├── src/
│   ├── crawl/          # TD1: WebCrawler (httpx + trafilatura)
│   ├── ie/             # TD1: NER + relation extraction (spaCy)
│   ├── kg/             # TD2: build_kg.py, entity_alignment.py, sparql_expansion.py
│   ├── reason/         # TD5: SWRL (OWLReady2) + SWRL vs KGE comparison
│   ├── kge/            # TD5: KGE training (PyKEEN/NumPy) + evaluation
│   └── rag/            # TD6: RAG pipeline + CLI
├── kg_artifacts/
│   ├── family.owl      # Family ontology for SWRL demo
│   ├── space_kg.ttl    # Initial KB (Turtle)
│   ├── alignment.ttl   # Entity + predicate alignments
│   ├── alignment_table.csv
│   ├── expanded_kb.nt  # Expanded KB (N-Triples, ~80k triples)
│   ├── kb_statistics.json
│   └── kge_data/       # train.txt, valid.txt, test.txt
│       └── kge_results/ # Embeddings + evaluation results
├── data/samples/       # Sample data files
├── reports/            # Final report PDF
├── main_pipeline.py    # Full pipeline entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Hardware Requirements

| Step | RAM | GPU | Time |
|------|-----|-----|------|
| TD1 (crawl) | 2 GB | — | ~2 min |
| TD2 (KB) | 4 GB | — | ~5 min |
| TD5 SWRL | 2 GB | — | ~1 min |
| TD5 KGE (PyKEEN) | 8 GB | optional | 1–3 h |
| TD5 KGE (lightweight) | 4 GB | — | ~10 min |
| TD6 RAG (Ollama gemma:2b) | 8 GB | optional | ~1 min/query |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/space-kg-project.git
cd space-kg-project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# or: venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model (TD1)
python -m spacy download en_core_web_trf

# 5. Install Ollama (TD6)
# → https://ollama.com/download
ollama pull gemma:2b
```

---

## How to Run

### Full pipeline
```bash
python main_pipeline.py
```

### Individual steps

**TD1 — Web Crawling**
```bash
python src/crawl/crawler.py
# Output: crawler_output.jsonl
```

**TD1 — NER + Relation Extraction**
```bash
python src/ie/ner_extraction.py
# Input:  crawler_output.jsonl
# Output: extracted_knowledge.csv, extracted_relations.csv
```

**TD2 — KB Construction**
```bash
python src/kg/build_kg.py          # Initial KB
python src/kg/entity_alignment.py  # Wikidata alignment
python src/kg/sparql_expansion.py  # SPARQL expansion (~80k triples)
```

**TD5 — SWRL Reasoning**
```bash
python src/reason/swrl_reasoning.py   # family.owl rules
python src/reason/swrl_vs_kge.py      # space KB rule + embedding comparison
```

**TD5 — KGE Training**
```bash
python src/kge/kge_training.py
# Trains TransE + DistMult
# Output: kg_artifacts/kge_data/, kg_artifacts/kge_results/
```

**TD6 — RAG Demo (CLI)**
```bash
# Start Ollama first:
ollama serve

# Then run the demo:
python src/rag/rag_pipeline.py

# Run evaluation (5 questions):
python src/rag/rag_pipeline.py eval
```

---

## Key Results

### KB Statistics (after SPARQL expansion)
| Metric | Value |
|--------|-------|
| Total triples | ~80,000 |
| Entities | ~8,000 |
| Relations | ~80 |
| Aligned to Wikidata | 30 core entities |

### KGE Results
| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
|-------|-----|--------|--------|---------|
| TransE | ~0.18 | ~0.12 | ~0.19 | ~0.32 |
| DistMult | ~0.22 | ~0.15 | ~0.23 | ~0.38 |

### SWRL Rules Applied
- `Person(?p) ∧ age(?p,?a) ∧ swrlb:greaterThan(?a,60) → OldPerson(?p)` → 2 inferences (Peter age=70, Marie age=69)
- `Person(?p) ∧ isBrotherOf(?p,?parent) ∧ Parent(?parent) → Uncle(?p)` → 1 inference (Paul is Uncle)

---

## Screenshot (RAG Demo)

```
Space Knowledge Graph RAG Demo
============================================================
✓ Ollama running | Model: gemma:2b
✓ Loaded RDF graph: 847 triples

Question: Which missions were launched by NASA?

--- Baseline (no RAG) ---
NASA has launched many missions including Apollo, Artemis, Hubble...

--- RAG (SPARQL-generation) ---
[SPARQL Query]
SELECT ?mission WHERE {
  ?mission rdf:type space:SpaceMission .
  ?mission space:launchedBy ent:NASA .
}
LIMIT 20

[Results] (4 rows)
  Apollo_11 | Artemis | Hubble_Space_Telescope | International_Space_Station
```

---

## Data

The expanded KB (~80k triples) is too large for direct GitHub hosting.  
A **sample (1,000 triples)** is provided in `data/samples/sample_kb.nt`.  
The full KB can be regenerated with `python src/kg/sparql_expansion.py`.

---

## License

MIT License — see `LICENSE`.
