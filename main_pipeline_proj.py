"""
Main Pipeline - runs the full project end-to-end
Space Exploration Knowledge Graph

Usage:
  python main_pipeline.py              # full pipeline
  python main_pipeline.py --step kg    # only KB construction
  python main_pipeline.py --step kge   # only KGE
  python main_pipeline.py --step rag   # only RAG CLI
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def step_kg():
    """TD2: Build initial KB + alignment + SPARQL expansion."""
    print("\n" + "="*60)
    print("STEP 1: Knowledge Base Construction (TD2)")
    print("="*60)

    from kg.build_kg import build_initial_kg
    from kg.entity_alignment import align_entities, align_predicates

    # Build initial KB
    g = build_initial_kg(
        entities_csv="extracted_knowledge.csv",
        output_ttl="kg_artifacts/space_kg.ttl",
    )

    # Entity + predicate alignment
    g, align_g = align_entities(
        kg_ttl="kg_artifacts/space_kg.ttl",
        output_alignment="kg_artifacts/alignment.ttl",
        output_csv="kg_artifacts/alignment_table.csv",
        use_api=False,
    )
    align_predicates("kg_artifacts/alignment.ttl")

    # SPARQL expansion (Wikidata + synthetic fallback)
    try:
        from kg.sparql_expansion import expand_kb
        triples, stats = expand_kb(
            kg_ttl="kg_artifacts/space_kg.ttl",
            output_nt="kg_artifacts/expanded_kb.nt",
            use_wikidata=True,
            target_size=80000,
        )
    except Exception as e:
        print(f"⚠ SPARQL expansion error: {e}")
        print("  Using synthetic data only...")
        from kg.sparql_expansion import expand_kb
        triples, stats = expand_kb(
            kg_ttl="kg_artifacts/space_kg.ttl",
            output_nt="kg_artifacts/expanded_kb.nt",
            use_wikidata=False,
            target_size=80000,
        )

    print("\n✓ KB construction complete")


def step_reasoning():
    """TD5 Part 1: SWRL reasoning on family.owl + space KB."""
    print("\n" + "="*60)
    print("STEP 2: Reasoning (TD5 Part 1)")
    print("="*60)

    from reason.swrl_reasoning import load_and_reason_family
    from reason.swrl_vs_kge import apply_swrl_rule_on_space_kb, compare_with_embeddings

    # SWRL on family.owl
    load_and_reason_family("kg_artifacts/family.owl")

    # SWRL on space KB
    apply_swrl_rule_on_space_kb("kg_artifacts/space_kg.ttl")

    print("\n✓ Reasoning complete")


def step_kge():
    """TD5 Part 2: KGE training and evaluation."""
    print("\n" + "="*60)
    print("STEP 3: Knowledge Graph Embeddings (TD5 Part 2)")
    print("="*60)

    from kge.kge_training import (
        prepare_kge_data, train_lightweight, run_size_sensitivity, analyze_embeddings
    )

    # Prepare data
    train, val, test, ent2id, rel2id = prepare_kge_data(
        nt_path="kg_artifacts/expanded_kb.nt",
        output_dir="kg_artifacts/kge_data",
    )

    # Try PyKEEN first, fall back to lightweight
    try:
        from kge.kge_training import train_with_pykeen
        results = train_with_pykeen(
            "kg_artifacts/kge_data/train.txt",
            "kg_artifacts/kge_data/valid.txt",
            "kg_artifacts/kge_data/test.txt",
            "kg_artifacts/kge_results",
        )
    except (ImportError, Exception):
        results, emb, ent2id, rel2id = train_lightweight(
            "kg_artifacts/kge_data/train.txt",
            "kg_artifacts/kge_data/valid.txt",
            "kg_artifacts/kge_data/test.txt",
            "kg_artifacts/kge_results",
        )

    # Size sensitivity
    run_size_sensitivity("kg_artifacts/expanded_kb.nt")

    # Embedding analysis
    emb_path = "kg_artifacts/kge_results/transe_entity_embeddings.npy"
    if Path(emb_path).exists():
        analyze_embeddings(emb_path, ent2id)

    # Rule vs embedding comparison
    from reason.swrl_vs_kge import compare_with_embeddings
    compare_with_embeddings("kg_artifacts/kge_results")

    print("\n✓ KGE complete")


def step_rag():
    """TD6: RAG CLI."""
    print("\n" + "="*60)
    print("STEP 4: RAG (TD6)")
    print("="*60)

    from rag.rag_pipeline import run_cli
    run_cli()


def main():
    parser = argparse.ArgumentParser(description="Space KG Project Pipeline")
    parser.add_argument("--step", choices=["kg", "reason", "kge", "rag", "all"],
                        default="all", help="Which step to run")
    args = parser.parse_args()

    Path("kg_artifacts").mkdir(exist_ok=True)
    Path("kg_artifacts/kge_data").mkdir(exist_ok=True)
    Path("kg_artifacts/kge_results").mkdir(exist_ok=True)

    if args.step in ("all", "kg"):
        step_kg()

    if args.step in ("all", "reason"):
        step_reasoning()

    if args.step in ("all", "kge"):
        step_kge()

    if args.step in ("rag",):
        step_rag()

    if args.step == "all":
        print("\n" + "="*60)
        print("✓ Full pipeline complete!")
        print("="*60)
        print("Generated files:")
        for f in sorted(Path("kg_artifacts").rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                print(f"  {f}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
