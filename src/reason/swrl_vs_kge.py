"""
td5 part 2 exercise 8 - compare swrl rules with embedding arithmetic
rule: SpaceAgency(?a) ∧ launchedBy(?m, ?a) → SpaceMission(?m)
embedding analogy: vec(launchedBy) + vec(SpaceAgency) ≈ vec(SpaceMission)
"""

import numpy as np
import json
from pathlib import Path


def apply_swrl_rule_on_space_kb(kg_ttl: str = "kg_artifacts/space_kg.ttl"):
    """
    simulate the swrl rule on the space kb using a sparql query.
    rule: SpaceAgency(?a) ∧ launchedBy(?m, ?a) → SpaceMission(?m)
    """
    print("="*60)
    print("SWRL RULE ON SPACE KB")
    print("="*60)

    rule = "SpaceAgency(?a) \u2227 launchedBy(?m, ?a) \u2192 SpaceMission(?m)"
    print(f"\nRule: {rule}\n")

    # use rdflib sparql to simulate the rule inference
    try:
        from rdflib import Graph, Namespace, RDF, RDFS, OWL

        SPACE = Namespace("http://space-exploration.org/ontology#")
        SPACE_ENT = Namespace("http://space-exploration.org/entity/")

        g = Graph()
        g.parse(kg_ttl, format="turtle")

        # find all missions that launchedBy a SpaceAgency - these should be SpaceMissions
        query = """
        PREFIX space: <http://space-exploration.org/ontology#>
        PREFIX ent: <http://space-exploration.org/entity/>

        SELECT DISTINCT ?mission ?agency WHERE {
            ?agency rdf:type space:SpaceAgency .
            ?mission space:launchedBy ?agency .
        }
        """
        results = list(g.query(query))

        print(f"Applying rule via SPARQL inference:")
        print(f"  SpaceAgency instances: {len(list(g.query('SELECT ?a WHERE { ?a rdf:type <http://space-exploration.org/ontology#SpaceAgency> }')))}")
        print(f"\nInferred SpaceMission instances (via launchedBy \u2192 SpaceAgency):")
        for row in results:
            mission = str(row[0]).split("/")[-1].replace("_", " ")
            agency = str(row[1]).split("/")[-1].replace("_", " ")
            print(f"  \u2192 {mission} (launchedBy {agency})")

        print(f"\nTotal new SpaceMission inferences: {len(results)}")

    except Exception as e:
        print(f"\u26a0 rdflib query failed: {e}")
        # manual fallback with known kb facts
        print("\nManual rule application on known KB:")
        known_missions = [
            ("Apollo_11", "NASA"),
            ("Artemis", "NASA"),
            ("Ariane_5", "European_Space_Agency"),
            ("ISS", "NASA"),
        ]
        print("  Entities inferred as SpaceMission:")
        for mission, agency in known_missions:
            print(f"  \u2192 {mission.replace('_', ' ')} (launchedBy {agency.replace('_', ' ')})")


def compare_with_embeddings(results_dir: str = "kg_artifacts/kge_results"):
    """
    exercise 8 - compare swrl rule results with transe embedding arithmetic.
    the idea: if the rule says launchedBy(mission, agency) then
    vec(agency) - vec(launchedBy) should be close to vec(mission)
    """
    print("\n" + "="*60)
    print("EMBEDDING COMPARISON (Rule-based vs KGE)")
    print("="*60)

    print("\nSWRL Rule:")
    print("  SpaceAgency(?a) \u2227 launchedBy(?m, ?a) \u2192 SpaceMission(?m)")

    print("\nKGE Embedding Analogy (TransE):")
    print("  TransE translates: head + relation \u2248 tail")
    print("  Expected: vec(Mission) + vec(launchedBy) \u2248 vec(Agency)")
    print("  Inverse: vec(Agency) - vec(launchedBy) \u2248 vec(Mission)")

    # load embeddings if available and do the arithmetic
    try:
        E = np.load(f"{results_dir}/transe_entity_embeddings.npy")
        R = np.load(f"{results_dir}/transe_relation_embeddings.npy")

        with open(f"{results_dir}/entity2id.json") as f:
            ent2id = json.load(f)
        with open(f"{results_dir}/relation2id.json") as f:
            rel2id = json.load(f)

        id2ent = {v: k for k, v in ent2id.items()}

        if "launchedBy" in rel2id:
            r_vec = R[rel2id["launchedBy"]]

            if "Agency_0" in ent2id:
                agency_vec = E[ent2id["Agency_0"]]
                # transe inverse: agency - launchedBy ≈ mission
                predicted_mission_vec = agency_vec - r_vec

                dists = np.linalg.norm(E - predicted_mission_vec, axis=1)
                top_ids = np.argsort(dists)[:5]

                print(f"\n  Embedding arithmetic: vec(Agency_0) - vec(launchedBy) \u2248 ?")
                print(f"  Top-5 nearest entities (TransE prediction):")
                for rank, eid in enumerate(top_ids, 1):
                    ent_name = id2ent.get(eid, f"ent_{eid}")
                    print(f"    {rank}. {ent_name} (dist={dists[eid]:.4f})")

                print("\n  \u2192 Missions should appear in top results if SWRL rule \u2248 KGE")
            else:
                print("\n  Agency_0 not found in embeddings \u2014 showing conceptual comparison")
        else:
            print("\n  'launchedBy' not found in relation embeddings \u2014 showing conceptual comparison")

    except FileNotFoundError:
        pass

    # always show this conceptual summary regardless of whether embeddings loaded
    print("\n" + "\u2500"*50)
    print("Conceptual Comparison:")
    print("\u2500"*50)
    print("""
  SWRL (rule-based):
    - Deterministic: if SpaceAgency(?a) \u2227 launchedBy(?m,?a) \u2192 SpaceMission(?m)
    - Interpretable: explicit logical condition
    - Limited: cannot discover unseen patterns

  TransE (embedding-based):
    - Probabilistic: scores all entities, ranks by distance
    - vec(NASA) + vec(launchedBy)\u207b\u00b9 \u2248 vec(Apollo_11)
    - Can generalize to unseen triples
    - Less interpretable but handles noisy data

  In practice:
    - Rule-based: perfect precision on clean KB
    - KGE: higher recall, tolerates noise, discovers analogies
    - Complementary: rules provide seeds, KGE extends coverage
    """)


if __name__ == "__main__":
    apply_swrl_rule_on_space_kb("kg_artifacts/space_kg.ttl")
    compare_with_embeddings("kg_artifacts/kge_results")
