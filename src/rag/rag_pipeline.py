"""
TD6 - RAG over RDF/SPARQL with Local LLM (Ollama)
- Loads RDF knowledge graph
- Builds schema summary
- Generates SPARQL from natural language
- Executes with self-repair loop
- CLI interface
- Baseline vs RAG evaluation
"""

import re
import json
import sys
import requests
from pathlib import Path
from rdflib import Graph, Namespace

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
TTL_FILE = "kg_artifacts/space_kg.ttl"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"   # fallback options: "deepseek-r1:1.5b", "qwen:0.5b", "llama3.2:1b"
MAX_PREDICATES = 60
MAX_CLASSES = 30
SAMPLE_TRIPLES = 15
MAX_REPAIR_ATTEMPTS = 2


# ─────────────────────────────────────────
# 0) LLM Client (Ollama)
# ─────────────────────────────────────────

def ask_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 60) -> str:
    """Send prompt to local Ollama LLM. Returns response string."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama not running. Start it with: ollama serve\n"
            "Then pull a model: ollama pull gemma:2b"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def check_ollama_available() -> bool:
    """Check if Ollama is running."""
    try:
        resp = requests.get("http://localhost:11434", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────
# 1) Load RDF Graph
# ─────────────────────────────────────────

def load_graph(ttl_path: str) -> Graph:
    """Load a Turtle RDF file."""
    g = Graph()
    g.parse(ttl_path, format="turtle")
    print(f"✓ Loaded RDF graph: {len(g)} triples from {ttl_path}")
    return g


# ─────────────────────────────────────────
# 2) Build Schema Summary
# ─────────────────────────────────────────

def build_schema_summary(g: Graph) -> str:
    """Extract classes, predicates, and sample triples for LLM prompting."""
    
    # Prefixes
    ns_lines = []
    for prefix, ns in g.namespace_manager.namespaces():
        ns_lines.append(f"PREFIX {prefix}: <{ns}>")
    prefix_block = "\n".join(sorted(set(ns_lines)))
    
    # Distinct predicates
    pred_q = f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o . }} LIMIT {MAX_PREDICATES}"
    predicates = [str(r.p) for r in g.query(pred_q)]
    
    # Distinct classes
    cls_q = f"SELECT DISTINCT ?cls WHERE {{ ?s a ?cls . }} LIMIT {MAX_CLASSES}"
    classes = [str(r.cls) for r in g.query(cls_q)]
    
    # Sample triples
    sample_q = f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . }} LIMIT {SAMPLE_TRIPLES}"
    samples = [(str(r.s), str(r.p), str(r.o)) for r in g.query(sample_q)]
    
    # Shorten URIs for readability
    def shorten(uri: str) -> str:
        for prefix, ns in g.namespace_manager.namespaces():
            if uri.startswith(str(ns)):
                local = uri[len(str(ns)):]
                if local:
                    return f"{prefix}:{local}"
        return uri
    
    pred_lines = "\n".join(f"  - {shorten(p)}" for p in predicates[:MAX_PREDICATES])
    cls_lines = "\n".join(f"  - {shorten(c)}" for c in classes[:MAX_CLASSES])
    sample_lines = "\n".join(f"  {shorten(s)} {shorten(p)} {shorten(o)}" for s, p, o in samples)
    
    summary = f"""
# RDF Graph Schema Summary

## Prefixes
{prefix_block}

## Classes (rdf:type targets)
{cls_lines}

## Predicates
{pred_lines}

## Sample Triples
{sample_lines}
""".strip()
    
    return summary


# ─────────────────────────────────────────
# 3) NL → SPARQL Prompting
# ─────────────────────────────────────────

SPARQL_SYSTEM = """You are a SPARQL 1.1 generator for an RDF knowledge graph about space exploration.
Given the schema summary and a natural language question, generate a valid SPARQL SELECT query.

STRICT RULES:
- Use ONLY the prefixes and predicates from the SCHEMA SUMMARY
- Return ONLY the SPARQL query inside a ```sparql``` code block
- No explanations outside the code block
- Use ?variable for unknowns
- Prefer simple queries with LIMIT 20
"""

def make_sparql_prompt(schema: str, question: str) -> str:
    return f"""{SPARQL_SYSTEM}

SCHEMA SUMMARY:
{schema}

QUESTION: {question}

Return only the SPARQL query in a ```sparql ... ``` code block.
"""

CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

def extract_sparql(text: str) -> str:
    """Extract SPARQL from code block, fallback to full text."""
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Try to find SELECT ... } pattern
    select_m = re.search(r"(SELECT\s+.*?}(?:\s*LIMIT\s+\d+)?)", text, re.DOTALL | re.IGNORECASE)
    if select_m:
        return select_m.group(1).strip()
    return text.strip()


def generate_sparql(question: str, schema: str) -> str:
    """Generate SPARQL query from natural language question."""
    prompt = make_sparql_prompt(schema, question)
    raw = ask_ollama(prompt)
    return extract_sparql(raw)


# ─────────────────────────────────────────
# 4) Execute SPARQL + Self-Repair
# ─────────────────────────────────────────

REPAIR_SYSTEM = """The SPARQL query below failed. Fix it using the schema summary.
Return ONLY the corrected SPARQL in a ```sparql ... ``` code block.
Keep the fix minimal and targeted at the error."""

def repair_sparql(schema: str, question: str, bad_query: str, error: str) -> str:
    """Ask LLM to fix a broken SPARQL query."""
    prompt = f"""{REPAIR_SYSTEM}

SCHEMA SUMMARY:
{schema}

ORIGINAL QUESTION: {question}

BAD SPARQL:
```sparql
{bad_query}
```

ERROR: {error}

Return only the corrected SPARQL in a code block.
"""
    raw = ask_ollama(prompt)
    return extract_sparql(raw)


def execute_sparql(g: Graph, query: str):
    """Execute a SPARQL query and return (vars, rows)."""
    result = g.query(query)
    vars_ = [str(v) for v in result.vars]
    rows = [tuple(str(cell) if cell else "" for cell in row) for row in result]
    return vars_, rows


def answer_with_rag(g: Graph, schema: str, question: str) -> dict:
    """
    Full RAG pipeline: generate SPARQL → execute → self-repair if needed.
    Returns dict with query, results, and metadata.
    """
    sparql = generate_sparql(question, schema)
    
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        try:
            vars_, rows = execute_sparql(g, sparql)
            return {
                "query": sparql,
                "vars": vars_,
                "rows": rows,
                "repaired": attempt > 0,
                "attempts": attempt + 1,
                "error": None,
            }
        except Exception as e:
            err_msg = str(e)
            if attempt < MAX_REPAIR_ATTEMPTS:
                print(f"    ⚠ Query failed (attempt {attempt+1}): {err_msg[:80]}...")
                print(f"    🔧 Attempting self-repair...")
                sparql = repair_sparql(schema, question, sparql, err_msg)
            else:
                return {
                    "query": sparql,
                    "vars": [],
                    "rows": [],
                    "repaired": attempt > 0,
                    "attempts": attempt + 1,
                    "error": err_msg,
                }


# ─────────────────────────────────────────
# 5) Baseline: Direct LLM answer (no RAG)
# ─────────────────────────────────────────

def answer_baseline(question: str) -> str:
    """Ask the LLM directly without any KB context."""
    prompt = f"Answer the following question about space exploration:\n\n{question}\n\nAnswer:"
    return ask_ollama(prompt)


# ─────────────────────────────────────────
# 6) Evaluation: 5 questions baseline vs RAG
# ─────────────────────────────────────────

EVAL_QUESTIONS = [
    "Which missions were launched by NASA?",
    "Who crewed the Apollo 11 mission?",
    "Which telescopes are operated by NASA?",
    "What missions is the Artemis program part of?",
    "Which organizations built the Hubble Space Telescope?",
]

def run_evaluation(g: Graph, schema: str, output_file: str = "rag_evaluation.json"):
    """Evaluate 5 questions: baseline vs RAG."""
    print("\n" + "="*60)
    print("RAG EVALUATION: Baseline vs SPARQL-Generation RAG")
    print("="*60)
    
    eval_results = []
    
    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")
        
        # Baseline
        print("  Baseline (no RAG)...")
        try:
            baseline_ans = answer_baseline(question)
        except RuntimeError as e:
            baseline_ans = f"[Ollama unavailable: {e}]"
        
        # RAG
        print("  RAG (SPARQL generation)...")
        try:
            rag_result = answer_with_rag(g, schema, question)
            rag_rows = rag_result["rows"]
            rag_ans = "; ".join([", ".join(r) for r in rag_rows[:5]]) if rag_rows else "[No results]"
            rag_correct = len(rag_rows) > 0 and not rag_result["error"]
        except RuntimeError as e:
            rag_ans = f"[Ollama unavailable: {e}]"
            rag_correct = False
            rag_result = {"query": "", "repaired": False, "error": str(e)}
        
        print(f"  Baseline: {baseline_ans[:80]}...")
        print(f"  RAG:      {rag_ans[:80]}")
        print(f"  Correct:  {'✓' if rag_correct else '✗'}")
        
        eval_results.append({
            "question": question,
            "baseline_answer": baseline_ans,
            "rag_answer": rag_ans,
            "sparql_query": rag_result.get("query", ""),
            "rag_correct": rag_correct,
            "repaired": rag_result.get("repaired", False),
        })
    
    # Summary
    n_correct = sum(1 for r in eval_results if r["rag_correct"])
    n_repaired = sum(1 for r in eval_results if r["repaired"])
    
    print("\n" + "─"*60)
    print(f"RESULTS: {n_correct}/{len(EVAL_QUESTIONS)} correct  |  {n_repaired} self-repairs used")
    print("─"*60)
    
    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Evaluation saved: {output_file}")
    
    return eval_results


# ─────────────────────────────────────────
# 7) Pretty printing
# ─────────────────────────────────────────

def pretty_print_rag_result(result: dict):
    """Print RAG query results in a readable format."""
    if result.get("error"):
        print(f"\n[Error] {result['error']}")
    
    print(f"\n[SPARQL Query]")
    print(result.get("query", "(none)"))
    
    if result.get("repaired"):
        print("[Note] Query was self-repaired")
    
    rows = result.get("rows", [])
    vars_ = result.get("vars", [])
    
    if not rows:
        print("\n[No results returned]")
        return
    
    print(f"\n[Results] ({len(rows)} rows)")
    if vars_:
        print("  " + " | ".join(v.upper() for v in vars_))
        print("  " + "─" * (sum(len(v) for v in vars_) + 3 * len(vars_)))
    for row in rows[:20]:
        # Shorten URIs for display
        display = []
        for cell in row:
            short = cell.split("/")[-1].replace("#", ":").replace("_", " ")
            display.append(short[:40])
        print("  " + " | ".join(display))
    if len(rows) > 20:
        print(f"  ... ({len(rows) - 20} more rows)")


# ─────────────────────────────────────────
# 8) CLI Demo
# ─────────────────────────────────────────

def run_cli():
    """Interactive CLI for RAG demo."""
    print("\n" + "="*60)
    print("Space Knowledge Graph RAG Demo")
    print("="*60)
    
    # Check Ollama
    if not check_ollama_available():
        print("\n⚠ Ollama is not running!")
        print("  Start with: ollama serve")
        print(f"  Pull model: ollama pull {OLLAMA_MODEL}")
        print("\n  Running in demo mode (showing schema only)...")
        demo_mode = True
    else:
        print(f"✓ Ollama running | Model: {OLLAMA_MODEL}")
        demo_mode = False
    
    # Load graph
    ttl = TTL_FILE
    if not Path(ttl).exists():
        print(f"⚠ Graph file not found: {ttl}")
        print("  Run src/kg/build_kg.py first to generate the KB")
        return
    
    g = load_graph(ttl)
    schema = build_schema_summary(g)
    
    print(f"\n✓ Schema built ({len(schema)} chars)")
    print("\nExample questions:")
    for q in EVAL_QUESTIONS[:3]:
        print(f"  - {q}")
    
    print("\nCommands: 'eval' to run evaluation, 'schema' to see schema, 'quit' to exit")
    print("-"*60)
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if question.lower() == "eval":
            run_evaluation(g, schema)
            continue
        if question.lower() == "schema":
            print(schema)
            continue
        
        if demo_mode:
            print("\n[Demo mode — Ollama required for actual generation]")
            print("The system would generate a SPARQL query like:")
            print(f"""
SELECT ?mission WHERE {{
    ?mission rdf:type space:SpaceMission .
    ?mission space:launchedBy ?agency .
    ?agency rdfs:label "NASA"@en .
}}
LIMIT 20
""")
            continue
        
        print(f"\n--- Baseline (no RAG) ---")
        try:
            baseline = answer_baseline(question)
            print(baseline[:300])
        except RuntimeError as e:
            print(f"Error: {e}")
        
        print(f"\n--- RAG (SPARQL-generation) ---")
        try:
            result = answer_with_rag(g, schema, question)
            pretty_print_rag_result(result)
        except RuntimeError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        g = load_graph(TTL_FILE)
        schema = build_schema_summary(g)
        run_evaluation(g, schema)
    else:
        run_cli()
