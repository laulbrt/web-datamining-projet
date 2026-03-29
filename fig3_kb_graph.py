import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os

os.makedirs("figures", exist_ok=True)

from rdflib import Graph as RDFGraph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL

SPACE = "http://space-exploration.org/ontology#"
ENT   = "http://space-exploration.org/entity/"

def short(uri):
    s = str(uri)
    if s.startswith(ENT):
        return s[len(ENT):]
    if s.startswith(SPACE):
        return s[len(SPACE):]
    return s.split('/')[-1].split('#')[-1]

print("Parsing space_kg.ttl with rdflib...")
g = RDFGraph()
g.parse("kg_artifacts/space_kg.ttl", format="turtle")
print(f"  {len(g)} triples loaded")

SEEDS = {
    'NASA', 'ESA', 'SpaceX', 'Roscosmos', 'ISRO', 'JAXA', 'Boeing',
    'Apollo_11', 'Hubble_Space_Telescope', 'James_Webb_Space_Telescope',
    'International_Space_Station', 'Artemis', 'Voyager_1',
    'Perseverance', 'Neil_Armstrong', 'Buzz_Aldrin',
    'Yuri_Gagarin', 'Valentina_Tereshkova', 'Ariane_5',
}

SKIP_PREDS = {str(RDF.type), str(RDFS.label), str(OWL.sameAs), str(RDFS.comment)}

G = nx.DiGraph()

for s, p, o in g:
    if str(p) in SKIP_PREDS:
        continue
    if not isinstance(s, URIRef) or not isinstance(o, URIRef):
        continue
    s_name = short(s)
    o_name = short(o)
    p_name = short(p)
    if s_name in SEEDS or o_name in SEEDS:
        G.add_edge(s_name, o_name, label=p_name)

print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Keep top 30 nodes by total degree
top_nodes = {n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:30]}
G2 = G.subgraph(top_nodes).copy()
G2.remove_nodes_from(list(nx.isolates(G2)))
print(f"  Subgraph: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

def node_color(n):
    nl = n.lower()
    if any(k in nl for k in ['nasa', 'esa', 'spacex', 'roscosmos', 'isro', 'jaxa', 'boeing']):
        return '#3fb950'
    if any(k in nl for k in ['apollo', 'artemis', 'hubble', 'webb', 'iss', 'international',
                               'voyager', 'perseverance', 'shuttle', 'ariane']):
        return '#58a6ff'
    if any(k in nl for k in ['armstrong', 'aldrin', 'gagarin', 'tereshkova', 'collins']):
        return '#ff7b72'
    return '#d2a8ff'

pos = nx.kamada_kawai_layout(G2, scale=3.0)
node_colors = [node_color(n) for n in G2.nodes()]

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.axis('off')

nx.draw_networkx_nodes(G2, pos, node_color=node_colors, node_size=750,
                       alpha=0.92, ax=ax)
nx.draw_networkx_edges(G2, pos, edge_color='#58a6ff', arrows=True,
                       arrowsize=20, width=1.3, alpha=0.5, ax=ax,
                       connectionstyle='arc3,rad=0.1',
                       min_source_margin=20, min_target_margin=20)

labels = {n: n.replace('_', ' ') for n in G2.nodes()}
nx.draw_networkx_labels(G2, pos, labels=labels, font_size=7.5,
                        font_color='white', font_weight='bold', ax=ax)

# Edge labels only between seed nodes
edge_labels = {(u, v): d['label']
               for u, v, d in G2.edges(data=True)
               if u in SEEDS and v in SEEDS}
nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels,
                              font_size=6.5, font_color='#ffa657', ax=ax,
                              bbox=dict(alpha=0, pad=0))

legend_items = [
    mpatches.Patch(color='#3fb950', label='Agency / Organization'),
    mpatches.Patch(color='#58a6ff', label='Mission / Spacecraft'),
    mpatches.Patch(color='#ff7b72', label='Astronaut / Person'),
    mpatches.Patch(color='#d2a8ff', label='Other entity'),
]
ax.legend(handles=legend_items, loc='lower left', framealpha=0.25,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=9)

ax.set_title('Knowledge Base Subgraph — Key Entities & Relations (top 30 nodes)',
             color='white', fontsize=13, fontweight='bold', pad=14)

plt.tight_layout()
plt.savefig('figures/fig3_kb_subgraph.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Saved: figures/fig3_kb_subgraph.png")
