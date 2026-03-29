"""
Generate 3 figures for the report:
1. Pipeline diagram
2. t-SNE embeddings plot
3. KB subgraph visualization
"""

import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from collections import defaultdict
import os

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────
# FIGURE 1 — Pipeline Diagram
# ─────────────────────────────────────────────
print("Generating Figure 1: Pipeline diagram...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

steps = [
    ("1. Web Crawling\n(lab1_crawler.py)", "10 Wikipedia\npages", "#1f6feb"),
    ("2. Info Extraction\n(ner_extraction.py)", "25,557 entities\n266 relations", "#388bfd"),
    ("3. KG Construction\n(build_kg.py)", "11,273 triples\n(space_kg.ttl)", "#58a6ff"),
    ("4. KB Enrichment\n(entity_alignment.py)", "62,757 triples\n31 Wikidata links", "#79c0ff"),
    ("5. KGE Training\n(kge_training.py)", "TransE MRR=0.30\nDistMult MRR~0", "#a5d6ff"),
]

box_w = 2.2
box_h = 1.6
gap = 0.4
start_x = 0.5
y_center = 2.5

for i, (title, subtitle, color) in enumerate(steps):
    x = start_x + i * (box_w + gap)
    # box shadow
    shadow = mpatches.FancyBboxPatch((x + 0.07, y_center - box_h/2 - 0.07),
        box_w, box_h, boxstyle="round,pad=0.1",
        facecolor='black', edgecolor='none', alpha=0.5, zorder=1)
    ax.add_patch(shadow)
    # main box
    box = mpatches.FancyBboxPatch((x, y_center - box_h/2),
        box_w, box_h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.92, zorder=2)
    ax.add_patch(box)
    # title
    ax.text(x + box_w/2, y_center + 0.25, title,
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            color='white', zorder=3, multialignment='center')
    # subtitle
    ax.text(x + box_w/2, y_center - 0.45, subtitle,
            ha='center', va='center', fontsize=7.5,
            color='#e6edf3', zorder=3, multialignment='center',
            style='italic')
    # arrow
    if i < len(steps) - 1:
        ax_x = x + box_w + 0.05
        ax.annotate('', xy=(ax_x + gap - 0.05, y_center),
                    xytext=(ax_x, y_center),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2),
                    zorder=4)

# title
ax.text(7, 4.6, 'Space Knowledge Graph — Full Pipeline',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color='white')

plt.tight_layout()
plt.savefig('figures/fig1_pipeline.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("  -> figures/fig1_pipeline.png")


# ─────────────────────────────────────────────
# FIGURE 2 — t-SNE Embeddings Plot
# ─────────────────────────────────────────────
print("Generating Figure 2: t-SNE plot...")

with open('kg_artifacts/kge_results/tsne_data.json') as f:
    tsne = json.load(f)

x = np.array(tsne['x'])
y = np.array(tsne['y'])
entities = tsne['entities']

def get_label(uri):
    name = uri.split('/')[-1].replace('_', ' ')
    return name

def classify(uri):
    e = uri.lower()
    name = uri.split('/')[-1].lower()
    if any(k in name for k in ['mission', 'apollo', 'artemis', 'voyager', 'curiosity',
                                'perseverance', 'hubble', 'webb', 'chandra', 'tess',
                                'cassini', 'juno', 'rosetta', 'new_horizons']):
        return 'Mission / Telescope'
    if any(k in name for k in ['astronaut', 'armstrong', 'aldrin', 'gagarin',
                                'tereshkova', 'hadfield', 'collins', 'person']):
        return 'Astronaut / Person'
    if any(k in name for k in ['nasa', 'esa', 'spacex', 'roscosmos', 'isro', 'jaxa',
                                'boeing', 'lockheed', 'organization', 'agency']):
        return 'Agency / Organization'
    if any(k in name for k in ['iss', 'station', 'spacecraft', 'rocket', 'ariane',
                                'shuttle', 'saturn', 'falcon']):
        return 'Spacecraft / Station'
    if any(k in e for k in ['wikidata', 'wd:']):
        return 'Wikidata Entity'
    return 'Other'

categories = [classify(e) for e in entities]

colors_map = {
    'Mission / Telescope':    '#58a6ff',
    'Astronaut / Person':     '#ff7b72',
    'Agency / Organization':  '#3fb950',
    'Spacecraft / Station':   '#d2a8ff',
    'Wikidata Entity':        '#ffa657',
    'Other':                  '#8b949e',
}

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

for cat, color in colors_map.items():
    mask = [c == cat for c in categories]
    xi = x[mask]
    yi = y[mask]
    ax.scatter(xi, yi, c=color, s=18, alpha=0.75, label=cat, linewidths=0)

# Annotate notable entities
notable_keywords = {
    'Neil_Armstrong': 'Neil Armstrong',
    'Buzz_Aldrin': 'Buzz Aldrin',
    'Yuri_Gagarin': 'Yuri Gagarin',
    'NASA': 'NASA',
    'ESA': 'ESA',
    'SpaceX': 'SpaceX',
    'ISS': 'ISS',
    'Hubble_Space_Telescope': 'Hubble',
    'James_Webb_Space_Telescope': 'JWST',
    'Apollo_11': 'Apollo 11',
    'Artemis': 'Artemis',
    'Perseverance': 'Perseverance',
    'Voyager_1': 'Voyager 1',
    'International_Space_Station': 'ISS',
}

annotated = set()
for i, ent in enumerate(entities):
    name = ent.split('/')[-1]
    for key, label in notable_keywords.items():
        if key.lower() in name.lower() and label not in annotated:
            cat = categories[i]
            color = colors_map.get(cat, 'white')
            ax.scatter([x[i]], [y[i]], c=color, s=80, zorder=5,
                       edgecolors='white', linewidths=0.8)
            ax.annotate(label, (x[i], y[i]),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=7.5, color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])
            annotated.add(label)
            break

ax.set_title('t-SNE Visualization of TransE Entity Embeddings (500 entities)',
             color='white', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('t-SNE Dimension 1', color='#8b949e', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', color='#8b949e', fontsize=10)
ax.tick_params(colors='#8b949e')
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')

legend = ax.legend(framealpha=0.3, facecolor='#161b22', edgecolor='#30363d',
                   labelcolor='white', fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('figures/fig2_tsne_embeddings.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("  -> figures/fig2_tsne_embeddings.png")


# ─────────────────────────────────────────────
# FIGURE 3 — KB Subgraph
# ─────────────────────────────────────────────
print("Generating Figure 3: KB subgraph...")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("  networkx not found, using manual layout")

# Parse a subset of triples from space_kg.ttl
triples_raw = []
pattern = re.compile(r'<([^>]+)>\s+<([^>]+)>\s+(?:<([^>]+)>|"([^"]+)")')

def short(uri):
    for prefix, short_p in [
        ('http://space-exploration.org/entity/', ''),
        ('http://space-exploration.org/ontology#', 'space:'),
        ('http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'rdf:'),
        ('http://www.w3.org/2000/01/rdf-schema#', 'rdfs:'),
        ('http://www.w3.org/2002/07/owl#', 'owl:'),
        ('http://www.wikidata.org/entity/', 'wd:'),
    ]:
        if uri.startswith(prefix):
            return uri[len(prefix):]
    return uri.split('/')[-1].split('#')[-1]

# Focus on key entities
focus = {
    'NASA', 'ESA', 'SpaceX', 'Roscosmos', 'ISRO', 'JAXA', 'Boeing',
    'Apollo_11', 'Hubble_Space_Telescope', 'James_Webb_Space_Telescope',
    'International_Space_Station', 'Artemis', 'Voyager_1', 'Perseverance',
    'Neil_Armstrong', 'Buzz_Aldrin', 'Yuri_Gagarin', 'Valentina_Tereshkova',
    'Ariane_5', 'Space_Shuttle',
}
focus_lower = {f.lower() for f in focus}

useful_preds = {'launchedBy', 'operatedBy', 'crewedBy', 'partOf',
                'developedBy', 'type', 'sameAs', 'member', 'hasMember',
                'owns', 'builds', 'launches', 'operates'}

G = nx.DiGraph() if HAS_NX else None

collected = []
with open('kg_artifacts/space_kg.ttl', encoding='utf-8') as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        s_uri, p_uri, o_uri, o_lit = m.group(1), m.group(2), m.group(3), m.group(4)
        s = short(s_uri)
        p = short(p_uri)
        o = short(o_uri) if o_uri else o_lit

        if s.lower() in focus_lower or o.lower() in focus_lower if o_uri else False:
            if p not in ('type', 'label') and len(o) < 50 and len(s) < 50:
                collected.append((s, p, o))
                if HAS_NX:
                    G.add_edge(s, o, label=p)
        if len(collected) >= 300:
            break

# Keep only nodes with degree >= 1, trim to top-degree
if HAS_NX and len(G.nodes) > 0:
    # keep top 35 nodes by degree
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:35]
    top_set = {n for n, _ in top_nodes}
    G2 = G.subgraph(top_set).copy()

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    try:
        pos = nx.kamada_kawai_layout(G2)
    except Exception:
        pos = nx.spring_layout(G2, seed=42, k=2.5)

    # color by type guess
    node_colors = []
    for n in G2.nodes():
        nl = n.lower()
        if any(k in nl for k in ['nasa', 'esa', 'spacex', 'roscosmos', 'isro', 'jaxa', 'boeing']):
            node_colors.append('#3fb950')
        elif any(k in nl for k in ['apollo', 'artemis', 'hubble', 'webb', 'iss', 'voyager',
                                    'perseverance', 'shuttle', 'ariane']):
            node_colors.append('#58a6ff')
        elif any(k in nl for k in ['armstrong', 'aldrin', 'gagarin', 'tereshkova', 'collins']):
            node_colors.append('#ff7b72')
        else:
            node_colors.append('#8b949e')

    nx.draw_networkx_nodes(G2, pos, node_color=node_colors, node_size=600,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G2, pos, edge_color='#30363d', arrows=True,
                           arrowsize=15, width=1.2, alpha=0.7, ax=ax,
                           connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G2, pos, font_size=7.5, font_color='white',
                            font_weight='bold', ax=ax)
    edge_labels = {(u, v): d['label'] for u, v, d in G2.edges(data=True)}
    nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels,
                                 font_size=6, font_color='#ffa657', ax=ax,
                                 bbox=dict(alpha=0))

    legend_items = [
        mpatches.Patch(color='#3fb950', label='Agency / Organization'),
        mpatches.Patch(color='#58a6ff', label='Mission / Spacecraft'),
        mpatches.Patch(color='#ff7b72', label='Astronaut / Person'),
        mpatches.Patch(color='#8b949e', label='Other'),
    ]
    ax.legend(handles=legend_items, loc='lower left', framealpha=0.3,
              facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=9)

    ax.set_title('Knowledge Base Subgraph — Key Entities & Relations (top 35 nodes)',
                 color='white', fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig('figures/fig3_kb_subgraph.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print("  -> figures/fig3_kb_subgraph.png")
else:
    print("  skipped (no networkx or no triples matched)")

print("\nDone. All figures saved in ./figures/")
