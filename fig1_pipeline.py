import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

steps = [
    ("1. Web Crawling",      "lab1_crawler.py",        "10 Wikipedia pages",              "#1f6feb"),
    ("2. Info Extraction",   "ner_extraction.py",      "25,557 entities\n266 relations",  "#388bfd"),
    ("3. KG Construction",   "build_kg.py",            "11,273 triples\n(space_kg.ttl)",  "#58a6ff"),
    ("4. KB Enrichment",     "entity_alignment.py",    "62,757 triples\n31 Wikidata links","#79c0ff"),
    ("5. KGE Training",      "kge_training.py",        "TransE MRR=0.30\nHits@10=76.6%", "#a5d6ff"),
]

box_w  = 2.5
box_h  = 1.8
gap    = 0.3
y      = 2.5
x0     = 0.4

for i, (title, script, stat, color) in enumerate(steps):
    x = x0 + i * (box_w + gap)
    # shadow
    ax.add_patch(mpatches.FancyBboxPatch(
        (x+0.06, y-box_h/2-0.06), box_w, box_h,
        boxstyle="round,pad=0.12", facecolor='black', edgecolor='none', alpha=0.45, zorder=1))
    # box
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y-box_h/2), box_w, box_h,
        boxstyle="round,pad=0.12", facecolor=color, edgecolor='white', linewidth=1.4, alpha=0.93, zorder=2))
    # step title
    ax.text(x+box_w/2, y+0.45, title,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=3)
    # script name
    ax.text(x+box_w/2, y+0.05, script,
            ha='center', va='center', fontsize=7.5, color='#cce3ff', zorder=3,
            fontstyle='italic')
    # stats
    ax.text(x+box_w/2, y-0.52, stat,
            ha='center', va='center', fontsize=7.5, color='#e6edf3', zorder=3,
            multialignment='center')
    # arrow
    if i < len(steps) - 1:
        arrow_x = x + box_w + 0.04
        ax.annotate('', xy=(arrow_x+gap-0.04, y), xytext=(arrow_x, y),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2.2), zorder=4)

ax.text(8, 4.65, 'Space Knowledge Graph — Full Pipeline',
        ha='center', va='center', fontsize=14, fontweight='bold', color='white')

plt.tight_layout(pad=0.5)
plt.savefig('figures/fig1_pipeline.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("Saved: figures/fig1_pipeline.png")
