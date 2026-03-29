import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

with open('kg_artifacts/kge_results/tsne_data.json') as f:
    tsne = json.load(f)

x = np.array(tsne['x'])
y = np.array(tsne['y'])
entities = tsne['entities']

def classify(uri):
    n = uri.split('/')[-1].lower()
    if any(k in n for k in ['mission', 'apollo', 'artemis', 'voyager', 'curiosity',
                              'perseverance', 'hubble', 'webb', 'chandra', 'tess',
                              'cassini', 'juno', 'rosetta', 'new_horizons', 'pioneer',
                              'telescope', 'probe', 'mars_']):
        return 'Mission / Telescope'
    if any(k in n for k in ['astronaut', 'armstrong', 'aldrin', 'gagarin', 'cosmonaut',
                              'tereshkova', 'hadfield', 'collins', 'person_', 'shepard']):
        return 'Astronaut / Person'
    if any(k in n for k in ['nasa', 'esa', 'spacex', 'roscosmos', 'isro', 'jaxa',
                              'boeing', 'lockheed', 'organization', 'agency', 'grumman']):
        return 'Agency / Organization'
    if any(k in n for k in ['iss', 'station', 'spacecraft', 'rocket', 'ariane',
                              'shuttle', 'saturn', 'falcon', 'dragon', 'soyuz']):
        return 'Spacecraft / Rocket'
    return 'Other'

colors_map = {
    'Mission / Telescope':   '#58a6ff',
    'Astronaut / Person':    '#ff7b72',
    'Agency / Organization': '#3fb950',
    'Spacecraft / Rocket':   '#d2a8ff',
    'Other':                 '#8b949e',
}
sizes_map = {
    'Mission / Telescope':   22,
    'Astronaut / Person':    22,
    'Agency / Organization': 22,
    'Spacecraft / Rocket':   22,
    'Other':                 10,
}

categories = [classify(e) for e in entities]

fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

for cat, color in colors_map.items():
    mask = np.array([c == cat for c in categories])
    ax.scatter(x[mask], y[mask], c=color, s=sizes_map[cat],
               alpha=0.75, label=cat, linewidths=0, zorder=2)

# Notable annotations
notable = {
    'Neil_Armstrong':              'Neil Armstrong',
    'Buzz_Aldrin':                 'Buzz Aldrin',
    'Yuri_Gagarin':                'Yuri Gagarin',
    'Valentina_Tereshkova':        'V. Tereshkova',
    'NASA':                        'NASA',
    'ESA':                         'ESA',
    'SpaceX':                      'SpaceX',
    'Roscosmos':                   'Roscosmos',
    'International_Space_Station': 'ISS',
    'Hubble_Space_Telescope':      'Hubble',
    'James_Webb_Space_Telescope':  'JWST',
    'Apollo_11':                   'Apollo 11',
    'Artemis_II':                  'Artemis II',
    'Perseverance':                'Perseverance',
    'Voyager_1':                   'Voyager 1',
}

annotated = set()
for i, ent in enumerate(entities):
    name = ent.split('/')[-1]
    for key, label in notable.items():
        if key.lower() == name.lower() and label not in annotated:
            cat = categories[i]
            color = colors_map.get(cat, 'white')
            ax.scatter([x[i]], [y[i]], c=color, s=90, zorder=5,
                       edgecolors='white', linewidths=1.0)
            ax.annotate(label, (x[i], y[i]),
                        textcoords='offset points', xytext=(7, 4),
                        fontsize=8, color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='black')],
                        zorder=6)
            annotated.add(label)
            break

ax.set_title('t-SNE Projection of TransE Entity Embeddings (500 entities)',
             color='white', fontsize=13, fontweight='bold', pad=14)
ax.set_xlabel('t-SNE Dimension 1', color='#8b949e', fontsize=10)
ax.set_ylabel('t-SNE Dimension 2', color='#8b949e', fontsize=10)
ax.tick_params(colors='#8b949e', labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')

ax.legend(framealpha=0.25, facecolor='#161b22', edgecolor='#30363d',
          labelcolor='white', fontsize=9, loc='lower right',
          markerscale=1.4)

# annotation note
ax.text(0.01, 0.01,
        'Notable entities annotated. Clusters reflect structural similarity in the KG.',
        transform=ax.transAxes, fontsize=7.5, color='#8b949e', va='bottom')

plt.tight_layout()
plt.savefig('figures/fig2_tsne_embeddings.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Saved: figures/fig2_tsne_embeddings.png")
