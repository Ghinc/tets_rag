"""
Script pour comparer les résultats LISA avec seuils 5% et 1%
"""

import json

# Charger les deux JSON
with open('lisa_clusters.json', 'r', encoding='utf-8') as f:
    d1 = json.load(f)

with open('lisa_clusters_1pct.json', 'r', encoding='utf-8') as f:
    d2 = json.load(f)

print("="*80)
print("COMPARAISON LISA : SEUIL 5% vs 1%")
print("="*80)
print()

print("SEUIL 5% (alpha=0.05):")
print(f"  Total communes: {len(d1['clusters'])}")
print(f"  Moran I: {d1['metadata']['moran_global_I']:.4f}")
print(f"  p-value: {d1['metadata']['moran_global_p']:.4f}")
sig_5pct = d1['metadata']['nb_significatives']
print(f"  Communes significatives: {sig_5pct} ({d1['metadata']['pourcent_significatives']:.1f}%)")

# Compter les clusters
clusters_5pct = {}
for c in d1['clusters'].values():
    cluster_type = c['cluster']
    clusters_5pct[cluster_type] = clusters_5pct.get(cluster_type, 0) + 1

print("  Distribution:")
for k, v in sorted(clusters_5pct.items()):
    print(f"    {k}: {v}")

print()

print("SEUIL 1% (alpha=0.01):")
print(f"  Total communes: {len(d2['clusters'])}")
print(f"  Moran I: {d2['metadata']['moran_global_I']:.4f}")
print(f"  p-value: {d2['metadata']['moran_global_p']:.4f}")
sig_1pct = d2['metadata']['nb_significatives']
print(f"  Communes significatives: {sig_1pct} ({d2['metadata']['pourcent_significatives']:.1f}%)")

# Compter les clusters
clusters_1pct = {}
for c in d2['clusters'].values():
    cluster_type = c['cluster']
    clusters_1pct[cluster_type] = clusters_1pct.get(cluster_type, 0) + 1

print("  Distribution:")
for k, v in sorted(clusters_1pct.items()):
    print(f"    {k}: {v}")

print()
print("="*80)
print(f"DIFFERENCE: {sig_5pct - sig_1pct} communes en moins avec le seuil 1%")
print("="*80)
