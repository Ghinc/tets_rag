"""
Test du QueryEnricher avec décomposition OppChoVec
"""

from query_enricher import QueryEnricher
from ontology_parser import OntologyParser
import json

# Initialiser
print("Chargement de l'ontologie...")
parser = OntologyParser("ontology_be_2010_bilingue_fr_en.ttl")
parser.extract_all()

enricher = QueryEnricher(parser)

# Tests avec indicateurs OppChoVec
test_queries = [
    "Pourquoi le score Vec est bas à Ajaccio ?",
    "Quel est le score Opp à Bastia ?",
    "Comment sont les indicateurs Cho à Porto-Vecchio ?",
    "Montre-moi les données sur Vec1 et Vec2"
]

print("\n" + "="*80)
print("TEST DECOMPOSITION OPPCHOVEC")
print("="*80)

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    result = enricher.enrich_query(query, max_expansions=10)

    print(f"\nOriginal: {result['original_query']}")
    print(f"Enriched: {result['enriched_query']}")

    print(f"\nOppChoVec Detected: {result['oppchovec_decomposition']['detected']}")
    if result['oppchovec_decomposition']['detected']:
        print(f"Indicators: {result['oppchovec_decomposition']['indicators']}")
        print(f"\nSub-indicators ({len(result['oppchovec_decomposition']['subindicators'])}):")
        for sub in result['oppchovec_decomposition']['subindicators']:
            print(f"  - {sub['name']}: {sub['label']}")
            print(f"    Keywords: {', '.join(sub['keywords'])}")

    print(f"\nExpansion terms: {result['expansion_terms']}")
    print(f"\nMetadata:")
    print(json.dumps(result['metadata'], indent=2, ensure_ascii=False))
