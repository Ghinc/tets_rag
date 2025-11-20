"""
Debug de l'enrichissement pour la requête Opp Bastia
"""
from ontology_parser import OntologyParser
from query_enricher import QueryEnricher
import json

# Initialiser
print("Chargement de l'ontologie...")
parser = OntologyParser("ontology_be_2010_bilingue_fr_en.ttl")
parser.extract_all()

enricher = QueryEnricher(parser)

# Tester la requête
question = "Quel est le score Opp à Bastia ?"
print(f"\nQuestion : {question}")
print("="*80)

result = enricher.enrich_query(question, max_expansions=10)

print("\nRésultat de l'enrichissement :")
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n" + "="*80)
print("ANALYSE")
print("="*80)
print(f"Requête originale : {result['original_query']}")
print(f"Requête enrichie  : {result['enriched_query']}")
print(f"\nOppChoVec détecté : {result['oppchovec_decomposition']['detected']}")
if result['oppchovec_decomposition']['detected']:
    print(f"Indicateurs : {result['oppchovec_decomposition']['indicators']}")
    print(f"Sous-indicateurs ({len(result['oppchovec_decomposition']['subindicators'])}) :")
    for sub in result['oppchovec_decomposition']['subindicators']:
        print(f"  - {sub['name']}: {sub['label']}")
