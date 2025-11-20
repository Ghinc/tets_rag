"""
Query Enricher - Enrichissement de requêtes basé sur l'ontologie

Ce module utilise l'ontologie du bien-être pour enrichir les requêtes
utilisateur avec des concepts et dimensions pertinents.

Stratégies d'enrichissement:
1. Expansion de mots-clés (synonymes via l'ontologie)
2. Ajout de termes liés aux dimensions identifiées
3. Expansion via les indicateurs associés

Auteur: Claude Code
Date: 2025-10-28
"""

from ontology_parser import OntologyParser
from typing import List, Dict, Set
import unicodedata


class QueryEnricher:
    """
    Enrichisseur de requêtes basé sur l'ontologie

    Utilise la structure de l'ontologie pour enrichir les requêtes
    et améliorer la qualité du retrieval dans le RAG.
    """

    def __init__(self, ontology_parser: OntologyParser):
        """
        Initialise l'enrichisseur

        Args:
            ontology_parser: Parser d'ontologie initialisé
        """
        self.parser = ontology_parser

    def normalize_text(self, text: str) -> str:
        """
        Normalise un texte pour la comparaison
        (enlève les accents, met en minuscules)

        Args:
            text: Texte à normaliser

        Returns:
            Texte normalisé
        """
        # Enlever les accents
        nfd = unicodedata.normalize('NFD', text)
        without_accents = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        return without_accents.lower()

    def enrich_query(self, query: str, max_expansions: int = 5) -> Dict:
        """
        Enrichit une requête avec des termes issus de l'ontologie
        Détecte et décompose automatiquement les indicateurs OppChoVec

        Args:
            query: Requête utilisateur originale
            max_expansions: Nombre max de termes d'expansion

        Returns:
            Dict contenant:
            - original_query: Requête originale
            - dimensions: Dimensions identifiées
            - expansion_terms: Termes d'expansion
            - enriched_query: Requête enrichie
            - metadata: Métadonnées sur l'enrichissement
            - oppchovec_decomposition: Décomposition des indicateurs OppChoVec détectés
        """
        # Normaliser la requête pour la recherche
        query_normalized = self.normalize_text(query)

        # 1. Détecter les indicateurs OppChoVec et leurs sous-composants
        oppchovec_info = self._detect_oppchovec_indicators(query_normalized)

        # 2. Identifier les dimensions pertinentes
        dimensions = self._find_dimensions(query_normalized)

        # 3. Extraire les termes d'expansion
        expansion_terms = set()

        # Ajouter les sous-composants OppChoVec comme termes d'expansion
        if oppchovec_info['detected']:
            for subindicator in oppchovec_info['subindicators']:
                expansion_terms.add(subindicator['label'])
                # Ajouter aussi les mots-clés liés
                if 'keywords' in subindicator:
                    for kw in subindicator['keywords'][:2]:  # Limiter à 2 mots-clés par sous-indicateur
                        expansion_terms.add(kw)

        for dim_uri in dimensions:
            # Ajouter le label de la dimension
            dim_info = self.parser.get_dimension_info(dim_uri)
            if dim_info.get('label'):
                expansion_terms.add(dim_info['label'])

            # Ajouter les indicateurs liés
            related = self.parser.get_related_concepts(dim_uri)
            for concept in related[:max_expansions]:
                expansion_terms.add(concept)

        # 4. Construire la requête enrichie
        enriched_query = query
        if expansion_terms:
            # Ajouter les termes d'expansion à la requête
            expansion_str = " ".join(list(expansion_terms)[:max_expansions])
            enriched_query = f"{query} {expansion_str}"

        # 5. Préparer les métadonnées
        metadata = {
            "dimensions_found": len(dimensions),
            "expansion_terms_count": len(expansion_terms),
            "dimension_labels": [
                self.parser.get_dimension_info(d).get('label', 'Unknown')
                for d in dimensions
            ],
            "oppchovec_detected": oppchovec_info['detected'],
            "oppchovec_indicators": oppchovec_info['indicators']
        }

        return {
            "original_query": query,
            "dimensions": dimensions,
            "expansion_terms": list(expansion_terms)[:max_expansions],
            "enriched_query": enriched_query,
            "metadata": metadata,
            "oppchovec_decomposition": oppchovec_info
        }

    def _detect_oppchovec_indicators(self, query_normalized: str) -> Dict:
        """
        Détecte les indicateurs OppChoVec et retourne leurs sous-composants

        Args:
            query_normalized: Requête normalisée

        Returns:
            Dict avec informations sur les indicateurs détectés
        """
        # Mapping des indicateurs vers leurs sous-composants et mots-clés associés
        oppchovec_mapping = {
            'opp': {
                'label': 'Opportunités',
                'subindicators': [
                    {'name': 'Opp1', 'label': 'Education', 'keywords': ['education', 'diplome', 'niveau scolaire']},
                    {'name': 'Opp2', 'label': 'Diversité sociale', 'keywords': ['diversite', 'mixite sociale', 'inegalite']},
                    {'name': 'Opp3', 'label': 'Transports', 'keywords': ['transport', 'mobilite', 'voiture', 'bus']},
                    {'name': 'Opp4', 'label': 'Technologies', 'keywords': ['technologie', 'numerique', 'internet', 'communication']}
                ]
            },
            'cho': {
                'label': 'Choix',
                'subindicators': [
                    {'name': 'Chx1', 'label': 'Choix de logement', 'keywords': ['logement', 'habitation', 'residence']},
                    {'name': 'Chx2', 'label': 'Choix emploi', 'keywords': ['emploi', 'travail', 'metier', 'profession']}
                ]
            },
            'vec': {
                'label': 'Vécu',
                'subindicators': [
                    {'name': 'Vec1', 'label': 'Sécurité', 'keywords': ['securite', 'insecurite', 'peur', 'danger']},
                    {'name': 'Vec2', 'label': 'Environnement', 'keywords': ['environnement', 'cadre de vie', 'qualite', 'pollution']},
                    {'name': 'Vec3', 'label': 'Santé', 'keywords': ['sante', 'medical', 'hopital', 'medecin']},
                    {'name': 'Vec4', 'label': 'Social', 'keywords': ['social', 'lien social', 'communaute', 'voisinage']}
                ]
            }
        }

        detected_indicators = []
        all_subindicators = []

        # Détecter les indicateurs principaux (Opp, Cho, Vec)
        for indicator_key, indicator_data in oppchovec_mapping.items():
            if indicator_key in query_normalized:
                detected_indicators.append(indicator_data['label'])
                all_subindicators.extend(indicator_data['subindicators'])

        # Détecter aussi les mentions directes de sous-indicateurs (opp1, vec2, etc.)
        for indicator_key, indicator_data in oppchovec_mapping.items():
            for sub in indicator_data['subindicators']:
                sub_name_normalized = self.normalize_text(sub['name'])
                if sub_name_normalized in query_normalized:
                    if sub not in all_subindicators:
                        all_subindicators.append(sub)
                    if indicator_data['label'] not in detected_indicators:
                        detected_indicators.append(indicator_data['label'])

        return {
            'detected': len(detected_indicators) > 0,
            'indicators': detected_indicators,
            'subindicators': all_subindicators
        }

    def _find_dimensions(self, query_normalized: str) -> List[str]:
        """
        Trouve les dimensions pertinentes dans une requête

        Args:
            query_normalized: Requête normalisée (sans accents, minuscules)

        Returns:
            Liste des URIs de dimensions
        """
        dimensions = []

        # Chercher dans le mapping de mots-clés
        for keyword, dim_uri in self.parser.keywords_to_dimension.items():
            keyword_normalized = self.normalize_text(keyword)

            if keyword_normalized in query_normalized:
                if dim_uri not in dimensions:
                    dimensions.append(dim_uri)

        return dimensions

    def enrich_query_simple(self, query: str) -> str:
        """
        Version simplifiée qui retourne juste la requête enrichie

        Args:
            query: Requête originale

        Returns:
            Requête enrichie sous forme de string
        """
        result = self.enrich_query(query)
        return result["enriched_query"]

    def explain_enrichment(self, query: str) -> str:
        """
        Explique comment une requête a été enrichie
        (utile pour le debugging et la documentation)

        Args:
            query: Requête à analyser

        Returns:
            Explication textuelle de l'enrichissement
        """
        result = self.enrich_query(query)

        lines = []
        lines.append(f"Requete originale: {query}")
        lines.append(f"\nDimensions identifiees ({result['metadata']['dimensions_found']}):")

        if result['metadata']['dimension_labels']:
            for label in result['metadata']['dimension_labels']:
                lines.append(f"  - {label}")
        else:
            lines.append("  (aucune)")

        lines.append(f"\nTermes d'expansion ({result['metadata']['expansion_terms_count']}):")
        if result['expansion_terms']:
            for term in result['expansion_terms']:
                lines.append(f"  - {term}")
        else:
            lines.append("  (aucun)")

        lines.append(f"\nRequete enrichie:")
        lines.append(f"  {result['enriched_query']}")

        return "\n".join(lines)


def main():
    """
    Fonction principale pour tester l'enrichisseur
    """
    print("="*80)
    print("QUERY ENRICHER - Test")
    print("="*80)

    # Charger l'ontologie
    print("\n1. Chargement de l'ontologie...")
    parser = OntologyParser("ontology_be_2010_bilingue_fr_en.ttl")
    parser.extract_all()

    # Créer l'enrichisseur
    print("\n2. Creation de l'enrichisseur...")
    enricher = QueryEnricher(parser)

    # Tests
    print("\n" + "="*80)
    print("TESTS D'ENRICHISSEMENT")
    print("="*80)

    test_queries = [
        "Combien y a-t-il de medecins a Afa ?",
        "Comment est le logement a Ajaccio ?",
        "Quels sont les problemes de transport a Bastia ?",
        "Comment est l'education a Corte ?",
        "Quel est le taux de chomage ?",
        "Comment est l'environnement a Porto-Vecchio ?"
    ]

    for query in test_queries:
        print("\n" + "-"*80)
        print(enricher.explain_enrichment(query))

    # Test avec export JSON
    print("\n" + "="*80)
    print("EXPORT JSON D'UN ENRICHISSEMENT")
    print("="*80)

    import json
    result = enricher.enrich_query("Comment est la sante a Ajaccio ?")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
