"""
Ontology Parser - Extraction et structuration de l'ontologie du bien-être

Ce module parse le fichier ontology_be_2010.ttl et extrait:
- Les concepts (Well-Being, Quality of Life, etc.)
- Les dimensions (Employment, Housing, Health, etc.)
- Les indicateurs (objectifs et subjectifs)
- Les relations entre eux

Auteur: Claude Code
Date: 2025-10-28
"""

from rdflib import Graph, Namespace, RDF, RDFS, OWL
from typing import Dict, List, Set, Tuple
import json


class OntologyParser:
    """
    Parser pour l'ontologie du bien-être territorial

    Extrait la structure conceptuelle de l'ontologie pour enrichir
    les requêtes du système RAG.
    """

    def __init__(self, ontology_path: str = "ontology_be_2010_bilingue_fr_en.ttl"):
        """
        Initialise le parser et charge l'ontologie

        Args:
            ontology_path: Chemin vers le fichier .ttl de l'ontologie
        """
        self.ontology_path = ontology_path
        self.graph = Graph()

        # Namespaces
        self.ns = Namespace("http://example.org/oppchovec#")

        # Structures de données extraites
        self.concepts = {}  # {concept_uri: {label, comment, type}}
        self.dimensions = {}  # {dimension_uri: {label, comment}}
        self.indicators = {}  # {indicator_uri: {label, comment, type, measures_dimension}}
        self.relations = []  # [(subject, predicate, object)]

        # Mapping label -> URI pour recherche rapide
        self.label_to_uri = {}
        self.keywords_to_dimension = {}  # {keyword: dimension_uri}

        # Charger l'ontologie
        self._load_ontology()

    def _load_ontology(self):
        """
        Charge le fichier RDF/Turtle de l'ontologie
        """
        print(f"Chargement de l'ontologie depuis {self.ontology_path}...")
        try:
            self.graph.parse(self.ontology_path, format="turtle")
            print(f"OK Ontologie chargee : {len(self.graph)} triples")
        except Exception as e:
            print(f"ERREUR lors du chargement : {e}")
            raise

    def extract_all(self):
        """
        Extrait tous les éléments de l'ontologie

        Cette méthode coordonne l'extraction de tous les composants
        de l'ontologie.
        """
        print("\nExtraction des éléments de l'ontologie...")

        self._extract_concepts()
        self._extract_dimensions()
        self._extract_indicators()
        self._extract_relations()
        self._build_keyword_mapping()

        print(f"\nOK Extraction terminee:")
        print(f"  - {len(self.concepts)} concepts")
        print(f"  - {len(self.dimensions)} dimensions")
        print(f"  - {len(self.indicators)} indicateurs")
        print(f"  - {len(self.relations)} relations")
        print(f"  - {len(self.keywords_to_dimension)} mots-cles mappes")

    def _extract_concepts(self):
        """
        Extrait les concepts principaux (Well-Being, Quality of Life, etc.)
        """
        # Classes qui sont des concepts
        concept_classes = [
            self.ns.WellBeing,
            self.ns.QualityOfLife,
            self.ns.Happiness,
            self.ns.BuenVivir,
            self.ns.EudaimonicWellBeing,
            self.ns.TerritorialWellBeing,
            self.ns.SubjectiveTerritorialWellBeing,
            self.ns.ObjectiveTerritorialWellBeing
        ]

        for concept_uri in concept_classes:
            label = self._get_label(concept_uri)
            comment = self._get_comment(concept_uri)

            self.concepts[str(concept_uri)] = {
                "label": label,
                "comment": comment,
                "uri": str(concept_uri)
            }

            if label:
                self.label_to_uri[label.lower()] = str(concept_uri)

    def _extract_dimensions(self):
        """
        Extrait les dimensions thématiques du bien-être territorial
        (Employment, Housing, Health, etc.)
        """
        # Classes qui sont des dimensions
        dimension_classes = [
            self.ns.Employment,
            self.ns.Housing,
            self.ns.Services,
            self.ns.Democracy,
            self.ns.Education,
            self.ns.Health,
            self.ns.EnvQuality,
            self.ns.Income,
            self.ns.WorkLifeBalance
        ]

        for dim_uri in dimension_classes:
            label = self._get_label(dim_uri)
            comment = self._get_comment(dim_uri)

            self.dimensions[str(dim_uri)] = {
                "label": label,
                "comment": comment,
                "uri": str(dim_uri)
            }

            if label:
                self.label_to_uri[label.lower()] = str(dim_uri)

    def _extract_indicators(self):
        """
        Extrait les indicateurs (objectifs et subjectifs) et leurs liens
        avec les dimensions
        """
        # Tous les indicateurs sont des instances de :Indicator ou ses sous-classes
        for indicator_uri in self.graph.subjects(RDF.type, OWL.Class):
            # Vérifier si c'est un indicateur (subClassOf :Indicator)
            if self._is_indicator(indicator_uri):
                label = self._get_label(indicator_uri)
                comment = self._get_comment(indicator_uri)

                # Déterminer le type d'indicateur
                indicator_type = "unknown"
                if self._is_subclass_of(indicator_uri, self.ns.ObjectiveIndicator):
                    indicator_type = "objective"
                elif self._is_subclass_of(indicator_uri, self.ns.SubjectiveIndicator):
                    indicator_type = "subjective"

                # Trouver la dimension mesurée
                measured_dimension = self._get_measured_dimension(indicator_uri)

                self.indicators[str(indicator_uri)] = {
                    "label": label,
                    "comment": comment,
                    "type": indicator_type,
                    "measures_dimension": measured_dimension,
                    "uri": str(indicator_uri)
                }

                if label:
                    self.label_to_uri[label.lower()] = str(indicator_uri)

    def _extract_relations(self):
        """
        Extrait les relations importantes entre concepts
        (hasDimension, measuresDimension, isOperationalizedBy, etc.)
        """
        # Propriétés importantes à extraire
        important_properties = [
            self.ns.hasDimension,
            self.ns.measuresDimension,
            self.ns.isOperationalizedBy,
            self.ns.appliesTo,
            self.ns.isEquivalentTo,
            self.ns.isAlternativeTo
        ]

        for prop in important_properties:
            for s, p, o in self.graph.triples((None, prop, None)):
                self.relations.append({
                    "subject": str(s),
                    "predicate": str(p),
                    "object": str(o),
                    "subject_label": self._get_label(s),
                    "object_label": self._get_label(o)
                })

    def _build_keyword_mapping(self):
        """
        Construit un mapping de mots-clés vers les dimensions

        Ce mapping permet d'enrichir les requêtes en identifiant
        automatiquement la dimension pertinente.

        Exemples:
        - "médecin", "santé", "hôpital" -> Health
        - "logement", "habitation" -> Housing
        - "école", "éducation" -> Education
        """
        # Mapping manuel de mots-clés français vers dimensions
        keyword_mapping = {
            # Health
            "medecin": self.ns.Health,
            "medecins": self.ns.Health,
            "docteur": self.ns.Health,
            "sante": self.ns.Health,
            "santé": self.ns.Health,
            "hopital": self.ns.Health,
            "hôpital": self.ns.Health,
            "soins": self.ns.Health,
            "pharmacie": self.ns.Health,

            # Housing
            "logement": self.ns.Housing,
            "logements": self.ns.Housing,
            "habitation": self.ns.Housing,
            "maison": self.ns.Housing,
            "appartement": self.ns.Housing,
            "loyer": self.ns.Housing,
            "prix immobilier": self.ns.Housing,

            # Education
            "ecole": self.ns.Education,
            "école": self.ns.Education,
            "ecoles": self.ns.Education,
            "écoles": self.ns.Education,
            "education": self.ns.Education,
            "éducation": self.ns.Education,
            "lycee": self.ns.Education,
            "lycée": self.ns.Education,
            "college": self.ns.Education,
            "collège": self.ns.Education,
            "universite": self.ns.Education,
            "université": self.ns.Education,
            "formation": self.ns.Education,

            # Services
            "service": self.ns.Services,
            "services": self.ns.Services,
            "commerce": self.ns.Services,
            "commerces": self.ns.Services,
            "administration": self.ns.Services,

            # Employment
            "emploi": self.ns.Employment,
            "travail": self.ns.Employment,
            "chomage": self.ns.Employment,
            "chômage": self.ns.Employment,
            "job": self.ns.Employment,

            # Income
            "revenu": self.ns.Income,
            "revenus": self.ns.Income,
            "salaire": self.ns.Income,
            "argent": self.ns.Income,

            # Democracy
            "democratie": self.ns.Democracy,
            "démocratie": self.ns.Democracy,
            "vote": self.ns.Democracy,
            "election": self.ns.Democracy,
            "élection": self.ns.Democracy,
            "politique": self.ns.Democracy,

            # Environment
            "environnement": self.ns.EnvQuality,
            "nature": self.ns.EnvQuality,
            "pollution": self.ns.EnvQuality,
            "air": self.ns.EnvQuality,
            "eau": self.ns.EnvQuality,

            # Transport (lié à Services)
            "transport": self.ns.Services,
            "transports": self.ns.Services,
            "bus": self.ns.Services,
            "train": self.ns.Services,
            "metro": self.ns.Services,
            "métro": self.ns.Services,
        }

        for keyword, dimension_uri in keyword_mapping.items():
            self.keywords_to_dimension[keyword.lower()] = str(dimension_uri)

    # Méthodes utilitaires

    def _get_label(self, uri) -> str:
        """Récupère le label rdfs:label d'un URI"""
        labels = list(self.graph.objects(uri, RDFS.label))
        return str(labels[0]) if labels else None

    def _get_comment(self, uri) -> str:
        """Récupère le commentaire rdfs:comment d'un URI"""
        comments = list(self.graph.objects(uri, RDFS.comment))
        return str(comments[0]) if comments else None

    def _is_indicator(self, uri) -> bool:
        """Vérifie si un URI est un indicateur"""
        return self._is_subclass_of(uri, self.ns.Indicator)

    def _is_subclass_of(self, uri, parent_class) -> bool:
        """Vérifie si uri est une sous-classe de parent_class"""
        # Chercher récursivement dans rdfs:subClassOf
        for subclass_stmt in self.graph.objects(uri, RDFS.subClassOf):
            if subclass_stmt == parent_class:
                return True
            if self._is_subclass_of(subclass_stmt, parent_class):
                return True
        return False

    def _get_measured_dimension(self, indicator_uri) -> str:
        """Trouve la dimension mesurée par un indicateur"""
        for dim in self.graph.objects(indicator_uri, self.ns.measuresDimension):
            return str(dim)
        return None

    # Méthodes publiques pour l'enrichissement de requêtes

    def find_dimension_for_query(self, query: str) -> List[str]:
        """
        Identifie les dimensions pertinentes pour une requête

        Args:
            query: Requête utilisateur

        Returns:
            Liste des URIs de dimensions pertinentes
        """
        query_lower = query.lower()
        dimensions_found = []

        for keyword, dimension_uri in self.keywords_to_dimension.items():
            if keyword in query_lower:
                if dimension_uri not in dimensions_found:
                    dimensions_found.append(dimension_uri)

        return dimensions_found

    def get_dimension_info(self, dimension_uri: str) -> Dict:
        """Récupère les informations sur une dimension"""
        return self.dimensions.get(dimension_uri, {})

    def get_related_concepts(self, dimension_uri: str) -> List[str]:
        """
        Trouve les concepts liés à une dimension

        Args:
            dimension_uri: URI de la dimension

        Returns:
            Liste des concepts reliés (labels)
        """
        related = []

        # Trouver les indicateurs qui mesurent cette dimension
        for indicator_uri, info in self.indicators.items():
            if info.get("measures_dimension") == dimension_uri:
                if info.get("label"):
                    related.append(info["label"])

        return related

    def export_to_json(self, output_path: str = "ontology_parsed.json"):
        """
        Exporte l'ontologie parsée en JSON pour inspection

        Args:
            output_path: Chemin du fichier JSON de sortie
        """
        export_data = {
            "concepts": self.concepts,
            "dimensions": self.dimensions,
            "indicators": self.indicators,
            "relations": self.relations,
            "keywords_to_dimension": self.keywords_to_dimension
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\nOK Ontologie exportee vers {output_path}")


def main():
    """
    Fonction principale pour tester le parser
    """
    print("="*80)
    print("ONTOLOGY PARSER - Test")
    print("="*80)

    # Créer le parser
    parser = OntologyParser("ontology_be_2010_bilingue_fr_en.ttl")

    # Extraire tous les éléments
    parser.extract_all()

    # Exporter en JSON pour inspection
    parser.export_to_json("ontology_parsed.json")

    # Tests d'enrichissement
    print("\n" + "="*80)
    print("TESTS D'ENRICHISSEMENT DE REQUÊTES")
    print("="*80)

    test_queries = [
        "Combien y a-t-il de médecins à Afa ?",
        "Comment est le logement à Ajaccio ?",
        "Quels sont les problèmes de transport à Bastia ?",
        "Comment est l'éducation à Corte ?"
    ]

    for query in test_queries:
        print(f"\nRequête: {query}")
        dimensions = parser.find_dimension_for_query(query)

        if dimensions:
            print(f"  Dimensions identifiées:")
            for dim_uri in dimensions:
                dim_info = parser.get_dimension_info(dim_uri)
                print(f"    - {dim_info.get('label', 'Unknown')}")
        else:
            print("  Aucune dimension identifiée")


if __name__ == "__main__":
    main()
