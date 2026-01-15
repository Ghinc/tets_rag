"""
Custom Graph Retriever qui utilise des requêtes Cypher directes
pour interroger l'ontologie BE-2010 dans Neo4j

Compatible avec l'architecture LlamaIndex (BaseRetriever)
"""

from typing import List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from neo4j import GraphDatabase


class CypherGraphRetriever(BaseRetriever):
    """
    Retriever personnalisé qui interroge Neo4j avec des requêtes Cypher
    pour récupérer des informations depuis l'ontologie BE-2010
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        similarity_top_k: int = 5
    ):
        """
        Args:
            neo4j_uri: URI Neo4j (bolt://localhost:7687)
            neo4j_user: Username Neo4j
            neo4j_password: Password Neo4j
            similarity_top_k: Nombre max de résultats
        """
        super().__init__()
        self._driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        self._similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Récupère les nœuds pertinents depuis Neo4j via Cypher

        Args:
            query_bundle: Bundle contenant la query string

        Returns:
            Liste de NodeWithScore
        """
        query_str = query_bundle.query_str

        # Extraire les mots-clés de la query
        keywords = self._extract_keywords(query_str)

        # Construire la requête Cypher
        cypher_query = self._build_cypher_query(keywords)

        # Exécuter la requête
        with self._driver.session() as session:
            result = session.run(cypher_query, {"keywords": keywords, "limit": self._similarity_top_k})

            nodes = []
            for i, record in enumerate(result):
                # Récupérer les données du nœud
                node_data = record.get("node", {})
                label = record.get("label", "")
                properties = record.get("properties", {})
                relations = record.get("relations", [])

                # Construire le texte du nœud (avec relations)
                text = self._format_node_text(label, properties, relations)

                # Créer un TextNode
                node = TextNode(
                    text=text,
                    id_=f"neo4j_node_{i}",
                    metadata={
                        "source": "neo4j_ontology",
                        "label": label,
                        "relations_count": len(relations),
                        **properties
                    }
                )

                # Score basé sur la position + nombre de relations (boost si bien connecté)
                base_score = 1.0 - (i * 0.1)
                relation_boost = min(len(relations) * 0.05, 0.2)  # Max +0.2 bonus
                score = min(base_score + relation_boost, 1.0)

                nodes.append(NodeWithScore(node=node, score=score))

            return nodes

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extrait des mots-clés de la query et les traduit en anglais

        Args:
            query: Texte de la requête

        Returns:
            Liste de mots-clés (FR + EN)
        """
        # Simple tokenization - lowercase et split
        words = query.lower().split()

        # Filtrer les stop words communs
        stop_words = {
            "le", "la", "les", "de", "du", "des", "un", "une",
            "et", "ou", "est", "sont", "dans", "pour", "sur",
            "à", "au", "aux", "que", "qui", "quels", "quel", "quelle",
            "comment", "quoi", "quelles"
        }

        keywords_fr = [w for w in words if w not in stop_words and len(w) > 2]

        # Mapping FR → EN pour termes clés de l'ontologie BE-2010
        translations = {
            "santé": "health",
            "bien-être": "wellbeing well-being",
            "bien": "well",
            "être": "being",
            "éducation": "education",
            "emploi": "employment",
            "revenu": "income",
            "logement": "housing",
            "environnement": "environmental",
            "démocratie": "democracy",
            "services": "services",
            "équilibre": "balance",
            "travail": "work",
            "vie": "life",
            "qualité": "quality",
            "bonheur": "happiness",
            "indicateurs": "indicators",
            "dimensions": "dimensions",
            "concepts": "concepts",
            "relations": "relationships relations"
        }

        # Ajouter traductions
        keywords_en = []
        for kw in keywords_fr:
            if kw in translations:
                # Ajouter les traductions (peuvent être multiples)
                keywords_en.extend(translations[kw].split())

        # Combiner FR + EN, dédupliquer
        all_keywords = list(set(keywords_fr + keywords_en))

        return all_keywords[:10]  # Limiter à 10 mots-clés

    def _build_cypher_query(self, keywords: List[str]) -> str:
        """
        Construit une requête Cypher pour chercher dans l'ontologie

        Args:
            keywords: Liste de mots-clés

        Returns:
            Requête Cypher
        """
        # Requête qui cherche des nœuds + leurs voisins connectés
        # Cela enrichit les résultats avec le contexte relationnel
        cypher = """
        MATCH (n)
        WHERE (n:Concept OR n:Dimension OR n:Indicator OR n:SubjectiveIndicator OR n:ObjectiveIndicator)
          AND ANY(kw IN $keywords WHERE
            toLower(n.uri) CONTAINS toLower(kw) OR
            toLower(n.label) CONTAINS toLower(kw) OR
            toLower(n.comment) CONTAINS toLower(kw)
        )
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE m:Concept OR m:Dimension OR m:Indicator
        WITH n,
             labels(n) AS label,
             properties(n) AS properties,
             collect(DISTINCT {type: type(r), node: m.label}) AS relations
        RETURN
            label,
            properties,
            n AS node,
            relations
        LIMIT $limit
        """

        return cypher

    def _format_node_text(self, label: str, properties: dict, relations: list = None) -> str:
        """
        Formate les données d'un nœud en texte lisible

        Args:
            label: Label du nœud
            properties: Propriétés du nœud
            relations: Liste des relations (optionnel)

        Returns:
            Texte formaté
        """
        parts = []

        # Label
        if label:
            parts.append(f"Type: {label}")

        # label (nom)
        if "label" in properties:
            parts.append(f"Nom: {properties['label']}")

        # comment (description)
        if "comment" in properties:
            parts.append(f"Description: {properties['comment']}")

        # URI
        if "uri" in properties:
            parts.append(f"URI: {properties['uri']}")

        # Relations (si présentes)
        if relations and len(relations) > 0:
            rel_strs = []
            for rel in relations[:5]:  # Limiter à 5 relations
                if rel.get('node'):
                    rel_type = rel.get('type', 'RELATED_TO')
                    rel_strs.append(f"{rel_type} → {rel['node']}")
            if rel_strs:
                parts.append(f"Relations: {', '.join(rel_strs)}")

        # Autres propriétés intéressantes
        for key, value in properties.items():
            if key not in ["uri", "label", "comment"] and value:
                # Nettoyer le nom de la propriété
                clean_key = key.replace("__", ":").replace("_", " ")
                parts.append(f"{clean_key}: {value}")

        return "\n".join(parts)

    def __del__(self):
        """Ferme la connexion Neo4j"""
        if hasattr(self, '_driver'):
            self._driver.close()
