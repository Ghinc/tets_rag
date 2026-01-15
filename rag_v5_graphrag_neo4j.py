"""
RAG v5 - Graph-RAG avec Neo4j

Ce module implémente un Graph-RAG qui :
1. Charge l'ontologie du bien-être dans Neo4j
2. Crée un graphe de connaissances avec les communes, indicateurs, et relations
3. Utilise le graphe pour enrichir le retrieval
4. Combine retrieval vectoriel (ChromaDB) + graphe (Neo4j)
5. Détection automatique des communes et filtrage ChromaDB

Architecture:
- Neo4j pour le graphe de connaissances (ontologie + données)
- ChromaDB pour le retrieval vectoriel classique
- Fusion des résultats pour une réponse enrichie

Auteur: Claude Code
Date: 2025-01-04
"""

import os
from typing import List, Dict, Tuple, Optional
import pandas as pd
from neo4j import GraphDatabase
import chromadb
import openai

# Détection de communes
from commune_detector import detect_commune

# Imports des modules existants
from rag_v2_improved import (
    ImprovedSemanticChunker,
    FrenchEmbeddingModel,
    CrossEncoderReranker,
    RetrievalResult,
    ImprovedPromptBuilder
)

from ontology_parser import OntologyParser


class Neo4jGraphManager:
    """
    Gestionnaire de graphe Neo4j pour Graph-RAG

    Gère la connexion, l'import de l'ontologie, et les requêtes Cypher
    """

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: Optional[str] = None):
        """
        Initialise la connexion Neo4j

        Args:
            uri: URI de connexion Neo4j
            user: Nom d'utilisateur (ignoré si password est None)
            password: Mot de passe (None pour connexion sans auth)
        """
        print(f"Connexion à Neo4j: {uri}")

        # Connexion avec ou sans authentification
        if password is None or password == "":
            print("  Mode: Sans authentification")
            self.driver = GraphDatabase.driver(uri)
        else:
            print(f"  Mode: Avec authentification (user: {user})")
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Vérifier la connexion
        try:
            self.driver.verify_connectivity()
            print("[OK] Connexion Neo4j établie")
        except Exception as e:
            print(f"[X] Erreur de connexion Neo4j: {e}")
            raise

    def close(self):
        """Ferme la connexion Neo4j"""
        self.driver.close()

    def clear_database(self):
        """Efface toutes les données de la base (DANGER !)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("[OK] Base de données effacée")

    def check_ontology_exists(self) -> bool:
        """
        Vérifie si l'ontologie est déjà importée dans Neo4j

        Détecte à la fois :
        - Labels personnalisés (Concept, Dimension, Indicator)
        - Labels neosemantics/n10s (owl__Class, owl__ObjectProperty, etc.)

        Returns:
            True si l'ontologie existe, False sinon
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE any(label IN labels(n) WHERE
                    label IN ['Concept', 'Dimension', 'Indicator'] OR
                    label STARTS WITH 'owl__' OR
                    label = 'Resource'
                )
                RETURN count(n) AS count
            """)
            record = result.single()
            count = record['count'] if record else 0
            return count > 0

    def import_ontology(self, ontology_parser: OntologyParser, force_reimport: bool = False):
        """
        Importe l'ontologie dans Neo4j

        Crée les nœuds :
        - Concepts (WellBeing, QualityOfLife, etc.)
        - Dimensions (Health, Housing, Education, etc.)
        - Indicateurs (ObjectiveIndicator, SubjectiveIndicator)

        Crée les relations :
        - (Concept)-[:HAS_DIMENSION]->(Dimension)
        - (Indicator)-[:MEASURES]->(Dimension)

        Args:
            ontology_parser: Parser d'ontologie
            force_reimport: Si True, réimporte même si déjà présente
        """
        # Vérifier si l'ontologie existe déjà
        if not force_reimport and self.check_ontology_exists():
            print("\n[OK] Ontologie déjà présente dans Neo4j (import skippé)")
            print("  Utilisez force_reimport=True pour réimporter")
            return

        print("\nImport de l'ontologie dans Neo4j...")

        with self.driver.session() as session:
            # 1. Créer les concepts
            for concept_uri, concept_data in ontology_parser.concepts.items():
                session.run("""
                    MERGE (c:Concept {uri: $uri})
                    SET c.label = $label,
                        c.comment = $comment
                """, uri=concept_uri,
                     label=concept_data.get('label'),
                     comment=concept_data.get('comment'))

            print(f"  [OK] {len(ontology_parser.concepts)} concepts importés")

            # 2. Créer les dimensions
            for dim_uri, dim_data in ontology_parser.dimensions.items():
                session.run("""
                    MERGE (d:Dimension {uri: $uri})
                    SET d.label = $label,
                        d.comment = $comment
                """, uri=dim_uri,
                     label=dim_data.get('label'),
                     comment=dim_data.get('comment'))

            print(f"  [OK] {len(ontology_parser.dimensions)} dimensions importées")

            # 3. Créer les indicateurs
            for ind_uri, ind_data in ontology_parser.indicators.items():
                indicator_type = ind_data.get('type', 'unknown')
                label_type = "ObjectiveIndicator" if indicator_type == "objective" else "SubjectiveIndicator"

                session.run(f"""
                    MERGE (i:Indicator:{label_type} {{uri: $uri}})
                    SET i.label = $label,
                        i.comment = $comment,
                        i.type = $type
                """, uri=ind_uri,
                     label=ind_data.get('label'),
                     comment=ind_data.get('comment'),
                     type=indicator_type)

            print(f"  [OK] {len(ontology_parser.indicators)} indicateurs importés")

            # 4. Créer les relations
            relation_count = 0
            for relation in ontology_parser.relations:
                rel_type = self._extract_relation_type(relation['predicate'])

                session.run(f"""
                    MATCH (s {{uri: $subject_uri}})
                    MATCH (o {{uri: $object_uri}})
                    MERGE (s)-[:{rel_type}]->(o)
                """, subject_uri=relation['subject'],
                     object_uri=relation['object'])

                relation_count += 1

            print(f"  [OK] {relation_count} relations créées")

    def _extract_relation_type(self, predicate_uri: str) -> str:
        """Extrait le type de relation depuis l'URI du prédicat"""
        # http://example.org/oppchovec#hasDimension -> HAS_DIMENSION
        relation_name = predicate_uri.split('#')[-1].split('/')[-1]
        # CamelCase -> SNAKE_CASE
        import re
        relation_name = re.sub('([a-z])([A-Z])', r'\1_\2', relation_name).upper()
        return relation_name

    def import_commune_data(self, df: pd.DataFrame):
        """
        Importe les données des communes et leurs indicateurs

        Args:
            df: DataFrame avec colonnes: commune, indicateur1, indicateur2, etc.
        """
        print("\nImport des données de communes...")

        with self.driver.session() as session:
            for _, row in df.iterrows():
                commune_name = row.get('commune')
                if not commune_name:
                    continue

                # Créer le nœud Commune
                session.run("""
                    MERGE (c:Commune {name: $name})
                """, name=commune_name)

                # Créer les valeurs d'indicateurs
                for col in df.columns:
                    if col == 'commune':
                        continue

                    value = row[col]
                    if pd.notna(value):
                        session.run("""
                            MATCH (c:Commune {name: $commune})
                            MERGE (i:IndicatorValue {
                                commune: $commune,
                                indicator_name: $indicator,
                                value: $value
                            })
                            MERGE (c)-[:HAS_INDICATOR_VALUE]->(i)
                        """, commune=commune_name,
                             indicator=col,
                             value=float(value))

        print(f"  [OK] {len(df)} communes importées")

    def find_related_dimensions(self, keywords: List[str], limit: int = 5) -> List[Dict]:
        """
        Trouve les dimensions pertinentes pour des mots-clés

        Args:
            keywords: Liste de mots-clés (ex: ['santé', 'médecin'])
            limit: Nombre max de dimensions à retourner

        Returns:
            Liste de dictionnaires avec label, uri, score
        """
        with self.driver.session() as session:
            # Requête Cypher pour trouver les dimensions par mots-clés
            result = session.run("""
                MATCH (d:Dimension)
                WHERE ANY(keyword IN $keywords WHERE
                    toLower(d.label) CONTAINS toLower(keyword) OR
                    toLower(d.comment) CONTAINS toLower(keyword)
                )
                RETURN d.label AS label, d.uri AS uri, d.comment AS comment
                LIMIT $limit
            """, keywords=keywords, limit=limit)

            return [dict(record) for record in result]

    def find_communes_by_indicator(self, indicator_name: str,
                                   top_n: int = 5,
                                   ascending: bool = False) -> List[Dict]:
        """
        Trouve les communes avec les meilleurs/pires scores pour un indicateur

        Args:
            indicator_name: Nom de l'indicateur
            top_n: Nombre de communes à retourner
            ascending: True pour les plus bas scores, False pour les plus hauts

        Returns:
            Liste de communes avec leurs scores
        """
        order = "ASC" if ascending else "DESC"

        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (c:Commune)-[:HAS_INDICATOR_VALUE]->(iv:IndicatorValue)
                WHERE iv.indicator_name = $indicator_name
                RETURN c.name AS commune, iv.value AS value
                ORDER BY iv.value {order}
                LIMIT $top_n
            """, indicator_name=indicator_name, top_n=top_n)

            return [dict(record) for record in result]

    def get_commune_full_context(self, commune_name: str) -> Dict:
        """
        Récupère le contexte complet d'une commune depuis le graphe

        Args:
            commune_name: Nom de la commune

        Returns:
            Dict avec tous les indicateurs et relations de la commune
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Commune {name: $commune})-[:HAS_INDICATOR_VALUE]->(iv:IndicatorValue)
                RETURN c.name AS commune,
                       collect({indicator: iv.indicator_name, value: iv.value}) AS indicators
            """, commune=commune_name)

            record = result.single()
            if record:
                return dict(record)
            return {}

    def graph_enhanced_query(self, question: str,
                            commune_filter: Optional[str] = None) -> str:
        """
        Requête enrichie par le graphe

        Utilise le graphe pour :
        1. Identifier les dimensions pertinentes
        2. Trouver les indicateurs liés
        3. Enrichir le contexte avec les relations ontologiques

        Args:
            question: Question utilisateur
            commune_filter: Commune spécifique (optionnel)

        Returns:
            Contexte enrichi sous forme de texte
        """
        # Extraire les mots-clés de la question
        keywords = question.lower().split()

        # Trouver les dimensions pertinentes
        dimensions = self.find_related_dimensions(keywords, limit=3)

        context_parts = []

        if dimensions:
            context_parts.append("=== CONTEXTE DU GRAPHE DE CONNAISSANCES ===\n")
            context_parts.append("Dimensions identifiées:")
            for dim in dimensions:
                context_parts.append(f"  - {dim['label']}: {dim.get('comment', '')}")

        # Si commune spécifiée, récupérer son contexte
        if commune_filter:
            commune_ctx = self.get_commune_full_context(commune_filter)
            if commune_ctx:
                context_parts.append(f"\nIndicateurs pour {commune_filter}:")
                for ind in commune_ctx.get('indicators', []):
                    context_parts.append(f"  - {ind['indicator']}: {ind['value']:.2f}")

        return "\n".join(context_parts)


class GraphRAGPipeline:
    """
    Pipeline Graph-RAG combinant Neo4j + ChromaDB
    """

    def __init__(self,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: Optional[str] = None,
                 chroma_path: str = "./chroma_v2",
                 collection_name: str = "communes_corses_v2",
                 embedding_model: str = "BAAI/bge-m3",
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 llm_model: str = "gpt-3.5-turbo",
                 openai_api_key: Optional[str] = None,
                 ontology_path: str = "ontology_be_2010_bilingue_fr_en.ttl"):
        """
        Initialise le pipeline Graph-RAG
        """
        print("="*80)
        print("INITIALISATION GRAPH-RAG (Neo4j + ChromaDB)")
        print("="*80)

        # Configuration OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key

        self.llm_model = llm_model

        # 1. Neo4j Graph Manager
        self.graph = Neo4jGraphManager(neo4j_uri, neo4j_user, neo4j_password)

        # 2. Ontology Parser
        print("\nChargement de l'ontologie...")
        self.ontology_parser = OntologyParser(ontology_path)
        self.ontology_parser.extract_all()

        # 3. Import dans Neo4j (détection automatique si déjà présente)
        self.graph.import_ontology(self.ontology_parser, force_reimport=False)

        # 4. Composants vectoriels (comme v2)
        self.chunker = ImprovedSemanticChunker(chunk_size=500, chunk_overlap=100)
        self.embed_model = FrenchEmbeddingModel(embedding_model)
        self.reranker = CrossEncoderReranker(reranker_model)

        # 5. ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print("\n[OK] Graph-RAG pipeline initialisé")

    def import_commune_data(self, csv_path: str):
        """
        Importe les données de communes dans Neo4j

        Args:
            csv_path: Chemin vers le CSV avec les données quantitatives
        """
        df = pd.read_csv(csv_path)
        self.graph.import_commune_data(df)

    def query(self, question: str,
              k: int = 5,
              use_graph: bool = True,
              use_reranking: bool = True,
              commune_filter: Optional[str] = None) -> Tuple[str, List[RetrievalResult]]:
        """
        Requête Graph-RAG complète

        Combine :
        1. Retrieval vectoriel (ChromaDB)
        2. Enrichissement par le graphe (Neo4j)
        3. Reranking
        4. Génération LLM

        Args:
            question: Question utilisateur
            k: Nombre de résultats vectoriels
            use_graph: Utiliser l'enrichissement par graphe
            use_reranking: Utiliser le reranking
            commune_filter: Commune spécifique

        Returns:
            (réponse, résultats)
        """
        print(f"\n{'='*80}")
        print(f"REQUÊTE GRAPH-RAG: {question}")
        print(f"{'='*80}")

        # 1. Enrichissement par le graphe (NOUVEAU)
        graph_context = ""
        if use_graph:
            print("Enrichissement par le graphe Neo4j...")
            graph_context = self.graph.graph_enhanced_query(question, commune_filter)
            print(f"[OK] Contexte graphe généré ({len(graph_context)} caractères)")

        # 2. Détecter commune dans la question
        detected_commune = detect_commune(question)
        if detected_commune:
            print(f"[AUTO-DETECT] Commune détectée: {detected_commune}")
            if not commune_filter:
                commune_filter = detected_commune

        # 3. Retrieval vectoriel classique
        print("Retrieval vectoriel...")
        query_embedding = self.embed_model.encode_query(question)

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": k*2,
            "include": ["documents", "metadatas", "distances"]
        }

        # Ajouter filtre par commune si spécifié
        if commune_filter:
            query_params["where"] = {"nom": commune_filter}
            print(f"[FILTRE v5] Recherche limitée à: {commune_filter}")

        dense_results = self.collection.query(**query_params)

        # Convertir en RetrievalResult
        results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            dense_results['documents'][0],
            dense_results['metadatas'][0],
            dense_results['distances'][0]
        )):
            score = 1.0 - distance  # Distance cosinus -> Similarité (plus correct)
            results.append(RetrievalResult(
                text=doc,
                metadata=metadata,
                score=score,
                source_type='vectoriel'
            ))

        # 3. Reranking
        if use_reranking:
            print("Reranking...")
            results = self.reranker.rerank(question, results, top_k=k)
        else:
            results = results[:k]

        # 4. Construire le prompt avec contexte graphe
        prompt = self._build_graph_rag_prompt(question, results, graph_context)

        # 5. Génération LLM
        print("Génération de la réponse...")
        response = self._generate_response(prompt)

        return response, results

    def _build_graph_rag_prompt(self, question: str,
                                results: List[RetrievalResult],
                                graph_context: str) -> str:
        """
        Construit un prompt enrichi par le graphe
        """
        prompt_parts = []

        # 1. Contexte du graphe (NOUVEAU)
        if graph_context:
            prompt_parts.append(graph_context)
            prompt_parts.append("\n")

        # 2. Résultats vectoriels
        prompt_parts.append("=== EXTRAITS D'ENTRETIENS (RETRIEVAL VECTORIEL) ===\n")
        for i, result in enumerate(results, 1):
            commune = result.metadata.get('nom', result.metadata.get('commune', 'N/A'))
            prompt_parts.append(f"\n[Extrait {i}]")
            prompt_parts.append(f"Commune: {commune}")
            prompt_parts.append(f"Score: {result.score:.3f}")
            prompt_parts.append(f"\n{result.text}\n")
            prompt_parts.append("-" * 80)

        # 3. Question
        prompt_parts.append(f"\n\n=== QUESTION ===\n{question}")

        # 4. Instructions
        prompt_parts.append("\n\n=== INSTRUCTIONS ===")
        prompt_parts.append("Réponds en te basant sur le contexte du graphe ET les extraits d'entretiens.")
        prompt_parts.append("Cite tes sources (commune, graphe de connaissances).")

        return "\n".join(prompt_parts)

    def _generate_response(self, prompt: str) -> str:
        """Génère une réponse avec le LLM"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Tu es un assistant spécialisé en analyse de données sur la qualité de vie. Tu as accès à un graphe de connaissances et des documents textuels. IMPORTANT: Ne force pas une analyse par commune si la question est générale ou conceptuelle. Réponds de manière appropriée au niveau de généralité de la question."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Erreur de génération: {str(e)}"

    def close(self):
        """Ferme les connexions"""
        self.graph.close()


def main():
    """
    Exemple d'utilisation
    """
    print("="*80)
    print("GRAPH-RAG avec Neo4j - Démonstration")
    print("="*80)

    # Configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Clé API OpenAI manquante")

    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    # Initialiser le pipeline
    rag = GraphRAGPipeline(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=NEO4J_PASSWORD,
        openai_api_key=OPENAI_API_KEY
    )

    # Importer les données de communes
    print("\nImport des données de communes...")
    rag.import_commune_data("df_mean_by_commune.csv")

    # Questions de test
    test_questions = [
        "Quelles sont les dimensions du bien-être territorial ?",
        "Comment est la santé à Ajaccio ?",
        "Quelles communes ont les meilleurs scores en éducation ?"
    ]

    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"Q: {question}")
        print(f"{'='*80}")

        response, results = rag.query(
            question,
            k=3,
            use_graph=True,
            use_reranking=True
        )

        print(f"\nRéponse:\n{response}")
        print(f"\nSources: {len(results)} documents")

    # Fermer les connexions
    rag.close()


if __name__ == "__main__":
    main()
