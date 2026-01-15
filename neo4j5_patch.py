"""
Patch pour rendre Neo4jPropertyGraphStore compatible avec Neo4j 5.x

Le problème : Neo4j 5.x a changé la syntaxe pour certaines fonctions:
- exists(n.prop) → n.prop IS NOT NULL
- Autres incompatibilités potentielles

Solution : Hériter de Neo4jPropertyGraphStore et override les méthodes
qui génèrent du Cypher incompatible.
"""

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from typing import Any, Dict, List, Optional
import re


class Neo4j5PropertyGraphStore(Neo4jPropertyGraphStore):
    """
    Version patché de Neo4jPropertyGraphStore pour Neo4j 5.x

    Override les méthodes qui génèrent du Cypher incompatible avec Neo4j 5.x
    """

    def _patch_cypher_query(self, query: str) -> str:
        """
        Patch une requête Cypher pour la rendre compatible Neo4j 5.x

        Transformations:
        - exists(n.prop) → n.prop IS NOT NULL
        - exists((pattern)) → EXISTS { (pattern) }
        """
        original = query

        # Pattern 1: exists(node.property) → node.property IS NOT NULL
        query = re.sub(
            r'exists\s*\(\s*(\w+)\.(\w+)\s*\)',
            r'\1.\2 IS NOT NULL',
            query,
            flags=re.IGNORECASE
        )

        # Pattern 2: exists((pattern)) pour relations - plus agressif
        # Cherche exists( suivi de n'importe quoi jusqu'au )
        def replace_exists_pattern(match):
            inner = match.group(1)
            return f'EXISTS {{ {inner} }}'

        query = re.sub(
            r'exists\s*\(\s*(\([^)]*\)(?:-[^)]*)?)\s*\)',
            replace_exists_pattern,
            query,
            flags=re.IGNORECASE
        )

        if query != original:
            print(f"\n[NEO4J5 PATCH] Requête modifiée")
            print(f"AVANT:\n{original}")
            print(f"\nAPRÈS:\n{query}\n")

        return query

    def structured_query(self, query: str, param_map: Optional[Dict[str, Any]] = None) -> Any:
        """
        Override structured_query pour patcher les requêtes
        """
        print(f"\n[NEO4J5 DEBUG] structured_query appelé:")
        print(f"Query: {query[:200]}...")

        patched = self._patch_cypher_query(query)
        return super().structured_query(patched, param_map)

    def vector_query(self, query: str, **kwargs: Any) -> Any:
        """
        Override vector_query
        """
        print(f"\n[NEO4J5 DEBUG] vector_query appelé")
        patched = self._patch_cypher_query(query)
        return super().vector_query(patched, **kwargs)

    def upsert_nodes(self, nodes: List[Any]) -> None:
        """
        Override si nécessaire - pour l'instant délègue au parent
        """
        return super().upsert_nodes(nodes)

    def upsert_relations(self, relations: List[Any]) -> None:
        """
        Override si nécessaire - pour l'instant délègue au parent
        """
        return super().upsert_relations(relations)
