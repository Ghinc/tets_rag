"""
Script pour appliquer les fixes de normalisation et fusion de scores dans rag_v6
"""

fix_retrieve_subgraph = '''    def retrieve_subgraph(self, query: str, k: int = 10) -> List[Dict]:
        """
        Récupère un sous-graphe pertinent pour une requête

        Args:
            query: Requête utilisateur
            k: Nombre de nœuds à récupérer

        Returns:
            Liste de nœuds pertinents avec leurs scores
        """
        if self.gnn_model is None:
            raise ValueError("Le GNN n'a pas été entraîné. Appelez train_gnn() d'abord.")

        # 1. Encoder la requête
        query_embedding = self.embed_model.encode_query(query)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float).to(self.device)

        # 2. Obtenir les embeddings GNN des nœuds
        self.gnn_model.eval()
        with torch.no_grad():
            node_embeddings = self.gnn_model(self.graph_data.x, self.graph_data.edge_index)

        # 3. Normaliser les embeddings pour cosine similarity correcte
        query_tensor = F.normalize(query_tensor.unsqueeze(0), p=2, dim=1)
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        # 4. Calculer la similarité cosinus (maintenant avec vecteurs normalisés)
        similarities = F.cosine_similarity(
            query_tensor,
            node_embeddings,
            dim=1
        )

        # 5. Récupérer les top-k nœuds
        top_k_scores, top_k_indices = torch.topk(similarities, k)

        # 6. Construire la réponse
        results = []
        for score, idx in zip(top_k_scores.cpu().numpy(), top_k_indices.cpu().numpy()):
            node = self.reverse_mapping[idx]
            results.append({
                'node': node,
                'score': float(score),
                'text': self._node_to_text(node)
            })

        return results'''

fix_query_fusion = '''        # 1. Retrieval via GNN
        gnn_results = []
        if use_gnn:
            print("Retrieval via GNN...")
            gnn_results = self.gretriever.retrieve_subgraph(question, k=k)
            print(f"[OK] {len(gnn_results)} nœuds récupérés du graphe")

        # 2. Retrieval vectoriel classique
        response, vector_results = super().query(question, k=k, use_graph=False, **kwargs)

        # 3. Fusionner les résultats GNN + vecteurs avec normalisation des scores
        # Extraire les scores pour normalisation
        gnn_scores = [r['score'] for r in gnn_results] if gnn_results else []
        vector_scores = [r.score for r in vector_results] if vector_results else []

        # Combiner tous les résultats avec pondération
        all_results = []

        # Ajouter les résultats GNN (pondération: 0.6)
        for gnn_result in gnn_results:
            all_results.append(RetrievalResult(
                text=gnn_result['text'],
                metadata={'source': 'gnn', 'node_type': gnn_result['node']['type'], 'properties': gnn_result['node']['properties']},
                score=gnn_result['score'] * 0.6,  # Pondération GNN
                source_type='gnn'
            ))

        # Ajouter les résultats vectoriels (pondération: 0.4)
        for vector_result in vector_results:
            all_results.append(RetrievalResult(
                text=vector_result.text,
                metadata=vector_result.metadata,
                score=vector_result.score * 0.4,  # Pondération vectoriel
                source_type=vector_result.source_type
            ))

        # Trier par score pondéré et prendre le top-k
        all_results.sort(key=lambda x: x.score, reverse=True)
        final_results = all_results[:k]

        # 4. Régénérer la réponse avec tous les résultats
        prompt = self._build_graph_rag_prompt(question, final_results, "")
        response = self._generate_response(prompt)

        return response, final_results'''

# Lire le fichier
with open('rag_v6_gretriever.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Appliquer fix 1: normalisation dans retrieve_subgraph
import re

# Pattern pour retrieve_subgraph
pattern1 = r'(    def retrieve_subgraph\(self, query: str, k: int = 10\) -> List\[Dict\]:.*?)(        return results)'
content = re.sub(pattern1, fix_retrieve_subgraph + '\n', content, flags=re.DOTALL)

# Pattern pour la fusion dans query()
pattern2 = r'(        # 1\. Retrieval via GNN.*?)(        return response, vector_results)'
content = re.sub(pattern2, fix_query_fusion + '\n', content, flags=re.DOTALL)

# Écrire le fichier
with open('rag_v6_gretriever.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] Fixes appliqués avec succès!")
print("  - Normalisation des embeddings avant cosine_similarity")
print("  - Pondération GNN (0.6) vs Vectoriel (0.4) pour fusion équilibrée")
