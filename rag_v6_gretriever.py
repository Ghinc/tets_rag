"""
RAG v6 - Graph-RAG avec G-Retriever (GNN-based)

Cette version utilise un Graph Neural Network (GNN) pour encoder
le graphe de connaissances et faire du retrieval plus sophistiqué.

Architecture:
1. Graphe Neo4j converti en représentation PyTorch Geometric
2. GNN encoder (GraphSAGE ou GAT) pour apprendre les embeddings de nœuds
3. Retrieval basé sur la similarité dans l'espace du graphe
4. Fusion avec retrieval vectoriel classique

Note: Cette implémentation nécessite PyTorch Geometric et torch-geometric-temporal

Auteur: Claude Code
Date: 2025-01-04
"""

import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from neo4j import GraphDatabase
import openai

# Imports des modules existants
from rag_v5_graphrag_neo4j import Neo4jGraphManager, GraphRAGPipeline
from rag_v2_improved import RetrievalResult
from commune_detector import detect_commune

# Dimension de l'espace latent partagé entre GNN et QueryEncoder.
# Tous les embeddings comparés par cosine similarity DOIVENT être dans cet espace.
LATENT_DIM = 512


class GraphEncoder(torch.nn.Module):
    """
    GNN Encoder pour apprendre les représentations des nœuds du graphe

    Utilise GraphSAGE pour agréger les informations du voisinage
    """

    def __init__(self, input_dim: int = 1024,
                 hidden_dim: int = 256,
                 output_dim: int = LATENT_DIM,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension des features d'entrée (BGE-M3 = 1024)
            hidden_dim: Dimension des couches cachées
            output_dim: Dimension de l'embedding final (doit == LATENT_DIM)
            num_layers: Nombre de couches GNN
            dropout: Taux de dropout
        """
        super().__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Première couche
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Couches intermédiaires
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Dernière couche
        self.convs.append(SAGEConv(hidden_dim, output_dim))

        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass

        Args:
            x: Features des nœuds [num_nodes, input_dim]
            edge_index: Arêtes du graphe [2, num_edges]

        Returns:
            Embeddings des nœuds [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Dernière couche (sans activation)
        x = self.convs[-1](x, edge_index)

        return x


class QueryEncoder(torch.nn.Module):
    """
    Projette les embeddings textuels de requêtes dans l'espace latent du GNN

    Architecture: MLP simple à 2 couches
    - Input: dim(embed_model) (BGE-M3 = 1024)
    - Hidden: 256 dim
    - Output: LATENT_DIM (== GNN output_dim)
    """

    def __init__(self, input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = LATENT_DIM):
        """
        Args:
            input_dim: Dimension de l'embedding textuel d'entrée (dérivée du modèle)
            hidden_dim: Dimension de la couche cachée
            output_dim: Dimension de sortie (doit == LATENT_DIM == GNN output_dim)
        """
        super().__init__()
        assert output_dim == LATENT_DIM, (
            f"QueryEncoder output_dim={output_dim} != LATENT_DIM={LATENT_DIM}. "
            f"Les espaces latents doivent correspondre."
        )

        # MLP à 2 couches : Simple mais efficace
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, query_embedding):
        """
        Forward pass

        Args:
            query_embedding: [batch_size, input_dim] - Embeddings textuels

        Returns:
            [batch_size, output_dim] - Embeddings projetés dans l'espace du graphe
        """
        return self.mlp(query_embedding)


class GRetrieverManager:
    """
    Gestionnaire G-Retriever pour Graph-RAG avancé

    Combine :
    - GNN pour encoder le graphe
    - Retrieval basé sur les embeddings de graphe
    - Fusion avec retrieval vectoriel
    """

    def __init__(self, neo4j_manager: Neo4jGraphManager,
                 embedding_model,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            neo4j_manager: Gestionnaire Neo4j
            embedding_model: Modèle d'embeddings (SentenceTransformer)
            device: Device PyTorch (cuda/cpu)
        """
        self.neo4j = neo4j_manager
        self.embed_model = embedding_model
        self.device = device

        print(f"G-Retriever initialisé (device: {device})")

        # Graphe PyTorch Geometric (sera construit)
        self.graph_data = None
        self.node_mapping = {}  # URI -> index
        self.reverse_mapping = {}  # index -> URI

        # Modèle GNN
        self.gnn_model = None

    def build_pytorch_graph(self):
        """
        Convertit le graphe Neo4j en format PyTorch Geometric

        Crée un graphe hétérogène avec différents types de nœuds:
        - Concept
        - Dimension
        - Indicator
        - Commune
        - IndicatorValue
        """
        print("\nConstruction du graphe PyTorch Geometric...")

        with self.neo4j.driver.session() as session:
            # 1. Récupérer tous les nœuds avec leurs types
            result = session.run("""
                MATCH (n)
                RETURN id(n) AS node_id, labels(n) AS labels, properties(n) AS props
            """)

            nodes = []
            for record in result:
                node_id = record['node_id']
                labels = record['labels']
                props = record['props']

                nodes.append({
                    'id': node_id,
                    'type': labels[0] if labels else 'Unknown',
                    'properties': props
                })

            print(f"  [OK] {len(nodes)} nœuds récupérés")

            # 2. Créer les embeddings de nœuds
            node_texts = []
            for node in nodes:
                # Créer une représentation textuelle du nœud
                text = self._node_to_text(node)
                node_texts.append(text)

            # Générer les embeddings
            print("  Génération des embeddings de nœuds...")
            node_features = self.embed_model.encode_documents(node_texts, show_progress=False)

            # 3. Récupérer les arêtes
            result = session.run("""
                MATCH (s)-[r]->(t)
                RETURN id(s) AS source, id(t) AS target, type(r) AS rel_type
            """)

            edges = []
            for record in result:
                edges.append({
                    'source': record['source'],
                    'target': record['target'],
                    'type': record['rel_type']
                })

            print(f"  [OK] {len(edges)} arêtes récupérées")

            # 4. Construire le graphe PyG
            # Mapping des IDs Neo4j vers indices PyTorch
            for idx, node in enumerate(nodes):
                self.node_mapping[node['id']] = idx
                self.reverse_mapping[idx] = node

            # Créer les edge_index
            edge_index = []
            for edge in edges:
                src_idx = self.node_mapping[edge['source']]
                tgt_idx = self.node_mapping[edge['target']]
                edge_index.append([src_idx, tgt_idx])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Créer le Data object
            x = torch.tensor(node_features, dtype=torch.float)

            self.graph_data = Data(x=x, edge_index=edge_index)
            self.graph_data = self.graph_data.to(self.device)

            print(f"  [OK] Graphe PyG créé: {self.graph_data.num_nodes} nœuds, {self.graph_data.num_edges} arêtes")

    def _node_to_text(self, node: Dict) -> str:
        """
        Convertit un nœud en représentation textuelle pour embedding

        Args:
            node: Dictionnaire avec id, type, properties

        Returns:
            Représentation textuelle
        """
        node_type = node['type']
        props = node['properties']

        # Construire le texte
        text_parts = [f"Type: {node_type}"]

        if 'label' in props:
            text_parts.append(f"Label: {props['label']}")

        if 'comment' in props:
            text_parts.append(f"Description: {props['comment']}")

        if 'name' in props:
            text_parts.append(f"Name: {props['name']}")

        return ". ".join(text_parts)

    def train_gnn(self, num_epochs: int = 50, lr: float = 0.01):
        """
        Entraîne le GNN de manière non-supervisée (reconstruction)

        Utilise une tâche de link prediction pour apprendre les embeddings

        Args:
            num_epochs: Nombre d'epochs
            lr: Learning rate
        """
        if self.graph_data is None:
            raise ValueError("Le graphe n'a pas été construit. Appelez build_pytorch_graph() d'abord.")

        print(f"\nEntraînement du GNN ({num_epochs} epochs)...")

        input_dim = self.graph_data.x.size(1)
        print(f"  Dimensions: input={input_dim}, output=LATENT_DIM={LATENT_DIM}")
        self.gnn_model = GraphEncoder(input_dim=input_dim, output_dim=LATENT_DIM).to(self.device)

        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=lr)

        self.gnn_model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward
            z = self.gnn_model(self.graph_data.x, self.graph_data.edge_index)

            # Loss : reconstruction du graphe (link prediction)
            # Échantillonner des arêtes positives et négatives
            pos_edge_index = self.graph_data.edge_index
            neg_edge_index = self._negative_sampling(pos_edge_index, self.graph_data.num_nodes)

            # Calculer les scores
            pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
            neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

            # Binary cross-entropy
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
            loss = pos_loss + neg_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        print("[OK] Entraînement terminé")

    def _negative_sampling(self, edge_index, num_nodes, num_neg_samples=None):
        """
        Échantillonnage négatif pour link prediction

        Args:
            edge_index: Arêtes positives [2, num_edges]
            num_nodes: Nombre de nœuds
            num_neg_samples: Nombre d'arêtes négatives (par défaut = num_edges)

        Returns:
            Arêtes négatives [2, num_neg_samples]
        """
        if num_neg_samples is None:
            num_neg_samples = edge_index.size(1)

        # Créer un set des arêtes existantes
        edge_set = set(map(tuple, edge_index.t().tolist()))

        neg_edges = []
        while len(neg_edges) < num_neg_samples:
            src = torch.randint(0, num_nodes, (1,)).item()
            tgt = torch.randint(0, num_nodes, (1,)).item()

            if (src, tgt) not in edge_set and src != tgt:
                neg_edges.append([src, tgt])

        return torch.tensor(neg_edges, dtype=torch.long).t().to(self.device)

    def contrastive_loss(self, q_proj, z_positive, z_negatives, temperature=0.07):
        """
        Loss InfoNCE pour aligner query_proj avec nodes pertinents

        Args:
            q_proj: [batch_size, embed_dim] - query projetée dans l'espace graphe
            z_positive: [batch_size, embed_dim] - nœuds pertinents
            z_negatives: [batch_size, num_neg, embed_dim] - nœuds non-pertinents
            temperature: température pour scaling (défaut: 0.07)

        Returns:
            loss: scalar - InfoNCE contrastive loss
        """
        # Similarités positives (query vs positifs)
        pos_sim = F.cosine_similarity(q_proj, z_positive, dim=1) / temperature

        # Similarités négatives (query vs négatifs)
        # z_negatives shape: [batch_size, num_neg, embed_dim]
        # q_proj shape: [batch_size, embed_dim] → [batch_size, embed_dim, 1]
        neg_sim = torch.bmm(
            z_negatives,                    # [batch_size, num_neg, embed_dim]
            q_proj.unsqueeze(2)             # [batch_size, embed_dim, 1]
        ).squeeze(2) / temperature          # [batch_size, num_neg]

        # InfoNCE loss : log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # Équivalent à cross_entropy avec label=0 (premier = positif)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1+num_neg]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=q_proj.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def train_query_encoder(
        self,
        query_node_pairs,  # Liste de (query_text, node_id_positive, [node_ids_negative])
        num_epochs=100,
        lr=0.001,
        temperature=0.07
    ):
        """
        Entraîne le QueryEncoder avec loss contrastive

        Phase 1 : GNN est FIGÉ (déjà entraîné sur link prediction)
        Phase 2 : On entraîne UNIQUEMENT le QueryEncoder

        Args:
            query_node_pairs: Liste de tuples (query_text, pos_node_id, [neg_node_ids])
            num_epochs: Nombre d'epochs d'entraînement
            lr: Learning rate
            temperature: Température pour loss contrastive
        """
        print(f"\n{'='*80}")
        print("ENTRAÎNEMENT QUERY ENCODER")
        print(f"{'='*80}")
        print(f"Paires d'entraînement: {len(query_node_pairs)}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {lr}")
        print(f"Temperature: {temperature}")

        # Dériver input_dim depuis le modèle d'embeddings (pas de hardcode)
        probe_dim = len(self.embed_model.encode_query("test"))
        print(f"  Dimension encode_query detectee: {probe_dim}")

        # Vérifier que le GNN sort bien LATENT_DIM
        with torch.no_grad():
            gnn_out = self.gnn_model(self.graph_data.x, self.graph_data.edge_index)
        gnn_out_dim = gnn_out.size(1)
        assert gnn_out_dim == LATENT_DIM, (
            f"GNN output_dim={gnn_out_dim} != LATENT_DIM={LATENT_DIM}. "
            f"Reentrainer le GNN avec output_dim=LATENT_DIM."
        )

        # Initialiser QueryEncoder
        self.query_encoder = QueryEncoder(
            input_dim=probe_dim,      # Dérivé du modèle d'embeddings
            hidden_dim=256,
            output_dim=LATENT_DIM     # == GNN output_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(self.query_encoder.parameters(), lr=lr)

        # FIGER le GNN (pas d'entraînement)
        self.gnn_model.eval()
        for param in self.gnn_model.parameters():
            param.requires_grad = False

        print("[OK] GNN figé (pas de gradient)")

        # Pré-calculer tous les embeddings de nœuds (puisque GNN figé)
        print("Pré-calcul des embeddings de nœuds...")
        with torch.no_grad():
            node_embeddings = self.gnn_model(
                self.graph_data.x,
                self.graph_data.edge_index
            )
        print(f"[OK] {node_embeddings.shape[0]} nœuds embedés en {node_embeddings.shape[1]}D")

        # Training loop
        self.query_encoder.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for query_text, pos_node_id, neg_node_ids in query_node_pairs:
                optimizer.zero_grad()

                # 1. Encoder la requête (texte → espace graphe)
                q_text = self.embed_model.encode_query(query_text)
                q_text_tensor = torch.tensor(q_text, dtype=torch.float).unsqueeze(0).to(self.device)
                q_proj = self.query_encoder(q_text_tensor)

                # 2. Récupérer les embeddings positifs/négatifs (GNN figé)
                z_pos = node_embeddings[pos_node_id].unsqueeze(0)  # [1, 512]

                # Gérer les négatifs
                if isinstance(neg_node_ids, list):
                    z_neg = torch.stack([node_embeddings[idx] for idx in neg_node_ids])
                else:
                    z_neg = node_embeddings[neg_node_ids]

                z_neg = z_neg.unsqueeze(0)  # [1, num_neg, 512]

                # 3. Loss contrastive
                loss = self.contrastive_loss(q_proj, z_pos, z_neg, temperature)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print(f"\n[OK] Entraînement terminé !")
        print(f"[OK] QueryEncoder prêt pour retrieval")

    def retrieve_subgraph(self, query: str, k: int = 10) -> List[Dict]:
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

        # 1. Encoder la requete (textuellement)
        query_embedding = self.embed_model.encode_query(query)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float).to(self.device)

        # 2. PROJETER dans l'espace du graphe avec QueryEncoder
        if not hasattr(self, 'query_encoder') or self.query_encoder is None:
            raise RuntimeError(
                "QueryEncoder non entraine. Impossible de projeter la requete "
                f"(dim={query_tensor.shape[-1]}) dans l'espace GNN (dim={LATENT_DIM}). "
                "Appelez train_query_encoder() d'abord."
            )

        with torch.no_grad():
            query_proj = self.query_encoder(query_tensor.unsqueeze(0))

        # 3. Obtenir les embeddings GNN des noeuds
        self.gnn_model.eval()
        with torch.no_grad():
            node_embeddings = self.gnn_model(self.graph_data.x, self.graph_data.edge_index)

        # 4. Assert dimensionnel explicite
        assert query_proj.shape[1] == node_embeddings.shape[1], (
            f"Dimension mismatch: query_proj={query_proj.shape[1]} vs "
            f"node_embeddings={node_embeddings.shape[1]}. "
            f"Les deux doivent etre == LATENT_DIM={LATENT_DIM}."
        )

        # 5. Normaliser les embeddings pour cosine similarity correcte
        query_proj = F.normalize(query_proj, p=2, dim=1)
        node_embeddings = F.normalize(node_embeddings, p=2, dim=1)

        # 6. Calculer la similarite cosinus (meme espace LATENT_DIM)
        similarities = F.cosine_similarity(
            query_proj,        # [1, LATENT_DIM]
            node_embeddings,   # [num_nodes, LATENT_DIM]
            dim=1
        )

        # 7. Recuperer les top-k noeuds
        top_k_scores, top_k_indices = torch.topk(similarities, k)

        # 8. Construire la reponse
        results = []
        for score, idx in zip(top_k_scores.cpu().numpy(), top_k_indices.cpu().numpy()):
            node = self.reverse_mapping[idx]
            results.append({
                'node': node,
                'score': float(score),
                'text': self._node_to_text(node)
            })

        return results





class GRetrieverRAGPipeline(GraphRAGPipeline):
    """
    Pipeline Graph-RAG avec G-Retriever (hérite de GraphRAGPipeline)
    """

    def __init__(self, *args, **kwargs):
        """Initialise le pipeline avec G-Retriever"""
        super().__init__(*args, **kwargs)

        # Ajouter le G-Retriever Manager
        print("\nInitialisation G-Retriever...")
        self.gretriever = GRetrieverManager(
            self.graph,
            self.embed_model
        )

        # Construire le graphe PyTorch
        self.gretriever.build_pytorch_graph()

        # Entraîner le GNN
        self.gretriever.train_gnn(num_epochs=50, lr=0.01)

        # Initialiser QueryEncoder à None (pas encore entraîné)
        # Pour entraîner, utiliser : self.gretriever.train_query_encoder(query_node_pairs)
        # Note: Le QueryEncoder sera créé lors de l'appel à train_query_encoder()
        print("[INFO] QueryEncoder non entraîné. Utilisera embedding brut BGE-M3 en fallback.")
        print("       Pour entraîner: gretriever.train_query_encoder(query_node_pairs)")

    def query(self, question: str, k: int = 5,
              use_gnn: bool = True,
              **kwargs) -> Tuple[str, List[RetrievalResult]]:
        """
        Requête avec G-Retriever

        Args:
            question: Question utilisateur
            k: Nombre de résultats
            use_gnn: Utiliser le GNN retrieval
            **kwargs: Autres arguments

        Returns:
            (réponse, résultats)
        """
        print(f"\n{'='*80}")
        print(f"REQUÊTE G-RETRIEVER: {question}")
        print(f"{'='*80}")

        # Détection automatique de commune dans la question
        detected_commune = detect_commune(question)
        if detected_commune:
            print(f"[AUTO-DETECT v6] Commune détectée: {detected_commune}")
            # Ajouter le filtre commune aux kwargs s'il n'est pas déjà présent
            if 'commune_filter' not in kwargs:
                kwargs['commune_filter'] = detected_commune

        # 1. Retrieval via GNN
        gnn_results = []
        if use_gnn:
            print("Retrieval via GNN...")
            gnn_results = self.gretriever.retrieve_subgraph(question, k=k)
            print(f"[OK] {len(gnn_results)} nœuds récupérés du graphe")

        # 2. Retrieval vectoriel classique (avec commune_filter si détecté)
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

        return response, final_results



def main():
    """Test de G-Retriever"""
    print("="*80)
    print("G-RETRIEVER GRAPH-RAG - Démonstration")
    print("="*80)

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    # Initialiser
    rag = GRetrieverRAGPipeline(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=NEO4J_PASSWORD,
        openai_api_key=OPENAI_API_KEY
    )

    # Importer les données
    rag.import_commune_data("df_mean_by_commune.csv")

    # Test
    response, results = rag.query(
        "Quelles dimensions du bien-être sont liées à la santé ?",
        use_gnn=True
    )

    print(f"\nRéponse:\n{response}")

    # Fermer
    rag.close()


if __name__ == "__main__":
    main()
