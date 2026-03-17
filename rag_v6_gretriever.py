"""
RAG v6 - Graph-RAG avec G-Retriever (GNN-based, graphe heterogene)

Cette version utilise un Graph Neural Network (GNN) pour encoder
le graphe de connaissances et faire du retrieval plus sophistique.

Architecture:
1. Graphe Neo4j converti en HeteroData PyTorch Geometric (types de noeuds + aretes)
2. GNN encoder (GraphSAGE) converti en heterogene via to_hetero()
3. Retrieval cible sur les types ontologiques (Concept, Dimension, Indicator)
4. Fusion avec retrieval vectoriel classique (ChromaDB)

Note: Cette implementation necessite PyTorch Geometric et torch-geometric-temporal

Auteur: Claude Code
Date: 2025-01-04
"""

import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from neo4j import GraphDatabase
import openai

# Imports des modules existants
from rag_v5_graphrag_neo4j import Neo4jGraphManager, GraphRAGPipeline
from rag_v2_improved import RetrievalResult
from commune_detector import detect_commune

# =============================================================================
# Constantes
# =============================================================================

# Dimension de l'espace latent partage entre GNN et QueryEncoder.
LATENT_DIM = 512

# Mapping Neo4j labels -> types PyG (groupement des petits types)
LABEL_TO_TYPE = {
    "Concept": "Concept",
    "Dimension": "Dimension",
    "Indicator": "Indicator",
    "ObjectiveIndicator": "Indicator",
    "SubjectiveIndicator": "Indicator",
    "Commune": "Commune",
    "Municipality": "Commune",
    "IndicatorValue": "IndicatorValue",
    "Country": "SpatialUnit",
    "Region": "SpatialUnit",
    "Department": "SpatialUnit",
    "QuantitativeSurvey": "DataSource",
    "Verbatim": "DataSource",
    "WikipediaArticle": "DataSource",
    "Interview": "DataSource",
    "StatisticalDataset": "DataSource",
}

# Labels prioritaires : le plus specifique d'abord (pour les noeuds multi-labels)
LABEL_PRIORITY = [
    "ObjectiveIndicator", "SubjectiveIndicator",
    "Municipality", "Department", "Region", "Country",
    "QuantitativeSurvey", "Verbatim", "WikipediaArticle",
    "Interview", "StatisticalDataset",
    "Concept", "Dimension", "Indicator", "Commune", "IndicatorValue",
]

# Types cibles pour le retrieval GNN (savoir ontologique utile au LLM)
RETRIEVAL_TARGET_TYPES = ["Concept", "Dimension", "Indicator"]


# =============================================================================
# Modeles
# =============================================================================

class GraphEncoder(torch.nn.Module):
    """
    GNN Encoder homogene (GraphSAGE), converti en heterogene via to_hetero().

    Pas de BatchNorm pour supporter les types avec peu de noeuds (ex: SpatialUnit=4).
    """

    def __init__(self, input_dim: int = 1024,
                 hidden_dim: int = 256,
                 output_dim: int = LATENT_DIM,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # Premiere couche
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        # Couches intermediaires
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        # Derniere couche
        self.convs.append(SAGEConv(hidden_dim, output_dim))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class QueryEncoder(torch.nn.Module):
    """
    Projette les embeddings textuels de requetes dans l'espace latent du GNN.

    Architecture: MLP 2 couches.
    - Input: dim(embed_model) (derivee dynamiquement)
    - Output: LATENT_DIM (== GNN output_dim)
    """

    def __init__(self, input_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = LATENT_DIM):
        super().__init__()
        assert output_dim == LATENT_DIM, (
            f"QueryEncoder output_dim={output_dim} != LATENT_DIM={LATENT_DIM}."
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, query_embedding):
        return self.mlp(query_embedding)


# =============================================================================
# GRetrieverManager
# =============================================================================

class GRetrieverManager:
    """
    Gestionnaire G-Retriever pour Graph-RAG avec graphe heterogene.

    - Construit un HeteroData a partir de Neo4j (7 types de noeuds)
    - Entraine un GNN heterogene (GraphSAGE + to_hetero)
    - Retrieval cible sur Concept/Dimension/Indicator
    """

    def __init__(self, neo4j_manager: Neo4jGraphManager,
                 embedding_model,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.neo4j = neo4j_manager
        self.embed_model = embedding_model
        self.device = device

        print(f"G-Retriever initialise (device: {device})")

        # Graphe heterogene PyTorch Geometric
        self.graph_data = None  # HeteroData

        # Mappings par type : {pyg_type: {neo4j_id: local_idx}}
        self.type_id_maps = {}
        # Reverse mappings : {pyg_type: {local_idx: node_dict}}
        self.type_reverse_maps = {}

        # Modeles
        self.gnn_model = None       # Heterogene (via to_hetero)
        self.query_encoder = None

    # -----------------------------------------------------------------
    # Resolution du type PyG
    # -----------------------------------------------------------------

    def _resolve_node_type(self, labels: List[str]) -> str:
        """Determine le type PyG a partir des labels Neo4j (priorite au plus specifique)."""
        for label in LABEL_PRIORITY:
            if label in labels:
                return LABEL_TO_TYPE[label]
        for label in labels:
            if label in LABEL_TO_TYPE:
                return LABEL_TO_TYPE[label]
        return "Unknown"

    # -----------------------------------------------------------------
    # Construction du graphe heterogene
    # -----------------------------------------------------------------

    def build_pytorch_graph(self):
        """
        Convertit le graphe Neo4j en HeteroData PyTorch Geometric.

        Cree un graphe heterogene avec types de noeuds groupes
        (Concept, Dimension, Indicator, Commune, IndicatorValue, SpatialUnit, DataSource)
        et aretes typees (src_type, rel_type, dst_type).
        """
        print("\nConstruction du graphe HeteroData...")

        with self.neo4j.driver.session() as session:
            # ---- 1. Recuperer tous les noeuds ----
            result = session.run("""
                MATCH (n)
                RETURN id(n) AS node_id, labels(n) AS labels, properties(n) AS props
            """)

            # Grouper les noeuds par type PyG
            nodes_by_type = defaultdict(list)  # {pyg_type: [(neo4j_id, node_dict)]}
            neo4j_id_to_type = {}              # {neo4j_id: pyg_type}

            for record in result:
                neo4j_id = record['node_id']
                labels = record['labels']
                props = record['props']
                pyg_type = self._resolve_node_type(labels)

                node_dict = {
                    'id': neo4j_id,
                    'type': pyg_type,
                    'labels': labels,
                    'properties': props
                }
                nodes_by_type[pyg_type].append((neo4j_id, node_dict))
                neo4j_id_to_type[neo4j_id] = pyg_type

            # Exclure 'Unknown' : ces noeuds n'ont pas de type ontologique
            # et ne sont destination d'aucune arete, ce qui fait crasher to_hetero()
            if "Unknown" in nodes_by_type:
                n_unknown = len(nodes_by_type["Unknown"])
                del nodes_by_type["Unknown"]
                # Nettoyer les id maps aussi
                for neo_id, ntype in list(neo4j_id_to_type.items()):
                    if ntype == "Unknown":
                        del neo4j_id_to_type[neo_id]
                print(f"  ({n_unknown} noeuds 'Unknown' exclus du graphe)")

            # Log
            total_nodes = sum(len(v) for v in nodes_by_type.values())
            print(f"  {total_nodes} noeuds recuperes, {len(nodes_by_type)} types:")
            for ntype, nodes in sorted(nodes_by_type.items()):
                print(f"    {ntype}: {len(nodes)} noeuds")

            # ---- 2. Creer les mappings et embeddings par type ----
            self.type_id_maps = {}
            self.type_reverse_maps = {}

            type_features = {}  # {pyg_type: np.ndarray}

            for pyg_type, nodes in nodes_by_type.items():
                id_map = {}
                reverse_map = {}
                texts = []

                for local_idx, (neo4j_id, node_dict) in enumerate(nodes):
                    id_map[neo4j_id] = local_idx
                    reverse_map[local_idx] = node_dict
                    texts.append(self._node_to_text(node_dict))

                self.type_id_maps[pyg_type] = id_map
                self.type_reverse_maps[pyg_type] = reverse_map

                # Embeddings pour ce type
                if texts:
                    type_features[pyg_type] = self.embed_model.encode_documents(
                        texts, show_progress=False
                    )

            # ---- 3. Recuperer les aretes avec types src/dst ----
            result = session.run("""
                MATCH (s)-[r]->(t)
                RETURN id(s) AS source, labels(s) AS src_labels,
                       type(r) AS rel_type,
                       id(t) AS target, labels(t) AS tgt_labels
            """)

            # Grouper les aretes par triplet (src_type, rel_type, dst_type)
            edges_by_type = defaultdict(list)  # {(src_t, rel, dst_t): [(src_local, dst_local)]}
            skipped_edges = 0

            for record in result:
                src_id = record['source']
                tgt_id = record['target']
                rel_type = record['rel_type']

                src_type = neo4j_id_to_type.get(src_id)
                dst_type = neo4j_id_to_type.get(tgt_id)

                if src_type is None or dst_type is None:
                    skipped_edges += 1
                    continue

                src_local = self.type_id_maps[src_type].get(src_id)
                dst_local = self.type_id_maps[dst_type].get(tgt_id)

                if src_local is None or dst_local is None:
                    skipped_edges += 1
                    continue

                edge_key = (src_type, rel_type, dst_type)
                edges_by_type[edge_key].append([src_local, dst_local])

            total_edges = sum(len(v) for v in edges_by_type.values())
            print(f"  {total_edges} aretes recuperees, {len(edges_by_type)} types:")
            for (st, rt, dt), edges in sorted(edges_by_type.items()):
                print(f"    ({st})-[{rt}]->({dt}): {len(edges)} aretes")
            if skipped_edges:
                print(f"  ({skipped_edges} aretes ignorees - noeuds non mappes)")

            # ---- 4. Construire HeteroData ----
            data = HeteroData()

            # Features par type
            for pyg_type, features in type_features.items():
                data[pyg_type].x = torch.tensor(features, dtype=torch.float)

            # Aretes par triplet
            for (src_type, rel_type, dst_type), edge_list in edges_by_type.items():
                ei = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                data[src_type, rel_type, dst_type].edge_index = ei

            self.graph_data = data.to(self.device)

            # Log final
            print(f"  [OK] HeteroData cree:")
            print(f"    Node types: {self.graph_data.node_types}")
            print(f"    Edge types: {[f'({s})-[{r}]->({d})' for s, r, d in self.graph_data.edge_types]}")
            for ntype in self.graph_data.node_types:
                print(f"    {ntype}.x: {self.graph_data[ntype].x.shape}")

    def _node_to_text(self, node: Dict) -> str:
        """Convertit un noeud en representation textuelle pour embedding."""
        node_type = node['type']
        props = node['properties']
        text_parts = [f"Type: {node_type}"]
        if 'label' in props:
            text_parts.append(f"Label: {props['label']}")
        if 'comment' in props:
            text_parts.append(f"Description: {props['comment']}")
        if 'name' in props:
            text_parts.append(f"Name: {props['name']}")
        return ". ".join(text_parts)

    # -----------------------------------------------------------------
    # Entrainement GNN
    # -----------------------------------------------------------------

    def train_gnn(self, num_epochs: int = 50, lr: float = 0.01):
        """
        Entraine le GNN heterogene (link prediction non-supervisee).

        1. Cree un GraphEncoder homogene
        2. Le convertit en heterogene via to_hetero(model, metadata)
        3. Entraine par link prediction sur chaque type d'arete
        """
        if self.graph_data is None:
            raise ValueError("Le graphe n'a pas ete construit. Appelez build_pytorch_graph() d'abord.")

        # Determiner input_dim (identique pour tous les types car meme embed_model)
        first_type = self.graph_data.node_types[0]
        input_dim = self.graph_data[first_type].x.size(1)

        print(f"\nEntrainement du GNN heterogene ({num_epochs} epochs)...")
        print(f"  Dimensions: input={input_dim}, output=LATENT_DIM={LATENT_DIM}")
        print(f"  Types de noeuds: {self.graph_data.node_types}")
        print(f"  Types d'aretes: {len(self.graph_data.edge_types)}")

        # 1. Creer le modele homogene de base
        base_model = GraphEncoder(input_dim=input_dim, output_dim=LATENT_DIM)

        # 2. Convertir en heterogene
        metadata = self.graph_data.metadata()
        self.gnn_model = to_hetero(base_model, metadata).to(self.device)

        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=lr)

        # 3. Training loop
        self.gnn_model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass heterogene -> z_dict {node_type: [n, LATENT_DIM]}
            z_dict = self.gnn_model(
                self.graph_data.x_dict,
                self.graph_data.edge_index_dict
            )

            # Link prediction loss par type d'arete
            total_loss = torch.tensor(0.0, device=self.device)
            num_edge_types = 0

            for (src_type, rel, dst_type) in self.graph_data.edge_types:
                ei = self.graph_data[src_type, rel, dst_type].edge_index

                if ei.size(1) == 0:
                    continue

                z_src = z_dict[src_type]
                z_dst = z_dict[dst_type]

                # Positive scores
                pos_score = (z_src[ei[0]] * z_dst[ei[1]]).sum(dim=1)

                # Negative sampling (dans les bons types)
                neg_ei = self._negative_sampling(
                    ei, z_src.size(0), z_dst.size(0)
                )
                neg_score = (z_src[neg_ei[0]] * z_dst[neg_ei[1]]).sum(dim=1)

                # BCE loss
                pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
                neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
                total_loss = total_loss + pos_loss + neg_loss
                num_edge_types += 1

            if num_edge_types > 0:
                total_loss = total_loss / num_edge_types
                total_loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")

        # Verifier les dimensions de sortie
        self.gnn_model.eval()
        with torch.no_grad():
            z_dict = self.gnn_model(
                self.graph_data.x_dict,
                self.graph_data.edge_index_dict
            )
        for ntype, z in z_dict.items():
            assert z.size(1) == LATENT_DIM, (
                f"GNN output pour '{ntype}': dim={z.size(1)} != LATENT_DIM={LATENT_DIM}"
            )
        print(f"  [OK] Entrainement termine, toutes les sorties en {LATENT_DIM}D")

    def _negative_sampling(self, edge_index, num_src_nodes, num_dst_nodes, num_neg_samples=None):
        """
        Echantillonnage negatif pour link prediction heterogene.

        Echantillonne des paires (src, dst) qui n'existent pas,
        avec src dans [0, num_src_nodes) et dst dans [0, num_dst_nodes).
        """
        if num_neg_samples is None:
            num_neg_samples = edge_index.size(1)

        edge_set = set(map(tuple, edge_index.t().tolist()))

        neg_edges = []
        max_attempts = num_neg_samples * 10
        attempts = 0
        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            src = torch.randint(0, num_src_nodes, (1,)).item()
            dst = torch.randint(0, num_dst_nodes, (1,)).item()
            if (src, dst) not in edge_set:
                neg_edges.append([src, dst])
            attempts += 1

        if not neg_edges:
            neg_edges = [[0, 0]]

        return torch.tensor(neg_edges, dtype=torch.long).t().to(self.device)

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------

    def retrieve_subgraph(self, query: str, k: int = 10) -> List[Dict]:
        """
        Recupere les noeuds ontologiques les plus pertinents pour une requete.

        Cible uniquement les types RETRIEVAL_TARGET_TYPES (Concept, Dimension, Indicator).
        Optionnel : enrichit les resultats avec les voisins 1-hop.
        """
        if self.gnn_model is None:
            raise ValueError("Le GNN n'a pas ete entraine. Appelez train_gnn() d'abord.")

        use_query_encoder = hasattr(self, 'query_encoder') and self.query_encoder is not None

        # 1. Encoder la requete avec BGE-M3 (1024-dim)
        query_embedding = self.embed_model.encode_query(query)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float).to(self.device)

        if use_query_encoder:
            # Projection apprise query -> LATENT_DIM (512)
            with torch.no_grad():
                query_proj = self.query_encoder(query_tensor.unsqueeze(0))

            # Forward GNN -> z_dict (LATENT_DIM par type)
            self.gnn_model.eval()
            with torch.no_grad():
                z_dict = self.gnn_model(
                    self.graph_data.x_dict,
                    self.graph_data.edge_index_dict
                )

            # Embeddings cibles = sorties GNN (512-dim)
            available_targets = [t for t in RETRIEVAL_TARGET_TYPES if t in z_dict]
            if not available_targets:
                raise ValueError(
                    f"Aucun type cible {RETRIEVAL_TARGET_TYPES} dans le graphe. "
                    f"Types disponibles: {list(z_dict.keys())}"
                )

            target_embeddings = []
            target_nodes = []
            for ntype in available_targets:
                z = z_dict[ntype]
                target_embeddings.append(z)
                for local_idx in range(z.size(0)):
                    target_nodes.append((ntype, local_idx))

            all_target = torch.cat(target_embeddings, dim=0)
            print(f"  [QueryEncoder] query {query_proj.shape[1]}D vs cibles {all_target.shape[1]}D")

        else:
            # Fallback : BGE-M3 direct contre embeddings originaux (1024-dim)
            # On cible quand meme uniquement les noeuds ontologiques
            query_proj = query_tensor.unsqueeze(0)  # [1, 1024]

            available_targets = [
                t for t in RETRIEVAL_TARGET_TYPES
                if t in self.graph_data.node_types
            ]
            if not available_targets:
                raise ValueError(
                    f"Aucun type cible {RETRIEVAL_TARGET_TYPES} dans le graphe. "
                    f"Types disponibles: {self.graph_data.node_types}"
                )

            target_embeddings = []
            target_nodes = []
            for ntype in available_targets:
                x = self.graph_data[ntype].x  # embeddings originaux [N, 1024]
                target_embeddings.append(x)
                for local_idx in range(x.size(0)):
                    target_nodes.append((ntype, local_idx))

            all_target = torch.cat(target_embeddings, dim=0)
            print(f"  [BGE-M3 fallback] query {query_proj.shape[1]}D vs "
                  f"{len(target_nodes)} noeuds ontologiques ({', '.join(available_targets)})")

        # Cosine similarity
        query_proj = F.normalize(query_proj, p=2, dim=1)
        all_target = F.normalize(all_target, p=2, dim=1)
        similarities = F.cosine_similarity(query_proj, all_target, dim=1)

        # 6. Top-k
        actual_k = min(k, similarities.size(0))
        top_k_scores, top_k_indices = torch.topk(similarities, actual_k)

        # 7. Construire les resultats avec mapping inverse
        results = []
        for score, flat_idx in zip(top_k_scores.cpu().numpy(), top_k_indices.cpu().numpy()):
            ntype, local_idx = target_nodes[flat_idx]
            node_dict = self.type_reverse_maps[ntype][local_idx]

            result = {
                'node': node_dict,
                'node_type': ntype,
                'score': float(score),
                'text': self._node_to_text(node_dict),
            }

            # 1-hop expansion : voisins directs pour enrichir les metadonnees
            neighbors = self._expand_one_hop(ntype, local_idx)
            if neighbors:
                result['neighbors'] = neighbors

            results.append(result)

        return results

    def _expand_one_hop(self, node_type: str, local_idx: int, max_neighbors: int = 3) -> List[Dict]:
        """
        Collecte les voisins 1-hop d'un noeud dans le graphe heterogene.

        Retourne au max `max_neighbors` voisins avec leur type et metadata.
        """
        neighbors = []

        for (src_t, rel, dst_t) in self.graph_data.edge_types:
            ei = self.graph_data[src_t, rel, dst_t].edge_index

            # Le noeud est source
            if src_t == node_type:
                mask = (ei[0] == local_idx)
                if mask.any():
                    dst_indices = ei[1][mask].cpu().tolist()
                    for dst_idx in dst_indices[:max_neighbors - len(neighbors)]:
                        if dst_t in self.type_reverse_maps and dst_idx in self.type_reverse_maps[dst_t]:
                            neighbor_dict = self.type_reverse_maps[dst_t][dst_idx]
                            neighbors.append({
                                'type': dst_t,
                                'relation': rel,
                                'direction': 'outgoing',
                                'label': neighbor_dict['properties'].get('label', ''),
                            })

            # Le noeud est destination
            if dst_t == node_type:
                mask = (ei[1] == local_idx)
                if mask.any():
                    src_indices = ei[0][mask].cpu().tolist()
                    for src_idx in src_indices[:max_neighbors - len(neighbors)]:
                        if src_t in self.type_reverse_maps and src_idx in self.type_reverse_maps[src_t]:
                            neighbor_dict = self.type_reverse_maps[src_t][src_idx]
                            neighbors.append({
                                'type': src_t,
                                'relation': rel,
                                'direction': 'incoming',
                                'label': neighbor_dict['properties'].get('label', ''),
                            })

            if len(neighbors) >= max_neighbors:
                break

        return neighbors

    # -----------------------------------------------------------------
    # Entrainement QueryEncoder
    # -----------------------------------------------------------------

    def contrastive_loss(self, q_proj, z_positive, z_negatives, temperature=0.07):
        """Loss InfoNCE pour aligner query_proj avec noeuds pertinents."""
        pos_sim = F.cosine_similarity(q_proj, z_positive, dim=1) / temperature
        neg_sim = torch.bmm(
            z_negatives,
            q_proj.unsqueeze(2)
        ).squeeze(2) / temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=q_proj.device)
        return F.cross_entropy(logits, labels)

    def train_query_encoder(
        self,
        query_node_pairs,  # Liste de (query_text, (node_type, local_idx)_pos, [(node_type, local_idx)_neg])
        num_epochs=100,
        lr=0.001,
        temperature=0.07
    ):
        """
        Entraine le QueryEncoder avec loss contrastive.

        Phase 1 : GNN est FIGE (deja entraine sur link prediction)
        Phase 2 : On entraine UNIQUEMENT le QueryEncoder

        Args:
            query_node_pairs: Liste de tuples :
                (query_text,
                 (node_type, local_idx) pour le positif,
                 [(node_type, local_idx), ...] pour les negatifs)
        """
        print(f"\n{'='*80}")
        print("ENTRAINEMENT QUERY ENCODER")
        print(f"{'='*80}")
        print(f"Paires d'entrainement: {len(query_node_pairs)}")

        # Deriver input_dim depuis le modele d'embeddings
        probe_dim = len(self.embed_model.encode_query("test"))
        print(f"  Dimension encode_query detectee: {probe_dim}")

        # Verifier que le GNN sort bien LATENT_DIM
        self.gnn_model.eval()
        with torch.no_grad():
            z_dict = self.gnn_model(
                self.graph_data.x_dict,
                self.graph_data.edge_index_dict
            )
        for ntype, z in z_dict.items():
            assert z.size(1) == LATENT_DIM, (
                f"GNN output '{ntype}': dim={z.size(1)} != LATENT_DIM={LATENT_DIM}."
            )

        # Initialiser QueryEncoder
        self.query_encoder = QueryEncoder(
            input_dim=probe_dim,
            output_dim=LATENT_DIM
        ).to(self.device)

        optimizer = torch.optim.Adam(self.query_encoder.parameters(), lr=lr)

        # Figer le GNN
        for param in self.gnn_model.parameters():
            param.requires_grad = False
        print("[OK] GNN fige")

        # Pre-calculer les embeddings de noeuds (GNN fige)
        # z_dict est deja calcule ci-dessus
        print(f"[OK] Embeddings pre-calcules pour {len(z_dict)} types")

        # Training loop
        self.query_encoder.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for query_text, pos_key, neg_keys in query_node_pairs:
                optimizer.zero_grad()

                # 1. Encoder la requete
                q_emb = self.embed_model.encode_query(query_text)
                q_tensor = torch.tensor(q_emb, dtype=torch.float).unsqueeze(0).to(self.device)
                q_proj = self.query_encoder(q_tensor)

                # 2. Embeddings positif
                pos_type, pos_idx = pos_key
                z_pos = z_dict[pos_type][pos_idx].unsqueeze(0)

                # 3. Embeddings negatifs
                neg_list = []
                for neg_type, neg_idx in neg_keys:
                    neg_list.append(z_dict[neg_type][neg_idx])
                z_neg = torch.stack(neg_list).unsqueeze(0)  # [1, num_neg, LATENT_DIM]

                # 4. Loss
                loss = self.contrastive_loss(q_proj, z_pos, z_neg, temperature)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0:
                avg = total_loss / num_batches if num_batches else 0
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg:.4f}")

        print(f"[OK] QueryEncoder entraine")


# =============================================================================
# Pipeline
# =============================================================================

class GRetrieverRAGPipeline(GraphRAGPipeline):
    """Pipeline Graph-RAG avec G-Retriever heterogene (herite de GraphRAGPipeline)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("\nInitialisation G-Retriever (heterogene)...")
        self.gretriever = GRetrieverManager(
            self.graph,
            self.embed_model
        )

        # Construire le graphe HeteroData
        self.gretriever.build_pytorch_graph()

        # Verifier que les types cibles existent
        available_targets = [
            t for t in RETRIEVAL_TARGET_TYPES
            if t in self.gretriever.graph_data.node_types
        ]
        if not available_targets:
            print(f"[WARNING] Aucun type cible {RETRIEVAL_TARGET_TYPES} dans le graphe.")
            print(f"  Types disponibles: {self.gretriever.graph_data.node_types}")
        else:
            print(f"  Types cibles disponibles: {available_targets}")

        # Entrainer le GNN
        self.gretriever.train_gnn(num_epochs=50, lr=0.01)

        print("[INFO] QueryEncoder non entraine.")
        print("       Pour entrainer: gretriever.train_query_encoder(query_node_pairs)")

    def query(self, question: str, k: int = 5,
              use_gnn: bool = True,
              **kwargs) -> Tuple[str, List[RetrievalResult]]:
        """
        Requete avec G-Retriever heterogene.

        Args:
            question: Question utilisateur
            k: Nombre de resultats
            use_gnn: Utiliser le GNN retrieval
            **kwargs: Autres arguments (commune_filter, etc.)

        Returns:
            (reponse, resultats)
        """
        print(f"\n{'='*80}")
        print(f"REQUETE G-RETRIEVER: {question}")
        print(f"{'='*80}")

        # Detection automatique de commune
        detected_commune = detect_commune(question)
        if detected_commune:
            print(f"[AUTO-DETECT v6] Commune detectee: {detected_commune}")
            if 'commune_filter' not in kwargs:
                kwargs['commune_filter'] = detected_commune

        # 1. Retrieval via GNN (types cibles ontologiques)
        gnn_results = []
        if use_gnn:
            try:
                print("Retrieval via GNN heterogene...")
                gnn_results = self.gretriever.retrieve_subgraph(question, k=k)
                print(f"[OK] {len(gnn_results)} noeuds recuperes du graphe")
            except RuntimeError as e:
                print(f"[SKIP GNN] {e}")

        # 2. Retrieval vectoriel classique
        response, vector_results = super().query(question, k=k, use_graph=False, **kwargs)

        # 3. Fusionner GNN + vecteurs
        all_results = []

        for gnn_result in gnn_results:
            # Enrichir le texte avec les voisins 1-hop
            text = gnn_result['text']
            neighbors = gnn_result.get('neighbors', [])
            if neighbors:
                neighbor_info = ", ".join(
                    f"{n['relation']}->{n['label']}" for n in neighbors if n['label']
                )
                if neighbor_info:
                    text += f" [Voisins: {neighbor_info}]"

            all_results.append(RetrievalResult(
                text=text,
                metadata={
                    'source': 'gnn',
                    'node_type': gnn_result['node_type'],
                    'properties': gnn_result['node']['properties'],
                    'neighbors': neighbors
                },
                score=gnn_result['score'] * 0.6,
                source_type='gnn'
            ))

        for vector_result in vector_results:
            all_results.append(RetrievalResult(
                text=vector_result.text,
                metadata=vector_result.metadata,
                score=vector_result.score * 0.4,
                source_type=vector_result.source_type
            ))

        # Trier et garder top-k
        all_results.sort(key=lambda x: x.score, reverse=True)
        final_results = all_results[:k]

        # 4. Regenerer la reponse
        prompt = self._build_graph_rag_prompt(question, final_results, "")
        response = self._generate_response(prompt)

        return response, final_results


# =============================================================================
# Main
# =============================================================================

def main():
    """Test de G-Retriever heterogene."""
    print("=" * 80)
    print("G-RETRIEVER GRAPH-RAG HETEROGENE - Demonstration")
    print("=" * 80)

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    rag = GRetrieverRAGPipeline(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=NEO4J_PASSWORD,
        openai_api_key=OPENAI_API_KEY
    )

    rag.import_commune_data("df_mean_by_commune.csv")

    response, results = rag.query(
        "Quelles dimensions du bien-etre sont liees a la sante ?",
        use_gnn=True
    )

    print(f"\nReponse:\n{response}")
    rag.close()


if __name__ == "__main__":
    main()
