"""
RAG v9 - RAPTOR-lite : vues analytiques hierarchiques sur verbatims citoyens.

Architecture NON-recursive : la hierarchie est definie explicitement par
6 vues analytiques (3 x 1D, 3 x 2D) sur les dimensions age_range, profession, commune.

Usage:
    python rag_v9_raptor.py --build          # Genere toutes les syntheses
    python rag_v9_raptor.py --stats          # Affiche les stats par vue
    python rag_v9_raptor.py --query "..."    # Test de retrieval
"""

import os
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Constantes
# ============================================================

VIEW_DEFINITIONS = [
    # 2D démographiques (specificity=2, testées en premier au retrieval)
    {"name": "age_range*profession", "dimensions": ["age_range", "profession"], "specificity": 2},
    {"name": "age_range*commune",    "dimensions": ["age_range", "nom"],        "specificity": 2},
    {"name": "profession*commune",   "dimensions": ["profession", "nom"],       "specificity": 2},
    # 2D thématiques (dimension QdV croisée avec démographie)
    {"name": "dimension*commune",    "dimensions": ["dimension", "nom"],        "specificity": 2},
    {"name": "dimension*age_range",  "dimensions": ["dimension", "age_range"],  "specificity": 2},
    {"name": "dimension*profession", "dimensions": ["dimension", "profession"], "specificity": 2},
    # 1D (specificity=1, fallback)
    {"name": "age_range",  "dimensions": ["age_range"],  "specificity": 1},
    {"name": "profession", "dimensions": ["profession"], "specificity": 1},
    {"name": "commune",    "dimensions": ["nom"],        "specificity": 1},
    {"name": "dimension",  "dimensions": ["dimension"],  "specificity": 1},
]

MATERIALIZATION_THRESHOLD = 3

CHROMA_PATH = "./chroma_portrait"
SOURCE_COLLECTION = "portrait_verbatims"
TARGET_COLLECTION = "raptor_summaries"

SUMMARIZATION_MODEL = "mistral-small-latest"
SUMMARIZATION_BASE_URL = "https://api.mistral.ai/v1"


# ============================================================
# Appel LLM local (max_tokens adapte pour les syntheses)
# ============================================================

def _call_llm(prompt: str, system_prompt: str,
              model: str = SUMMARIZATION_MODEL,
              base_url: str = SUMMARIZATION_BASE_URL,
              max_tokens: int = 1500,
              temperature: float = 0.3,
              max_retries: int = 5) -> str:
    """Appel LLM via API OpenAI-compatible avec retry exponentiel sur 429."""
    from openai import OpenAI
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY non definie")
    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt * 2  # 2, 4, 8, 16, 32 secondes
                print(f"    [RATE LIMIT] Attente {wait}s avant retry ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


# ============================================================
# RaptorBuilder
# ============================================================

SYSTEM_PROMPT_SUMMARY = (
    "Tu es un analyste specialise dans l'analyse de verbatims citoyens sur la qualite de vie. "
    "Tu produis des syntheses analytiques structurees, factuelles et nuancees. "
    "Tu ne dois JAMAIS inventer d'information. Base-toi uniquement sur les verbatims fournis."
)


def _build_user_prompt(view_def: Dict, dim_values: Dict[str, str],
                       chunks: List[Tuple[str, str, Dict]]) -> str:
    """Construit le prompt utilisateur pour la generation de synthese."""
    dims_desc = "\n".join(f"- {d} : {dim_values[d]}" for d in view_def["dimensions"])
    dim_label = ", ".join(f"{d}={dim_values[d]}" for d in view_def["dimensions"])

    verbatims = []
    for i, (chunk_id, text, meta) in enumerate(chunks, 1):
        verbatims.append(f"[Verbatim {i}] {text[:500]}")

    return f"""Voici {len(chunks)} verbatims de citoyens correspondant au groupe suivant :
- Vue : {view_def['name']}
{dims_desc}

=== VERBATIMS ===
{chr(10).join(verbatims)}

Produis une synthese analytique de ce groupe en suivant EXACTEMENT ce format :

**Groupe : {dim_label} (N={len(chunks)})**

**Themes dominants** : Liste les 2-4 themes les plus frequents dans ces verbatims.

**Points de convergence** : Quels avis ou constats sont partages par plusieurs personnes ?

**Points de divergence** : Quelles opinions contradictoires ou minoritaires apparaissent ?

**Signaux faibles** : Y a-t-il des mentions isolees mais potentiellement significatives ?

**Limites de l'echantillon** : Indique les biais possibles (taille du groupe, surrepresentation, etc.)

Reste factuel. Cite des extraits courts entre guillemets quand c'est pertinent."""


def _make_doc_id(view_name: str, dim_values: Dict[str, str]) -> str:
    """Cree un ID deterministe et slugifie pour un document RAPTOR."""
    slug = "_".join(
        v.lower().replace(" ", "-").replace("(", "").replace(")", "")[:30]
        for v in dim_values.values()
    )
    return f"raptor_{view_name}_{slug}"


class RaptorBuilder:
    """Genere les vues analytiques RAPTOR-lite et les stocke dans ChromaDB."""

    def __init__(self,
                 chroma_path: str = CHROMA_PATH,
                 source_collection: str = SOURCE_COLLECTION,
                 target_collection: str = TARGET_COLLECTION,
                 min_chunks: int = MATERIALIZATION_THRESHOLD):
        self.chroma_path = chroma_path
        self.source_collection_name = source_collection
        self.target_collection_name = target_collection
        self.min_chunks = min_chunks

    def build_all(self, incremental: bool = False) -> Dict:
        """Genere toutes les vues RAPTOR. Retourne les statistiques.
        Si incremental=True, ne regenere pas les syntheses deja presentes."""
        import chromadb
        from sentence_transformers import SentenceTransformer

        print("=" * 70)
        print("RAPTOR-lite : Construction des vues analytiques")
        if incremental:
            print("  Mode INCREMENTAL : seules les syntheses manquantes seront generees")
        print("=" * 70)

        # 1. Charger les chunks source
        client = chromadb.PersistentClient(path=self.chroma_path)
        source = client.get_collection(self.source_collection_name)
        all_data = source.get(include=["documents", "metadatas"])

        chunks = list(zip(all_data["ids"], all_data["documents"], all_data["metadatas"]))
        print(f"Source : {len(chunks)} chunks depuis {self.source_collection_name}")

        # 2. Collection cible : recreer ou reutiliser
        if incremental:
            target = client.get_or_create_collection(
                self.target_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            existing_ids = set(target.get()["ids"])
            print(f"Collection cible : {self.target_collection_name} ({len(existing_ids)} syntheses existantes)")
        else:
            try:
                client.delete_collection(self.target_collection_name)
            except Exception:
                pass
            target = client.create_collection(
                self.target_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            existing_ids = set()
            print(f"Collection cible : {self.target_collection_name} (recree)")

        # 3. Charger le modele d'embeddings
        print("Chargement du modele d'embeddings (BGE-M3)...")
        embed_model = SentenceTransformer("BAAI/bge-m3")

        # 4. Construire chaque vue
        stats = {"total_views": 0, "total_summaries": 0, "by_view": {}}

        for view_def in VIEW_DEFINITIONS:
            print(f"\n--- Vue : {view_def['name']} ---")
            n_materialized = self._build_view(view_def, chunks, target, embed_model, existing_ids)
            stats["by_view"][view_def["name"]] = n_materialized
            stats["total_summaries"] += n_materialized
            stats["total_views"] += 1

        print(f"\n{'=' * 70}")
        print(f"BUILD TERMINE : {stats['total_summaries']} syntheses generees")
        for view_name, count in stats["by_view"].items():
            print(f"  {view_name}: {count}")
        print("=" * 70)

        return stats

    def _build_view(self, view_def: Dict, all_chunks: List,
                    target_collection, embed_model,
                    existing_ids: set = None) -> int:
        """Groupe et materialise une vue. Retourne le nombre de groupes."""
        dims = view_def["dimensions"]
        if existing_ids is None:
            existing_ids = set()

        # Grouper par dimensions
        groups = defaultdict(list)
        for chunk_id, text, meta in all_chunks:
            key_values = tuple(meta.get(d, "") for d in dims)
            if all(key_values):  # skip si une dimension est vide
                groups[key_values].append((chunk_id, text, meta))

        # Filtrer par seuil
        valid_groups = {k: v for k, v in groups.items() if len(v) >= self.min_chunks}
        print(f"  {len(groups)} groupes, {len(valid_groups)} avec >= {self.min_chunks} chunks")

        materialized = 0
        for key_values, group_chunks in valid_groups.items():
            dim_values = {d: v for d, v in zip(dims, key_values)}

            # Skip si deja present (mode incremental)
            doc_id = _make_doc_id(view_def["name"], dim_values)
            if doc_id in existing_ids:
                dim_label = ", ".join(f"{d}={v}" for d, v in dim_values.items())
                print(f"  [SKIP] {dim_label} (deja present)")
                materialized += 1
                continue

            # Generer la synthese
            try:
                prompt = _build_user_prompt(view_def, dim_values, group_chunks)
                summary = _call_llm(prompt, SYSTEM_PROMPT_SUMMARY)
            except Exception as e:
                print(f"  [ERREUR] {dim_values}: {e}")
                continue

            # Embedder la synthese
            embedding = embed_model.encode(f"passage: {summary}").tolist()

            # Construire les metadonnees (doc_id deja calcule plus haut)
            meta = {
                "view_name": view_def["name"],
                "specificity": view_def["specificity"],
                "dim1_name": dims[0],
                "dim1_value": key_values[0],
                "dim2_name": dims[1] if len(dims) > 1 else "",
                "dim2_value": key_values[1] if len(key_values) > 1 else "",
                "num_chunks": len(group_chunks),
                "source_chunk_ids": json.dumps([c[0] for c in group_chunks]),
                "built_at": datetime.now().isoformat(),
            }

            # Stocker
            target_collection.add(
                ids=[doc_id],
                documents=[summary],
                embeddings=[embedding],
                metadatas=[meta]
            )
            materialized += 1

            dim_label = ", ".join(f"{d}={v}" for d, v in dim_values.items())
            print(f"  [OK] {dim_label} (N={len(group_chunks)})")

            # Rate limiting
            time.sleep(0.3)

        return materialized


# ============================================================
# RaptorRetriever
# ============================================================

class RaptorRetriever:
    """Retrieval RAPTOR-lite en 2 etapes : detection dimensions -> fallback hierarchique."""

    def __init__(self,
                 chroma_path: str = CHROMA_PATH,
                 source_collection: str = SOURCE_COLLECTION,
                 summary_collection: str = TARGET_COLLECTION,
                 n_evidence_chunks: int = 5,
                 oppchovec_col=None):
        self.chroma_path = chroma_path
        self.source_collection_name = source_collection
        self.summary_collection_name = summary_collection
        self.n_evidence = n_evidence_chunks
        self._embed_model = None
        self._source = None
        self._summaries = None
        self._oppchovec = oppchovec_col  # collection ChromaDB oppchovec_scores (optionnelle)

    # Collections supplémentaires et leurs paramètres de retrieval
    _EXTRA_COLLECTIONS = [
        {"name": "portrait_entretiens",   "data_type": "quali",  "label": "Entretien semi-directif (quali)",       "threshold": 0.55},
        {"name": "enquete_responses",     "data_type": "quanti", "label": "Réponse enquête (quanti)",              "threshold": 0.50},
        {"name": "enquete_scores_commune","data_type": "quanti", "label": "Scores enquête commune (quanti)",       "threshold": 0.50},
        {"name": "communes_wiki",         "data_type": "mixte",  "label": "Fiche commune Wikipedia (mixte)",       "threshold": 0.60},
        {"name": "communes_equipements",  "data_type": "quanti", "label": "Équipements et services commune (quanti)", "threshold": 0.80},
        {"name": "communes_geo",          "data_type": "mixte",  "label": "Géographie commune (mixte)",               "threshold": 0.75},
        {"name": "zones_epci",            "data_type": "mixte",  "label": "Zone intercommunale (mixte)",              "threshold": 0.75},
    ]

    def init(self):
        """Charge les modeles et connecte aux collections ChromaDB."""
        import chromadb
        from sentence_transformers import SentenceTransformer

        self._embed_model = SentenceTransformer("BAAI/bge-m3")
        client = chromadb.PersistentClient(path=self.chroma_path)
        self._source = client.get_collection(self.source_collection_name)
        self._summaries = client.get_collection(self.summary_collection_name)
        self.summary_count = self._summaries.count()

        # Charger oppchovec_scores depuis le même client (évite les conflits multi-client)
        try:
            self._oppchovec = client.get_collection("oppchovec_scores")
        except Exception:
            self._oppchovec = None

        # Charger les collections supplémentaires
        self._extra_cols: Dict[str, object] = {}
        for cfg in self._EXTRA_COLLECTIONS:
            try:
                self._extra_cols[cfg["name"]] = client.get_collection(cfg["name"])
            except Exception:
                pass

        loaded = list(self._extra_cols.keys())
        opp_info = f" + oppchovec ({self._oppchovec.count()} docs)" if self._oppchovec else ""
        extra_info = f" + {len(loaded)} collections extra ({', '.join(loaded)})" if loaded else ""
        print(f"RAPTOR Retriever : {self.summary_count} syntheses{opp_info}{extra_info}")

    def query(self, question: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Retrieval RAPTOR en 2 etapes.
        Retourne (context_str, sources_list) compatible RAGAdapter.
        """
        # Etape 1 : Detecter les dimensions dans la question
        detected = self._detect_dimensions(question)
        print(f"  [RAPTOR] Dimensions detectees: {detected if detected else 'aucune'}")

        # Etape 2 : Trouver la vue la plus specifique
        summary_text, summary_meta = self._find_best_view(question, detected)

        if summary_text:
            view_name = summary_meta.get("view_name", "?")
            num_chunks = summary_meta.get("num_chunks", "?")
            print(f"  [RAPTOR] Vue trouvee: {view_name} (N={num_chunks})")
        else:
            print("  [RAPTOR] Aucune vue trouvee, fallback semantique")

        # Etape 3 : Recuperer les chunks evidence
        evidence_docs = []
        evidence_metas = []
        if summary_meta and summary_meta.get("source_chunk_ids"):
            chunk_ids = json.loads(summary_meta["source_chunk_ids"])
            # Prendre un sous-ensemble des chunks comme evidence
            evidence_ids = chunk_ids[:k]
            try:
                evidence = self._source.get(
                    ids=evidence_ids,
                    include=["documents", "metadatas"]
                )
                evidence_docs = evidence["documents"]
                evidence_metas = evidence["metadatas"]
            except Exception:
                pass

        # Etape 4 : Assembler le contexte
        context_parts = []
        if summary_text:
            context_parts.append(
                f"[Synthese RAPTOR (quali) - vue {summary_meta.get('view_name', 'N/A')}]\n{summary_text}"
            )

        for i, (doc, meta) in enumerate(zip(evidence_docs, evidence_metas), 1):
            context_parts.append(f"[Verbatim (quali) {i}] {doc[:300]}...")

        # Etape 5 : Sources supplémentaires (entretiens, enquête, wiki)
        extra_results = self.query_extra_sources(question, k=2)
        for r in extra_results:
            context_parts.append(f"[{r['label']}]\n{r['text'][:1500]}")

        context_str = "\n\n".join(context_parts)

        # Construire la liste de sources
        sources = []
        if summary_text:
            sources.append({
                "rank": 0,
                "type": "raptor_summary",
                "view": summary_meta.get("view_name"),
                "num_chunks": summary_meta.get("num_chunks"),
                "extrait": summary_text
            })
        for i, (doc, meta) in enumerate(zip(evidence_docs, evidence_metas), 1):
            sources.append({
                "rank": i,
                "commune": meta.get("nom", "N/A"),
                "genre": meta.get("genre", "N/A"),
                "age": meta.get("age_exact", "N/A"),
                "profession": meta.get("profession", "N/A"),
                "dimension": meta.get("dimension", "N/A"),
                "source_type": "verbatim_evidence",
                "extrait": doc
            })
        for i, r in enumerate(extra_results):
            sources.append({
                "rank": len(sources) + i,
                "source_type": r["collection"],
                "data_type": r["data_type"],
                "commune": r["meta"].get("commune", r["meta"].get("nom", "N/A")),
                "distance": round(r["distance"], 3),
                "extrait": r["text"][:1500],
            })

        return context_str, sources

    def query_extra_sources(self, question: str, k: int = 2) -> List[Dict]:
        """
        Recherche sémantique dans les collections supplémentaires (entretiens, enquête, wiki).
        Retourne les résultats pertinents (distance < seuil) avec label et data_type.
        """
        if not self._extra_cols:
            return []
        q_emb = self._embed_model.encode(f"query: {question}").tolist()
        results = []
        for cfg in self._EXTRA_COLLECTIONS:
            col = self._extra_cols.get(cfg["name"])
            if col is None:
                continue
            n = min(k, col.count())
            if n == 0:
                continue
            try:
                res = col.query(
                    query_embeddings=[q_emb],
                    n_results=n,
                    include=["documents", "metadatas", "distances"]
                )
                for doc, meta, dist in zip(
                    res["documents"][0], res["metadatas"][0], res["distances"][0]
                ):
                    if dist <= cfg["threshold"]:
                        results.append({
                            "text": doc,
                            "meta": meta,
                            "distance": dist,
                            "collection": cfg["name"],
                            "data_type": cfg["data_type"],
                            "label": cfg["label"],
                        })
            except Exception:
                pass
        return results

    def query_oppchovec(self, question: str, k: int = 3) -> List[Dict]:
        """
        Cherche les scores OppChoVec les plus pertinents pour la question.
        Retourne toujours le doc méthodologique (par ID fixe) en premier,
        puis k-1 communes pertinentes par recherche sémantique.
        Retourne [] si la collection n'est pas disponible.
        """
        if self._oppchovec is None:
            return []

        results = []

        # 1. Toujours inclure le doc méthodologique (par ID fixe, zéro risque de non-retrieval)
        try:
            meth = self._oppchovec.get(
                ids=["oppchovec_methodology"],
                include=["documents", "metadatas"]
            )
            if meth["documents"]:
                results.append({
                    "text": meth["documents"][0],
                    "meta": meth["metadatas"][0],
                    "distance": 0.0
                })
        except Exception:
            pass

        # 2. Recherche sémantique pour les communes pertinentes (k-1 résultats)
        k_communes = max(1, k - 1)
        q_emb = self._embed_model.encode(f"query: {question}").tolist()
        n = min(k_communes, self._oppchovec.count())
        if n > 0:
            try:
                res = self._oppchovec.query(
                    query_embeddings=[q_emb],
                    n_results=n,
                    include=["documents", "metadatas", "distances"],
                    where={"source": {"$in": ["oppchovec_betti_0_10", "oppchovec_aggregate"]}}
                )
                for doc, meta, dist in zip(
                    res["documents"][0],
                    res["metadatas"][0],
                    res["distances"][0]
                ):
                    results.append({"text": doc, "meta": meta, "distance": dist})
            except Exception:
                pass

        return results


    # Mots-clés pour détecter la dimension QdV dans une question
    _DIMENSION_KEYWORDS: Dict[str, list] = {
        "Santé":    ["santé", "médecin", "hôpital", "soins", "médical", "désert médical", "sanitaire", "bien-être physique"],
        "Environnement": ["environnement", "nature", "paysage", "pollution", "déchets", "écologie", "vert", "cadre naturel"],
        "Culture":  ["culture", "patrimoine", "animation", "loisir", "sport", "artistique", "festival", "culturel"],
        "Logement": ["logement", "habitat", "immobilier", "loyer", "hébergement", "appartement", "maison", "logements"],
        "Services de proximité": ["services de proximité", "commerces", "supermarchés", "boutique", "magasin", "épicerie", "proximité"],
        "Réseau":   ["réseau", "internet", "connexion", "fibre", "numérique", "digital", "4g", "5g", "couverture réseau"],
        "Sécurité": ["sécurité", "violence", "criminalité", "délinquance", "insécurité", "sûreté"],
        "Ratio vie pro/vie perso": ["vie professionnelle", "vie pro", "équilibre travail", "télétravail", "conciliation", "work-life"],
        "Education": ["éducation", "école", "université", "formation", "enseignement", "lycée", "collège", "scolaire"],
        "Revenus":  ["revenus", "salaires", "coût de la vie", "pouvoir d'achat", "prix", "cherté", "rémunération"],
        "Emploi":   ["emploi", "chômage", "offres d'emploi", "recrutement", "travail", "marché du travail"],
        "Transports": ["transports", "mobilité", "voiture", "bus", "routes", "trafic", "déplacements", "route"],
        "Communauté et relations": ["communauté", "liens sociaux", "convivialité", "voisinage", "relations sociales", "solidarité", "lien social"],
        "Tourisme (ressenti localement)": ["tourisme", "touristes", "saison touristique", "masse touristique", "flux touristique"],
    }

    @classmethod
    def _detect_dimension_theme(cls, question: str) -> Optional[str]:
        """Detecte la dimension QdV (theme) mentionnee dans la question."""
        q = question.lower()
        for dim, keywords in cls._DIMENSION_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return dim
        return None

    def _detect_dimensions(self, question: str) -> Dict[str, str]:
        """Detecte les dimensions (age_range, profession, commune, dimension QdV) dans la question."""
        detected = {}

        try:
            from portrait_detector import detect_portrait_filters
            filters = detect_portrait_filters(question)
            if filters.get("age_range"):
                detected["age_range"] = filters["age_range"]
            elif filters.get("age_min") is not None:
                detected["age_range"] = self._map_age_to_range(
                    filters["age_min"], filters.get("age_max", 100)
                )
            if filters.get("profession"):
                detected["profession"] = filters["profession"]
        except ImportError:
            pass

        try:
            from commune_detector import detect_commune
            commune = detect_commune(question)
            if commune:
                detected["nom"] = commune
        except ImportError:
            pass

        theme = self._detect_dimension_theme(question)
        if theme:
            detected["dimension"] = theme

        return detected

    def _find_best_view(self, question: str,
                        detected: Dict[str, str]) -> Tuple[Optional[str], Dict]:
        """
        Trouve la vue la plus specifique avec fallback.
        Retourne (summary_text, summary_metadata) ou (None, {}).
        """
        detected_dims = set(detected.keys())

        # Essayer les vues de la plus specifique a la moins specifique
        for view_def in VIEW_DEFINITIONS:
            view_dims = set(view_def["dimensions"])

            # Toutes les dimensions de cette vue doivent etre detectees
            if not view_dims.issubset(detected_dims):
                continue

            # Construire le filtre ChromaDB
            conditions = [{"view_name": view_def["name"]}]
            for i, dim in enumerate(view_def["dimensions"]):
                dim_key = f"dim{i+1}_value"
                conditions.append({dim_key: detected[dim]})

            where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]

            try:
                results = self._summaries.get(
                    where=where_filter,
                    include=["documents", "metadatas"]
                )
                if results["documents"]:
                    return results["documents"][0], results["metadatas"][0]
            except Exception:
                continue

        # Fallback : recherche semantique sur toutes les syntheses
        # Seuil de distance : ne pas injecter un RAPTOR non pertinent
        FALLBACK_DISTANCE_THRESHOLD = 0.55
        if self._summaries.count() > 0:
            query_embedding = self._embed_model.encode(f"query: {question}").tolist()
            results = self._summaries.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )
            if results["documents"] and results["documents"][0]:
                distance = results["distances"][0][0]
                if distance <= FALLBACK_DISTANCE_THRESHOLD:
                    return results["documents"][0][0], results["metadatas"][0][0]
                else:
                    print(f"  [RAPTOR] Fallback ignore (distance={distance:.3f} > seuil {FALLBACK_DISTANCE_THRESHOLD})")

        return None, {}

    @staticmethod
    def _map_age_to_range(age_min: int, age_max: int) -> str:
        """Mappe une tranche d'age numerique vers le bucket standard."""
        midpoint = (age_min + age_max) / 2
        if midpoint < 25:
            return "15-24"
        elif midpoint < 35:
            return "25-34"
        elif midpoint < 50:
            return "35-49"
        elif midpoint < 65:
            return "50-64"
        return "65+"

    def close(self):
        """Libere les ressources."""
        pass


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAPTOR-lite : vues analytiques sur verbatims citoyens"
    )
    parser.add_argument("--build", action="store_true",
                        help="Construire toutes les vues RAPTOR")
    parser.add_argument("--incremental", action="store_true",
                        help="Mode incremental : ne regenere que les syntheses manquantes")
    parser.add_argument("--stats", action="store_true",
                        help="Afficher les statistiques des vues")
    parser.add_argument("--query", type=str,
                        help="Tester le retrieval sur une question")
    parser.add_argument("--k", type=int, default=5,
                        help="Nombre de chunks evidence (defaut: 5)")
    args = parser.parse_args()

    if args.build:
        builder = RaptorBuilder()
        stats = builder.build_all(incremental=args.incremental)
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    elif args.stats:
        import chromadb
        from collections import Counter
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            coll = client.get_collection(TARGET_COLLECTION)
        except Exception:
            print("Collection raptor_summaries introuvable. Lancez --build d'abord.")
            return
        all_docs = coll.get(include=["metadatas"])
        print(f"Total syntheses : {len(all_docs['ids'])}")
        views = Counter(m["view_name"] for m in all_docs["metadatas"])
        for view, count in views.most_common():
            print(f"  {view}: {count} groupes")

    elif args.query:
        retriever = RaptorRetriever(n_evidence_chunks=args.k)
        retriever.init()
        context, sources = retriever.query(args.query, k=args.k)
        print("\n=== CONTEXTE ===")
        print(context[:2000])
        print(f"\n=== SOURCES ({len(sources)}) ===")
        for s in sources:
            print(f"  {s}")
        retriever.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
