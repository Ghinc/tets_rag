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


def _count_persons(chunks: List[Tuple[str, str, Dict]]) -> int:
    """Compte les personnes uniques dans un groupe de chunks.
    Utilise num_questionnaire si disponible, sinon déduplique sur (nom, age_exact, genre, profession)."""
    ids = set()
    for _, _, meta in chunks:
        qid = meta.get("num_questionnaire")
        if qid is not None:
            ids.add(qid)
        else:
            ids.add((meta.get("nom", ""), meta.get("age_exact", ""),
                     meta.get("genre", ""), meta.get("profession", "")))
    return len(ids)


def _build_user_prompt(view_def: Dict, dim_values: Dict[str, str],
                       chunks: List[Tuple[str, str, Dict]]) -> str:
    """Construit le prompt utilisateur pour la generation de synthese."""
    dims_desc = "\n".join(f"- {d} : {dim_values[d]}" for d in view_def["dimensions"])
    dim_label = ", ".join(f"{d}={dim_values[d]}" for d in view_def["dimensions"])
    n_persons = _count_persons(chunks)

    verbatims = []
    for i, (chunk_id, text, meta) in enumerate(chunks, 1):
        verbatims.append(f"[Verbatim {i}] {text[:500]}")

    return f"""Voici {len(chunks)} verbatims issus de {n_persons} questionnaires, correspondant au groupe suivant :
- Vue : {view_def['name']}
{dims_desc}

=== VERBATIMS ===
{chr(10).join(verbatims)}

Produis une synthese analytique de ce groupe en suivant EXACTEMENT ce format :

**Groupe : {dim_label} (N={n_persons} personnes)**

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

        # 2. Collection cible : vider ou reutiliser
        # Note: on evite delete_collection + create_collection (bug ChromaDB Rust sur Windows).
        # A la place : get_or_create + suppression individuelle des docs existants si non-incremental.
        target = client.get_or_create_collection(
            self.target_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        if incremental:
            existing_ids = set(target.get()["ids"])
            print(f"Collection cible : {self.target_collection_name} ({len(existing_ids)} syntheses existantes)")
        else:
            old_ids = target.get()["ids"]
            if old_ids:
                target.delete(ids=old_ids)
                print(f"Collection cible : {self.target_collection_name} ({len(old_ids)} anciens docs supprimes)")
            else:
                print(f"Collection cible : {self.target_collection_name} (vide)")
            existing_ids = set()

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
            n_persons = _count_persons(group_chunks)
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
                "n_persons": n_persons,
                "source_chunk_ids": json.dumps([c[0] for c in group_chunks]),
                "built_at": datetime.now().isoformat(),
            }

            # Stocker (upsert pour etre idempotent)
            target_collection.upsert(
                ids=[doc_id],
                documents=[summary],
                embeddings=[embedding],
                metadatas=[meta]
            )
            materialized += 1

            dim_label = ", ".join(f"{d}={v}" for d, v in dim_values.items())
            print(f"  [OK] {dim_label} (N={n_persons} personnes, {len(group_chunks)} verbatims)")

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
    # wellbeing_type : "subjectif" (perceptions citoyens), "objectif" (indicateurs territoriaux), "contextuel" (données factuelles de contexte)
    _EXTRA_COLLECTIONS = [
        {"name": "portrait_entretiens",        "data_type": "quali",  "wellbeing_type": "subjectif",  "label": "Entretien semi-directif (subjectif/quali)",          "threshold": 0.55},
        {"name": "raptor_entretiens_summaries","data_type": "quali",  "wellbeing_type": "subjectif",  "label": "Synthèse RAPTOR entretiens (subjectif/quali)",       "threshold": 0.55},
        {"name": "enquete_responses",          "data_type": "quanti", "wellbeing_type": "subjectif",  "label": "Réponse enquête individuelle (subjectif/quanti)",    "threshold": 0.55},
        {"name": "enquete_scores_commune",     "data_type": "quanti", "wellbeing_type": "subjectif",  "label": "Scores satisfaction commune (subjectif/quanti)",     "threshold": 0.75},
        # raptor_enquete_summaries est géré par retrieval structuré (_find_best_enquete_view)
        # et non plus par recherche sémantique générique
        {"name": "communes_wiki",              "data_type": "mixte",  "wellbeing_type": "contextuel", "label": "Fiche commune Wikipedia (contextuel/mixte)",          "threshold": 0.60},
        {"name": "communes_equipements",       "data_type": "quanti", "wellbeing_type": "objectif",   "label": "Équipements et services commune (objectif/quanti)",  "threshold": 0.65},
        {"name": "communes_geo",               "data_type": "mixte",  "wellbeing_type": "objectif",   "label": "Géographie commune (objectif/mixte)",                "threshold": 0.75},
        {"name": "zones_epci",                 "data_type": "mixte",  "wellbeing_type": "contextuel", "label": "Zone intercommunale (contextuel/mixte)",              "threshold": 0.75},
        {"name": "communes_profil",            "data_type": "quanti", "wellbeing_type": "contextuel", "label": "Profil démographique des répondants (contextuel/quanti)", "threshold": 0.70},
    ]

    def init(self):
        """Charge les modeles et connecte aux collections ChromaDB."""
        # SentenceTransformer DOIT être importé avant chromadb (conflit de libs natives sur Windows)
        from sentence_transformers import SentenceTransformer
        self._embed_model = SentenceTransformer("BAAI/bge-m3")

        import chromadb
        client = chromadb.PersistentClient(path=self.chroma_path)
        self._source = client.get_collection(self.source_collection_name)
        self._summaries = client.get_collection(self.summary_collection_name)
        self.summary_count = self._summaries.count()

        # Charger oppchovec_scores depuis le même client (évite les conflits multi-client)
        try:
            self._oppchovec = client.get_collection("oppchovec_scores")
        except Exception:
            self._oppchovec = None

        # Charger raptor_enquete_summaries pour retrieval structuré
        try:
            self._enquete_summaries = client.get_collection("raptor_enquete_summaries")
        except Exception:
            self._enquete_summaries = None

        # Charger les collections supplémentaires
        self._extra_cols: Dict[str, object] = {}
        for cfg in self._EXTRA_COLLECTIONS:
            try:
                self._extra_cols[cfg["name"]] = client.get_collection(cfg["name"])
            except Exception:
                pass

        loaded = list(self._extra_cols.keys())
        opp_info = f" + oppchovec ({self._oppchovec.count()} docs)" if self._oppchovec else ""
        enq_info = f" + raptor_enquete ({self._enquete_summaries.count()} syntheses)" if self._enquete_summaries else ""
        extra_info = f" + {len(loaded)} collections extra ({', '.join(loaded)})" if loaded else ""
        print(f"RAPTOR Retriever : {self.summary_count} syntheses{opp_info}{enq_info}{extra_info}")

    # ── Hybrid retrieval helpers ──────────────────────────────────────────────

    _HYBRID_RRF_K: int = 60  # constante RRF standard

    def _encode_query(self, question: str) -> list:
        """Vecteur dense pour une question (compatible ChromaDB)."""
        return self._embed_model.encode(f"query: {question}").tolist()

    def _hybrid_rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = None,
    ) -> List[Dict]:
        """
        Re-rank candidates par RRF fusion : rang dense (ChromaDB cosinus) + rang BM25 (lexical).
        Chaque candidate est un dict avec au moins 'text' et 'distance'.
        Retourne la liste triée (meilleur en premier), champ 'distance' inchangé.
        """
        if len(candidates) <= 1:
            return candidates

        from rank_bm25 import BM25Okapi
        import re

        def _tokenize(text: str) -> List[str]:
            return re.findall(r'\w+', text.lower())

        corpus_tokens = [_tokenize(c['text']) for c in candidates]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(_tokenize(query))

        # Rang dense : distance croissante (plus petit = meilleur)
        dense_order = sorted(range(len(candidates)), key=lambda i: candidates[i]['distance'])
        dense_rank = {idx: rank for rank, idx in enumerate(dense_order)}
        # Rang BM25 : score décroissant (plus grand = meilleur)
        bm25_order = sorted(range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank = {idx: rank for rank, idx in enumerate(bm25_order)}

        K = self._HYBRID_RRF_K
        rrf = [
            1.0 / (K + dense_rank[i]) + 1.0 / (K + bm25_rank[i])
            for i in range(len(candidates))
        ]
        reranked = sorted(range(len(candidates)), key=lambda i: rrf[i], reverse=True)
        result = [candidates[i] for i in reranked]
        return result[:top_k] if top_k else result

    def _query_epci(self, question: str, detected: Dict,
                    k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Retrieval en deux temps pour les questions portant sur un EPCI.

        Étape 1 (structurelle, toujours exécutée) :
          - Lookup direct du doc zones_epci par ID → liste des communes membres,
            intercommunalités voisines, micro-région.
          - Récupération de la liste des communes depuis communes_geo (filtre metadata).

        Étape 2 (enrichissement QdV, conditionnel) :
          - Si la question porte aussi sur la qualité de vie / perceptions :
            pour chaque commune membre (≤ _MAX_COMMUNES_EPCI_ENRICHMENT),
            lookup structuré dans raptor_summaries et raptor_enquete_summaries.
          - Le fallback sémantique est désactivé (no_fallback=True) pour éviter
            qu'une commune dominante (ex: Ajaccio) contamine tout le contexte.

        Les collections wiki / entretiens / equip. sont interrogées normalement
        avec le filtre EPCI pour les collections compatibles.
        """
        epci_name = detected["epci"]
        context_parts: List[str] = []
        sources: List[Dict] = []

        # ── Étape 1 : lookup structurel ──────────────────────────────────────

        # 1a. Doc zones_epci par ID direct (zéro embedding)
        epci_col = self._extra_cols.get("zones_epci")
        member_communes: List[str] = []
        if epci_col is not None:
            try:
                res = epci_col.get(
                    ids=[f"epci_{epci_name}"],
                    include=["documents", "metadatas"]
                )
                if res["documents"]:
                    epci_doc_text = res["documents"][0]
                    context_parts.append(
                        f"[Structure territoriale — {epci_name}]\n{epci_doc_text}"
                    )
                    sources.append({
                        "rank": 0,
                        "type": "zones_epci",
                        "epci": epci_name,
                        "extrait": epci_doc_text[:400],
                    })
                    print(f"  [RAPTOR/EPCI] Doc zones_epci trouvé : {epci_name}")
            except Exception:
                pass

        # 1b. Communes membres depuis communes_geo (filtre metadata exact)
        member_communes = self._get_epci_member_communes(epci_name)
        print(f"  [RAPTOR/EPCI] {len(member_communes)} communes membres : {member_communes}")

        # ── Étape 2 : enrichissement QdV par commune (conditionnel) ─────────

        is_qdv = self._is_qdv_question(question)
        enriched = 0
        communes_with_data: List[str] = []
        communes_no_data: List[str] = []

        if is_qdv and member_communes:
            print(f"  [RAPTOR/EPCI] Question QdV → enrichissement per-commune "
                  f"(max {self._MAX_COMMUNES_EPCI_ENRICHMENT})")
            for commune in member_communes:
                if enriched >= self._MAX_COMMUNES_EPCI_ENRICHMENT:
                    break
                det_c = {k_: v for k_, v in detected.items()
                         if k_ not in ("nom", "noms", "epci")}
                det_c["nom"] = commune

                got_data = False

                # Synthèse RAPTOR verbatims — no_fallback pour éviter contamination
                t, m = self._find_best_view(question, det_c, no_fallback=True)
                if t:
                    context_parts.append(
                        f"[Synthèse RAPTOR — {commune} "
                        f"(vue {m.get('view_name', '?')})]\n{t}"
                    )
                    sources.append({
                        "rank": len(sources),
                        "type": "raptor_summary",
                        "commune": commune,
                        "view": m.get("view_name"),
                        "extrait": t[:300],
                    })
                    got_data = True

                # Scores enquête par commune — no_fallback idem
                et, em = self._find_best_enquete_view(det_c, no_fallback=True)
                if et:
                    context_parts.append(
                        f"[Scores enquête — {commune} "
                        f"(vue {em.get('view_name', '?')})]\n{et}"
                    )
                    sources.append({
                        "rank": len(sources),
                        "type": "raptor_enquete_summary",
                        "commune": commune,
                        "view": em.get("view_name"),
                        "extrait": et[:300],
                    })
                    got_data = True

                if got_data:
                    communes_with_data.append(commune)
                    enriched += 1
                else:
                    communes_no_data.append(commune)

            # Note de couverture pour le LLM
            remaining_communes = [c for c in member_communes
                                  if c not in communes_with_data
                                  and c not in communes_no_data[:0]]
            no_data_all = [c for c in member_communes if c not in communes_with_data]
            if no_data_all:
                context_parts.append(
                    f"[NOTE couverture enquête] Données d'enquête disponibles pour "
                    f"{len(communes_with_data)}/{len(member_communes)} communes de {epci_name} : "
                    f"{', '.join(communes_with_data) or 'aucune'}. "
                    f"Communes sans données suffisantes : {', '.join(no_data_all)}."
                )

        # ── Sources supplémentaires (entretiens, wiki, etc.) ─────────────────
        # Passer les communes membres (≤3) + l'EPCI pour guider les filtres
        extra_results = self.query_extra_sources(
            question, k=2,
            communes=member_communes[:3] if member_communes else None,
            epci=epci_name,
        )
        for r in extra_results:
            context_parts.append(f"[{r['label']}]\n{r['text'][:1500]}")
            sources.append({
                "rank": len(sources),
                "source_type": r["collection"],
                "data_type": r["data_type"],
                "commune": r["meta"].get("commune", r["meta"].get("nom", "N/A")),
                "distance": round(r["distance"], 3),
                "extrait": r["text"][:1500],
            })

        # ── Scores OppChoVec pour les communes membres ───────────────────────
        if self._oppchovec and member_communes:
            opp_results = self.query_oppchovec(
                question, k=min(len(member_communes), 3),
                communes=member_communes,
            )
            for r in opp_results:
                context_parts.append(
                    f"[Scores OppChoVec — {r['meta'].get('commune', 'N/A')}]\n{r['text'][:1500]}"
                )
                sources.append({
                    "rank": len(sources),
                    "type": "oppchovec_score",
                    "source_type": "oppchovec_scores",
                    "commune": r["meta"].get("commune"),
                    "oppchovec_0_10": r["meta"].get("oppchovec_0_10"),
                    "opp_0_10": r["meta"].get("opp_0_10"),
                    "cho_0_10": r["meta"].get("cho_0_10"),
                    "vec_0_10": r["meta"].get("vec_0_10"),
                    "distance": round(r.get("distance", 0), 3),
                    "extrait": r["text"][:1500],
                })

        context_str = "\n\n".join(context_parts)
        return context_str, sources

    def query(self, question: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Retrieval RAPTOR en 2 etapes.
        Retourne (context_str, sources_list) compatible RAGAdapter.
        """
        # Etape 1 : Detecter les dimensions dans la question
        detected = self._detect_dimensions(question)
        print(f"  [RAPTOR] Dimensions detectees: {detected if detected else 'aucune'}")

        # ── Dispatch EPCI : retrieval en deux temps ───────────────────────────
        if detected.get("epci"):
            return self._query_epci(question, detected, k)

        # Cas multi-communes : construire un detected par commune, collecter les vues de chacune
        extra_communes_summaries: List[Tuple[str, str, Dict]] = []  # (commune, text, meta)
        extra_communes_enquete:   List[Tuple[str, str, Dict]] = []
        multi_communes = detected.get("noms", [])
        if len(multi_communes) > 1:
            print(f"  [RAPTOR] Multi-communes détectées : {multi_communes}")
            for commune in multi_communes[1:]:  # le premier est géré par le flux normal
                det_c = {k: v for k, v in detected.items() if k not in ("nom", "noms")}
                det_c["nom"] = commune
                t, m = self._find_best_view(question, det_c)
                if t:
                    extra_communes_summaries.append((commune, t, m))
                et, em = self._find_best_enquete_view(det_c)
                if et:
                    extra_communes_enquete.append((commune, et, em))

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

        # Etape 4 : Synthèse enquête citoyenne structurée (scores par dimension)
        enquete_summary_text, enquete_meta = self._find_best_enquete_view(detected)

        # Etape 4bis : Classement dimensions si question comparative
        classement_text = None
        classement_meta: Dict = {}
        if self._is_dimension_ranking_question(question):
            classement_text, classement_meta = self._find_classement_doc(detected)
            if classement_text:
                print(f"  [RAPTOR] Classement dimensions trouvé (scope={classement_meta.get('scope', '?')})")
            else:
                print("  [RAPTOR] Question classement détectée mais aucun doc classement disponible")

        # Etape 5 : Assembler le contexte
        def _scope_label(view_name: str) -> str:
            has_commune = bool(detected.get("nom"))
            has_csp = bool(detected.get("profession"))
            if has_commune and has_csp:
                if view_name == "commune":
                    return f"vue {view_name} — {detected['nom']} (tous résidents, pas spécifique à la CSP)"
                elif view_name == "profession":
                    return f"vue {view_name} — Corse entière (pas spécifique à {detected['nom']})"
            return f"vue {view_name}"

        context_parts = []
        if summary_text:
            context_parts.append(
                f"[Synthèse RAPTOR enquête citoyenne (verbatims/subjectif/quali) - {_scope_label(summary_meta.get('view_name', 'N/A'))}]\n{summary_text}"
            )

        if enquete_summary_text:
            context_parts.append(
                f"[Synthèse RAPTOR scores enquête citoyenne (subjectif/quanti) - {_scope_label(enquete_meta.get('view_name', 'N/A'))}]\n{enquete_summary_text}"
            )

        if classement_text:
            context_parts.append(
                f"[Classement des dimensions enquête citoyenne — {classement_meta.get('scope', '?')}]\n{classement_text}"
            )

        # Etape 5b : Injecter les vues des communes supplémentaires (multi-communes)
        for commune, t, m in extra_communes_summaries:
            context_parts.append(
                f"[Synthèse RAPTOR enquête citoyenne — {commune} (vue {m.get('view_name', 'N/A')})]\n{t}"
            )
        for commune, et, em in extra_communes_enquete:
            context_parts.append(
                f"[Synthèse RAPTOR scores enquête — {commune} (vue {em.get('view_name', 'N/A')})]\n{et}"
            )

        # Etape 5c : Classements dimensions des communes supplémentaires (multi-communes)
        if len(multi_communes) > 1 and self._is_dimension_ranking_question(question):
            extra_classements = self._find_classement_docs_multi(multi_communes[1:])
            for commune, ct, cm in extra_classements:
                context_parts.append(
                    f"[Classement des dimensions enquête citoyenne — {commune}]\n{ct}"
                )

        # Etape 6 : Scores OppChoVec (objectif/quanti) — injecté EN PREMIER parmi les sources
        # complémentaires pour éviter la troncature du contexte (5000 chars dans answer_subquestion)
        opp_results = []
        if self._oppchovec:
            opp_results = self.query_oppchovec(
                question, k=2,
                communes=detected.get("noms") or None,
            )
            for r in opp_results:
                # Docs de classement global/composantes : plus longs pour permettre le filtrage EPCI
                _is_classement = r['meta'].get('type') in ('classement', 'aggregate')
                _text_limit = 8000 if _is_classement else 1500
                context_parts.append(
                    f"[Scores OppChoVec — {r['meta'].get('commune', 'Corse entière') if _is_classement else r['meta'].get('commune', 'N/A')} (indicateurs territoriaux objectifs)]\n"
                    f"{r['text'][:_text_limit]}"
                )

        for i, (doc, meta) in enumerate(zip(evidence_docs, evidence_metas), 1):
            context_parts.append(f"[Verbatim citoyen (subjectif/quali) {i}] {doc[:300]}...")

        # Etape 7 : Sources supplémentaires (entretiens, enquête, wiki, geo)
        extra_results = self.query_extra_sources(
            question, k=2,
            communes=detected.get("noms") or None,
            epci=detected.get("epci"),
        )

        # Si la question porte explicitement sur les entretiens, les injecter EN TÊTE
        # (avant les verbatims et synthèses génériques) pour éviter la troncature du contexte
        import unicodedata as _ud3
        _q3 = "".join(c for c in _ud3.normalize("NFD", question.lower()) if _ud3.category(c) != "Mn")
        _entretien_priority = any(kw in _q3 for kw in ("entretien", "qualitatif", "interview"))

        entretien_parts = []
        other_extra_parts = []
        for r in extra_results:
            text_to_inject = r['text'][:1500]
            if r.get("collection") == "raptor_entretiens_summaries":
                text_to_inject = (
                    "[NOTE : dans cette synthèse, N = nombre d'extraits de transcription "
                    "d'entretiens semi-directifs, PAS le nombre de répondants individuels. "
                    "Le nombre d'entretiens (personnes interviewées) est indiqué séparément.]\n\n"
                    + text_to_inject
                )
            part = f"[{r['label']}]\n{text_to_inject}"
            if _entretien_priority and r.get("collection") in ("portrait_entretiens", "raptor_entretiens_summaries"):
                entretien_parts.append(part)
            else:
                other_extra_parts.append(part)

        if _entretien_priority and entretien_parts:
            context_parts = entretien_parts + context_parts
        context_parts.extend(other_extra_parts)

        # Injection méthodologie enquête si synthèse enquête utilisée ou question sur l'enquête
        import unicodedata as _ud
        def _strip_acc(s: str) -> str:
            return "".join(c for c in _ud.normalize("NFD", s.lower()) if _ud.category(c) != "Mn")
        q_stripped = _strip_acc(question)

        _ENQUETE_META_KEYWORDS = [
            "enquete", "repondant", "questionnaire", "combien de personnes",
            "qui a repondu", "methodologie", "comment fonctionne", "comment a ete realise",
        ]
        _methodology_doc = None
        if (summary_text or enquete_summary_text or
                any(_strip_acc(kw) in q_stripped for kw in _ENQUETE_META_KEYWORDS)):
            methodology = self._get_enquete_methodology()
            if methodology:
                _methodology_doc = methodology
                context_parts.append(
                    f"[Description de l'enquête QdV — méthodologie et répondants]\n{methodology[:2000]}"
                )

        # Injection baseline Corse entière si question comparative avec commune détectée
        if detected.get("noms") and any(_strip_acc(kw) in q_stripped for kw in self._COMPARE_KEYWORDS):
            baseline = self._get_global_baseline()
            if baseline:
                context_parts.append(
                    f"[Référence Corse entière — baseline comparatif (N=246)]\n{baseline[:1500]}"
                )

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
        if enquete_summary_text:
            sources.append({
                "rank": len(sources),
                "type": "raptor_enquete_summary",
                "view": enquete_meta.get("view_name"),
                "num_chunks": enquete_meta.get("num_chunks"),
                "extrait": enquete_summary_text
            })
        if classement_text:
            sources.append({
                "rank": len(sources),
                "type": "classement_dimensions",
                "scope": classement_meta.get("scope"),
                "commune": classement_meta.get("commune"),
                "cat_age": classement_meta.get("cat_age"),
                "csp": classement_meta.get("csp"),
                "extrait": classement_text[:300],
            })
        for commune, t, m in extra_communes_summaries:
            sources.append({
                "rank": len(sources),
                "type": "raptor_summary",
                "commune": commune,
                "view": m.get("view_name"),
                "num_chunks": m.get("num_chunks"),
                "extrait": t[:300],
            })
        for commune, et, em in extra_communes_enquete:
            sources.append({
                "rank": len(sources),
                "type": "raptor_enquete_summary",
                "commune": commune,
                "view": em.get("view_name"),
                "num_chunks": em.get("num_chunks"),
                "extrait": et[:300],
            })
        for i, (doc, meta) in enumerate(zip(evidence_docs, evidence_metas), 1):
            sources.append({
                "rank": len(sources),
                "commune": meta.get("nom", "N/A"),
                "genre": meta.get("genre", "N/A"),
                "age": meta.get("age_exact", "N/A"),
                "profession": meta.get("profession", "N/A"),
                "dimension": meta.get("dimension", "N/A"),
                "source_type": "verbatim_evidence",
                "extrait": doc
            })
        for i, r in enumerate(extra_results):
            meta = r["meta"]
            # Chercher la commune dans plusieurs champs possibles (dim1_value pour les synthèses RAPTOR)
            commune_val = (meta.get("commune") or meta.get("nom")
                           or (meta.get("dim1_value") if meta.get("dim1_name") in ("commune", "nom") else None)
                           or None)
            sources.append({
                "rank": len(sources),
                "source_type": r["collection"],
                "data_type": r["data_type"],
                "commune": commune_val,
                "num_chunks": meta.get("num_chunks"),
                "view_name": meta.get("view_name"),
                "distance": round(r["distance"], 3),
                "extrait": r["text"][:1500],
            })

        for r in opp_results:
            sources.append({
                "rank": len(sources),
                "type": "oppchovec_score",
                "source_type": "oppchovec_scores",
                "commune": r["meta"].get("commune"),
                "oppchovec_0_10": r["meta"].get("oppchovec_0_10"),
                "opp_0_10":  r["meta"].get("opp_0_10"),
                "cho_0_10":  r["meta"].get("cho_0_10"),
                "vec_0_10":  r["meta"].get("vec_0_10"),
                "distance": round(r.get("distance", 0), 3),
                "extrait": r["text"][:1500],
            })

        if _methodology_doc:
            sources.append({
                "rank": len(sources),
                "type": "methodology",
                "source_type": "methodology",
                "view_name": "enquete_methodology",
                "extrait": _methodology_doc[:400],
            })

        return context_str, sources

    # Règles de filtrage par collection :
    # "commune_field"  : champ metadata à filtrer quand des communes sont détectées
    # "epci_field"     : champ metadata à filtrer quand un EPCI est détecté
    # "epci_k"         : nombre de résultats à retourner quand filtre EPCI actif
    _FILTERABLE_COLLECTIONS: Dict[str, Dict] = {
        "portrait_entretiens":          {"commune_field": "nom"},
        "raptor_entretiens_summaries":  {"commune_field": "dim1_value"},
        "enquete_responses":            {"commune_field": "commune"},
        "communes_wiki":                {"commune_field": "commune"},
        "communes_geo":                 {"commune_field": "commune", "epci_field": "epci", "epci_k": 5},
        "zones_epci":                   {"epci_field": "epci", "epci_k": 1},
        "communes_profil":              {"commune_field": "commune"},
        "communes_equipements":         {"commune_field": "commune"},
    }

    def query_extra_sources(self, question: str, k: int = 2,
                            communes: Optional[List[str]] = None,
                            epci: Optional[str] = None) -> List[Dict]:
        """
        Recherche dans les collections supplémentaires (entretiens, enquête, wiki, geo).

        Filtres appliqués quand des entités géographiques sont détectées :
        - communes : filtre strict sur portrait_entretiens (nom) et enquete_responses (commune).
          Pour communes_geo, filtre par commune si 1 seule commune et pas d'EPCI.
        - epci : filtre strict sur zones_epci (epci) et communes_geo (epci).
          Prioritaire sur le filtre commune pour communes_geo.
        - Autres collections (wiki, equipements, zones_epci sans EPCI détecté) :
          recherche sémantique libre — leurs docs peuvent couvrir plusieurs communes.
        """
        if not self._extra_cols:
            return []
        q_emb = self._encode_query(question)
        results = []
        # Expansion factor pour le retrieval hybride (on récupère plus de candidats, on rerank)
        HYBRID_EXPANSION = 3

        # Boost seuil pour entretiens si la question les mentionne explicitement
        import unicodedata as _ud2
        _q_norm = "".join(
            c for c in _ud2.normalize("NFD", question.lower()) if _ud2.category(c) != "Mn"
        )
        _entretien_boost = any(kw in _q_norm for kw in ("entretien", "qualitatif", "interview"))

        # Boost équipements : question portant sur des indicateurs objectifs territoriaux
        _equipements_boost = any(kw in _q_norm for kw in (
            "medecin", "hopital", "sante", "ecole", "service", "commerce",
            "emploi", "chomage", "actif", "activite", "salarie", "taux",
            "infrastructure", "equipement", "transport", "logement",
            "nombre de", "combien", "indicateur", "objectif", "territoire",
            "boulangerie", "supermarche", "poste", "pharmacie",
        ))

        for cfg in self._EXTRA_COLLECTIONS:
            col_name = cfg["name"]
            col = self._extra_cols.get(col_name)
            if col is None:
                continue

            filter_rules = self._FILTERABLE_COLLECTIONS.get(col_name, {})
            n_final = k
            where_filter = None
            post_filter_field = None   # champ pour post-filtrage Python (multi-communes)
            post_filter_values = None

            # ── Construire le filtre WHERE ────────────────────────────────────
            epci_field = filter_rules.get("epci_field")
            commune_field = filter_rules.get("commune_field")

            if epci_field and epci:
                # Filtre EPCI strict (zones_epci, communes_geo avec EPCI détecté)
                where_filter = {epci_field: {"$eq": epci}}
                n_final = filter_rules.get("epci_k", k)

            elif commune_field and communes:
                if len(communes) == 1:
                    where_filter = {commune_field: {"$eq": communes[0]}}
                else:
                    # Multi-communes : agrandir n + post-filtrage Python
                    n_final = k * len(communes)
                    post_filter_field = commune_field
                    post_filter_values = set(communes)
                    # pas de where_filter (ChromaDB $in non garanti)

            # Expansion pour hybrid rerank (sauf si filtre strict sur petite collection)
            n_expanded = min(n_final * HYBRID_EXPANSION, col.count())
            if n_expanded == 0:
                continue

            try:
                query_kwargs: Dict = dict(
                    query_embeddings=[q_emb],
                    n_results=n_expanded,
                    include=["documents", "metadatas", "distances"],
                )
                if where_filter:
                    query_kwargs["where"] = where_filter

                res = col.query(**query_kwargs)
                candidates = []
                for doc, meta, dist in zip(
                    res["documents"][0], res["metadatas"][0], res["distances"][0]
                ):
                    # Post-filtrage Python pour multi-communes
                    if post_filter_field and post_filter_values:
                        if meta.get(post_filter_field, "") not in post_filter_values:
                            continue
                    candidates.append({
                        "text": doc,
                        "meta": meta,
                        "distance": dist,
                        "collection": col_name,
                        "data_type": cfg["data_type"],
                        "wellbeing_type": cfg["wellbeing_type"],
                        "label": cfg["label"],
                    })

                # Hybrid rerank sur les candidats expandés, puis filtre seuil dense
                reranked = self._hybrid_rerank(question, candidates, top_k=n_final)
                _entretien_col = col_name in ("portrait_entretiens", "raptor_entretiens_summaries")
                _equipements_col = col_name == "communes_equipements"
                # Injection forcée si commune détectée dans la question — docs déjà filtrés par WHERE
                _entretien_commune_forced = _entretien_col and bool(communes)
                if _entretien_commune_forced:
                    effective_threshold = 1.5
                elif _entretien_col and _entretien_boost:
                    effective_threshold = 0.75
                elif _equipements_col and _equipements_boost:
                    effective_threshold = 1.5
                else:
                    effective_threshold = cfg["threshold"]
                for c in reranked:
                    if c["distance"] <= effective_threshold:
                        results.append(c)

            except Exception:
                pass

        return results

    # Mots-clés indiquant une question de classement global (nécessite la vision d'ensemble)
    _RANKING_KEYWORDS = [
        "meilleur", "mieux", "meilleure", "mieux classé",
        "premier", "première", "premières", "premiers",
        "dernier", "dernière", "dernières", "derniers",
        "pire", "moins bon", "moins bonne",
        "classement", "classée", "classées", "classés",
        "top ", "rang ", "rang?", "quel rang",
        "mieux placé", "moins bien placé",
        "quelle commune", "quelles communes",
        "plus haut score", "plus bas score", "score le plus",
        "toutes les communes", "ensemble des communes",
        "comparer", "comparaison",
    ]

    def _is_ranking_question(self, question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in self._RANKING_KEYWORDS)

    # Mots-clés indiquant une question de classement des DIMENSIONS (enquête citoyenne)
    _DIMENSION_RANKING_KEYWORDS = [
        "classement", "dimension la plus", "dimension la moins",
        "la plus faible", "la plus elevee", "la plus eleve",
        "le plus faible", "le plus eleve",
        "la note la plus", "la note la moins",
        "moins bien note", "mieux note",
        "meilleure dimension", "pire dimension",
        "quelle dimension", "quel score le plus",
        "toutes les dimensions", "comparaison des dimensions",
        "note la plus basse", "note la plus haute",
        "dimension obtient", "dimension recoit",
    ]

    def _is_dimension_ranking_question(self, question: str) -> bool:
        import unicodedata
        def _strip(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                           if unicodedata.category(c) != "Mn")
        q = _strip(question)
        return any(_strip(kw) in q for kw in self._DIMENSION_RANKING_KEYWORDS)

    def _find_classement_doc(self, detected: Dict) -> Tuple[Optional[str], Dict]:
        """
        Cherche le(s) document(s) de classement des dimensions les plus spécifiques
        selon le contexte détecté (commune > cat_age > csp > global).

        Pour les questions multi-communes (detected["noms"] contient plusieurs valeurs),
        retourne le classement de la première commune ; les suivantes sont gérées par
        _find_classement_docs_multi().

        Les classements sont stockés dans enquete_scores_commune avec
        source_type='classement'.
        """
        col = self._extra_cols.get("enquete_scores_commune")
        if col is None:
            return None, {}

        if detected.get("nom"):
            where = {"$and": [
                {"source_type": {"$eq": "classement"}},
                {"commune": {"$eq": detected["nom"]}},
            ]}
        elif detected.get("age_range"):
            where = {"$and": [
                {"source_type": {"$eq": "classement"}},
                {"cat_age": {"$eq": detected["age_range"]}},
            ]}
        elif detected.get("profession"):
            where = {"$and": [
                {"source_type": {"$eq": "classement"}},
                {"csp": {"$eq": detected["profession"]}},
            ]}
        else:
            where = {"$and": [
                {"source_type": {"$eq": "classement"}},
                {"scope": {"$eq": "global"}},
            ]}

        try:
            res = col.get(where=where, include=["documents", "metadatas"])
            if res["documents"]:
                return res["documents"][0], res["metadatas"][0]
        except Exception:
            pass
        return None, {}

    def _find_classement_docs_multi(self, communes: List[str]) -> List[Tuple[str, str, Dict]]:
        """
        Retourne les classements de dimensions pour chaque commune de la liste.
        Utilisé pour les questions comparatives multi-communes.
        Retourne [(commune, text, meta), ...] — uniquement les communes qui ont un doc.
        """
        col = self._extra_cols.get("enquete_scores_commune")
        if col is None:
            return []
        out = []
        for commune in communes:
            try:
                res = col.get(
                    where={"$and": [
                        {"source_type": {"$eq": "classement"}},
                        {"commune": {"$eq": commune}},
                    ]},
                    include=["documents", "metadatas"]
                )
                if res["documents"]:
                    out.append((commune, res["documents"][0], res["metadatas"][0]))
            except Exception:
                pass
        return out

    def query_oppchovec(self, question: str, k: int = 3,
                        communes: Optional[List[str]] = None) -> List[Dict]:
        """
        Cherche les scores OppChoVec les plus pertinents pour la question.
        - Pour les questions de classement global : retourne le doc de synthèse classement
          (top/bottom 20, stats descriptives, classement complet) à la place du doc méthodo.
        - Sinon : retourne le doc méthodologique + k-1 communes pertinentes.
        - communes : liste de communes détectées dans la question ; si fournie, la recherche
          sémantique est restreinte à ces communes (filtre metadata exact).
        Retourne [] si la collection n'est pas disponible.
        """
        if self._oppchovec is None:
            return []

        # Détecter les communes si non fournies en paramètre
        if communes is None:
            try:
                from commune_detector import detect_communes as _dc
                communes = _dc(question) or None
            except ImportError:
                communes = None

        results = []
        is_ranking = self._is_ranking_question(question)

        # 1a-bis. Question sur la moyenne/score global Corse (pas de commune) → doc agrégat Corse
        import unicodedata as _ud_opp
        _q_opp = "".join(c for c in _ud_opp.normalize("NFD", question.lower()) if _ud_opp.category(c) != "Mn")
        _is_global_average = not communes and any(kw in _q_opp for kw in (
            "moyen", "moyenne", "general", "global", "ensemble", "niveau",
            "corse entiere", "ile entiere", "l ensemble", "toutes les communes",
            "score global", "score corse", "indicateur corse",
        ))
        if _is_global_average:
            try:
                agg_doc = self._oppchovec.get(
                    ids=["oppchovec_aggregate_corse"],
                    include=["documents", "metadatas"]
                )
                if agg_doc["documents"]:
                    results.append({
                        "text": agg_doc["documents"][0],
                        "meta": agg_doc["metadatas"][0],
                        "distance": 0.0,
                    })
            except Exception:
                pass

        # 1a. Question de classement → doc synthèse classement global (ID fixe)
        if is_ranking:
            try:
                ranking_doc = self._oppchovec.get(
                    ids=["oppchovec_classement_global"],
                    include=["documents", "metadatas"]
                )
                if ranking_doc["documents"]:
                    results.append({
                        "text": ranking_doc["documents"][0],
                        "meta": ranking_doc["metadatas"][0],
                        "distance": 0.0
                    })
            except Exception:
                pass

        # 1b. Question sur les composantes/sous-scores → doc classement composantes
        if self._is_dimension_ranking_question(question) or any(
            kw in question.lower() for kw in ["composante", "opp ", "cho ", "vec ", "sous-score", "sous score"]
        ):
            try:
                comp_doc = self._oppchovec.get(
                    ids=["oppchovec_classement_composantes"],
                    include=["documents", "metadatas"]
                )
                if comp_doc["documents"]:
                    results.insert(0, {
                        "text": comp_doc["documents"][0],
                        "meta": comp_doc["metadatas"][0],
                        "distance": 0.0,
                    })
            except Exception:
                pass

        # 1c. Toujours inclure le doc méthodologique (par ID fixe)
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
        #    Si des communes sont explicitement citées, on filtre par commune pour garantir
        #    que les docs retournés correspondent aux communes demandées.
        k_communes = max(1, k - 1) if not communes else max(len(communes), k - 1)
        q_emb = self._encode_query(question)

        # Construire le filtre WHERE : source_type restreint + commune si connue
        source_filter = {"source": {"$in": ["oppchovec_betti_0_10", "oppchovec_aggregate"]}}
        if communes:
            if len(communes) == 1:
                where_sem = {"$and": [source_filter, {"commune": {"$eq": communes[0]}}]}
            else:
                # Pour N communes, on lance une requête par commune pour garantir
                # au moins 1 résultat par commune (pas de $in sur commune dans cette version)
                for com in communes:
                    n_c = min(1, self._oppchovec.count())
                    if n_c == 0:
                        continue
                    try:
                        res_c = self._oppchovec.query(
                            query_embeddings=[q_emb],
                            n_results=n_c,
                            include=["documents", "metadatas", "distances"],
                            where={"$and": [source_filter, {"commune": {"$eq": com}}]}
                        )
                        for doc, meta, dist in zip(
                            res_c["documents"][0], res_c["metadatas"][0], res_c["distances"][0]
                        ):
                            results.append({"text": doc, "meta": meta, "distance": dist})
                    except Exception:
                        pass
                return results
        else:
            where_sem = source_filter

        # Expansion + hybrid rerank pour la recherche sémantique générale
        n_expanded = min(k_communes * 3, self._oppchovec.count())
        if n_expanded > 0:
            try:
                res = self._oppchovec.query(
                    query_embeddings=[q_emb],
                    n_results=n_expanded,
                    include=["documents", "metadatas", "distances"],
                    where=where_sem
                )
                candidates = [
                    {"text": doc, "meta": meta, "distance": dist}
                    for doc, meta, dist in zip(
                        res["documents"][0], res["metadatas"][0], res["distances"][0]
                    )
                ]
                for c in self._hybrid_rerank(question, candidates, top_k=k_communes):
                    results.append(c)
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
        "Tourisme": ["tourisme", "touristes", "saison touristique", "masse touristique", "flux touristique"],
    }

    @classmethod
    def _detect_dimension_theme(cls, question: str) -> Optional[str]:
        """Detecte la dimension QdV (theme) mentionnee dans la question."""
        import unicodedata
        def _strip_accents(s: str) -> str:
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            )
        q = _strip_accents(question.lower())
        for dim, keywords in cls._DIMENSION_KEYWORDS.items():
            if any(_strip_accents(kw) in q for kw in keywords):
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
            from commune_detector import detect_communes
            communes = detect_communes(question)
            if communes:
                detected["nom"] = communes[0]   # rétrocompatibilité (vue unique)
                detected["noms"] = communes      # liste complète pour comparaisons
        except ImportError:
            pass

        try:
            from epci_detector import detect_epci
            epci = detect_epci(question)
            if epci:
                detected["epci"] = epci
        except ImportError:
            pass

        theme = self._detect_dimension_theme(question)
        if theme:
            detected["dimension"] = theme

        return detected

    # Vues enquête ordonnées du plus spécifique au moins spécifique.
    # Les noms de vues et les dims utilisent les noms publics (age_range, profession)
    # alignés sur raptor_summaries (portrait verbatims).
    _ENQUETE_VIEW_DEFINITIONS = [
        # 2D avec dimension QdV (priorité haute)
        {"name": "enquete_dimension*commune",    "dims": ["dimension", "commune"]},
        {"name": "enquete_dimension*age_range",  "dims": ["dimension", "age_range"]},
        {"name": "enquete_dimension*profession", "dims": ["dimension", "profession"]},
        # 2D démographiques
        {"name": "enquete_age_range*commune",    "dims": ["age_range", "commune"]},
        {"name": "enquete_profession*commune",   "dims": ["profession", "commune"]},
        {"name": "enquete_age_range*profession", "dims": ["age_range", "profession"]},
        # 1D
        {"name": "enquete_commune",              "dims": ["commune"]},
        {"name": "enquete_age_range",            "dims": ["age_range"]},
        {"name": "enquete_profession",           "dims": ["profession"]},
        {"name": "enquete_dimension",            "dims": ["dimension"]},
    ]

    # Corrections des noms de dimensions : raptor_summaries stocke des noms légèrement différents
    _RAPTOR_SUMMARIES_DIM_FIX: Dict[str, str] = {
        "Tourisme":              "Tourisme (ressenti localement)",
        "Ratio vie pro/vie perso": "Ratio vie pro/ vie perso",
    }
    # Corrections pour raptor_enquete_summaries (accents différents)
    _ENQUETE_SUMMARIES_DIM_FIX: Dict[str, str] = {
        "Education": "Éducation",
    }

    def _find_best_enquete_view(self, detected: Dict[str, str],
                                no_fallback: bool = False) -> Tuple[Optional[str], Dict]:
        """
        Cherche la synthèse enquête citoyenne la plus spécifique via filtre metadata.
        Mappe les dimensions détectées (age_range, nom, profession) vers les champs enquête
        (cat_age, commune, csp) et cherche dans _ENQUETE_VIEW_DEFINITIONS par ordre de priorité.
        no_fallback=True : désactive le fallback sémantique.
        Retourne (summary_text, metadata) ou (None, {}).
        """
        if self._enquete_summaries is None:
            return None, {}

        # Construire le dict de recherche dans les métadonnées enquête.
        # age_range utilise désormais les mêmes buckets que raptor_summaries (18-24…65+).
        enq = {}
        if detected.get("age_range"):
            enq["age_range"] = detected["age_range"]
        if detected.get("nom"):
            enq["commune"] = detected["nom"]
        if detected.get("profession"):
            enq["profession"] = detected["profession"]
        if detected.get("dimension"):
            dim_val = detected["dimension"]
            # Corriger les noms de dimensions pour raptor_enquete_summaries
            enq["dimension"] = self._ENQUETE_SUMMARIES_DIM_FIX.get(dim_val, dim_val)

        for view_def in self._ENQUETE_VIEW_DEFINITIONS:
            dims = view_def["dims"]
            if not all(enq.get(d) for d in dims):
                continue
            conditions = [{"view_name": view_def["name"]}]
            for i, dim in enumerate(dims):
                conditions.append({f"dim{i+1}_value": enq[dim]})
            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            try:
                res = self._enquete_summaries.get(
                    where=where, include=["documents", "metadatas"]
                )
                if res["documents"]:
                    return res["documents"][0], res["metadatas"][0]
            except Exception:
                continue

        if no_fallback:
            return None, {}

        # Fallback commune : si on connaît la commune mais pas la dimension, chercher
        # n'importe quelle synthèse enquête pour cette commune (dimension*commune)
        if enq.get("commune") and not enq.get("dimension"):
            try:
                res = self._enquete_summaries.get(
                    where={"$and": [
                        {"view_name": {"$eq": "enquete_dimension*commune"}},
                        {"dim2_value": {"$eq": enq["commune"]}},
                    ]},
                    include=["documents", "metadatas"]
                )
                if res["documents"]:
                    return res["documents"][0], res["metadatas"][0]
            except Exception:
                pass

        # Fallback sémantique sur enquete_summaries (scoped à la commune si connue)
        if self._embed_model and self._enquete_summaries.count() > 0:
            try:
                q_emb = self._encode_query(' '.join(str(v) for v in enq.values() if v))
                kwargs: dict = {
                    "query_embeddings": [q_emb], "n_results": 1,
                    "include": ["documents", "metadatas", "distances"],
                }
                if enq.get("commune"):
                    kwargs["where"] = {"dim2_value": {"$eq": enq["commune"]}}
                res = self._enquete_summaries.query(**kwargs)
                if res["documents"] and res["documents"][0]:
                    if res["distances"][0][0] <= 0.60:
                        return res["documents"][0][0], res["metadatas"][0][0]
            except Exception:
                pass

        # Dernier recours : vue globale Corse entière si elle existe
        try:
            res = self._enquete_summaries.get(
                where={"view_name": {"$eq": "enquete_global"}},
                include=["documents", "metadatas"]
            )
            if res["documents"]:
                return res["documents"][0], res["metadatas"][0]
        except Exception:
            pass

        return None, {}

    def _find_best_view(self, question: str,
                        detected: Dict[str, str],
                        no_fallback: bool = False) -> Tuple[Optional[str], Dict]:
        """
        Trouve la vue la plus specifique avec fallback sémantique optionnel.
        no_fallback=True : retourne (None, {}) si aucune vue structurée trouvée,
        sans déclencher la recherche sémantique (évite la contamination EPCI).
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
                val = detected[dim]
                # Corriger les noms de dimensions pour raptor_summaries
                if dim == "dimension":
                    val = self._RAPTOR_SUMMARIES_DIM_FIX.get(val, val)
                conditions.append({dim_key: val})

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

        if no_fallback:
            return None, {}

        # Fallback : recherche semantique sur toutes les syntheses
        # Seuil de distance : ne pas injecter un RAPTOR non pertinent
        FALLBACK_DISTANCE_THRESHOLD = 0.55
        if self._summaries.count() > 0:
            query_embedding = self._encode_query(question)
            try:
                results = self._summaries.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as _hnsw_err:
                # Index HNSW binaire absent ou incompatible (ex: migration ChromaDB 0.x→1.x)
                # On retourne None pour continuer sans contexte RAPTOR plutôt que 500
                print(f"  [RAPTOR] Fallback sémantique indisponible (HNSW): {_hnsw_err}")
                return None, {}
            if results["documents"] and results["documents"][0]:
                distance = results["distances"][0][0]
                if distance <= FALLBACK_DISTANCE_THRESHOLD:
                    _fallback_meta = results["metadatas"][0][0]
                    _fallback_view = _fallback_meta.get("view_name", "")
                    # Bloquer les vues commune-spécifiques non pertinentes
                    if not detected and "commune" in _fallback_view:
                        print(f"  [RAPTOR] Fallback ignoré (question globale, vue '{_fallback_view}')")
                    elif detected.get("nom") and "commune" in _fallback_view:
                        _fb_comm = _fallback_meta.get("dim1_value") or _fallback_meta.get("dim2_value") or ""
                        if _fb_comm and _fb_comm != detected["nom"]:
                            print(f"  [RAPTOR] Fallback ignoré (vue '{_fallback_view}' pour '{_fb_comm}' ≠ '{detected['nom']}')")
                        else:
                            return results["documents"][0][0], _fallback_meta
                    else:
                        return results["documents"][0][0], _fallback_meta
                else:
                    print(f"  [RAPTOR] Fallback ignore (distance={distance:.3f} > seuil {FALLBACK_DISTANCE_THRESHOLD})")

        return None, {}

    # Nombre max de communes à enrichir par requête EPCI (évite l'explosion de contexte)
    _MAX_COMMUNES_EPCI_ENRICHMENT: int = 5

    # Mots-clés indiquant que la question porte sur la qualité de vie / perceptions
    _QDV_KEYWORDS = [
        "qualite de vie", "bien-etre", "satisfaction", "perception",
        "habitants", "vivent", "ressentent", "ressenti",
        "score", "indice", "note", "enquete", "opinion", "avis",
        "vecu", "conditions de vie", "problemes", "difficultes",
        "transports", "sante", "logement", "emploi", "securite",
        "education", "culture", "revenus", "services", "tourisme",
        "communaute", "relations", "institutions", "reseaux",
    ]

    def _is_qdv_question(self, question: str) -> bool:
        """Détecte si la question porte sur la QdV / perceptions (pas juste la géographie)."""
        import unicodedata
        def _strip(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                           if unicodedata.category(c) != "Mn")
        q = _strip(question)
        return any(_strip(kw) in q for kw in self._QDV_KEYWORDS)

    def _get_epci_member_communes(self, epci_name: str) -> List[str]:
        """
        Retourne la liste des communes membres d'un EPCI depuis communes_geo.
        Utilise un filtre metadata exact (pas d'embedding).
        """
        col = self._extra_cols.get("communes_geo")
        if col is None:
            return []
        try:
            res = col.get(
                where={"epci": {"$eq": epci_name}},
                include=["metadatas"]
            )
            return sorted(m["commune"] for m in res["metadatas"])
        except Exception:
            return []

    _COMPARE_KEYWORDS = [
        "compar", "moyenne", "reste", "île", "corse entiere", "corse entière",
        "par rapport", "reference", "référence", "representatif", "représentatif", "typique",
    ]

    def _get_enquete_methodology(self) -> Optional[str]:
        """Retourne le document de description méthodologique de l'enquête QdV."""
        try:
            res = self._enquete_summaries.get(
                ids=["enquete_methodology"], include=["documents"]
            )
            return res["documents"][0] if res["documents"] else None
        except Exception:
            return None

    def _get_global_baseline(self) -> Optional[str]:
        """Retourne la synthèse globale Corse entière si disponible."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_path)
            col = client.get_collection(self.summary_collection_name)
            res = col.get(where={"view_name": {"$eq": "enquete_global"}}, include=["documents"])
            return res["documents"][0] if res["documents"] else None
        except Exception:
            return None

    @staticmethod
    def _map_age_to_range(age_min: int, age_max: int) -> str:
        """Mappe une tranche d'age numerique vers le bucket standard."""
        midpoint = (age_min + age_max) / 2
        if midpoint < 25:
            return "18-24"
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
