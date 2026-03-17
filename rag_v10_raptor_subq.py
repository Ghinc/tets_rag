"""
RAG v10 - RAPTOR-lite + Sous-questions.

Pipeline en 3 etapes :
  1. Mistral Large decompose la question initiale en N sous-questions
  2. RaptorRetriever + Claude Haiku repond a chaque sous-question
  3. Mistral Large synthetise toutes les reponses et filtre le bruit

Usage:
    python rag_v10_raptor_subq.py --query "..."
    python rag_v10_raptor_subq.py --query "..." --n-subquestions 5
"""

import os
import json
import time
import argparse
from typing import List, Tuple, Dict

from dotenv import load_dotenv
load_dotenv()

from rag_v9_raptor import (
    RaptorRetriever,
    CHROMA_PATH,
    SOURCE_COLLECTION,
    TARGET_COLLECTION,
)

# ============================================================
# Constantes
# ============================================================

DECOMPOSER_MODEL  = "mistral-large-latest"
ANSWERER_MODEL    = "claude-haiku-4-5-20251001"
SYNTHESIZER_MODEL = "mistral-large-latest"
MISTRAL_BASE_URL  = "https://api.mistral.ai/v1"
DEFAULT_N_SUBQUESTIONS = 5


# ============================================================
# Appels LLM
# ============================================================

def _call_mistral(prompt: str, system_prompt: str,
                  model: str = DECOMPOSER_MODEL,
                  max_tokens: int = 1000,
                  temperature: float = 0.3,
                  max_retries: int = 5) -> str:
    """Appel Mistral via API OpenAI-compatible avec retry exponentiel sur 429."""
    from openai import OpenAI
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY non definie")
    client = OpenAI(api_key=api_key, base_url=MISTRAL_BASE_URL)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt * 2
                print(f"    [RATE LIMIT] Mistral : attente {wait}s (tentative {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def _call_claude(prompt: str, system_prompt: str,
                 model: str = ANSWERER_MODEL,
                 max_tokens: int = 800,
                 temperature: float = 0.2) -> str:
    """Appel Claude via SDK Anthropic."""
    import anthropic
    # Support des deux noms de variable (ANTHROPIC_API_KEY ou CLAUDE_API_KEY)
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY (ou CLAUDE_API_KEY) non definie")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ============================================================
# Etape 1 : Decomposition en sous-questions (Mistral Large)
# ============================================================

_SYSTEM_DECOMPOSER = (
    "Tu es un expert en analyse de donnees qualitatives sur la qualite de vie. "
    "Ton role est de decomposer une question complexe en sous-questions ciblees, "
    "chacune portant sur un aspect distinct de la question initiale. "
    "Reponds UNIQUEMENT avec un JSON valide contenant une liste de sous-questions."
)


def decompose_question(question: str, n: int = DEFAULT_N_SUBQUESTIONS) -> List[str]:
    """
    Utilise Mistral Large pour decomposer la question en N sous-questions.
    Chaque sous-question porte sur un aspect distinct (groupe demographique,
    dimension thematique, comparaison geographique, etc.).
    """
    prompt = (
        f"Decompose cette question en exactement {n} sous-questions ciblees et complementaires.\n"
        f"Chaque sous-question doit porter sur un aspect distinct "
        f"(ex: un groupe demographique, une dimension thematique, une comparaison geographique).\n\n"
        f"Question initiale : {question}\n\n"
        f"Reponds UNIQUEMENT avec ce JSON (sans texte avant ou apres) :\n"
        f'{{\n  "sub_questions": [\n    "sous-question 1",\n    "sous-question 2"\n  ]\n}}'
    )

    raw = _call_mistral(prompt, _SYSTEM_DECOMPOSER, max_tokens=600, temperature=0.4)

    # Parser le JSON (nettoyer si entoure de markdown)
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(cleaned)
        sub_questions = data.get("sub_questions", [])
        if not sub_questions:
            raise ValueError("Liste sub_questions vide")
        return sub_questions[:n]
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Fallback : extraire les lignes qui ressemblent a des questions
        print(f"  [WARN] Parsing JSON decomposition echoue ({e}), extraction manuelle...")
        lines = [
            line.strip().lstrip("-•*0123456789. ")
            for line in raw.split("\n")
            if "?" in line and len(line.strip()) > 10
        ]
        if lines:
            return lines[:n]
        raise RuntimeError(f"Impossible de decomposer la question. Reponse brute : {raw[:300]}")


# ============================================================
# Etape 2 : Reponse aux sous-questions (Claude Haiku + RAPTOR)
# ============================================================

_SYSTEM_ANSWERER = (
    "Tu es un analyste specialise dans les verbatims citoyens sur la qualite de vie en Corse. "
    "Reponds a la sous-question posee en te basant UNIQUEMENT sur le contexte fourni. "
    "Sois concis (3-5 phrases max), factuel et cite des extraits courts entre guillemets si pertinent. "
    "Si le contexte ne contient pas d'information pertinente, dis-le clairement en une phrase."
)


def answer_subquestion(sub_question: str, context: str) -> str:
    """
    Utilise Claude Haiku pour repondre a une sous-question a partir du contexte RAPTOR.
    Le contexte est tronque a 3000 caracteres pour maitriser les tokens.
    """
    prompt = f"Contexte (syntheses et verbatims) :\n{context[:3000]}\n\nSous-question : {sub_question}"
    return _call_claude(prompt, _SYSTEM_ANSWERER)


# ============================================================
# Etape 3 : Synthese finale (Mistral Large)
# ============================================================

_SYSTEM_SYNTHESIZER = (
    "Tu es un expert en analyse de donnees qualitatives. "
    "Tu recois une question initiale et des reponses a des sous-questions derivees. "
    "Ton role est de produire une synthese finale coherente, en eliminant les redondances et le bruit, "
    "et en repondant directement a la question initiale. "
    "Sois analytique, nuance et factuel. Structure ta reponse avec des points cles si pertinent."
)


def synthesize_answers(initial_question: str,
                        sub_qa_pairs: List[Tuple[str, str]]) -> str:
    """
    Utilise Mistral Large pour synthetiser les reponses aux sous-questions.
    Elimine le bruit et repond directement a la question initiale.
    """
    sub_answers_text = "\n\n".join(
        f"**Sous-question {i+1}** : {sq}\n**Reponse** : {ans}"
        for i, (sq, ans) in enumerate(sub_qa_pairs)
    )

    prompt = (
        f"Question initiale : {initial_question}\n\n"
        f"Voici les reponses aux sous-questions derivees :\n\n"
        f"{sub_answers_text}\n\n"
        f"Produis une synthese finale qui :\n"
        f"1. Repond directement a la question initiale\n"
        f"2. Integre les informations pertinentes des sous-reponses\n"
        f"3. Elimine les redondances et les elements non pertinents (bruit)\n"
        f"4. Signale si certaines sous-reponses manquaient d'information utile"
    )

    return _call_mistral(prompt, _SYSTEM_SYNTHESIZER, max_tokens=1500, temperature=0.3)


# ============================================================
# Etape 4 : Notation de la dimension evaluee (Mistral Large, optionnelle)
# ============================================================

_SYSTEM_SCORER = (
    "Tu es un expert en evaluation de la qualite de vie. "
    "On te fournit une question et une synthese analytique issues de verbatims citoyens. "
    "Ta tache : identifier la ou les dimensions evaluables dans la synthese "
    "(ex : environnement, transports, logement, services publics, securite, culture, lien social, sante, "
    "bien-etre general, etc.) pour un territoire ou un groupe donne, "
    "et attribuer une note si les donnees sont suffisantes. "
    "Reponds UNIQUEMENT avec un JSON valide, sans texte avant ou apres."
)


def score_dimension(question: str, synthesis: str) -> Dict:
    """
    Utilise Mistral Large pour identifier la dimension principale evaluable
    dans la synthese et lui attribuer une note de 1 a 5 avec justification.

    La notation est applicable des que la synthese contient assez de donnees
    sur un territoire, un groupe, ou une dimension — meme si la question est
    generale. La dimension notee peut etre le bien-etre global d'un groupe
    ou la qualite de vie sur un territoire.

    Retourne un dict :
      {
        "applicable": bool,        # True sauf si les donnees sont vraiment insuffisantes
        "dimension": str,          # ex: "bien-etre des seniors (65+)"
        "score": int | None,       # 1-5, None si vraiment non applicable
        "justification": str       # "Si je devais noter X, je mettrais Y/5 car..."
      }
    """
    prompt = (
        f"Question initiale : {question}\n\n"
        f"Synthese analytique :\n{synthesis[:3000]}\n\n"
        f"En te basant sur les informations presentes dans la synthese, identifie la dimension "
        f"principale ou le sujet central qui peut etre note (ex : bien-etre des seniors, "
        f"qualite de vie a Ajaccio, qualite de l'environnement local, acces aux services, etc.).\n\n"
        f"Attribue une note de 1 a 5 si les donnees sont suffisantes pour former un jugement, "
        f"meme partiel. N'attribue PAS de note UNIQUEMENT si les donnees sont totalement absentes "
        f"ou hors-sujet (ex : corpus ne couvrant pas du tout le sujet demande).\n\n"
        f"La justification doit etre formulee ainsi : "
        f"'Si je devais noter [dimension], je mettrais [score]/5 car [raison concise en 2-3 phrases].'\n\n"
        f"Reponds avec ce JSON :\n"
        f'{{\n'
        f'  "applicable": true,\n'
        f'  "dimension": "dimension principale ou sujet central note",\n'
        f'  "score": 3,\n'
        f'  "justification": "Si je devais noter X, je mettrais 3/5 car..."\n'
        f'}}\n'
        f'ou si les donnees sont vraiment insuffisantes (corpus hors-sujet) :\n'
        f'{{\n'
        f'  "applicable": false,\n'
        f'  "dimension": null,\n'
        f'  "score": null,\n'
        f'  "justification": "raison courte (1 phrase)"\n'
        f'}}'
    )

    raw = _call_mistral(prompt, _SYSTEM_SCORER, max_tokens=400, temperature=0.2)

    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(cleaned)
        return {
            "applicable": bool(data.get("applicable", False)),
            "dimension": data.get("dimension"),
            "score": data.get("score"),
            "justification": data.get("justification", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "applicable": False,
            "dimension": None,
            "score": None,
            "justification": f"Parsing JSON echoue. Reponse brute : {raw[:200]}",
        }


# ============================================================
# Pipeline principal
# ============================================================

class RaptorSubQuestionPipeline:
    """
    Pipeline RAG v10 : RAPTOR-lite + Sous-questions.

    Etape 1 : Mistral Large decompose la question en N sous-questions.
    Etape 2 : Pour chaque sous-question, RaptorRetriever recupere le contexte
              et Claude Haiku produit une reponse ciblee.
    Etape 3 : Mistral Large synthetise toutes les reponses et filtre le bruit
              pour produire la reponse finale.
    Etape 4 : Mistral Large evalue si une dimension est evaluable et attribue
              une note 1-5 avec justification (optionnel, selon pertinence).
    """

    def __init__(self,
                 chroma_path: str = CHROMA_PATH,
                 source_collection: str = SOURCE_COLLECTION,
                 summary_collection: str = TARGET_COLLECTION,
                 n_evidence_chunks: int = 5):
        self.retriever = RaptorRetriever(
            chroma_path=chroma_path,
            source_collection=source_collection,
            summary_collection=summary_collection,
            n_evidence_chunks=n_evidence_chunks,
        )
        self._initialized = False

    def init(self):
        """Charge le RaptorRetriever (modele d'embeddings + connexion ChromaDB)."""
        self.retriever.init()
        self._initialized = True
        print(f"RAG v10 initialise ({self.retriever.summary_count} syntheses RAPTOR disponibles)")

    def query(self, question: str, k: int = 5,
              n_subquestions: int = DEFAULT_N_SUBQUESTIONS) -> Tuple[str, List[Dict], Dict]:
        """
        Pipeline complet v10.

        Returns:
            (final_answer, all_sources, scoring)
            all_sources : liste de dicts avec les champs habituels RAPTOR
                          + sub_question_idx et sub_question pour tracabilite.
            scoring     : dict {applicable, dimension, score, justification}
        """
        if not self._initialized:
            raise RuntimeError("Pipeline non initialise. Appelez init() d'abord.")

        print(f"\n[v10] Question : {question}")

        # --- Etape 1 : Decomposition ---
        print(f"[v10] Etape 1/4 : Decomposition en {n_subquestions} sous-questions (Mistral Large)...")
        sub_questions = decompose_question(question, n=n_subquestions)
        print(f"[v10] Sous-questions :")
        for i, sq in enumerate(sub_questions, 1):
            print(f"  {i}. {sq}")

        # --- Etape 2 : Retrieval + reponse par sous-question ---
        print(f"\n[v10] Etape 2/4 : Reponses aux sous-questions (RAPTOR + Claude Haiku)...")
        sub_qa_pairs: List[Tuple[str, str]] = []
        all_sources: List[Dict] = []

        for i, sq in enumerate(sub_questions, 1):
            print(f"  [{i}/{len(sub_questions)}] {sq[:80]}...")
            context_str, sources = self.retriever.query(sq, k=k)
            ans = answer_subquestion(sq, context_str)
            sub_qa_pairs.append((sq, ans))
            for s in sources:
                s["sub_question_idx"] = i
                s["sub_question"] = sq
            all_sources.extend(sources)

        # --- Etape 3 : Synthese finale ---
        print(f"\n[v10] Etape 3/4 : Synthese finale (Mistral Large)...")
        final_answer = synthesize_answers(question, sub_qa_pairs)

        # --- Etape 4 : Notation de la dimension (optionnelle) ---
        print(f"\n[v10] Etape 4/4 : Notation de la dimension (Mistral Large)...")
        scoring = score_dimension(question, final_answer)
        if scoring["applicable"]:
            print(f"  -> {scoring['dimension']} : {scoring['score']}/5")
            print(f"     {scoring['justification'][:100]}...")
        else:
            print(f"  -> Notation non applicable ({scoring['justification'][:80]}...)")

        print("[v10] Pipeline termine.")
        return final_answer, all_sources, scoring

    def close(self):
        self.retriever.close()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG v10 : RAPTOR-lite + Sous-questions (Mistral Large / Claude Haiku)"
    )
    parser.add_argument("--query", type=str, required=True,
                        help="Question a traiter")
    parser.add_argument("--n-subquestions", type=int, default=DEFAULT_N_SUBQUESTIONS,
                        help=f"Nombre de sous-questions (defaut: {DEFAULT_N_SUBQUESTIONS})")
    parser.add_argument("--k", type=int, default=5,
                        help="Nombre de chunks evidence par sous-question (defaut: 5)")
    args = parser.parse_args()

    pipeline = RaptorSubQuestionPipeline()
    pipeline.init()

    answer, sources, scoring = pipeline.query(
        question=args.query,
        k=args.k,
        n_subquestions=args.n_subquestions,
    )

    print("\n" + "=" * 70)
    print("REPONSE FINALE")
    print("=" * 70)
    print(answer)
    if scoring["applicable"]:
        print(f"\n--- NOTE : {scoring['dimension']} ---")
        print(f"{scoring['score']}/5 — {scoring['justification']}")
    print(f"\n=== SOURCES ({len(sources)}) ===")
    for s in sources[:10]:
        sq_idx = s.get("sub_question_idx", "?")
        print(f"  [SQ{sq_idx}] {s.get('type', s.get('source_type', '?'))} — {s.get('extrait', '')[:100]}...")

    pipeline.close()


if __name__ == "__main__":
    main()
