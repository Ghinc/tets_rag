"""
rag_selfask.py — Self-Ask (Press et al. 2022) comme baseline externe.

Boucle séquentielle et adaptative :
  1. Mistral Large (temperature=0) génère un Follow up conditionné par les
     réponses intermédiaires précédentes.
  2. _get_raw_sources (chunks bruts, sans RAPTOR) fait le retrieval au hop.
  3. Claude Haiku génère la réponse intermédiaire (cohérence avec v_decomp).
  4. Itérer jusqu'à "So the final answer is:" ou MAX_HOPS.

Format conversation multi-turn fidèle à l'article :
    [user]      Question: X\nAre there follow up questions needed here:
    [assistant] Yes.\nFollow up: Y?
    [user]      Intermediate answer: Z
    [assistant] Follow up: W?   ← conditionné par Z
    [user]      Intermediate answer: V
    [assistant] So the final answer is: [réponse complète]
"""

import os, re, sys, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

from rag_v10_raptor_subq import (
    CHROMA_PATH, DECOMPOSER_MODEL,
    answer_subquestion,
)
from rag_ablations import _get_raw_sources

MAX_HOPS_DEFAULT = 5
K_DEFAULT = 5

# ── Système Self-Ask ─────────────────────────────────────────────────────────

_SYSTEM_SELFASK = """\
Tu es un expert en analyse du bien-être en Corse, spécialisé dans les données
territoriales objectives (indicateurs OppChoVec, équipements communaux) et
subjectives (verbatims citoyens, enquêtes, entretiens semi-directifs).

Pour répondre à une question, tu procèdes par décomposition séquentielle :
— Si tu as besoin d'une information intermédiaire avant de pouvoir répondre,
  génère EXACTEMENT une seule ligne : "Follow up: [ta sous-question]?"
— Après chaque réponse intermédiaire fournie, génère soit un nouveau "Follow up:"
  soit la réponse finale : "So the final answer is: [réponse complète en français]"
— Si tu peux répondre directement sans étape intermédiaire, commence immédiatement
  par : "So the final answer is: [réponse]"

CONTRAINTES ABSOLUES :
— Un seul Follow up par message
— Ne génère jamais de sous-questions sur des données qui n'existent pas :
  les scores OppChoVec ne sont disponibles qu'à l'échelle de la commune entière
  (pas de ventilation par CSP, âge, genre ou sous-population)
— Si la question mentionne une commune spécifique, reste sur cette commune
— La réponse finale doit être en français, complète, bien rédigée, sourcée,
  et répondre directement à la question initiale
— Pour les questions factuelles (scores, chiffres), cite les valeurs précises
— Pour les questions qualitatives (perceptions, ressentis), appuie-toi sur
  les verbatims et entretiens
"""

_FEWSHOT = """\
Voici un exemple de raisonnement Self-Ask appliqué à ce domaine :

---
Question: Quel est le profil de bien-être de Bastia selon les indicateurs objectifs et les perceptions citoyennes ?
Are there follow up questions needed here: Yes.
Follow up: Quel est le score OppChoVec global de la commune de Bastia et ses composantes ?
Intermediate answer: Bastia obtient un score OppChoVec global de 5,1/10 (Opp: 5,4, Cho: 6,2, Vec: 3,7), la positionnant légèrement sous la médiane des communes corses. La composante Vécu (conditions matérielles : revenu, logement, emploi, accès services) est la plus faible à 3,7.
Follow up: Que disent les verbatims et l'enquête citoyenne sur le bien-être perçu à Bastia ?
Intermediate answer: Les verbatims de Bastia révèlent une satisfaction mitigée : les habitants valorisent la dynamique commerciale et le cadre urbain, mais expriment des frustrations concernant le coût de la vie, le trafic et des inégalités entre quartiers. Le score de satisfaction global issu de l'enquête citoyenne est de 3,1/5.
So the final answer is: Bastia présente un profil de bien-être contrasté. Sur le plan objectif, son score OppChoVec de 5,1/10 (Opp: 5,4, Cho: 6,2, Vec: 3,7) la situe légèrement en dessous de la médiane corse, avec une composante Vécu (conditions matérielles de vie) particulièrement faible à 3,7/10. Les perceptions citoyennes recueillies dans l'enquête confirment cette ambivalence : si la dynamique commerciale et le tissu urbain sont valorisés, les habitants pointent le coût de la vie élevé, les difficultés de circulation et des disparités entre quartiers. Le score de satisfaction global de 3,1/5 issu de l'enquête citoyenne est cohérent avec ces indicateurs objectifs modérés. La convergence entre données objectives et perceptions subjectives indique une situation de bien-être réelle mais fragilisée sur le plan matériel.
---

"""


# ── Appel Mistral multi-tour ─────────────────────────────────────────────────

def _call_mistral_multiturn(
    messages: List[Dict],
    temperature: float = 0.0,
    max_tokens: int = 2000,
    max_retries: int = 5,
) -> str:
    """Appel Mistral Large multi-tour (liste de messages complète)."""
    from openai import OpenAI
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY non définie")
    client = OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")

    full_messages = [{"role": "system", "content": _SYSTEM_SELFASK}] + messages

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=DECOMPOSER_MODEL,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt * 2
                print(f"    [RATE LIMIT] Mistral multi-turn : attente {wait}s "
                      f"(tentative {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


# ── Parsing Self-Ask ─────────────────────────────────────────────────────────

def _extract_follow_up(text: str) -> Optional[str]:
    """Extrait la sous-question après 'Follow up:' (tolérant aux variantes)."""
    m = re.search(r'Follow(?:\s+|-)?up:\s*(.+?)(?:\?|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip("?") + "?"
    return None


def _extract_final_answer(text: str) -> Optional[str]:
    """Extrait la réponse finale après 'So the final answer is:'."""
    m = re.search(r'So the final answer is:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _source_key(s: Dict) -> tuple:
    """Clé de déduplication inter-hops pour une source."""
    meta = s.get("metadata", {})
    return (
        s.get("source_type", ""),
        meta.get("commune", ""),
        str(meta.get("chunk_idx", meta.get("id", ""))),
        meta.get("source", ""),
    )


# ── Classe principale ────────────────────────────────────────────────────────

class SelfAskRAG:
    """
    V_selfask — Self-Ask (Press et al. 2022) — baseline externe.

    Paramètres :
      max_hops : nombre maximal de follow-ups (défaut 5, comparable à N=5 sous-questions)
      k        : chunks récupérés par hop (défaut 5, identique aux autres configs)
    """

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        max_hops: int = MAX_HOPS_DEFAULT,
        k: int = K_DEFAULT,
    ):
        self.chroma_path = chroma_path
        self.max_hops = max_hops
        self.k = k
        self._retriever = None
        self._initialized = False

    def init(self):
        from rag_v9_raptor import RaptorRetriever
        self._retriever = RaptorRetriever(chroma_path=self.chroma_path)
        self._retriever.init()
        self._initialized = True
        print("SelfAskRAG (V_selfask) initialisé")

    def query(self, question: str) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Lance la boucle Self-Ask sur question.

        Retourne :
          final_answer   : str — réponse finale synthétisée
          all_sources    : List[Dict] — sources dédupliquées de tous les hops
                           format {content, metadata, source_type, label}
          hops           : List[Dict] — [{hop, follow_up, model_raw, sources, intermediate_answer}]
                           sources = sources brutes intégrales du hop
        """
        assert self._initialized, "Appelez init() d'abord."

        # Message initial : few-shot + question
        initial_content = (
            _FEWSHOT
            + f"Question: {question}\n"
            "Are there follow up questions needed here: "
        )
        messages: List[Dict] = [{"role": "user", "content": initial_content}]

        hops: List[Dict] = []
        all_sources: List[Dict] = []
        seen_keys: set = set()
        final_answer: Optional[str] = None

        print(f"\n[v_selfask] Question : {question}")
        print(f"[v_selfask] Boucle Self-Ask (max {self.max_hops} hops, k={self.k})...")

        for hop_i in range(self.max_hops):
            raw = _call_mistral_multiturn(messages, temperature=0.0, max_tokens=2000)

            # Priorité : réponse finale ?
            final = _extract_final_answer(raw)
            if final:
                final_answer = final
                print(f"[v_selfask] → Réponse finale après {hop_i} hop(s).")
                break

            # Follow-up ?
            follow_up = _extract_follow_up(raw)
            if not follow_up:
                # Réponse inattendue sans marqueur — traiter comme finale
                print(f"[v_selfask] → Pas de follow-up parsé (hop {hop_i + 1}) "
                      "— traité comme réponse finale.")
                final_answer = raw.strip()
                break

            print(f"  [hop {hop_i + 1}/{self.max_hops}] Follow up: {follow_up[:80]}...")

            # Retrieval chunks bruts (sans RAPTOR)
            context_str, hop_sources = _get_raw_sources(self._retriever, follow_up, self.k)

            # Déduplication inter-hops
            for s in hop_sources:
                key = _source_key(s)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_sources.append(s)

            # Haiku : réponse intermédiaire
            intermediate = answer_subquestion(follow_up, context_str)
            print(f"           → {len(hop_sources)} src │ {intermediate[:120]}...")

            hops.append({
                "hop": hop_i + 1,
                "follow_up": follow_up,
                "model_raw": raw,
                "sources": hop_sources,
                "intermediate_answer": intermediate,
            })

            # Mise à jour conversation
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Intermediate answer: {intermediate}",
            })

        else:
            # Max hops atteint sans réponse finale
            print(f"[v_selfask] Max hops ({self.max_hops}) atteint — forçage synthèse finale.")
            messages.append({
                "role": "user",
                "content": (
                    "Tu as maintenant toutes les informations nécessaires. "
                    "Génère la réponse finale complète à la question initiale. "
                    'Commence impérativement par : "So the final answer is:"'
                ),
            })
            raw_final = _call_mistral_multiturn(messages, temperature=0.0, max_tokens=3000)
            final_answer = _extract_final_answer(raw_final) or raw_final.strip()

        n_src = len(all_sources)
        print(f"[v_selfask] Pipeline terminé ({len(hops)} hops, {n_src} sources uniques).")
        return final_answer or "", all_sources, hops
