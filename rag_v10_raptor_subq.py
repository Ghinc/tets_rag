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
                 temperature: float = 0.2,
                 max_retries: int = 5) -> str:
    """Appel Claude via SDK Anthropic avec retry exponentiel sur 429/overload."""
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY (ou CLAUDE_API_KEY) non definie")
    client = anthropic.Anthropic(api_key=api_key)
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            err = str(e)
            is_retryable = "429" in err or "529" in err or "overloaded" in err.lower() or "rate" in err.lower()
            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** attempt * 3
                print(f"    [RATE LIMIT] Claude : attente {wait}s (tentative {attempt+1}/{max_retries}) — {err[:80]}...")
                time.sleep(wait)
            else:
                print(f"    [ERREUR Claude] {err[:200]}")
                raise


# ============================================================
# Etape 1 : Decomposition en sous-questions (Mistral Large)
# ============================================================

_SYSTEM_DECOMPOSER = (
    "Tu es un expert specialisé UNIQUEMENT dans l'analyse de données sur le bien-être en Corse, "
    "a partir de verbatims citoyens et d'indicateurs territoriaux (scores OppChoVec, données d'enquêtes). "
    "Ton rôle est de decomposer une question en sous-questions ciblees. "
    "REGLES ABSOLUES : "
    "(1) Chaque sous-question doit rester du MEME TYPE que la question initiale : "
    "si la question est factuelle/chiffree (ex: score, rang, indicateur), les sous-questions le sont aussi — "
    "ne derives JAMAIS vers du qualitatif/percetions/verbatims pour une question factuelle. "
    "Si la question est qualitative (ex: que pensent les habitants), les sous-questions portent sur des groupes ou dimensions. "
    "EXCEPTION : si la question porte sur le bien-etre global ou la qualite de vie generale d'une commune, "
    "tu DOIS generer un mix de sous-questions qualitatives (perceptions, verbatims citoyens) ET quantitatives "
    "(scores d'enquete, indicateurs OppChoVec) pour couvrir les deux dimensions. "
    "Si la question comporte des verbes de percetion, considère que la question est qualitative ; si non, et si elle ne demande pas excplicitement l'opinion des habitants, elle est générale."
    "(2) Ne genere PAS de sous-questions sur des donnees que tu n'as pas (ex: sous-indicateurs detailles, "
    "ventilation demographique d'un score chiffre si non demandee). "
    "Les scores OppChoVec sont UNIQUEMENT disponibles à l'échelle de la commune — "
    "jamais par CSP, tranche d'âge, genre ou autre sous-population. "
    "Ne genere donc JAMAIS de sous-question du type 'quel est le score OppChoVec des entrepreneurs/seniors/femmes à X'. "
    "Si la question initiale concerne une sous-population et les indicateurs objectifs, "
    "formule la sous-question sur le score communal global en précisant que la ventilation n'est pas disponible. "
    "(3) Reste strictement dans le domaine de la question — ne glisse pas vers d'autres sujets. "
    "(4) RÈGLE GÉOGRAPHIQUE CRITIQUE : si la question initiale ne mentionne aucune commune ou EPCI "
    "spécifique de Corse, les sous-questions doivent rester au niveau GLOBAL CORSE. "
    "Ne génère JAMAIS de sous-questions ciblant une commune particulière (ex: Ajaccio, Bastia, Corte…) "
    "si elle n'est pas explicitement mentionnée dans la question initiale. "
    "Reponds UNIQUEMENT avec un JSON valide contenant une liste de sous-questions."
)


def decompose_question(question: str, n: int = DEFAULT_N_SUBQUESTIONS,
                        extra_context: str = "",
                        force_mixed: bool = False) -> List[str]:
    """
    Utilise Mistral Large pour decomposer la question en N sous-questions.
    Chaque sous-question porte sur un aspect distinct (groupe demographique,
    dimension thematique, comparaison geographique, etc.).
    Si extra_context est fourni (ex: donnees OppChoVec), il guide la decomposition
    pour eviter les sous-questions hors-scope.
    Si force_mixed=True (question bien-être global), force au moins une sous-question
    portant spécifiquement sur les indicateurs quantitatifs disponibles.
    """
    context_hint = ""
    if extra_context and force_mixed:
        context_hint = (
            f"DONNEES QUANTITATIVES DISPONIBLES — tu DOIS generer au moins une sous-question "
            f"portant specifiquement sur ces indicateurs chiffres :\n"
            f"{extra_context[:1500]}\n\n"
        )
    elif extra_context:
        context_hint = (
            f"DONNEES DISPONIBLES (utilise-les pour orienter la decomposition) :\n"
            f"{extra_context[:1500]}\n\n"
        )

    mix_instruction = (
        "IMPORTANT : cette question porte sur le bien-etre global, tu DOIS generer un mix : "
        "au moins une sous-question qualitative (perceptions citoyens/verbatims) ET "
        "au moins une sous-question quantitative (scores d'enquete ou indicateurs OppChoVec).\n"
    ) if force_mixed else (
        "Chaque sous-question doit rester du meme type que la question initiale "
        "(factuelle si factuelle, qualitative si qualitative) et porter sur un aspect distinct.\n"
        "INTERDIT : melanger questions chiffrees et questions de perception.\n"
    )

    prompt = (
        f"{context_hint}"
        f"Decompose cette question en exactement {n} sous-questions ciblees et complementaires.\n"
        f"{mix_instruction}"
        f"INTERDIT : deriver vers des dimensions non mentionnees dans la question initiale, "
        f"inventer des sous-categories demographiques si elles ne sont pas demandees.\n"
        f"IMPORTANT : ne genere QUE des sous-questions auxquelles les donnees disponibles permettent de repondre.\n\n"
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
    "Tu es un analyste specialise dans l'analyse du bien-être en Corse, "
    "a partir de verbatims citoyens et d'indicateurs territoriaux chiffres. "
    "Reponds a la sous-question posee en te basant UNIQUEMENT sur le contexte fourni. "
    "Le contexte contient des donnees d'enquete citoyenne (perceptions et satisfaction declarees par les habitants) "
    "et des indicateurs territoriaux objectifs (rang, scores territoriaux). "
    "Pour une sous-question chiffree sur le bien-etre territorial (rang, score), "
    "appuie-toi prioritairement sur les indicateurs objectifs. "
    "Pour une sous-question de perception ou de satisfaction, priorise les donnees d'enquete citoyenne. "
    "REGLE ABSOLUE : n'utilise JAMAIS les labels techniques internes dans ta reponse — "
    "ne mentionne pas 'RAPTOR', 'subjectif/quali', 'objectif/quanti'. "
    "Les scores OppChoVec (blocs [Scores OppChoVec ...] dans le contexte) sont LES indicateurs objectifs "
    "territoriaux de référence — indépendants des opinions, agrégés sur 0-10 (Opp/Cho/Vec). "
    "RÈGLE ABSOLUE : si un bloc [Scores OppChoVec ...] est présent dans le contexte et que la sous-question "
    "porte sur les indicateurs objectifs, le bien-être territorial ou une comparaison quanti/quali, "
    "tu DOIS commenter ces scores — ne jamais écrire 'aucun indicateur objectif' quand ce bloc est présent. "
    "De même, les blocs [Équipements et services commune ...] sont des indicateurs objectifs factuels "
    "(médecins, écoles, commerces, taux d'activité, pauvreté…) — cite-les quand pertinents. "
    "ATTENTION — définitions opérationnelles des sous-indicateurs OppChoVec (ne pas surinterprèter) : "
    "Opp = éducation moyenne + diversité CSP + accessibilité mobilité + couverture TIC/haut débit. "
    "Cho = % population avec droit de vote + absence de quartiers prioritaires (QPV) — "
    "PAS une mesure de libertés individuelles au sens large. "
    "Vec = revenu fiscal moyen + qualité du logement + stabilité de l'emploi + accès aux services en <20 min. "
    "Ces scores sont des proxies statistiques (0-10, relatif aux 360 communes corses uniquement). "
    "Ne pas extrapoler leur signification au-delà de ces composantes concrètes. "
    "IMPORTANT : les scores OppChoVec sont des indicateurs à l'échelle de la COMMUNE entière — "
    "ils ne sont pas ventilés par CSP, tranche d'âge, genre ou toute autre sous-population. "
    "Si la sous-question demande le score OppChoVec d'un groupe spécifique (ex : 'des entrepreneurs', 'des seniors'), "
    "précise clairement que ce score n'existe pas à ce niveau de granularité et fournis le score communal global à la place. "
    "Utilise des formulations naturelles : 'selon l'enquete citoyenne', 'les habitants estiment que', "
    "'les indicateurs territoriaux montrent', 'le score territorial est de X'. "
    "RÈGLE ATTRIBUTION OBLIGATOIRE : pour tout chiffre, score ou constat cité, tu DOIS indiquer sa provenance "
    "en une courte mention intégrée : "
    "(a) données d'enquête citoyenne → 'selon l'enquête citoyenne' ; "
    "(b) scores OppChoVec ou équipements → 'selon les données territoriales objectives' ou 'selon les indicateurs OppChoVec' ; "
    "(c) entretiens semi-directifs → 'selon les entretiens qualitatifs'. "
    "Ne cite JAMAIS un chiffre ou un constat sans sa source. "
    "RÈGLE PORTÉE GÉOGRAPHIQUE OBLIGATOIRE : si les données d'enquête citoyenne disponibles couvrent "
    "la Corse entière (contexte indiquant 'enquete_global', 'Corse entière', ou N=246 répondants tous horizons), "
    "tu DOIS le signaler explicitement : formule-le comme "
    "'selon l'enquête citoyenne — données pour la Corse entière, aucune donnée spécifique à [commune] n'est disponible'. "
    "Ne présente JAMAIS des perceptions ou scores Corse-entière comme s'ils étaient propres à la commune interrogée. "
    "RÈGLE GÉOGRAPHIQUE STRICTE : réponds UNIQUEMENT sur la/les commune(s) mentionnée(s) dans la sous-question. "
    "Ne cite JAMAIS d'autres communes comme exemples. "
    "Si les données sont limitées (pas de données CSP-spécifiques pour cette commune), dis-le sobrement en une phrase "
    "et utilise ce qui est disponible (score OppChoVec communal, données globales CSP) sans te diluer sur d'autres communes. "
    "Sois concis (3-5 phrases max), factuel et cite des extraits courts entre guillemets si pertinent. "
    "Si le contexte ne contient pas d'information pertinente pour cette sous-question, dis-le clairement en une phrase — "
    "ne brode pas et n'invente aucun chiffre, nom ou fait absent du contexte."
)


def answer_subquestion(sub_question: str, context: str) -> str:
    """
    Utilise Claude Haiku pour repondre a une sous-question a partir du contexte RAPTOR.
    Le contexte est tronque a 3000 caracteres pour maitriser les tokens.
    """
    prompt = f"Contexte (syntheses et verbatims) :\n{context[:15000]}\n\nSous-question : {sub_question}"
    return _call_claude(prompt, _SYSTEM_ANSWERER)


# ============================================================
# Etape 3 : Synthese finale (Mistral Large)
# ============================================================

_SYSTEM_SYNTHESIZER = (
    "Tu es un expert en analyse de donnees qualitatives et quantitatives sur la qualite de vie en Corse. "
    "Tu recois une question initiale, un bilan structure des sources disponibles par sous-question, "
    "et les reponses a ces sous-questions. "
    "Les donnees d'enquete citoyenne representent les perceptions et ressentis declares par les habitants interroges. "
    "Les indicateurs territoriaux sont des mesures objectives indépendantes des opinions — "
    "les scores OppChoVec (Opp/Cho/Vec, 0-10) en sont la source principale et structurée. "
    "Ils peuvent être complétés par des données d'équipements communaux (médecins, écoles, commerces, "
    "taux d'activité, pauvreté, etc.) si présentes dans les réponses aux sous-questions. "
    "Si une sous-question a reçu des données OppChoVec, intègre-les dans la synthèse — "
    "ne jamais conclure à l'absence d'indicateurs objectifs si ces données figurent dans les réponses. "
    "ATTENTION — définitions opérationnelles des sous-indicateurs OppChoVec (ne pas surinterprèter) : "
    "Opp = éducation moyenne + diversité CSP + accessibilité mobilité + couverture TIC/haut débit. "
    "Cho = % population avec droit de vote + absence de quartiers prioritaires (QPV) — "
    "PAS une mesure de libertés individuelles au sens large. "
    "Vec = revenu fiscal moyen + qualité du logement + stabilité de l'emploi + accès aux services en <20 min. "
    "Ces scores sont des proxies (0-10, relatif aux 360 communes corses uniquement) — ne pas extrapoler. "
    "Produis une synthese finale coherente, analytique et nuancee, en repondant directement a la question initiale. "
    "Respecte scrupuleusement le bilan des sources : si le bilan indique des donnees d'enquete presentes "
    "pour une sous-question, ne pretends pas qu'elles sont absentes. "
    "REGLE ABSOLUE : n'utilise JAMAIS les termes techniques internes dans ta reponse — "
    "ne mentionne pas 'RAPTOR', 'subjectif/quali', 'objectif/quanti', 'OppChoVec integre', etc. "
    "Utilise des formulations naturelles : 'selon l'enquete citoyenne', 'les indicateurs territoriaux montrent'. "
    "RÈGLE ATTRIBUTION OBLIGATOIRE : pour tout chiffre, score ou constat intégré dans la synthèse, "
    "tu DOIS mentionner sa provenance en une courte mention intégrée : "
    "(a) données d'enquête citoyenne → 'selon l'enquête citoyenne' ; "
    "(b) scores OppChoVec ou équipements → 'selon les données territoriales objectives' ou 'selon les indicateurs OppChoVec' ; "
    "(c) entretiens semi-directifs → 'selon les entretiens qualitatifs'. "
    "Ne cite JAMAIS un chiffre ou un constat sans sa source. "
    "RÈGLE PORTÉE GÉOGRAPHIQUE OBLIGATOIRE : si les données d'enquête citoyenne utilisées couvrent "
    "la Corse entière (et non la commune interrogée spécifiquement), tu DOIS le signaler clairement dans la synthèse — "
    "par exemple : 'Les données d'enquête citoyenne disponibles couvrent la Corse entière ; "
    "aucune donnée d'enquête spécifique à [commune] n'est disponible.' "
    "Ne présente JAMAIS des perceptions ou scores Corse-entière comme propres à la commune interrogée. "
    "Structure ta reponse avec des points cles si pertinent. "
    "RÈGLE GÉOGRAPHIQUE STRICTE : réponds UNIQUEMENT sur la/les commune(s) mentionnée(s) dans la question initiale. "
    "Ne cite JAMAIS d'autres communes. Si les données sont limitées à une portée moins spécifique (tous résidents, "
    "Corse entière), indique-le clairement sans inventer de données complémentaires d'autres communes. "
    "IMPORTANT — honnêteté sur les lacunes : si une sous-question n'a pas de réponse dans le contexte, "
    "indique-le explicitement ('aucune donnée disponible sur ce point') plutôt que de broder ou d'extrapoler. "
    "Ne fabrique jamais de chiffres, de rangs ou de faits absents des réponses aux sous-questions."
)

# Identique à _SYSTEM_SYNTHESIZER mais sans les deux lignes qui référencent le bilan.
# Utilisé pour V_decomp et V_decomp_raptor (ablations sans bilan déterministe).
_SYSTEM_SYNTHESIZER_NO_BILAN = (
    "Tu es un expert en analyse de donnees qualitatives et quantitatives sur la qualite de vie en Corse. "
    "Tu recois une question initiale et les reponses a des sous-questions derivees. "
    "Les donnees d'enquete citoyenne representent les perceptions et ressentis declares par les habitants interroges. "
    "Les indicateurs territoriaux sont des mesures objectives indépendantes des opinions — "
    "les scores OppChoVec (Opp/Cho/Vec, 0-10) en sont la source principale et structurée. "
    "Ils peuvent être complétés par des données d'équipements communaux (médecins, écoles, commerces, "
    "taux d'activité, pauvreté, etc.) si présentes dans les réponses aux sous-questions. "
    "Si une sous-question a reçu des données OppChoVec, intègre-les dans la synthèse — "
    "ne jamais conclure à l'absence d'indicateurs objectifs si ces données figurent dans les réponses. "
    "ATTENTION — définitions opérationnelles des sous-indicateurs OppChoVec (ne pas surinterprèter) : "
    "Opp = éducation moyenne + diversité CSP + accessibilité mobilité + couverture TIC/haut débit. "
    "Cho = % population avec droit de vote + absence de quartiers prioritaires (QPV) — "
    "PAS une mesure de libertés individuelles au sens large. "
    "Vec = revenu fiscal moyen + qualité du logement + stabilité de l'emploi + accès aux services en <20 min. "
    "Ces scores sont des proxies (0-10, relatif aux 360 communes corses uniquement) — ne pas extrapoler. "
    "Produis une synthese finale coherente, analytique et nuancee, en repondant directement a la question initiale. "
    "REGLE ABSOLUE : n'utilise JAMAIS les termes techniques internes dans ta reponse — "
    "ne mentionne pas 'RAPTOR', 'subjectif/quali', 'objectif/quanti', 'OppChoVec integre', etc. "
    "Utilise des formulations naturelles : 'selon l'enquete citoyenne', 'les indicateurs territoriaux montrent'. "
    "RÈGLE ATTRIBUTION OBLIGATOIRE : pour tout chiffre, score ou constat intégré dans la synthèse, "
    "tu DOIS mentionner sa provenance en une courte mention intégrée : "
    "(a) données d'enquête citoyenne → 'selon l'enquête citoyenne' ; "
    "(b) scores OppChoVec ou équipements → 'selon les données territoriales objectives' ou 'selon les indicateurs OppChoVec' ; "
    "(c) entretiens semi-directifs → 'selon les entretiens qualitatifs'. "
    "Ne cite JAMAIS un chiffre ou un constat sans sa source. "
    "RÈGLE PORTÉE GÉOGRAPHIQUE OBLIGATOIRE : si les données d'enquête citoyenne utilisées couvrent "
    "la Corse entière (et non la commune interrogée spécifiquement), tu DOIS le signaler clairement dans la synthèse — "
    "par exemple : 'Les données d'enquête citoyenne disponibles couvrent la Corse entière ; "
    "aucune donnée d'enquête spécifique à [commune] n'est disponible.' "
    "Ne présente JAMAIS des perceptions ou scores Corse-entière comme propres à la commune interrogée. "
    "Structure ta reponse avec des points cles si pertinent. "
    "RÈGLE GÉOGRAPHIQUE STRICTE : réponds UNIQUEMENT sur la/les commune(s) mentionnée(s) dans la question initiale. "
    "Ne cite JAMAIS d'autres communes. Si les données sont limitées à une portée moins spécifique (tous résidents, "
    "Corse entière), indique-le clairement sans inventer de données complémentaires d'autres communes. "
    "IMPORTANT — honnêteté sur les lacunes : si une sous-question n'a pas de réponse dans le contexte, "
    "indique-le explicitement ('aucune donnée disponible sur ce point') plutôt que de broder ou d'extrapoler. "
    "Ne fabrique jamais de chiffres, de rangs ou de faits absents des réponses aux sous-questions."
)


def synthesize_answers(initial_question: str,
                        sub_qa_pairs: List[Tuple[str, str]],
                        source_bilan: Dict[int, Dict] = None,
                        use_bilan: bool = True) -> str:
    """
    Utilise Mistral Large pour synthetiser les reponses aux sous-questions.
    source_bilan : dict {idx_sq -> {has_subjective, has_objective}} calcule
    deterministement depuis les sources retrievees (pas de LLM).
    use_bilan=False : ablation — skip du bloc bilan + system prompt sans références au bilan.
    """
    sub_answers_text = "\n\n".join(
        f"**Sous-question {i+1}** : {sq}\n**Reponse** : {ans[:1500]}"
        for i, (sq, ans) in enumerate(sub_qa_pairs)
    )

    # Bloc bilan structurel — calculé en Python, injecté comme fait établi.
    # Skippé entièrement quand use_bilan=False (ablation V_decomp / V_decomp_raptor).
    bilan_block = ""
    if use_bilan and source_bilan:
        lines = []
        for i, (sq, _) in enumerate(sub_qa_pairs, 1):
            b = source_bilan.get(i, {})
            subj = "OUI" if b.get("has_subjective") else "NON"
            obj  = "OUI" if b.get("has_objective")  else "NON"
            lines.append(
                f"  [SQ{i}] \"{sq[:80]}\"\n"
                f"    -> Donnees enquete citoyenne (perceptions habitants) : {subj}"
                f"  |  Indicateurs territoriaux objectifs : {obj}"
            )
        bilan_block = (
            "=== BILAN DES SOURCES PAR SOUS-QUESTION ===\n"
            + "\n".join(lines)
            + "\n===========================================\n\n"
        )

    system_prompt = _SYSTEM_SYNTHESIZER if use_bilan else _SYSTEM_SYNTHESIZER_NO_BILAN

    prompt = (
        f"Question initiale : {initial_question}\n\n"
        f"{bilan_block}"
        f"Voici les reponses aux sous-questions derivees :\n\n"
        f"{sub_answers_text}\n\n"
        f"Produis une synthese finale qui :\n"
        f"1. Repond directement a la question initiale\n"
        f"2. Integre les informations pertinentes des sous-reponses\n"
        f"3. Distingue donnees subjectives (perceptions citoyens) et donnees objectives (indicateurs)\n"
        f"4. Ne mentionne PAS l'absence d'un type de donnees si le bilan indique sa presence"
    )

    return _call_mistral(prompt, system_prompt, max_tokens=1500, temperature=0.3)


# ============================================================
# Etape 4 : Notation de la dimension evaluee (Mistral Large, optionnelle)
# ============================================================

_SYSTEM_SCORER = (
    "Tu es un expert en evaluation de la qualite de vie. "
    "On te fournit une question et une synthese analytique issues de ressentis citoyens ou de scores objectifs. "
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
                 n_evidence_chunks: int = 5,
                 oppchovec_col=None):
        self.retriever = RaptorRetriever(
            chroma_path=chroma_path,
            source_collection=source_collection,
            summary_collection=summary_collection,
            n_evidence_chunks=n_evidence_chunks,
            oppchovec_col=oppchovec_col,
        )
        self._initialized = False

    def init(self):
        """Charge le RaptorRetriever (modele d'embeddings + connexion ChromaDB)."""
        self.retriever.init()
        self._initialized = True
        print(f"RAG v10 initialise ({self.retriever.summary_count} syntheses RAPTOR disponibles)")

    def query(self, question: str, k: int = 5,
              n_subquestions: int = DEFAULT_N_SUBQUESTIONS,
              extra_context: str = "",
              force_mixed: bool = False,
              use_bilan: bool = True) -> Tuple[str, List[Dict], Dict, List[Dict]]:
        """
        Pipeline complet v10.

        Returns:
            (final_answer, all_sources, scoring, sub_qa_pairs)
            all_sources  : liste de dicts avec les champs habituels RAPTOR
                           + sub_question_idx et sub_question pour tracabilite.
            scoring      : dict {applicable, dimension, score, justification}
            sub_qa_pairs : liste de dicts {idx, question, answer} pour affichage
        """
        if not self._initialized:
            raise RuntimeError("Pipeline non initialise. Appelez init() d'abord.")

        print(f"\n[v10] Question : {question}")

        # --- Détection question globale Corse (pas de commune) ---
        # Pour éviter que Mistral hallucine des communes dans les sous-questions,
        # on court-circuite la décomposition et on passe directement à RAPTOR global.
        import unicodedata as _ud_v10
        _q_v10 = "".join(c for c in _ud_v10.normalize("NFD", question.lower()) if _ud_v10.category(c) != "Mn")
        try:
            from commune_detector import detect_communes as _dc_v10
            _communes_in_q = _dc_v10(question)
        except ImportError:
            _communes_in_q = []
        _is_global_q = not _communes_in_q and any(kw in _q_v10 for kw in (
            "moyen", "moyenne", "general", "global", "ensemble", "niveau",
            "corse entiere", "ile entiere", "l ensemble", "toutes les communes",
            "score global", "score corse", "indicateur corse",
        ))
        if _is_global_q:
            print(f"[v10] Question globale Corse détectée (pas de commune) — bypass décomposition")
            context_str, sources = self.retriever.query(question, k=k)
            # Injecter le doc agrégat Corse si disponible
            _global_extra = extra_context
            if self.retriever._oppchovec:
                try:
                    _agg = self.retriever._oppchovec.get(
                        ids=["oppchovec_aggregate_corse"],
                        include=["documents", "metadatas"]
                    )
                    if _agg["documents"]:
                        _global_extra = ("[Scores OppChoVec — Corse entière (indicateurs territoriaux objectifs)]\n"
                                         + _agg["documents"][0] + ("\n\n" + extra_context if extra_context else ""))
                        print("[v10] Doc agrégat Corse injecté")
                except Exception:
                    pass
            if _global_extra:
                context_str = _global_extra + "\n\n" + context_str
            _single_answer = answer_subquestion(question, context_str)
            # Construire la réponse finale via synthétiseur (1 sous-question = la question elle-même)
            _single_pair = [(question, _single_answer)]
            final_answer = synthesize_answers(question, _single_pair, {1: {"has_subjective": True, "has_objective": True}})
            scoring = {"applicable": False, "dimension": None, "score": None, "justification": "Question globale — scoring non applicable"}
            sub_qa_pairs_out = [{"idx": 1, "question": question, "answer": _single_answer}]
            print(f"[v10] Pipeline global termine.")
            return final_answer, sources, scoring, sub_qa_pairs_out

        # --- Etape 1 : Decomposition ---
        print(f"[v10] Etape 1/4 : Decomposition en {n_subquestions} sous-questions (Mistral Large)...")
        try:
            sub_questions = decompose_question(question, n=n_subquestions, extra_context=extra_context, force_mixed=force_mixed)
        except RuntimeError as _e:
            print(f"[v10] Question hors-domaine ou indécomposable : {_e}")
            _refusal = _call_mistral(
                f"Question : {question}",
                "Tu es un assistant spécialisé en qualité de vie en Corse. "
                "Cette question ne relève pas de ton domaine d'expertise. "
                "Réponds poliment que tu ne peux pas répondre à cette question.",
                max_tokens=300, temperature=0.3,
            )
            _empty_scoring = {"applicable": False, "dimension": None, "score": None,
                              "justification": "Question hors-domaine"}
            return _refusal, [], _empty_scoring, []
        print(f"[v10] Sous-questions :")
        for i, sq in enumerate(sub_questions, 1):
            print(f"  {i}. {sq}")

        # --- Etape 1bis : pré-injection classement OppChoVec si question de classement ---
        # La sous-question générée peut ne pas contenir "meilleur" → is_ranking=False dans retriever.
        # On force l'injection ici au niveau v10 pour toutes les sous-questions.
        opp_extra = ""
        if self.retriever._oppchovec and self.retriever._is_ranking_question(question):
            try:
                _cl = self.retriever._oppchovec.get(
                    ids=["oppchovec_classement_global"],
                    include=["documents", "metadatas"]
                )
                if _cl["documents"]:
                    opp_extra = ("[Classement OppChoVec des communes corses — référence pour filtrer par EPCI/commune]\n"
                                 + _cl["documents"][0][:8000])
                    print("[v10] Classement OppChoVec global injecté dans le contexte des sous-questions")
            except Exception:
                pass

        # Pré-injection OppChoVec pour communes détectées — toujours, même si classement global déjà injecté.
        # Le classement (8000 chars) peut enterrer le score d'une petite commune → on injecte toujours le score spécifique.
        if _communes_in_q and self.retriever._oppchovec:
            try:
                q_emb = self.retriever._encode_query(question)
                for _com in _communes_in_q[:2]:
                    _res_c = self.retriever._oppchovec.query(
                        query_embeddings=[q_emb], n_results=1,
                        where={"$and": [
                            {"source": {"$in": ["oppchovec_betti_0_10", "oppchovec_aggregate"]}},
                            {"commune": {"$eq": _com}}
                        ]},
                        include=["documents", "metadatas", "distances"]
                    )
                    if _res_c["documents"][0]:
                        opp_extra += f"\n\n[Scores OppChoVec — {_com}]\n{_res_c['documents'][0][0][:1500]}"
                        print(f"[v10] OppChoVec pré-injecté pour commune : {_com}")
                opp_extra = opp_extra.strip()
            except Exception as _e:
                print(f"[v10] Erreur pré-injection OppChoVec communes : {_e}")

        # --- Etape 2 : Retrieval + reponse par sous-question ---
        print(f"\n[v10] Etape 2/4 : Reponses aux sous-questions (RAPTOR + Claude Haiku)...")
        sub_qa_pairs: List[Tuple[str, str]] = []
        all_sources: List[Dict] = []

        for i, sq in enumerate(sub_questions, 1):
            print(f"  [{i}/{len(sub_questions)}] {sq[:80]}...")
            context_str, sources = self.retriever.query(sq, k=k)
            merged_extra = "\n\n".join(x for x in [opp_extra, extra_context] if x)
            if merged_extra:
                context_str = merged_extra + "\n\n" + context_str
            ans = answer_subquestion(sq, context_str)
            sub_qa_pairs.append((sq, ans))
            for s in sources:
                s["sub_question_idx"] = i
                s["sub_question"] = sq
            all_sources.extend(sources)

        # Bilan déterministe des sources par sous-question (calculé en Python, pas via LLM).
        # Skippé quand use_bilan=False (ablation V_decomp_raptor).
        source_bilan: Dict[int, Dict] = {}
        if use_bilan:
            for s in all_sources:
                idx = s.get("sub_question_idx", 0)
                if idx not in source_bilan:
                    source_bilan[idx] = {"has_subjective": False, "has_objective": False}
                t = s.get("type", "") or s.get("source_type", "")
                if "raptor" in t or "enquete" in t or "verbatim" in t:
                    source_bilan[idx]["has_subjective"] = True
                if "opp" in t or "objectif" in t or "equipement" in t:
                    source_bilan[idx]["has_objective"] = True

        # --- Etape 3 : Synthese finale ---
        print(f"\n[v10] Etape 3/4 : Synthese finale (Mistral Large)...")
        final_answer = synthesize_answers(question, sub_qa_pairs, source_bilan, use_bilan=use_bilan)

        # --- Etape 4 : Notation de la dimension (optionnelle) ---
        print(f"\n[v10] Etape 4/4 : Notation de la dimension (Mistral Large)...")
        scoring = score_dimension(question, final_answer)
        if scoring["applicable"]:
            print(f"  -> {scoring['dimension']} : {scoring['score']}/5")
            print(f"     {str(scoring.get('justification', ''))[:100]}...")
        else:
            print(f"  -> Notation non applicable ({str(scoring.get('justification', ''))[:80]}...)")

        print("[v10] Pipeline termine.")
        sub_qa_list = [
            {"idx": i + 1, "question": sq, "answer": ans}
            for i, (sq, ans) in enumerate(sub_qa_pairs)
        ]
        return final_answer, all_sources, scoring, sub_qa_list

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

    answer, sources, scoring, sub_qa = pipeline.query(
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
