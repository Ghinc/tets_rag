# Rapport : Agentic RAG — Principe, Technologies et Faisabilité d'une v11

> **Projet :** Système RAG multi-version pour l'analyse de bien-être citoyen en Corse
> **Date :** Avril 2026
> **Versions couvertes :** v1 → v10 (existantes) + v11 (proposition)
> **Auteur :** Généré via Claude Code (claude-sonnet-4-6)

---

## Table des matières

1. [Du RAG classique à l'Agentic RAG](#1-du-rag-classique-à-lagentic-rag)
2. [Patterns clés de l'Agentic RAG](#2-patterns-clés-de-lagentic-rag)
3. [Frameworks et technologies](#3-frameworks-et-technologies)
4. [Analyse de l'existant — ce qui est déjà agentique](#4-analyse-de-lexistant--ce-qui-est-déjà-agentique)
5. [Design de la v11 — Agentic RAG](#5-design-de-la-v11--agentic-rag)
6. [Faisabilité et roadmap d'implémentation](#6-faisabilité-et-roadmap-dimplémentation)
7. [Conclusion et recommandation](#7-conclusion-et-recommandation)

---

## 1. Du RAG classique à l'Agentic RAG

### 1.1 Le pipeline RAG standard

La Retrieval-Augmented Generation (RAG) repose sur un pipeline en trois étapes fixes :

```
Question utilisateur
      ↓
  [RETRIEVAL]  — recherche dans une base vectorielle (ex. ChromaDB)
      ↓
  [AUGMENTATION] — concaténation question + contexte récupéré
      ↓
  [GENERATION] — LLM produit une réponse à partir du contexte
```

Cette architecture, introduite par Lewis et al. (2020), a rapidement montré sa valeur : elle ancre le LLM dans des faits vérifiables, réduit les hallucinations et permet de mettre à jour les connaissances sans réentraîner le modèle.

### 1.2 Les limites fondamentales du RAG passif

Malgré ses qualités, le RAG standard souffre de plusieurs limitations structurelles :

| Limitation | Description | Impact |
|------------|-------------|--------|
| **Requête unique** | Une seule passe de retrieval, quelles que soient la complexité ou l'ambiguïté | Échec sur les questions multi-aspects |
| **Contexte statique** | Le contexte récupéré ne peut pas être enrichi si insuffisant | Réponses incomplètes ou inventées |
| **Pas de raisonnement** | Aucune logique pour décomposer, planifier, vérifier | Confond faits, déductions et suppositions |
| **Outil unique** | Un seul index vectoriel, même si plusieurs sources seraient utiles | Perte d'information disponible ailleurs |
| **Pas d'auto-correction** | Si le contexte est hors sujet, la réponse sera quand même générée | Hallucinations non détectées |

Dans ce projet, ces limites sont particulièrement sensibles : les questions sur le bien-être citoyen mêlent souvent des données qualitatives (verbatims), quantitatives (scores OppChoVec), géographiques (communes) et démographiques (âge, profession) — une requête unique vers un seul index ne peut pas satisfaire ce besoin de façon fiable.

### 1.3 Définition de l'Agentic RAG

L'**Agentic RAG** désigne un système RAG dans lequel le LLM joue un rôle actif de décision à chaque étape du pipeline — choisissant quand et comment récupérer du contexte, évaluant la qualité de ce qu'il a obtenu, et itérant jusqu'à disposer d'une base suffisante pour répondre.

Trois caractéristiques fondamentales le distinguent du RAG passif :

1. **Boucle de raisonnement** : le pipeline n'est plus linéaire mais itératif. Le LLM peut décider de faire plusieurs appels de retrieval successifs, de reformuler une requête ou de changer d'outil.
2. **Arsenal d'outils** : au lieu d'un seul index vectoriel, plusieurs outils sont disponibles (différentes collections, SQL, web search, calculatrice…). Le LLM choisit lequel appeler en fonction du besoin.
3. **Auto-évaluation** : après chaque retrieval, le système évalue si le contexte obtenu est pertinent, suffisant, et non contradictoire. Si non, il corrige.

### 1.4 Le continuum d'agentivité

Il n'y a pas de rupture franche entre RAG classique et Agentic RAG, mais un continuum :

```
Niveau 0 — RAG passif (v1-v2)
  → Un index, une requête, une réponse. Aucune décision LLM.

Niveau 1 — RAG avec routing (v7-v8)
  → Le LLM choisit entre deux moteurs (vecteur ou graphe).
  → Une décision, mais pas de boucle.

Niveau 2 — RAG avec décomposition (v10)
  → La question est décomposée en sous-questions traitées en parallèle.
  → Plusieurs retrieval indépendants, synthèse finale.
  → Séquentiel, pas itératif.

Niveau 3 — Agentic RAG (v11 proposée)
  → Boucle ReAct : le LLM pense, agit, observe, réévalue.
  → Sélection dynamique d'outils.
  → Auto-évaluation du contexte (CRAG gate).
  → Reformulation automatique si retrieval insuffisant.
```

---

## 2. Patterns clés de l'Agentic RAG

### 2.1 ReAct — Reasoning + Acting

ReAct (Yao et al., 2022) est le pattern fondateur de l'Agentic RAG. Il définit une boucle alternant raisonnement en langage naturel et appels d'outils :

```
THINK  : "Pour répondre à cette question, j'ai besoin des verbatims sur la santé à Ajaccio."
ACT    : verbatim_search(query="santé Ajaccio", filters={"commune": "Ajaccio"})
OBSERVE: [contexte retourné par l'outil]
THINK  : "Le contexte couvre bien la santé mais pas les soins d'urgence, il faut compléter."
ACT    : summary_search(query="urgences médicales Ajaccio", view="dimension×commune")
OBSERVE: [contexte supplémentaire]
THINK  : "J'ai suffisamment d'éléments pour répondre."
ANSWER : [réponse finale avec citations]
```

**Points clés :**
- Le LLM génère explicitement ses pensées avant chaque action (chain-of-thought implicite)
- La boucle s'arrête quand le LLM juge le contexte suffisant, ou après N itérations max
- Le choix de l'outil et des paramètres est entièrement à la discrétion du LLM
- Implémenté nativement dans LangChain AgentExecutor, LlamaIndex ReActAgent, et Anthropic Tool Use

**Limite :** sans garde-fous, le LLM peut boucler indéfiniment ou choisir des outils inadaptés.

---

### 2.2 Self-RAG — Self-Reflective RAG

Self-RAG (Asai et al., 2023) est une approche différente : le modèle LLM lui-même est fine-tuné pour émettre des **jetons de réflexion spéciaux** à chaque étape :

| Jeton | Signification | Valeurs possibles |
|-------|---------------|-------------------|
| `[Retrieve]` | Faut-il récupérer du contexte ? | Yes / No |
| `[IsREL]` | Le contexte récupéré est-il pertinent ? | Relevant / Irrelevant |
| `[IsSUP]` | La réponse générée est-elle supportée ? | Fully / Partially / No |
| `[IsUse]` | La réponse est-elle utile ? | 5 / 4 / 3 / 2 / 1 |

**Avantage majeur :** le retrieval n'est déclenché que si nécessaire (pas de sur-retrieval), et chaque fragment de réponse est évalué avant d'être inclus.

**Limite principale :** nécessite un modèle fine-tuné spécifiquement avec ces jetons. Non applicable directement avec des LLMs commerciaux (GPT, Claude, Mistral) sans fine-tuning.

**Adaptation possible ici :** simuler ces jetons via des prompts structurés ("Évalue si ce contexte est suffisant sur une échelle de 0 à 1") sans fine-tuning.

---

### 2.3 CRAG — Corrective RAG

CRAG (Yan et al., 2024) introduit un **évaluateur de pertinence post-retrieval** qui agit comme un filtre correcteur :

```
Retrieval initial
      ↓
[ÉVALUATEUR]
  ├─ Score > 0.7 → RELEVANT   → génération directe
  ├─ Score 0.3-0.7 → AMBIGUOUS → combine contexte original + contexte reformulé
  └─ Score < 0.3 → IRRELEVANT → reformule la requête + nouveau retrieval
                                 (voire web search externe)
```

**Ce qui distingue CRAG de ReAct :** CRAG est plus ciblé — il ne boucle pas sur l'ensemble du pipeline, mais corrige spécifiquement la qualité du contexte récupéré avant de générer.

**Pertinence pour ce projet :** très haute. Avec 5 collections ChromaDB distinctes (verbatims, résumés, scores, géo, entretiens), il est fréquent qu'une requête mal formulée atterrisse dans la mauvaise collection. Le CRAG gate permettrait de détecter ce cas et de reformuler automatiquement.

---

### 2.4 FLARE — Forward-Looking Active Retrieval

FLARE (Jiang et al., 2023) adopte une approche radicalement différente : au lieu de récupérer du contexte *avant* de générer, il déclenche le retrieval *pendant* la génération.

```
Génération token par token :
  "Le score OppChoVec d'Ajaccio est de [INCERTITUDE DÉTECTÉE]"
       → Probabilité faible sur le prochain token
       → Déclenche un retrieval : score_lookup("Ajaccio", "OppChoVec global")
       → Résultat injecté : "5.78"
  → Continue : "Le score OppChoVec d'Ajaccio est de 5.78, ce qui..."
```

**Avantage :** le retrieval est chirurgical — déclenché exactement au moment où le LLM hésite, sur le fragment précis dont il a besoin.

**Limite :** nécessite un accès aux probabilités de tokens du LLM (logprobs), ce qui n'est possible qu'avec OpenAI API (param `logprobs=True`) ou des modèles locaux. Claude et Mistral ne l'exposent pas actuellement.

**Applicable ici :** uniquement avec GPT-4o/GPT-4-turbo via l'option `logprobs`.

---

### 2.5 Adaptive RAG

L'Adaptive RAG (Jeong et al., 2024) ajoute une couche de **classification de la complexité de la question** en amont du pipeline, afin de router vers la stratégie la moins coûteuse suffisante :

```
Question
  ↓
[CLASSIFIER] (petit LLM ou modèle fine-tuné)
  ├─ Simple / factuelle       → RAG direct (1 retrieval, 1 génération)
  ├─ Complexe / multi-aspects → ReAct multi-steps
  └─ Hors domaine             → Refus explicite (pas de retrieval)
```

**Pertinence pour ce projet :** très forte. Les 103 questions d'évaluation couvrent des cas très hétérogènes : questions binaires simples ("Ajaccio a-t-il un score supérieur à Bastia ?"), questions analytiques complexes ("Quelles sont les dimensions où les agriculteurs de moins de 35 ans se distinguent ?") et questions hors domaine ("Quel est le PIB de la Corse ?"). Un classifier en amont éviterait de lancer une boucle ReAct coûteuse sur une question à réponse binaire immédiate.

---

### 2.6 Multi-Agent RAG

Dans une architecture multi-agents, chaque agent est spécialisé dans un rôle précis et peut communiquer avec les autres :

```
[ORCHESTRATEUR] — reçoit la question, délègue
      ├─ [AGENT PLANNER]    — décompose, planifie les étapes
      ├─ [AGENT RETRIEVER]  — expert en accès aux sources (outils)
      ├─ [AGENT CRITIC]     — évalue la qualité du contexte/réponse
      └─ [AGENT SYNTHESIZER] — agrège et formate la réponse finale
```

Chaque agent peut avoir sa propre mémoire, ses propres outils, et son propre LLM. La communication peut être synchrone (chaîne) ou asynchrone (graph de messages).

**Frameworks associés :** AutoGen (Microsoft), CrewAI, LangGraph multi-agent.

**Limite :** la complexité de coordination entre agents peut dépasser le gain en qualité pour la plupart des cas d'usage simples. Le multi-agent est pertinent quand les agents ont réellement des domaines de compétence distincts.

**Note :** la v10 de ce projet est déjà un système multi-agents séquentiel (Mistral Large Planner → Claude Haiku Answerer × N → Mistral Large Synthesizer). La v11 irait plus loin en rendant l'orchestration dynamique et itérative.

---

## 3. Frameworks et technologies

### 3.1 LangGraph

**LangGraph** (LangChain, 2024) modélise le pipeline RAG comme un **graphe d'états** orienté :

- **Nœuds** : fonctions Python (retrieval, évaluation, génération…)
- **Arêtes** : transitions conditionnelles entre nœuds (ex. `if score < 0.4 → nœud reformulation`)
- **État** : dictionnaire partagé et muté à chaque nœud (question, contexte accumulé, historique des appels d'outils, score…)
- **Checkpointing** : sauvegarde de l'état à chaque nœud pour reprise ou debug

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("planner", plan_retrieval)
graph.add_node("retriever", execute_tool_call)
graph.add_node("evaluator", score_context)
graph.add_node("generator", generate_answer)

graph.add_conditional_edges("evaluator", route_on_score, {
    "sufficient": "generator",
    "insufficient": "retriever",
    "max_iterations": "generator",
})
```

**Points forts :** très expressif pour des workflows complexes avec boucles, branches, parallélisme. Compatible avec tous les LLMs via LangChain.

**Limite ici :** dépendance à LangChain (déjà présente pour BM25), mais ajoute une couche d'abstraction supplémentaire. Overhead de configuration notable.

---

### 3.2 LlamaIndex AgentRunner / ReActAgent

LlamaIndex expose une abstraction `ReActAgent` qui encapsule la boucle ReAct autour de n'importe quel ensemble d'outils :

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

verbatim_tool = QueryEngineTool.from_defaults(
    query_engine=verbatim_engine,  # VectorStoreIndex.as_query_engine()
    description="Recherche dans les verbatims citoyens"
)
summary_tool = QueryEngineTool.from_defaults(
    query_engine=raptor_engine,
    description="Recherche dans les résumés hiérarchiques RAPTOR"
)

agent = ReActAgent.from_tools([verbatim_tool, summary_tool], llm=llm, verbose=True)
response = agent.chat("Quels aspects de la santé ressortent chez les agriculteurs ?")
```

**Pertinence pour ce projet :** forte. Les `QueryEngineTool` sont déjà utilisés en v7/v8. La migration vers `ReActAgent` consiste essentiellement à ajouter la boucle autour de ces outils existants.

**Limite :** l'implémentation ReAct de LlamaIndex peut être fragile avec des LLMs non-GPT (parsing du format Thought/Action/Observation).

---

### 3.3 Anthropic Tool Use API (natif)

L'API Anthropic supporte nativement le **tool use** (function calling) depuis Claude 3 :

```python
tools = [
    {
        "name": "verbatim_search",
        "description": "Recherche dans les verbatims citoyens avec filtres démographiques",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "commune": {"type": "string"},
                        "age_range": {"type": "string"},
                        "profession": {"type": "string"},
                        "dimension": {"type": "string"}
                    }
                },
                "k": {"type": "integer", "default": 7}
            },
            "required": ["query"]
        }
    }
    # ... autres outils
]

# Boucle ReAct en ~25 lignes
messages = [{"role": "user", "content": question}]
while True:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        tools=tools,
        messages=messages
    )
    if response.stop_reason == "end_turn":
        break
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            ]})
            break
```

**Avantages décisifs pour ce projet :**
- `anthropic` est **déjà installé** (utilisé pour Claude Haiku en v10)
- Aucun nouveau framework à installer
- Claude Sonnet 4.6 est l'un des meilleurs modèles disponibles pour le tool use structuré
- Zéro dépendance supplémentaire

---

### 3.4 DSPy

DSPy (Stanford, 2023) propose une approche **déclarative** : on définit des signatures (entrée → sortie) et des modules (ChainOfThought, ReAct, MIPRO), et un optimiseur compile automatiquement les prompts sur un dataset d'exemples.

```python
class RAGSignature(dspy.Signature):
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField()

rag = dspy.ChainOfThought(RAGSignature)
optimized_rag = dspy.MIPROv2().compile(rag, trainset=examples)
```

**Limite pour ce projet :** l'optimisation nécessite un dataset labellisé (le pipeline d'évaluation existe, mais l'intégration demanderait un travail spécifique). DSPy est excellent pour optimiser des prompts à grande échelle, moins pour des pipelines de retrieval complexes avec de nombreux outils.

---

### 3.5 OpenAI Assistants API

L'API Assistants d'OpenAI (2023) offre des threads persistants, la gestion automatique de la mémoire, et un code interpreter intégré. La boucle `run → retrieve steps → submit tool outputs` est gérée côté serveur.

**Limite ici :** faible contrôle sur le retrieval (ChromaDB local n'est pas directement intégrable), et la logique CRAG nécessite une instrumentation que l'API Assistants n'expose pas facilement.

---

### 3.6 AutoGen (Microsoft)

AutoGen définit des agents conversationnels qui s'envoient des messages. Un `UserProxyAgent` peut appeler des fonctions locales ; un `AssistantAgent` génère des réponses.

**Pertinence :** faible pour ce projet. AutoGen brille dans des scénarios où plusieurs agents spécialisés doivent collaborer de façon non déterministe (ex. un agent code + un agent test + un agent review). Pour un pipeline RAG avec boucle de retrieval, c'est une surcouche inutile.

---

### Synthèse comparative

| Framework | Complexité | Tool Use | Boucle ReAct | CRAG natif | Pertinence v11 |
|-----------|------------|----------|--------------|------------|----------------|
| Anthropic Tool Use natif | Faible | ✅ natif | ✅ manuel | À implémenter | **★★★★★** |
| LangGraph | Moyenne | ✅ via LangChain | ✅ natif | ✅ natif | ★★★★☆ |
| LlamaIndex ReActAgent | Moyenne | ✅ natif | ✅ natif | Partiel | ★★★☆☆ |
| DSPy | Forte | Partiel | ✅ natif | Non | ★★☆☆☆ |
| AutoGen | Forte | ✅ natif | Complexe | Non | ★☆☆☆☆ |

**Recommandation : Anthropic Tool Use natif**, sans framework intermédiaire. Zéro nouvelle dépendance, contrôle total sur la boucle, Claude Sonnet déjà accessible dans le projet.

---

## 4. Analyse de l'existant — ce qui est déjà agentique

### 4.1 Ce que v10 a déjà

La v10 (`rag_v10_raptor_subq.py`) est en réalité un **système multi-agents linéaire** — souvent présenté comme le niveau 2 du continuum agentique :

```
Question
  ↓
[Agent Planner — Mistral Large]
  → Décompose en N sous-questions (1-8, JSON structuré)
  → Raisonnement explicite par sous-question
  ↓
[Agent Answerer — Claude Haiku × N, en parallèle]
  → Pour chaque sous-question :
     - RaptorRetriever.query() comme outil de retrieval
     - Score de pertinence calculé
     - Filtrage des sous-réponses à score faible
  ↓
[Agent Synthesizer — Mistral Large]
  → Fusion des sous-réponses
  → Élimination des redondances et contradictions
  → Réponse finale cohérente
```

La v10 implémente donc déjà :
- ✅ Décomposition de question (Planning)
- ✅ Retrieval par sous-tâche (outil RaptorRetriever)
- ✅ Scoring de pertinence par réponse partielle
- ✅ Synthèse multi-sources
- ✅ Filtrage des réponses de mauvaise qualité
- ✅ Multi-LLM orchestration (3 modèles différents)

En v7/v8, on trouve également :
- ✅ Routing LLM entre moteurs (vecteur vs graphe) via `RouterQueryEngine`
- ✅ Pattern QueryEngineTool (outil de retrieval comme fonction)

En v2.2 :
- ✅ Extraction automatique de filtres depuis la question (`PortraitDetector`)

En v9 :
- ✅ Détection de type de question (`_is_oppchovec_question`) → enrichissement conditionnel

### 4.2 Ce qui manque encore

| Capacité | v10 | v11 visée |
|----------|-----|-----------|
| Pipeline | Linéaire et figé | Boucle itérative dynamique |
| Sélection d'outils | Fixe (uniquement RaptorRetriever) | Dynamique parmi 5 outils |
| Auto-évaluation du contexte | Non | CRAG gate après chaque retrieval |
| Reformulation de requête | Non | Automatique si score < seuil |
| Mémoire de session | Non | Context accumulation dans la boucle |
| Fast path (questions simples) | Non | Classifier → bypass agentique |
| Critique post-génération | Non | Optionnel (Mistral Large) |

---

## 5. Design de la v11 — Agentic RAG

### 5.1 Architecture globale

La v11 propose un **pipeline ReAct + CRAG hybride**, avec un fast path pour les questions simples :

```
Question utilisateur
      ↓
┌─────────────────────────────────────────────────────┐
│  [CLASSIFICATEUR]  (regex + Mistral Small)          │
│  question_type : quanti | quali | comparatif | géo  │
│  complexity    : simple | complex | hors_domaine    │
└─────────────┬──────────────────┬────────────────────┘
              │ simple           │ complex
              ▼                  ▼
    ┌──────────────┐   ┌───────────────────────────────────────────┐
    │  FAST PATH   │   │  [PLANNER]  Mistral Large                 │
    │  v9 direct   │   │  → Plan JSON : liste ordonnée d'outils    │
    │  ~4-8s       │   │  → Estimation nb itérations nécessaires   │
    └──────────────┘   └────────────────┬──────────────────────────┘
                                        ↓
                       ┌────────────────────────────────────────────┐
                       │  [AGENT EXECUTOR]  Claude Sonnet 4.6       │
                       │  Boucle ReAct  (max 5 itérations)          │
                       │                                            │
                       │  ┌─ THINK : contexte suffisant ?          │
                       │  ├─ ACT   : appel outil                   │
                       │  │   verbatim_search(query, filters, k)   │
                       │  │   summary_search(query, view, k)       │
                       │  │   score_lookup(commune, indicator)     │
                       │  │   geo_neighbors(commune, radius)       │
                       │  │   decompose(question)                  │
                       │  ├─ OBSERVE : contexte retourné           │
                       │  └─ [CRAG GATE]                           │
                       │      score < 0.4  → reformuler + retry    │
                       │      score 0.4-0.7 → continuer            │
                       │      score > 0.7  → suffisant             │
                       └────────────────┬───────────────────────────┘
                                        ↓
                       ┌────────────────────────────────────────────┐
                       │  [GÉNÉRATION]  Claude Sonnet               │
                       │  Réponse avec citations des sources        │
                       └────────────────┬───────────────────────────┘
                                        ↓
                       ┌────────────────────────────────────────────┐
                       │  [CRITIQUE]  Mistral Large  (optionnel)    │
                       │  Score 1-5 sur : pertinence, fondement,   │
                       │  nuance, cohérence quali/quanti            │
                       │  Si score < 3 → re-retrieval ciblé        │
                       └────────────────┬───────────────────────────┘
                                        ↓
                                 Réponse finale
```

### 5.2 Les 5 outils disponibles

| Outil | Paramètres | Implémentation réutilisée | Collection |
|-------|------------|--------------------------|------------|
| `verbatim_search` | `query`, `filters` (commune, âge, profession, dimension), `k` | `PortraitHybridRetriever` (v2.2) + `PortraitDetector` | `portrait_verbatims` |
| `summary_search` | `query`, `view` (1D/2D), `k` | `RaptorRetriever.query()` (v9) | `raptor_summaries` |
| `score_lookup` | `commune`, `indicator` (global/opp/cho/vec) | `query_oppchovec()` (API v9/v10) | `oppchovec_scores` |
| `geo_neighbors` | `commune`, `radius_km` | Nouveau — requête `communes_geo` par EPCI/coords | `communes_geo` |
| `decompose` | `question` | Nouveau — appel Mistral Large (logique v10 Planner) | — |

**Exemple de trace d'exécution** pour "Comment les personnes âgées vivant dans les communes rurales perçoivent-elles les transports ?" :

```
THINK: Question complexe, deux dimensions : démographique (personnes âgées) + géographique (rural) + thématique (transports)

ACT: geo_neighbors(commune="toutes", radius_km=0, epci_type="rural")
OBSERVE: [liste des communes rurales corses]

CRAG: score=0.6 (contexte géo ok, mais pas encore de verbatims)

ACT: verbatim_search(
  query="transports mobilité déplacements",
  filters={"age_range": "65+", "communes": ["Zonza", "Sartène", ...], "dimension": "mobilité"},
  k=8
)
OBSERVE: [7 verbatims pertinents sur la mobilité des seniors en milieu rural]

CRAG: score=0.82 → SUFFISANT

GENERATE: "Les personnes âgées des communes rurales expriment [...]"
```

### 5.3 Le CRAG gate

Le CRAG gate est implémenté comme une évaluation BGE-reranker entre la question et chaque document récupéré :

```python
def crag_score(question: str, docs: list[str]) -> float:
    """Retourne un score de pertinence agrégé [0, 1]."""
    if not docs:
        return 0.0
    scores = reranker.predict([(question, doc) for doc in docs])
    return float(np.mean(scores))
```

Le reranker BGE-reranker-v2-m3 est **déjà chargé** en mémoire pour les versions v2.2 et v6. Sa réutilisation est donc sans coût supplémentaire en RAM ni en latence de chargement.

Seuils :
- `< 0.4` : contexte non pertinent → reformulation automatique de la requête + appel d'un autre outil
- `0.4 – 0.7` : pertinent mais incomplet → continuer la boucle pour enrichir
- `> 0.7` : suffisant → passer à la génération

### 5.4 Le classificateur de complexité

Un simple classifier en deux temps :
1. **Regex rapide** : détecter les questions binaires, les requêtes de score chiffré, les hors-domaine évidents
2. **Mistral Small** (si ambigu) : classer en `simple | complex | hors_domaine`

Ce classifier n'ajoute que ~0.3s de latence, mais évite de lancer une boucle ReAct complète (5-15s) sur une question à laquelle v9 répond en 2s.

---

## 6. Faisabilité et roadmap d'implémentation

### 6.1 Ce qui est acquis (réutilisation directe)

| Composant | Fichier source | Rôle en v11 |
|-----------|---------------|-------------|
| `RaptorRetriever` | `rag_v9_raptor.py` | Outil `summary_search` |
| `PortraitHybridRetriever` + `PortraitDetector` | `rag_v2_2_portrait.py` | Outil `verbatim_search` + filtres auto |
| `query_oppchovec()` | `api_server_multi_version.py` | Outil `score_lookup` |
| `BGE-reranker-v2-m3` | `rag_v2_2_portrait.py` | CRAG gate |
| `anthropic.Anthropic()` + Claude Haiku | `rag_v10_raptor_subq.py` | Executor → passer à Claude Sonnet |
| `Mistral Large` | `rag_v10_raptor_subq.py` | Planner + Critic |
| Route `/api/query` + dispatch par version | `api_server_multi_version.py` | Ajouter `v11` |
| Dropdown version dans le frontend | `example_frontend_multi_version.html` | Rien à faire |
| Pipeline d'évaluation | `eval_from_excel.py` | Évaluer v11 sans modification |
| Streaming SSE | `api_server_multi_version.py` | Streamer les étapes de la boucle |

### 6.2 Ce qui est à créer

**Fichier principal : `rag_v11_agentic.py`** (~350 lignes)

```
Classe AgenticRAGPipeline :
  __init__()
    - Charger RaptorRetriever (réutilisé de v9)
    - Charger PortraitHybridRetriever (réutilisé de v2.2)
    - Charger BGE-reranker (réutilisé)
    - Initialiser clients Anthropic + Mistral

  classify_question(question) → {type, complexity}
    - Regex patterns + Mistral Small si ambigu

  execute_tool(name, params) → str
    - Switch sur les 5 outils
    - Appel vers retrievers existants

  crag_score(question, docs) → float
    - BGE-reranker agrégé

  run_react_loop(question, plan) → {context, tool_calls, iterations}
    - Boucle Anthropic tool use (max 5)
    - CRAG gate après chaque observation

  query(question, **kwargs) → (answer, sources, metadata)
    - Fast path si simple
    - Plan → ReAct loop → génération → critique optionnelle
```

**Intégration API : `api_server_multi_version.py`** (~35 lignes supplémentaires)
- Ajout `"v11": None` dans `rag_pipelines`
- Initialisation dans lifespan : `AgenticRAGPipeline(...)`
- Branche `elif rag_version == "v11"` dans `query_rag()`
- La réponse inclut `tool_calls` (liste des outils appelés) + `iterations` dans `metadata`

**Frontend** : rien à faire. La version v11 apparaîtra automatiquement dans le dropdown dès qu'elle est déclarée dans l'API (`/api/versions`). Le tab "Sous-questions" du panneau de résultats peut afficher les `tool_calls`.

### 6.3 Risques et mitigations

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Latence élevée (boucle × N LLM calls) | Moyen | Haute | Fast path pour questions simples ; cap strict à 5 itérations |
| Coût LLM multiplié | Moyen | Haute | Claude Haiku pour CRAG gate (scoring) au lieu de Sonnet |
| Hallucination dans sélection d'outils | Fort | Moyenne | JSON Schema strict ; validation Pydantic des paramètres |
| Boucle infinie (LLM choisit toujours le même outil) | Fort | Faible | Historique des appels dans le contexte ; ban d'outil après 2 appels identiques |
| Dépassement de fenêtre contextuelle | Fort | Moyenne | Résumé du contexte accumulé toutes les 3 itérations |
| Instabilité sur Windows (GIL) | Faible | Moyenne | Même pattern que v10 (sync def dans anyio) |
| Chargement mémoire (2 retrievers simultanés) | Faible | Faible | Partage des instances déjà chargées pour v2.2 et v9 |

### 6.4 Comparatif v10 vs v11

| Critère | v10 (Sub-questions) | v11 (Agentic ReAct+CRAG) |
|---------|--------------------|-----------------------------|
| **Type de pipeline** | Linéaire, fixe (3 phases) | Boucle dynamique (N phases) |
| **Sélection d'outils** | Fixe — uniquement RaptorRetriever | Dynamique — 5 outils selon besoin |
| **Auto-correction retrieval** | Non | Oui — CRAG gate après chaque observation |
| **Reformulation de requête** | Non | Automatique si score < 0.4 |
| **Mémoire inter-étapes** | Partielle (sub-answers passées au synthesizer) | Complète (context window accumulé) |
| **Fast path questions simples** | Non — toujours 3 phases | Oui — bypass via v9 direct |
| **Latence moyenne** | ~10-20s | ~6-10s (simple) / ~18-35s (complex) |
| **Coût LLM** | Moyen (3 appels minimum) | Moyen-élevé (5-10 appels max) |
| **Pertinence attendue** | Haute | Très haute sur questions complexes |
| **Explicabilité** | Bonne (sous-questions visibles) | Excellente (tool_calls traçables) |
| **Complexité d'implémentation** | Faible | Moyenne (~350 lignes) |

### 6.5 Estimation d'effort

| Tâche | Effort estimé |
|-------|--------------|
| `rag_v11_agentic.py` — squelette + outils | 3-4h |
| Boucle ReAct + CRAG gate | 2-3h |
| Classificateur de complexité | 1h |
| Intégration API (`api_server_multi_version.py`) | 1h |
| Tests sur les 103 questions d'évaluation | 2-3h |
| **Total** | **~1 journée** |

---

## 7. Conclusion et recommandation

### Ce que l'architecture existante a accompli

En 10 versions successives, ce projet a parcouru l'essentiel du spectre des architectures RAG modernes :

- **v1-v2** : fondations solides (hybride dense+sparse, reranking cross-encoder)
- **v2.2** : enrichissement sémantique par profil démographique (unique dans la littérature pour ce domaine)
- **v5-v6** : intégration de graphes de connaissance (Neo4j + GNN)
- **v7-v8** : routing LLM multi-moteurs (premiers patterns agentiques)
- **v9** : RAPTOR hiérarchique original (vues 1D/2D pré-calculées)
- **v10** : orchestration multi-LLM (Mistral + Claude + GPT), véritable pipeline multi-agents linéaire

**La v10 représente déjà un niveau de sophistication que peu de projets académiques ou industriels atteignent** pour un corpus de cette taille.

### Ce que la v11 apporterait

La v11 comblerait précisément ce qui manque à la v10 : la **rétroaction**. Aujourd'hui, si RaptorRetriever ne trouve rien d'utile pour une sous-question, le pipeline continue quand même — et le synthesizer produit une réponse fondée sur un contexte insuffisant.

La v11 détecterait ce cas (CRAG gate), reformulerait la requête, essaierait un autre outil (`verbatim_search` au lieu de `summary_search`, par exemple), et ne générerait la réponse finale que lorsque le contexte est jugé suffisant.

### Recommandation technique

1. **Implémenter la v11 avec Anthropic Tool Use natif** — zéro nouvelle dépendance, contrôle total sur la boucle, Claude Sonnet 4.6 comme exécuteur.
2. **Ne pas installer LangGraph** dans un premier temps — la complexité supplémentaire n'est pas justifiée pour ce nombre d'outils. LangGraph devient pertinent si le graphe d'états dépasse 5-6 nœuds avec de nombreuses conditions.
3. **Conserver tous les fichiers existants** (v1-v10, API, évaluation, frontend) — la v11 est un ajout, pas un remplacement.
4. **Commencer par le fast path** : un classifier simple qui route les questions binaires vers v9 directement représente déjà 80% du gain en latence pour un coût d'implémentation faible.
5. **Utiliser le pipeline d'évaluation existant** (`eval_from_excel.py`) pour mesurer le gain v10 → v11 sur les 103 questions de référence.

### Positionnement scientifique

Dans le contexte d'une thèse, la v11 permettrait de positionner le travail sur l'**Agentic RAG appliqué aux données de bien-être citoyen** — un angle encore peu exploré dans la littérature. Les contributions originales restent la combinaison :
- RAPTOR multi-vues démographiques×thématiques (v9)
- Multi-LLM orchestration spécialisée (v10)
- Agentic loop avec 5 outils hétérogènes sur corpus citoyen multilingue (v11)

---

*Rapport généré le 7 avril 2026 — Claude Code (claude-sonnet-4-6)*
*Basé sur l'exploration complète de l'architecture v1-v10 du projet*
