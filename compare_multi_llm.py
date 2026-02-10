"""
Comparaison multi-modèles LLM avec RAG v2.2
Compare ChatGPT, Grok, Kimi, Claude et Mistral sur 5 questions
"""

import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Questions à tester
QUESTIONS = [
    "Comment définir le bien-être ?",
    "Y a-t-il un écart entre les scores objectifs et les ressentis des habitants d'Ajaccio ?",
    "Y a-t-il une station de ski à Bastia ?",
    "Les retours sur le bien-être à Ajaccio varient-ils selon les tranches d'âge ?",
    "Quelles dimensions du bien-être sont jugées les plus importantes à Ajaccio ?"
]

# Configuration des modèles par provider
MODELS_CONFIG = {
    "OpenAI": {
        "api_key_env": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "base_url": None
    },
    "Grok": {
        "api_key_env": "GROK_API_KEY",
        "models": ["grok-3", "grok-3-mini"],
        "base_url": "https://api.x.ai/v1"
    },
    "Kimi": {
        "api_key_env": "KIMIK2_API_KEY",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k"],
        "base_url": "https://api.moonshot.ai/v1"
    },
    "Claude": {
        "api_key_env": "CLAUDE_API_KEY",
        "models": ["claude-sonnet-4-20250514", "claude-3-haiku-20240307"],
        "base_url": None,
        "use_anthropic": True
    },
    "Mistral": {
        "api_key_env": "MISTRAL_API_KEY",
        "models": ["mistral-large-latest", "mistral-small-latest"],
        "base_url": "https://api.mistral.ai/v1"
    }
}

def call_openai_compatible(api_key: str, model: str, prompt: str, system_prompt: str, base_url: Optional[str] = None) -> str:
    """Appelle une API compatible OpenAI"""
    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content

def call_anthropic(api_key: str, model: str, prompt: str, system_prompt: str) -> str:
    """Appelle l'API Anthropic (Claude)"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text

def get_rag_context(question: str, k: int = 5) -> tuple:
    """Récupère le contexte RAG pour une question"""
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Charger le modèle d'embeddings
    model = SentenceTransformer("BAAI/bge-m3")

    # Connexion ChromaDB
    client = chromadb.PersistentClient(path="./chroma_portrait")
    collection = client.get_collection("portrait_verbatims")

    # Encoder la question
    query_embedding = model.encode(f"query: {question}").tolist()

    # Recherche
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )

    # Formater le contexte
    context_parts = []
    sources = []

    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        context_parts.append(f"[Source {i}] {doc[:300]}...")
        sources.append({
            "rank": i,
            "commune": meta.get("nom", "N/A"),
            "genre": meta.get("genre", "N/A"),
            "age": meta.get("age_exact", "N/A"),
            "profession": meta.get("profession", "N/A"),
            "dimension": meta.get("dimension", "N/A"),
            "extrait": doc[:200] + "..."
        })

    context = "\n\n".join(context_parts)
    return context, sources

def build_prompt(question: str, context: str) -> tuple:
    """Construit le prompt système et utilisateur"""
    system_prompt = """Tu es un assistant spécialisé dans l'analyse de données sur la qualité de vie en Corse.
Tu as accès à des verbatims d'habitants avec leurs profils démographiques (âge, genre, profession).
Base tes réponses UNIQUEMENT sur les informations fournies.
Si les données sont insuffisantes, indique-le clairement.
Réponds de manière concise et factuelle."""

    user_prompt = f"""=== CONTEXTE (verbatims d'habitants) ===
{context}

=== QUESTION ===
{question}

Réponds à la question en te basant sur le contexte ci-dessus."""

    return system_prompt, user_prompt

def test_model(provider: str, model: str, question: str, context: str, sources: List[Dict]) -> Dict[str, Any]:
    """Teste un modèle sur une question"""
    config = MODELS_CONFIG[provider]
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        return {
            "provider": provider,
            "model": model,
            "question": question,
            "answer": f"ERREUR: Clé API {config['api_key_env']} non trouvée",
            "time_seconds": 0,
            "sources": [],
            "error": True
        }

    system_prompt, user_prompt = build_prompt(question, context)

    start_time = time.time()

    try:
        if config.get("use_anthropic"):
            answer = call_anthropic(api_key, model, user_prompt, system_prompt)
        else:
            answer = call_openai_compatible(api_key, model, user_prompt, system_prompt, config.get("base_url"))

        elapsed = time.time() - start_time

        return {
            "provider": provider,
            "model": model,
            "question": question,
            "answer": answer,
            "time_seconds": round(elapsed, 2),
            "sources": sources,
            "error": False
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "provider": provider,
            "model": model,
            "question": question,
            "answer": f"ERREUR: {str(e)}",
            "time_seconds": round(elapsed, 2),
            "sources": [],
            "error": True
        }

def run_comparison():
    """Exécute la comparaison complète"""
    print("="*70)
    print("COMPARAISON MULTI-MODÈLES LLM")
    print("="*70)

    all_results = []

    # Pré-charger le modèle d'embeddings une seule fois
    print("\nChargement du modèle d'embeddings...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("BAAI/bge-m3")
    print("OK")

    for q_idx, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"QUESTION {q_idx}/{len(QUESTIONS)}: {question[:60]}...")
        print("="*70)

        # Récupérer le contexte RAG une fois par question
        print("Récupération du contexte RAG...")
        context, sources = get_rag_context(question)
        print(f"OK - {len(sources)} sources récupérées")

        # Tester chaque provider et modèle
        for provider, config in MODELS_CONFIG.items():
            for model in config["models"]:
                print(f"\n  Testing {provider}/{model}...", end=" ", flush=True)

                result = test_model(provider, model, question, context, sources)
                all_results.append(result)

                if result["error"]:
                    print(f"ERREUR ({result['time_seconds']}s)")
                else:
                    print(f"OK ({result['time_seconds']}s)")

    return all_results

def export_to_excel(results: List[Dict], output_path: str):
    """Exporte les résultats en Excel"""
    print(f"\nExport vers {output_path}...")

    # Feuille 1: Résumé par question
    summary_data = []
    for q in QUESTIONS:
        row = {"Question": q}
        q_results = [r for r in results if r["question"] == q]
        for r in q_results:
            col_name = f"{r['provider']}/{r['model']}"
            if r["error"]:
                row[col_name] = f"ERREUR ({r['time_seconds']}s)"
            else:
                row[col_name] = f"{r['time_seconds']}s"
        summary_data.append(row)
    df_summary = pd.DataFrame(summary_data)

    # Feuille 2: Réponses complètes
    responses_data = []
    for r in results:
        responses_data.append({
            "Question": r["question"][:50] + "...",
            "Provider": r["provider"],
            "Modèle": r["model"],
            "Temps (s)": r["time_seconds"],
            "Erreur": "Oui" if r["error"] else "Non",
            "Réponse": r["answer"][:1000] if r["answer"] else "N/A"
        })
    df_responses = pd.DataFrame(responses_data)

    # Feuille 3: Sources utilisées (une par question)
    sources_data = []
    seen_questions = set()
    for r in results:
        if r["question"] not in seen_questions and r["sources"]:
            seen_questions.add(r["question"])
            for s in r["sources"]:
                sources_data.append({
                    "Question": r["question"][:50] + "...",
                    "Rang": s["rank"],
                    "Commune": s["commune"],
                    "Genre": s["genre"],
                    "Âge": s["age"],
                    "Profession": s["profession"],
                    "Dimension": s["dimension"],
                    "Extrait": s["extrait"]
                })
    df_sources = pd.DataFrame(sources_data)

    # Feuille 4: Comparaison détaillée par question
    comparison_sheets = {}
    for q in QUESTIONS:
        q_results = [r for r in results if r["question"] == q]
        q_data = []
        for r in q_results:
            q_data.append({
                "Provider": r["provider"],
                "Modèle": r["model"],
                "Temps (s)": r["time_seconds"],
                "Réponse complète": r["answer"] if not r["error"] else f"ERREUR: {r['answer']}"
            })
        comparison_sheets[q[:25]] = pd.DataFrame(q_data)

    # Écrire le fichier Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Résumé_temps', index=False)
        df_responses.to_excel(writer, sheet_name='Toutes_réponses', index=False)
        df_sources.to_excel(writer, sheet_name='Sources_RAG', index=False)

        # Ajouter une feuille par question
        for i, (q_name, df_q) in enumerate(comparison_sheets.items(), 1):
            sheet_name = f"Q{i}_{q_name[:20]}"
            df_q.to_excel(writer, sheet_name=sheet_name, index=False)

        # Ajuster largeur colonnes
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = max(len(str(cell.value or "")) for cell in column)
                adjusted_width = min(max_length + 2, 80)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    print(f"OK - Fichier exporté: {output_path}")

def main():
    # Exécuter la comparaison
    results = run_comparison()

    # Exporter en Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"comparaisons_rag/comparison_multi_llm_{timestamp}.xlsx"
    os.makedirs("comparaisons_rag", exist_ok=True)

    export_to_excel(results, output_path)

    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ FINAL")
    print("="*70)

    # Temps moyen par provider
    providers_times = {}
    for r in results:
        if not r["error"]:
            if r["provider"] not in providers_times:
                providers_times[r["provider"]] = []
            providers_times[r["provider"]].append(r["time_seconds"])

    print("\nTemps moyen par provider:")
    for provider, times in sorted(providers_times.items(), key=lambda x: sum(x[1])/len(x[1])):
        avg = sum(times) / len(times)
        print(f"  {provider}: {avg:.2f}s (sur {len(times)} réponses)")

    # Erreurs
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"\nErreurs ({len(errors)}):")
        for e in errors:
            print(f"  - {e['provider']}/{e['model']}: {e['answer'][:50]}...")

    print(f"\nRésultats exportés dans: {output_path}")

if __name__ == "__main__":
    main()
