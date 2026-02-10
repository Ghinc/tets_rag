"""
Script de comparaison RAG v2 vs v2.2 (Portrait)
Compare les réponses et extraits pour une question sur les jeunes
"""

import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Question de test
QUESTION = "Quelles dimensions du bien-être sont importantes pour les jeunes ?"

def test_rag_v2():
    """Test avec RAG v2 (sans filtrage portrait) - utilise chroma_portrait sans filtre"""
    print("\n" + "="*60)
    print("TEST RAG v2.2 SANS FILTRE (baseline)")
    print("="*60)

    try:
        from rag_v2_2_portrait import PortraitRAGPipeline

        rag = PortraitRAGPipeline(
            chroma_path="./chroma_portrait",
            collection_name="portrait_verbatims",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            llm_model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            quant_data_path="df_mean_by_commune.csv"
        )

        print(f"\nQuestion: {QUESTION}")
        # Requête SANS auto-détection des filtres portrait
        answer, results, _ = rag.query(
            question=QUESTION,
            k=5,
            use_reranking=True,
            include_quantitative=False,
            auto_detect_filters=False  # PAS de filtrage par âge
        )

        return {
            "version": "v2.2 (sans filtre)",
            "answer": answer,
            "results": results,
            "error": None
        }

    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return {
            "version": "v2.2 (sans filtre)",
            "answer": None,
            "results": [],
            "error": str(e)
        }

def test_rag_v2_2():
    """Test avec RAG v2.2 (avec filtrage portrait pour jeunes)"""
    print("\n" + "="*60)
    print("TEST RAG v2.2 (Portrait - filtrage jeunes)")
    print("="*60)

    try:
        from rag_v2_2_portrait import PortraitRAGPipeline

        rag = PortraitRAGPipeline(
            chroma_path="./chroma_portrait",
            collection_name="portrait_verbatims",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            llm_model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            quant_data_path="df_mean_by_commune.csv"
        )

        print(f"\nQuestion: {QUESTION}")
        answer, results, detected_filters = rag.query(
            question=QUESTION,
            k=5,
            use_reranking=True,
            include_quantitative=False,
            auto_detect_filters=True  # Auto-détection des "jeunes"
        )

        print(f"Filtres détectés: {detected_filters}")

        return {
            "version": "v2.2",
            "answer": answer,
            "results": results,
            "filters": detected_filters,
            "error": None
        }

    except Exception as e:
        print(f"Erreur RAG v2.2: {e}")
        import traceback
        traceback.print_exc()
        return {
            "version": "v2.2",
            "answer": None,
            "results": [],
            "error": str(e)
        }

def export_to_excel(results_v2, results_v2_2, output_path):
    """Exporte les résultats dans un fichier Excel"""

    # Feuille 1: Résumé des réponses
    summary_data = []

    for result in [results_v2, results_v2_2]:
        summary_data.append({
            "Version RAG": result["version"],
            "Question": QUESTION,
            "Réponse": result["answer"] if result["answer"] else f"ERREUR: {result.get('error', 'Inconnue')}",
            "Nb extraits": len(result.get("results", [])),
            "Filtres détectés": str(result.get("filters", "N/A"))
        })

    df_summary = pd.DataFrame(summary_data)

    # Feuille 2: Extraits RAG v2
    extraits_v2 = []
    for i, r in enumerate(results_v2.get("results", []), 1):
        extraits_v2.append({
            "Rang": i,
            "Score": round(r.score, 4) if hasattr(r, 'score') else 0,
            "Commune": r.metadata.get("nom", r.metadata.get("commune", "N/A")) if hasattr(r, 'metadata') else "N/A",
            "Source": r.metadata.get("source", "N/A") if hasattr(r, 'metadata') else "N/A",
            "Contenu": r.text[:500] + "..." if len(r.text) > 500 else r.text if hasattr(r, 'text') else str(r)
        })
    df_v2 = pd.DataFrame(extraits_v2) if extraits_v2 else pd.DataFrame(columns=["Rang", "Score", "Commune", "Source", "Contenu"])

    # Feuille 3: Extraits RAG v2.2
    extraits_v2_2 = []
    for i, r in enumerate(results_v2_2.get("results", []), 1):
        meta = r.metadata if hasattr(r, 'metadata') else {}
        extraits_v2_2.append({
            "Rang": i,
            "Score": round(r.score, 4) if hasattr(r, 'score') else 0,
            "Commune": meta.get("nom", "N/A"),
            "Genre": meta.get("genre", "N/A"),
            "Âge": meta.get("age_exact", "N/A"),
            "Tranche": meta.get("age_range", "N/A"),
            "Profession": meta.get("profession", "N/A"),
            "Dimension": meta.get("dimension", "N/A"),
            "Contenu": r.text[:500] + "..." if len(r.text) > 500 else r.text if hasattr(r, 'text') else str(r)
        })
    df_v2_2 = pd.DataFrame(extraits_v2_2) if extraits_v2_2 else pd.DataFrame(columns=["Rang", "Score", "Commune", "Genre", "Âge", "Tranche", "Profession", "Dimension", "Contenu"])

    # Écrire dans Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Résumé', index=False)
        df_v2.to_excel(writer, sheet_name='Extraits_RAG_v2', index=False)
        df_v2_2.to_excel(writer, sheet_name='Extraits_RAG_v2.2', index=False)

        # Ajuster largeur colonnes
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = max(len(str(cell.value or "")) for cell in column)
                adjusted_width = min(max_length + 2, 80)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    print(f"\nExcel exporté: {output_path}")

def main():
    print("="*60)
    print("COMPARAISON: Sans filtre vs Avec filtre jeunes")
    print(f"Question: {QUESTION}")
    print("="*60)

    # Tester les deux versions
    results_v2 = test_rag_v2()  # Sans filtre
    results_v2_2 = test_rag_v2_2()  # Avec filtre jeunes

    # Exporter en Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"comparaisons_rag/comparison_portrait_{timestamp}.xlsx"

    # Créer le dossier si nécessaire
    os.makedirs("comparaisons_rag", exist_ok=True)

    export_to_excel(results_v2, results_v2_2, output_path)

    # Afficher un résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)

    print(f"\n--- RAG v2 ---")
    if results_v2["answer"]:
        print(f"Réponse: {results_v2['answer'][:300]}...")
        print(f"Nb extraits: {len(results_v2['results'])}")
    else:
        print(f"Erreur: {results_v2['error']}")

    print(f"\n--- RAG v2.2 (Portrait) ---")
    if results_v2_2["answer"]:
        print(f"Filtres: {results_v2_2.get('filters', {})}")
        print(f"Réponse: {results_v2_2['answer'][:300]}...")
        print(f"Nb extraits: {len(results_v2_2['results'])}")
    else:
        print(f"Erreur: {results_v2_2['error']}")

    print(f"\n\nRésultats exportés dans: {output_path}")

if __name__ == "__main__":
    main()
