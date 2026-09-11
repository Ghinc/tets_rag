"""
test_guard.py — Test unitaire déterministe du HallucinationGuard.

Principe : on fabrique un contexte et une réponse qui contiennent exactement
une violation par règle. Les 3 doivent se déclencher. Aucune ne doit manquer.
Si une règle ne se déclenche pas sur un cas construit pour elle, le guard est cassé.

Exécution :
    python local/test_guard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.hallucination_guard import (
    HallucinationGuard,
    check_entity_silence,
    check_numbers,
    check_quotes,
    build_silence_header,
    Violation,
)


# ── Cas de test ───────────────────────────────────────────────────────────────

# Contexte de synthèse réel : contient SEULEMENT ces chiffres
SYNTH_CONTEXT = """
Enquête Ajaccio : 3,74/5 satisfaction globale. 51 répondants.
Score bien-être : 3,5/5. Emploi : 68 % satisfaits.
"""

# Sources de verbatims : contiennent SEULEMENT ces textes
SOURCES = [
    {"extrait": "La mer est magnifique ici, on apprécie la qualité du cadre naturel."},
    {"extrait": "Les transports en commun sont insuffisants selon les habitants interrogés."},
]

# Entités silencées (pas de données dans le corpus)
SILENCE_LIST = ["Aïti", "Ghisoni"]

# ── Réponse fabriquée avec 3 violations ──────────────────────────────────────
# Violation 1 (silence) : phrase sur Aïti sans marqueur d'absence
# Violation 2 (number)  : chiffre 6,8 absent du SYNTH_CONTEXT
# Violation 3 (quote)   : citation >5 mots inventée, absente des SOURCES

FABRICATED_ANSWER = """\
Le bien-être à Ajaccio est satisfaisant avec un score de 3,74/5.
À Aïti, les habitants expriment une satisfaction modérée face aux services locaux.
Le score environnemental atteint 6,8/10 dans les données objectives.
Selon les témoignages recueillis : «on étouffe en été et les poubelles débordent».
"""


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tests():
    print("=" * 65)
    print("TEST UNITAIRE — HallucinationGuard")
    print("=" * 65)

    # ── Règle 1 : entity silence ──────────────────────────────────────────
    print("\n[R1] Règle silence — entité 'Aïti' sans marqueur d'absence")
    v1 = check_entity_silence(FABRICATED_ANSWER, SILENCE_LIST)
    silence_violations = [v for v in v1 if v.rule == "silence"]
    if silence_violations:
        for v in silence_violations:
            print(f"  ✓ DÉCLENCHÉ : {v.detail}")
            print(f"    Evidence : {v.evidence[:100]}")
    else:
        print("  ✗ MANQUÉ — la règle silence ne s'est pas déclenchée")

    # ── Règle 2 : number grounding ────────────────────────────────────────
    print("\n[R2] Règle chiffres — '6,8' absent du contexte de synthèse")
    v2 = check_numbers(FABRICATED_ANSWER, SYNTH_CONTEXT)
    num_violations = [v for v in v2 if v.rule == "number"]
    # Vérification : 3,74 EST dans le contexte → doit passer
    v2_ctx = check_numbers("Score : 3,74/5.", SYNTH_CONTEXT)
    num_false_positives = [v for v in v2_ctx if v.rule == "number"]

    if num_violations:
        for v in num_violations:
            print(f"  ✓ DÉCLENCHÉ : {v.detail}")
    else:
        print("  ✗ MANQUÉ — '6,8' aurait dû être signalé")

    if num_false_positives:
        print(f"  ✗ FAUX POSITIF : '3,74' est dans le contexte mais signalé quand même")
        for v in num_false_positives:
            print(f"    {v.detail}")
    else:
        print("  ✓ Pas de faux positif sur '3,74' (présent dans le contexte)")

    # Vérification supplémentaire : '3,5' et '68' sont dans le contexte → silencieux
    v2_ok = check_numbers("Score : 3,5/5 et taux de 68 %.", SYNTH_CONTEXT)
    if not v2_ok:
        print("  ✓ '3,5' et '68' reconnus comme fondés (dans le contexte)")
    else:
        print(f"  ✗ FAUX POSITIF sur nombres du contexte : {[v.detail for v in v2_ok]}")

    # ── Règle 3 : quote grounding ────────────────────────────────────────
    print("\n[R3] Règle citations — citation inventée >5 mots absente des sources")
    v3 = check_quotes(FABRICATED_ANSWER, SOURCES)
    quote_violations = [v for v in v3 if v.rule == "quote"]
    # Vérification : une vraie citation dans les sources ne doit pas déclencher
    real_quote = "«La mer est magnifique ici, on apprécie la qualité du cadre naturel.»"
    v3_ok = check_quotes(f"Comme disent les habitants : {real_quote}", SOURCES)
    quote_false_positives = [v for v in v3_ok if v.rule == "quote"]

    if quote_violations:
        for v in quote_violations:
            print(f"  ✓ DÉCLENCHÉ : {v.detail}")
            print(f"    Evidence : {v.evidence[:80]}")
    else:
        print("  ✗ MANQUÉ — citation inventée non détectée")

    if quote_false_positives:
        print(f"  ✗ FAUX POSITIF : citation réelle signalée à tort")
    else:
        print("  ✓ Pas de faux positif sur citation issue des sources")

    # ── Runner complet via HallucinationGuard.run_all ────────────────────
    print("\n[FULL] run_all() sur la réponse fabriquée")
    guard = HallucinationGuard()
    report = guard.run_all(
        final_answer=FABRICATED_ANSWER,
        silence_list=SILENCE_LIST,
        synth_context=SYNTH_CONTEXT,
        sources=SOURCES,
    )
    guard.print_report(report)

    # ── Vérification : réponse propre → aucune violation ─────────────────
    print("\n[CLEAN] run_all() sur une réponse fondée — zéro violation attendu")
    clean_answer = """\
Le bien-être à Ajaccio est satisfaisant avec un score de 3,74/5 (51 répondants).
Aucune donnée n'est disponible pour Aïti dans le corpus.
Les habitants apprécient «la qualité du cadre naturel» selon les verbatims.
"""
    report_clean = guard.run_all(
        final_answer=clean_answer,
        silence_list=SILENCE_LIST,
        synth_context=SYNTH_CONTEXT,
        sources=SOURCES,
    )
    guard.print_report(report_clean)

    # ── Synthèse ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    n_rules_ok = sum([
        bool(silence_violations),
        bool(num_violations),
        bool(quote_violations),
    ])
    n_fp = sum([
        bool(num_false_positives),
        bool(quote_false_positives),
    ])
    status = "✓ GUARD OPÉRATIONNEL" if n_rules_ok == 3 and n_fp == 0 else "✗ GUARD DÉFAILLANT"
    print(f"{status}  ({n_rules_ok}/3 règles déclenchées, {n_fp} faux positifs)")
    if n_rules_ok < 3:
        print("  → Ne pas relancer le smoke test tant que ce test ne passe pas à 3/3.")
    print("=" * 65)
    return n_rules_ok == 3 and n_fp == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
