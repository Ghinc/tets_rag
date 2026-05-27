"""
epci_detector.py — Détection des EPCI (intercommunalités corses) dans les questions.

Mappe les acronymes, noms courts et noms complets vers le nom canonique stocké
dans ChromaDB (collection zones_epci, champ metadata `epci`).
"""

import re
import unicodedata
from typing import Optional


# Mapping alias (terme utilisateur normalisé) → nom canonique ChromaDB
_EPCI_ALIASES: dict = {
    # ── Acronymes officiels ──────────────────────────────────────────────────
    "capa":              "CA du Pays Ajaccien",
    "cab":               "CA de Bastia",
    # ── Noms courts / territoires ────────────────────────────────────────────
    "niolu":             "CC du Niolu",
    "balagne":           "CC de Calvi Balagne",
    "calvi":             "CC de Calvi Balagne",
    "alta rocca":        "CC de l'Alta Rocca",
    "cap corse":         "CC du Cap Corse",
    "sartenais":         "CC du Sartenais Valinco",
    "valinco":           "CC du Sartenais Valinco",
    "sud corse":         "CC du Sud Corse",
    "casinca":           "CC de la Casinca",
    "costa verde":       "CC de la Costa Verde",
    "centre corse":      "CC du Centre Corse",
    "nebbiu":            "CC du Nebbiu",
    "ile rousse":        "CC du Bassin de Vie de l'Ile Rousse",
    "l ile-rousse":      "CC du Bassin de Vie de l'Ile Rousse",
    "ile-rousse":        "CC du Bassin de Vie de l'Ile Rousse",
    "fiumorbu":          "CC de Fium'orbu Castellu",
    "fium orbu":         "CC de Fium'orbu Castellu",
    "castellu":          "CC de Fium'orbu Castellu",
    "orezza":            "CC Orezza-Ampugnani",
    "ampugnani":         "CC Orezza-Ampugnani",
    "conca d oro":       "CC de la Conca d'Oro",
    "conca d or":        "CC de la Conca d'Oro",
    "conca d'or":        "CC de la Conca d'Oro",
    "taravu":            "CC du Taravu",
    "ornano":            "CC de la Pieve de L'ornano",
    "gravona":           "CC de la Haute Vallée de la Gravona",
    "prunelli":          "CC de la Vallée du Prunelli",
    "marana":            "CC de Marana-Golo",
    "golo":              "CC de la Vallée du Golo",
    "casaconi":          "CC du Casacconi à Golu Suttanu",
    "casacconi":         "CC du Casacconi à Golu Suttanu",
    "liamone":           "CC Communaute des Communes du Liamone",
    "aghja nova":        "CC Aghja Nova",
    "oriente":           "CC de l'Oriente",
    "cote des nacres":   "CC de la Côte des Nacres",
    "nacres":            "CC de la Côte des Nacres",
    "boziu":             "CC Di E Tre Pieve : Boziu, Mercoriu E Rogna",
    "mercoriu":          "CC Di E Tre Pieve : Boziu, Mercoriu E Rogna",
    "deux sevi":         "CC des Deux Sevi",
    "sevi":              "CC des Deux Sevi",
}

# Liste canonique (du plus long au plus court) pour matching sur nom complet
_CANONICAL_NAMES: list = sorted(set(_EPCI_ALIASES.values()), key=len, reverse=True)


def _normalize(s: str) -> str:
    """Supprime les accents et met en minuscules."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def detect_epci(text: str) -> Optional[str]:
    """
    Détecte le nom canonique d'un EPCI dans un texte.

    Stratégie :
    1. Cherche les alias (acronymes + noms courts normalisés) — ex: "capa" → "CA du Pays Ajaccien"
    2. Cherche les noms canoniques complets (normalisés) dans le texte

    Args:
        text: Question ou texte à analyser

    Returns:
        Nom canonique ChromaDB de l'EPCI détecté, ou None
    """
    t = _normalize(text)

    # Étape 1 : matching sur les alias (ordre itération = ordre d'insertion → acronymes en premier)
    for alias, canonical in _EPCI_ALIASES.items():
        if re.search(r"\b" + re.escape(_normalize(alias)) + r"\b", t):
            return canonical

    # Étape 2 : matching sur les noms canoniques complets (du plus long au plus court)
    for canonical in _CANONICAL_NAMES:
        pat = re.escape(_normalize(canonical))
        pat = pat.replace(r"\ ", r"[\s\-]?")   # espaces/tirets optionnels
        pat = pat.replace(r"\'", r"[\s']?")     # apostrophes optionnelles
        if re.search(pat, t):
            return canonical

    return None


if __name__ == "__main__":
    tests = [
        "Quelles communes appartiennent à la CAPA ?",
        "Communes de la CAB",
        "Communes membres de la CC du Niolu",
        "Quelle est la situation dans le Sartenais-Valinco ?",
        "Qualité de vie en Balagne",
        "Contexte géographique du Cap Corse",
        "Quel est le score OppChoVec pour la Conca d'Or ?",
        "Que sait-on de la Côte des Nacres ?",
        "Quelles communes font partie du Fiumorbu ?",
        "Situation générale en Corse",
    ]
    for q in tests:
        result = detect_epci(q)
        mark = "OK" if result else "--"
        print(f"  [{mark}] {q!r}")
        if result:
            print(f"       -> {result!r}")
