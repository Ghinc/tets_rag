"""
Module de détection de communes dans les requêtes utilisateur

Permet de détecter automatiquement les noms de communes mentionnés
et de filtrer les résultats de recherche en conséquence.
"""

import os
import re
import unicodedata
from typing import Optional, List


def _normalize_str(s: str) -> str:
    """Minuscules + suppression des accents diacritiques."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def _levenshtein(a: str, b: str) -> int:
    """Distance de Levenshtein entre deux chaînes (O(n*m))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


# Mapping gentilé (forme adjectivale) → nom officiel de la commune corse
# Couvre les principales communes + formes féminines/plurielles
_GENTILES: dict = {
    # Ajaccio
    "ajaccien": "Ajaccio", "ajaccienne": "Ajaccio",
    "ajacciens": "Ajaccio", "ajacciennes": "Ajaccio",
    # Bastia
    "bastiais": "Bastia", "bastiaise": "Bastia",
    "bastiaises": "Bastia", "bastiasi": "Bastia",
    # Corte
    "cortinche": "Corte", "cortinchi": "Corte",
    "cortenais": "Corte", "cortenaise": "Corte", "cortenaises": "Corte",
    # Porto-Vecchio
    "porto-vecchiais": "Porto-Vecchio", "porto-vecchiaise": "Porto-Vecchio",
    "portivecchiais": "Porto-Vecchio", "portivecchiaise": "Porto-Vecchio",
    # Bonifacio
    "bonifacien": "Bonifacio", "bonifacienne": "Bonifacio",
    "bonifaciens": "Bonifacio", "bonifaciennes": "Bonifacio",
    # Calvi
    "calvais": "Calvi", "calvaise": "Calvi", "calvaises": "Calvi",
    # Sartène
    "sartenais": "Sartène", "sartenaise": "Sartène", "sartenaises": "Sartène",
    # Propriano
    "proprianais": "Propriano", "proprianaise": "Propriano",
    # L'Île-Rousse
    "isoroussin": "L'Île-Rousse", "isoroussine": "L'Île-Rousse",
    "isoroussins": "L'Île-Rousse",
    # Ghisonaccia
    "ghisonacciais": "Ghisonaccia", "ghisonacciaise": "Ghisonaccia",
    # Aléria
    "alérien": "Aléria", "alérienne": "Aléria",
    "alériens": "Aléria", "alériennes": "Aléria",
    # Cervione
    "cervionais": "Cervione", "cervionaise": "Cervione",
    # Zonza
    "zonzain": "Zonza", "zonzaine": "Zonza",
    # Piedicorte-di-Gaggio — alias abrégés fréquents
    "piedicorte": "Piedicorte-di-Gaggio", "pedicorte": "Piedicorte-di-Gaggio",
    "piedicorte-di-gaggio": "Piedicorte-di-Gaggio",
    # Corse générique → pas de commune spécifique (None = ignorer)
    "corse": None, "corses": None,
}


def load_commune_names(communes_dir: str = "./communes_chatbot") -> List[str]:
    """
    Charge la liste de toutes les communes depuis le répertoire

    Args:
        communes_dir: Répertoire contenant les fichiers .txt des communes

    Returns:
        Liste des noms de communes (sans .txt)
    """
    if not os.path.exists(communes_dir):
        return []

    commune_names = []
    for filename in os.listdir(communes_dir):
        if filename.endswith('.txt'):
            commune_name = filename.replace('.txt', '')
            commune_names.append(commune_name)

    return commune_names


def detect_commune_in_text(text: str, commune_names: Optional[List[str]] = None) -> Optional[str]:
    """
    Détecte le nom d'une commune dans un texte

    Args:
        text: Texte dans lequel chercher
        commune_names: Liste des noms de communes (si None, charge depuis ./communes_chatbot)

    Returns:
        Nom de la commune détectée ou None
    """
    # Charger les noms si pas fournis
    if commune_names is None:
        commune_names = load_commune_names()

    if not commune_names:
        return None

    # Normaliser le texte
    text_lower = text.lower()

    # Étape 1 : détecter les gentilés (formes adjectivales des communes)
    for gentile, commune_name in _GENTILES.items():
        if commune_name and re.search(r'\b' + re.escape(gentile) + r'\b', text_lower):
            return commune_name

    # Étape 2 : matching exact sur les noms de communes
    # Trier par longueur décroissante pour matcher les noms composés en premier
    for commune_name in sorted(commune_names, key=len, reverse=True):
        commune_lower = commune_name.lower()

        # Recherche flexible: avec ou sans tirets, apostrophes, accents
        # Créer un pattern qui tolère les variations
        pattern = re.escape(commune_lower)
        pattern = pattern.replace(r'\-', r'[\s\-]?')  # Tirets optionnels/espaces
        pattern = pattern.replace(r"\'", r"[\s']?")   # Apostrophes optionnelles

        # Recherche avec limites de mots pour éviter les faux positifs
        if re.search(r'\b' + pattern + r'\b', text_lower):
            return commune_name

    # Étape 2b/2c : matching sur le premier token des communes composées
    # Gère les abréviations exactes ("Piedicorte") et les typos proches ("Pedicorte" → LD=1)
    _words_norm = [_normalize_str(w) for w in re.split(r'\W+', text) if len(w) >= 6]
    for commune_name in sorted(commune_names, key=len, reverse=True):
        if "-" not in commune_name:
            continue
        first_token = _normalize_str(commune_name.split("-")[0])
        if len(first_token) < 5:
            continue
        for word in _words_norm:
            if _levenshtein(word, first_token) <= 1:
                return commune_name

    return None


# Charger les noms au démarrage du module pour performance
_COMMUNE_NAMES = load_commune_names()


def detect_commune(text: str) -> Optional[str]:
    """
    Détecte rapidement une commune dans un texte (utilise cache)

    Args:
        text: Texte dans lequel chercher

    Returns:
        Nom de la commune détectée ou None
    """
    return detect_commune_in_text(text, _COMMUNE_NAMES)


def detect_communes(text: str) -> List[str]:
    """
    Détecte TOUTES les communes mentionnées dans un texte.

    Contrairement à detect_commune() qui retourne seulement la première,
    cette fonction retourne la liste complète (dans l'ordre d'apparition dans le texte).
    Utile pour les questions comparatives : "Compare Ajaccio et Corte".

    Args:
        text: Texte dans lequel chercher

    Returns:
        Liste des noms de communes détectées (peut être vide)
    """
    if not _COMMUNE_NAMES:
        return []

    text_lower = text.lower()
    found: List[str] = []
    seen: set = set()

    # Étape 1 : gentilés
    for gentile, commune_name in _GENTILES.items():
        if commune_name and re.search(r'\b' + re.escape(gentile) + r'\b', text_lower):
            if commune_name not in seen:
                seen.add(commune_name)
                found.append(commune_name)

    # Étape 2 : matching exact sur les noms, du plus long au plus court
    for commune_name in sorted(_COMMUNE_NAMES, key=len, reverse=True):
        if commune_name in seen:
            continue
        commune_lower = commune_name.lower()
        pattern = re.escape(commune_lower)
        pattern = pattern.replace(r'\-', r'[\s\-]?')
        pattern = pattern.replace(r"\'", r"[\s']?")
        if re.search(r'\b' + pattern + r'\b', text_lower):
            seen.add(commune_name)
            found.append(commune_name)

    # Étape 2b/2c : matching sur le premier token des communes composées
    _words_norm = [_normalize_str(w) for w in re.split(r'\W+', text) if len(w) >= 6]
    for commune_name in sorted(_COMMUNE_NAMES, key=len, reverse=True):
        if commune_name in seen:
            continue
        if "-" not in commune_name:
            continue
        first_token = _normalize_str(commune_name.split("-")[0])
        if len(first_token) < 5:
            continue
        for word in _words_norm:
            if _levenshtein(word, first_token) <= 1:
                seen.add(commune_name)
                found.append(commune_name)
                break

    return found
