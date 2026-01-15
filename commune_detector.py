"""
Module de détection de communes dans les requêtes utilisateur

Permet de détecter automatiquement les noms de communes mentionnés
et de filtrer les résultats de recherche en conséquence.
"""

import os
import re
from typing import Optional, List


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

    # Chercher chaque commune dans le texte
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
