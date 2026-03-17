"""
Détecteur de filtres portrait dans les questions
Extrait automatiquement: âge, genre, profession, dimension
"""
import re
from typing import Optional, Dict, Any

# Mapping des mots-clés vers les tranches d'âge
AGE_PATTERNS = {
    # Patterns exacts avec âges
    r'\b(\d{1,2})\s*[-à]\s*(\d{1,2})\s*ans\b': 'range',  # "18-25 ans", "18 à 25 ans"
    r'\bmoins\s+de\s+(\d{1,2})\s*ans\b': 'less_than',    # "moins de 30 ans"
    r'\bplus\s+de\s+(\d{1,2})\s*ans\b': 'more_than',     # "plus de 60 ans"
    r'\b(\d{1,2})\s*ans\b': 'exact',                      # "25 ans"
}

# Mapping des mots-clés vers les catégories d'âge
AGE_KEYWORDS = {
    'jeunes': ('15-24', 15, 24),
    'jeune': ('15-24', 15, 24),
    'jeunes adultes': ('25-34', 25, 34),
    'jeune adulte': ('25-34', 25, 34),
    'adultes': ('35-49', 35, 49),
    'adulte': ('35-49', 35, 49),
    'jeunes seniors': ('50-64', 50, 64),
    'jeune senior': ('50-64', 50, 64),
    'seniors': ('65+', 65, 100),
    'senior': ('65+', 65, 100),
    'personnes âgées': ('65+', 65, 100),
    'personne âgée': ('65+', 65, 100),
    'retraités': ('65+', 65, 100),  # Souvent associé aux seniors
    'retraité': ('65+', 65, 100),
    'étudiants': ('15-24', 15, 24),  # Souvent jeunes
    'étudiant': ('15-24', 15, 24),
}

# Mapping des mots-clés vers le genre
GENDER_KEYWORDS = {
    'femmes': 'Femme',
    'femme': 'Femme',
    'féminin': 'Femme',
    'féminines': 'Femme',
    'hommes': 'Homme',
    'homme': 'Homme',
    'masculin': 'Homme',
    'masculins': 'Homme',
}

# Mapping des mots-clés vers les professions (du jeu de données)
PROFESSION_KEYWORDS = {
    # Étudiant(e)
    'étudiants': 'Étudiant(e)',
    'étudiant': 'Étudiant(e)',
    'étudiante': 'Étudiant(e)',
    'étudiantes': 'Étudiant(e)',

    # Salarié(e) – Cadre ou profession intermédiaire
    'cadres': 'Salarié(e) – Cadre ou profession intermédiaire',
    'cadre': 'Salarié(e) – Cadre ou profession intermédiaire',
    'professions intermédiaires': 'Salarié(e) – Cadre ou profession intermédiaire',

    # Salarié(e) – Employé(e)
    'employés': 'Salarié(e) – Employé(e)',
    'employé': 'Salarié(e) – Employé(e)',
    'employée': 'Salarié(e) – Employé(e)',
    'employées': 'Salarié(e) – Employé(e)',
    'salariés': 'Salarié(e) – Employé(e)',  # Par défaut employé
    'salarié': 'Salarié(e) – Employé(e)',

    # Fonctionnaire
    'fonctionnaires': 'Fonctionnaire',
    'fonctionnaire': 'Fonctionnaire',
    'agents publics': 'Fonctionnaire',
    'fonction publique': 'Fonctionnaire',

    # Travailleur(se) indépendant(e) / Entrepreneur(e)
    'indépendants': 'Travailleur(se) indépendant(e) / Entrepreneur(e)',
    'indépendant': 'Travailleur(se) indépendant(e) / Entrepreneur(e)',
    'entrepreneurs': 'Travailleur(se) indépendant(e) / Entrepreneur(e)',
    'entrepreneur': 'Travailleur(se) indépendant(e) / Entrepreneur(e)',
    'auto-entrepreneurs': 'Travailleur(se) indépendant(e) / Entrepreneur(e)',

    # Retraité(e)
    'retraités': 'Retraité(e)',
    'retraité': 'Retraité(e)',
    'retraitée': 'Retraité(e)',
    'retraitées': 'Retraité(e)',
    'retraites': 'Retraité(e)',
    'retraite': 'Retraité(e)',

    # Agriculteur(trice), artisan(e) ou commerçant(e)
    'agriculteurs': 'Agriculteur(trice), artisan(e) ou commerçant(e)',
    'agriculteur': 'Agriculteur(trice), artisan(e) ou commerçant(e)',
    'artisans': 'Agriculteur(trice), artisan(e) ou commerçant(e)',
    'artisan': 'Agriculteur(trice), artisan(e) ou commerçant(e)',
    'commerçants': 'Agriculteur(trice), artisan(e) ou commerçant(e)',
    'commerçant': 'Agriculteur(trice), artisan(e) ou commerçant(e)',

    # Sans emploi (en recherche d'emploi)
    'chômeurs': 'Sans emploi (en recherche d\'emploi)',
    'chômeur': 'Sans emploi (en recherche d\'emploi)',
    'demandeurs d\'emploi': 'Sans emploi (en recherche d\'emploi)',
    'sans emploi': 'Sans emploi (en recherche d\'emploi)',
}

# Mapping des mots-clés vers les dimensions de qualité de vie
DIMENSION_KEYWORDS = {
    'santé': 'Santé',
    'sante': 'Santé',
    'médecin': 'Santé',
    'hôpital': 'Santé',
    'soins': 'Santé',
    'maladie': 'Santé',

    'environnement': 'Environnement',
    'nature': 'Environnement',
    'écologie': 'Environnement',
    'pollution': 'Environnement',
    'climat': 'Environnement',

    'culture': 'Culture',
    'musée': 'Culture',
    'théâtre': 'Culture',
    'cinéma': 'Culture',
    'concert': 'Culture',
    'loisirs': 'Culture',

    'logement': 'Logement',
    'loger': 'Logement',
    'maison': 'Logement',
    'appartement': 'Logement',
    'habitat': 'Logement',

    'services': 'Services de proximité',
    'commerces': 'Services de proximité',
    'proximité': 'Services de proximité',
    'courses': 'Services de proximité',

    'réseau': 'Réseau',
    'internet': 'Réseau',
    'téléphone': 'Réseau',
    'connexion': 'Réseau',

    'sécurité': 'Sécurité',
    'securite': 'Sécurité',
    'tranquillité': 'Sécurité',
    'violence': 'Sécurité',

    'vie pro': 'Ratio vie pro/ vie perso',
    'vie perso': 'Ratio vie pro/ vie perso',
    'équilibre': 'Ratio vie pro/ vie perso',
    'temps libre': 'Ratio vie pro/ vie perso',

    'éducation': 'Education',
    'education': 'Education',
    'école': 'Education',
    'université': 'Education',
    'formation': 'Education',

    'revenus': 'Revenus',
    'argent': 'Revenus',
    'salaire': 'Revenus',
    'finances': 'Revenus',

    'emploi': 'Emploi',
    'travail': 'Emploi',
    'chômage': 'Emploi',

    'transports': 'Transports',
    'transport': 'Transports',
    'bus': 'Transports',
    'train': 'Transports',
    'voiture': 'Transports',
    'mobilité': 'Transports',

    'communauté': 'Communauté et relations',
    'relations': 'Communauté et relations',
    'famille': 'Communauté et relations',
    'amis': 'Communauté et relations',
    'voisins': 'Communauté et relations',
    'lien social': 'Communauté et relations',

    'tourisme': 'Tourisme (ressenti localement)',
    'touristes': 'Tourisme (ressenti localement)',
}


def normalize_text(text: str) -> str:
    """Normalise le texte pour la comparaison"""
    text = text.lower()
    # Garder les accents pour le matching français
    return text


def detect_age(question: str) -> Dict[str, Any]:
    """
    Détecte les filtres d'âge dans la question.
    Retourne: {'age_min': int|None, 'age_max': int|None, 'age_range': str|None}
    """
    question_lower = normalize_text(question)
    result = {'age_min': None, 'age_max': None, 'age_range': None}

    # 1. Chercher les patterns numériques
    # Pattern "X-Y ans" ou "X à Y ans"
    match = re.search(r'\b(\d{1,2})\s*[-à]\s*(\d{1,2})\s*ans\b', question_lower)
    if match:
        result['age_min'] = int(match.group(1))
        result['age_max'] = int(match.group(2))
        return result

    # Pattern "moins de X ans"
    match = re.search(r'\bmoins\s+de\s+(\d{1,2})\s*ans\b', question_lower)
    if match:
        result['age_max'] = int(match.group(1)) - 1
        result['age_min'] = 15  # Âge minimum raisonnable
        return result

    # Pattern "plus de X ans"
    match = re.search(r'\bplus\s+de\s+(\d{1,2})\s*ans\b', question_lower)
    if match:
        result['age_min'] = int(match.group(1)) + 1
        result['age_max'] = 100
        return result

    # 2. Chercher les mots-clés d'âge (du plus spécifique au moins spécifique)
    for keyword, (age_range, age_min, age_max) in sorted(AGE_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in question_lower:
            result['age_range'] = age_range
            result['age_min'] = age_min
            result['age_max'] = age_max
            return result

    return result


def detect_gender(question: str) -> Optional[str]:
    """
    Détecte le genre dans la question.
    Retourne: 'Femme' | 'Homme' | None
    """
    question_lower = normalize_text(question)

    for keyword, gender in sorted(GENDER_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in question_lower:
            return gender

    return None


def detect_profession(question: str) -> Optional[str]:
    """
    Détecte la profession dans la question.
    Retourne: la profession du jeu de données ou None
    """
    question_lower = normalize_text(question)

    for keyword, profession in sorted(PROFESSION_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in question_lower:
            return profession

    return None


def detect_dimension(question: str) -> Optional[str]:
    """
    Détecte la dimension de qualité de vie dans la question.
    Retourne: la dimension ou None
    """
    question_lower = normalize_text(question)

    for keyword, dimension in sorted(DIMENSION_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in question_lower:
            return dimension

    return None


def detect_portrait_filters(question: str) -> Dict[str, Any]:
    """
    Détecte tous les filtres portrait dans une question.

    Args:
        question: La question de l'utilisateur

    Returns:
        Dict avec les clés:
        - age_min: int | None
        - age_max: int | None
        - age_range: str | None
        - genre: str | None ('Femme' ou 'Homme')
        - profession: str | None
        - dimension: str | None
        - has_portrait_filter: bool (True si au moins un filtre détecté)
    """
    age_filters = detect_age(question)
    genre = detect_gender(question)
    profession = detect_profession(question)
    dimension = detect_dimension(question)

    result = {
        'age_min': age_filters['age_min'],
        'age_max': age_filters['age_max'],
        'age_range': age_filters['age_range'],
        'genre': genre,
        'profession': profession,
        'dimension': dimension,
    }

    # Déterminer si des filtres portrait sont actifs
    result['has_portrait_filter'] = any([
        result['age_min'] is not None,
        result['age_max'] is not None,
        result['genre'] is not None,
        result['profession'] is not None,
        result['dimension'] is not None,
    ])

    return result


# Tests si exécuté directement
if __name__ == "__main__":
    test_questions = [
        "Que pensent les jeunes de 18-25 ans de la santé ?",
        "Quel est l'avis des étudiants sur le logement ?",
        "Comment les femmes perçoivent-elles les transports ?",
        "Quelles sont les priorités des retraités à Bastia ?",
        "Les hommes salariés sont-ils satisfaits de leur qualité de vie ?",
        "Que pensent les jeunes femmes de l'environnement ?",
        "L'avis des fonctionnaires sur la sécurité",
        "Les seniors et la santé",
        "Personnes de moins de 30 ans sur le logement",
        "Plus de 60 ans et les transports",
    ]

    print("=" * 60)
    print("TEST DU DÉTECTEUR DE FILTRES PORTRAIT")
    print("=" * 60)

    for q in test_questions:
        print(f"\nQuestion: {q}")
        filters = detect_portrait_filters(q)
        print(f"  Âge: {filters['age_min']}-{filters['age_max']} ({filters['age_range']})")
        print(f"  Genre: {filters['genre']}")
        print(f"  Profession: {filters['profession']}")
        print(f"  Dimension: {filters['dimension']}")
        print(f"  Filtre actif: {filters['has_portrait_filter']}")
