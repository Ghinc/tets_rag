"""
Script pour peupler l'ontologie avec les instances de communes corses.

Ce script:
1. Lit TOUTES les communes depuis communes_corse_wikipedia.csv (360 communes)
2. Enrichit avec les données d'enquête de df_mean_by_commune.csv si disponibles
3. Détecte automatiquement le département depuis le texte Wikipedia
4. Crée des instances OWL de type :Municipality
5. Met à jour le fichier onto_be_instances.ttl

Usage:
    python populate_communes.py
"""

import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
import unicodedata


def normalize_uri(name: str) -> str:
    """
    Normalise un nom pour en faire un URI valide.
    - Supprime les accents
    - Remplace les espaces et caractères spéciaux par des underscores
    - Met en CamelCase
    """
    # Supprimer les accents
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')

    # Remplacer les caractères spéciaux
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)

    # Convertir en CamelCase
    words = name.split()
    name = ''.join(word.capitalize() for word in words)

    return name


def escape_turtle_string(s: str) -> str:
    """Échappe les caractères spéciaux pour Turtle."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    return s


def detect_department(wiki_text: str, commune_name: str) -> str:
    """
    Détecte le département depuis le texte Wikipedia.

    Args:
        wiki_text: Texte Wikipedia de la commune
        commune_name: Nom de la commune (pour fallback)

    Returns:
        ':CorseDuSud' ou ':HauteCorse'
    """
    if not wiki_text or pd.isna(wiki_text):
        # Fallback: liste de communes connues en Corse-du-Sud
        corse_du_sud_communes = [
            'Ajaccio', 'Afa', 'Alata', 'Bastelicaccia', 'Albitreccia',
            'Grosseto-Prugna', 'Porticcio', 'Propriano', 'Sartène', 'Bonifacio',
            'Porto-Vecchio', 'Lecci', 'Zonza', 'Levie', 'Sollacaro',
            'Sarrola-Carcopino', 'Cuttoli-Corticchiato', 'Peri', 'Valle-di-Mezzana',
            'Appietto', 'Cauro', 'Eccica-Suarella', 'Coggia', 'Cargèse',
            'Olmeto', 'Pietrosella', 'Viggianello', 'Quenza', 'Serra-di-Ferro',
            'Piana', 'Osani', 'Partinello', 'Evisa', 'Cristinacce', 'Vico',
            'Soccia', 'Orto', 'Guagno', 'Azzana', 'Balogna', 'Letia',
            'Renno', 'Marignana', 'Serriera', 'Ota', 'Calcatoggio',
            'Sari-Solenzara', 'Conca', 'Monacia-d\'Aullène', 'Figari',
            'Pianotolli-Caldarello', 'Sotta', 'San-Gavino-di-Carbini'
        ]
        if commune_name in corse_du_sud_communes:
            return ':CorseDuSud'
        return ':HauteCorse'

    wiki_lower = wiki_text.lower()

    # Chercher les mentions explicites du département
    if 'corse-du-sud' in wiki_lower or 'corse du sud' in wiki_lower:
        return ':CorseDuSud'
    elif 'haute-corse' in wiki_lower or 'haute corse' in wiki_lower:
        return ':HauteCorse'

    # Fallback
    return ':HauteCorse'


def generate_commune_ttl(
    commune_name: str,
    wiki_data: Optional[Dict] = None,
    survey_data: Optional[Dict] = None,
    insee_code: Optional[str] = None
) -> str:
    """
    Génère le TTL pour une commune.

    Args:
        commune_name: Nom de la commune
        wiki_data: Données Wikipedia (résumé, contenu)
        survey_data: Données d'enquête (optionnel, pour numberOfRespondents)
        insee_code: Code INSEE de la commune (optionnel)

    Returns:
        String TTL pour cette commune
    """
    uri = normalize_uri(commune_name)

    lines = []
    lines.append(f":{uri} a :Municipality ;")
    lines.append(f'    rdfs:label "{escape_turtle_string(commune_name)}"@fr ;')
    lines.append(f'    rdfs:label "{escape_turtle_string(commune_name)}"@en ;')
    lines.append(f'    :officialName "{escape_turtle_string(commune_name)}" ;')

    # Ajouter le code INSEE si disponible
    if insee_code:
        lines.append(f'    :inseeCode "{insee_code}" ;')

    # Ajouter le nombre de répondants si disponible (depuis données d'enquête)
    if survey_data and 'total_respondants' in survey_data and pd.notna(survey_data['total_respondants']):
        lines.append(f'    :numberOfRespondents {int(survey_data["total_respondants"])} ;')

    # Déterminer le département depuis le texte Wikipedia
    wiki_text = ""
    if wiki_data:
        wiki_text = wiki_data.get('résumé', '') or wiki_data.get('contenu_wiki', '') or ''

    department = detect_department(wiki_text, commune_name)
    lines.append(f'    :isPartOfSpatialUnit {department} ;')

    # Note: On n'ajoute pas :isPartOfSpatialUnit :Corse car c'est redondant
    # (les départements CorseDuSud et HauteCorse sont déjà rattachés à :Corse)

    # Ajouter un résumé Wikipedia si disponible
    if wiki_data and 'résumé' in wiki_data and pd.notna(wiki_data['résumé']):
        resume = str(wiki_data['résumé'])[:500]  # Limiter à 500 caractères
        lines.append(f'    rdfs:comment "{escape_turtle_string(resume)}"@fr ;')

    # Fermer l'instance
    lines[-1] = lines[-1].rstrip(' ;') + ' .'

    return '\n'.join(lines)


def generate_dimension_instances() -> str:
    """Génère les instances pour les dimensions du bien-être."""
    return """
# === DIMENSIONS DU BIEN-ÊTRE ===

:dim_employment a :Employment ;
    rdfs:label "Emploi"@fr ;
    rdfs:label "Employment"@en .

:dim_housing a :Housing ;
    rdfs:label "Logement"@fr ;
    rdfs:label "Housing"@en .

:dim_services a :Services ;
    rdfs:label "Services"@fr ;
    rdfs:label "Services"@en .

:dim_democracy a :Democracy ;
    rdfs:label "Démocratie"@fr ;
    rdfs:label "Democracy"@en .

:dim_education a :Education ;
    rdfs:label "Éducation"@fr ;
    rdfs:label "Education"@en .

:dim_transport a :Transport ;
    rdfs:label "Transports"@fr ;
    rdfs:label "Transport"@en .

:dim_health a :Health ;
    rdfs:label "Santé"@fr ;
    rdfs:label "Health"@en .

:dim_env_quality a :EnvQuality ;
    rdfs:label "Qualité environnementale"@fr ;
    rdfs:label "Environmental quality"@en .

:dim_income a :Income ;
    rdfs:label "Revenu"@fr ;
    rdfs:label "Income"@en .

:dim_work_life_balance a :WorkLifeBalance ;
    rdfs:label "Équilibre vie pro/vie perso"@fr ;
    rdfs:label "Work-life balance"@en .

:dim_social_diversity a :SocialDiversity ;
    rdfs:label "Diversité sociale"@fr ;
    rdfs:label "Social diversity"@en .
"""


def generate_indicator_instances() -> str:
    """Génère les instances pour les indicateurs subjectifs et objectifs."""
    return """
# === INDICATEURS SUBJECTIFS ===

:ind_enoughFreeTimeFeeling a :SubjectiveIndicator ;
    rdfs:label "Sentiment d'avoir assez de temps libre"@fr ;
    rdfs:label "Feeling of having enough free time"@en ;
    :measuresDimension :dim_work_life_balance .

:ind_incomeSatisfaction a :SubjectiveIndicator ;
    rdfs:label "Satisfaction vis-à-vis du revenu"@fr ;
    rdfs:label "Satisfaction with income"@en ;
    :measuresDimension :dim_income .

:ind_pleasantEnvironmentFeeling a :SubjectiveIndicator ;
    rdfs:label "Perception d'un environnement agréable"@fr ;
    rdfs:label "Perceived pleasant environment"@en ;
    :measuresDimension :dim_env_quality .

:ind_selfRatedHealth a :SubjectiveIndicator ;
    rdfs:label "Auto-évaluation de la santé"@fr ;
    rdfs:label "Self-rated health"@en ;
    :measuresDimension :dim_health .

:ind_educationOfferSatisfaction a :SubjectiveIndicator ;
    rdfs:label "Satisfaction vis-à-vis de l'offre éducative"@fr ;
    rdfs:label "Satisfaction with the educational offer"@en ;
    :measuresDimension :dim_education .

:ind_heardByPublicAuthoritiesFeeling a :SubjectiveIndicator ;
    rdfs:label "Sentiment d'être écouté par les autorités publiques"@fr ;
    rdfs:label "Feeling of being heard by public authorities"@en ;
    :measuresDimension :dim_democracy .

:ind_serviceAvailabilitySatisfaction a :SubjectiveIndicator ;
    rdfs:label "Satisfaction vis-à-vis des services disponibles"@fr ;
    rdfs:label "Satisfaction with available services"@en ;
    :measuresDimension :dim_services .

:ind_selfRatedHousingComfort a :SubjectiveIndicator ;
    rdfs:label "Confort du logement (auto-évalué)"@fr ;
    rdfs:label "Self-rated housing comfort"@en ;
    :measuresDimension :dim_housing .

:ind_selfRatedJobFulfilment a :SubjectiveIndicator ;
    rdfs:label "Épanouissement au travail (auto-évalué)"@fr ;
    rdfs:label "Self-rated job fulfilment"@en ;
    :measuresDimension :dim_employment .

# === INDICATEURS OBJECTIFS ===

:ind_unemploymentRate a :ObjectiveIndicator ;
    rdfs:label "Taux de chômage"@fr ;
    rdfs:label "Unemployment rate"@en ;
    :measuresDimension :dim_employment ;
    :hasUnit :unit_percent .

:ind_housingSanitationRate a :ObjectiveIndicator ;
    rdfs:label "Taux de logements équipés d'installations sanitaires"@fr ;
    rdfs:label "Percentage of dwellings with sanitary facilities"@en ;
    :measuresDimension :dim_housing ;
    :hasUnit :unit_percent .

:ind_serviceAvailability a :ObjectiveIndicator ;
    rdfs:label "Disponibilité des services"@fr ;
    rdfs:label "Availability of services"@en ;
    :measuresDimension :dim_services ;
    :hasUnit :unit_index .

:ind_voterTurnout a :ObjectiveIndicator ;
    rdfs:label "Taux de participation électorale"@fr ;
    rdfs:label "Voter turnout"@en ;
    :measuresDimension :dim_democracy ;
    :hasUnit :unit_percent .

:ind_theilCSPDiversityIndex a :ObjectiveIndicator ;
    rdfs:label "Indice de Theil de diversité des CSP"@fr ;
    rdfs:label "Theil index of socio-professional diversity"@en ;
    :measuresDimension :dim_social_diversity ;
    :hasUnit :unit_index .

:ind_highSchoolGraduatesRate a :ObjectiveIndicator ;
    rdfs:label "Taux de diplômés du baccalauréat"@fr ;
    rdfs:label "Share of residents holding a high-school diploma"@en ;
    :measuresDimension :dim_education ;
    :hasUnit :unit_percent .

:ind_healthProfessionalProximity a :ObjectiveIndicator ;
    rdfs:label "Proximité des professionnels de santé"@fr ;
    rdfs:label "Proximity of health professionals"@en ;
    :measuresDimension :dim_health ;
    :hasUnit :unit_minutes .

:ind_waterQuality a :ObjectiveIndicator ;
    rdfs:label "Qualité de l'eau"@fr ;
    rdfs:label "Water quality"@en ;
    :measuresDimension :dim_env_quality ;
    :hasUnit :unit_index .

:ind_gdpPerCapita a :ObjectiveIndicator ;
    rdfs:label "PIB par habitant"@fr ;
    rdfs:label "GDP per capita"@en ;
    :measuresDimension :dim_income ;
    :hasUnit :unit_euros .

:ind_averageWeeklyWorkingHours a :ObjectiveIndicator ;
    rdfs:label "Nombre moyen d'heures travaillées par semaine"@fr ;
    rdfs:label "Average weekly working hours"@en ;
    :measuresDimension :dim_work_life_balance ;
    :hasUnit :unit_hours .

# === UNITÉS DE MESURE ===

:unit_percent a qudt:Unit ;
    rdfs:label "Pourcentage"@fr ;
    rdfs:label "Percent"@en ;
    qudt:symbol "%" .

:unit_index a qudt:Unit ;
    rdfs:label "Indice"@fr ;
    rdfs:label "Index"@en ;
    qudt:symbol "idx" .

:unit_minutes a qudt:Unit ;
    rdfs:label "Minutes"@fr ;
    rdfs:label "Minutes"@en ;
    qudt:symbol "min" .

:unit_hours a qudt:Unit ;
    rdfs:label "Heures"@fr ;
    rdfs:label "Hours"@en ;
    qudt:symbol "h" .

:unit_euros a qudt:Unit ;
    rdfs:label "Euros"@fr ;
    rdfs:label "Euros"@en ;
    qudt:symbol "€" .
"""


def generate_department_instances() -> str:
    """Génère les instances pour les départements et la région."""
    return """
# === RÉGION ET DÉPARTEMENTS ===

:France a :Country ;
    rdfs:label "France"@fr ;
    rdfs:label "France"@en ;
    :officialName "République française" .

:Corse a :Region ;
    rdfs:label "Corse"@fr ;
    rdfs:label "Corsica"@en ;
    :officialName "Collectivité de Corse" ;
    :isPartOfSpatialUnit :France .

:CorseDuSud a :Department ;
    rdfs:label "Corse-du-Sud"@fr ;
    rdfs:label "South Corsica"@en ;
    :officialName "Corse-du-Sud" ;
    :inseeCode "2A" ;
    :isPartOfSpatialUnit :Corse .

:HauteCorse a :Department ;
    rdfs:label "Haute-Corse"@fr ;
    rdfs:label "Upper Corsica"@en ;
    :officialName "Haute-Corse" ;
    :inseeCode "2B" ;
    :isPartOfSpatialUnit :Corse .
"""


def generate_datasource_instances(rag_be_path: str, insee_data: Dict[str, str]) -> str:
    """
    Génère les instances de sources de données à partir des fichiers d'enquête.

    Args:
        rag_be_path: Chemin vers le dossier rag_be contenant les fichiers d'enquête
        insee_data: Dictionnaire {commune_name: insee_code}

    Returns:
        String TTL pour les sources de données (enquêtes quantitatives et verbatims)
    """
    import os
    import uuid

    lines = []
    lines.append("# === SOURCES DE DONNÉES - ENQUÊTES ET VERBATIMS ===\n")

    # Lister les fichiers dans rag_be
    if not os.path.exists(rag_be_path):
        return ""

    files = sorted([f for f in os.listdir(rag_be_path) if f.endswith('.txt')])

    for filename in files:
        commune_name = filename.replace('.txt', '')
        commune_uri = normalize_uri(commune_name)
        insee_code = insee_data.get(commune_name, "")

        # Générer un UUID unique pour chaque source
        survey_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"survey_{commune_name}"))
        verbatim_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"verbatim_{commune_name}"))

        # Créer une instance de source d'enquête quantitative avec verbatims
        lines.append(f":source_survey_{commune_uri} a :QuantitativeSurvey ;")
        lines.append(f'    :sourceId "{survey_uuid}" ;')
        if insee_code:
            lines.append(f'    :inseeCode "{insee_code}" ;')
        lines.append(f'    rdfs:label "Enquête bien-être - {commune_name}"@fr ;')
        lines.append(f'    rdfs:label "Well-being survey - {commune_name}"@en ;')
        lines.append(f'    :appliesToSpatialUnit :{commune_uri} ;')
        lines.append(f'    rdfs:comment "Données d\'enquête quantitative sur le bien-être territorial pour la commune de {commune_name}."@fr .')
        lines.append("")

        # Créer une instance de verbatims associés
        lines.append(f":source_verbatim_{commune_uri} a :Verbatim ;")
        lines.append(f'    :sourceId "{verbatim_uuid}" ;')
        if insee_code:
            lines.append(f'    :inseeCode "{insee_code}" ;')
        lines.append(f'    rdfs:label "Verbatims - {commune_name}"@fr ;')
        lines.append(f'    rdfs:label "Verbatim responses - {commune_name}"@en ;')
        lines.append(f'    :appliesToSpatialUnit :{commune_uri} ;')
        lines.append(f'    rdfs:comment "Réponses libres des participants à l\'enquête pour la commune de {commune_name}."@fr .')
        lines.append("")

    return '\n'.join(lines)


def generate_wikipedia_instances(wiki_df: 'pd.DataFrame', insee_data: Dict[str, str]) -> str:
    """
    Génère les instances WikipediaArticle pour chaque commune.

    Args:
        wiki_df: DataFrame avec les données Wikipedia des communes
        insee_data: Dictionnaire {commune_name: insee_code}

    Returns:
        String TTL pour les articles Wikipedia
    """
    import uuid

    lines = []
    lines.append("# === SOURCES DE DONNÉES - WIKIPEDIA ===\n")

    for _, row in wiki_df.iterrows():
        commune_name = row['commune']
        commune_uri = normalize_uri(commune_name)
        insee_code = insee_data.get(commune_name, "")

        # Générer un UUID unique
        wiki_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"wiki_{commune_name}"))

        lines.append(f":source_wiki_{commune_uri} a :WikipediaArticle ;")
        lines.append(f'    :sourceId "{wiki_uuid}" ;')
        if insee_code:
            lines.append(f'    :inseeCode "{insee_code}" ;')
        lines.append(f'    rdfs:label "Article Wikipedia - {commune_name}"@fr ;')
        lines.append(f'    rdfs:label "Wikipedia article - {commune_name}"@en ;')
        lines.append(f'    :appliesToSpatialUnit :{commune_uri} ;')
        lines.append(f'    rdfs:comment "Article Wikipedia décrivant la commune de {commune_name}."@fr .')
        lines.append("")

    return '\n'.join(lines)


def generate_interview_instances(entretiens_path: str, insee_data: Dict[str, str]) -> str:
    """
    Génère les instances Interview à partir du fichier d'entretiens.

    Args:
        entretiens_path: Chemin vers le fichier entretiens_lea.txt
        insee_data: Dictionnaire {commune_name: insee_code}

    Returns:
        String TTL pour les entretiens
    """
    import re
    import os
    import uuid

    lines = []
    lines.append("# === SOURCES DE DONNÉES - ENTRETIENS ===\n")

    if not os.path.exists(entretiens_path):
        return ""

    with open(entretiens_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Trouver tous les entretiens avec le pattern "### Commune : X | Entretien Y"
    pattern = r'### Commune : ([^|]+) \| Entretien (\d+)'
    matches = re.findall(pattern, content)

    for commune_name, entretien_num in matches:
        commune_name = commune_name.strip()
        commune_uri = normalize_uri(commune_name)
        # Essayer plusieurs variantes pour trouver le code INSEE
        insee_code = insee_data.get(commune_name, "")
        if not insee_code:
            # Essayer avec tiret au lieu d'espace
            insee_code = insee_data.get(commune_name.replace(" ", "-"), "")
        if not insee_code:
            # Cas spécifique: Grossetto Prugna -> Grosseto-Prugna
            name_with_dash = commune_name.replace(" ", "-")
            # Simplifier seulement les doubles 't' (Grossetto -> Grosseto)
            simplified = name_with_dash.replace("tt", "t")
            insee_code = insee_data.get(simplified, "")

        # Générer un UUID unique
        interview_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"interview_{commune_name}_{entretien_num}"))

        lines.append(f":source_interview_{commune_uri}_{entretien_num} a :Interview ;")
        lines.append(f'    :sourceId "{interview_uuid}" ;')
        if insee_code:
            lines.append(f'    :inseeCode "{insee_code}" ;')
        lines.append(f'    rdfs:label "Entretien {entretien_num} - {commune_name}"@fr ;')
        lines.append(f'    rdfs:label "Interview {entretien_num} - {commune_name}"@en ;')
        lines.append(f'    :appliesToSpatialUnit :{commune_uri} ;')
        lines.append(f'    rdfs:comment "Entretien qualitatif n°{entretien_num} réalisé avec un habitant de {commune_name}."@fr .')
        lines.append("")

    return '\n'.join(lines)


def generate_statistical_dataset_instances() -> str:
    """
    Génère les instances StatisticalDataset pour les données chiffrées.

    Returns:
        String TTL pour les jeux de données statistiques
    """
    import uuid

    # Générer des UUIDs uniques pour les datasets régionaux
    insee_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "insee_corse"))
    survey_agg_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "survey_aggregated_corse"))

    return f"""# === SOURCES DE DONNÉES - JEUX DE DONNÉES STATISTIQUES ===

:source_insee_corse a :StatisticalDataset ;
    :sourceId "{insee_uuid}" ;
    rdfs:label "Données INSEE Corse"@fr ;
    rdfs:label "INSEE Corsica data"@en ;
    :appliesToSpatialUnit :Corse ;
    rdfs:comment "Données statistiques INSEE pour les communes de Corse (codes INSEE, données socio-économiques)."@fr .

:source_survey_aggregated a :StatisticalDataset ;
    :sourceId "{survey_agg_uuid}" ;
    rdfs:label "Données agrégées d'enquête bien-être"@fr ;
    rdfs:label "Aggregated well-being survey data"@en ;
    :appliesToSpatialUnit :Corse ;
    rdfs:comment "Moyennes par commune des réponses aux enquêtes de bien-être territorial."@fr .
"""


def generate_source_mapping(
    rag_be_path: str,
    wiki_df: 'pd.DataFrame',
    entretiens_path: str,
    insee_data: Dict[str, str]
) -> Dict:
    """
    Génère un fichier JSON de mapping entre les fichiers sources et les identifiants de l'ontologie.
    Ce mapping sera utilisé par ChromaDB pour enrichir les métadonnées des chunks.

    Returns:
        Dictionnaire de mapping
    """
    import os
    import re
    import uuid
    import json

    mapping = {
        "description": "Mapping entre fichiers sources et identifiants ontologie pour ChromaDB",
        "generated_at": datetime.now().isoformat(),
        "sources": {}
    }

    # 1. Mapping pour les enquêtes/verbatims (dossier rag_be)
    if os.path.exists(rag_be_path):
        files = sorted([f for f in os.listdir(rag_be_path) if f.endswith('.txt')])
        for filename in files:
            commune_name = filename.replace('.txt', '')
            commune_uri = normalize_uri(commune_name)
            insee_code = insee_data.get(commune_name, "")

            survey_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"survey_{commune_name}"))
            verbatim_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"verbatim_{commune_name}"))

            # Clé = nom du fichier (pour retrouver facilement)
            mapping["sources"][filename] = {
                "commune": commune_name,
                "insee_code": insee_code,
                "survey": {
                    "source_id": survey_uuid,
                    "source_uri": f"source_survey_{commune_uri}",
                    "type": "QuantitativeSurvey"
                },
                "verbatim": {
                    "source_id": verbatim_uuid,
                    "source_uri": f"source_verbatim_{commune_uri}",
                    "type": "Verbatim"
                }
            }

    # 2. Mapping pour les articles Wikipedia
    for _, row in wiki_df.iterrows():
        commune_name = row['commune']
        commune_uri = normalize_uri(commune_name)
        insee_code = insee_data.get(commune_name, "")
        wiki_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"wiki_{commune_name}"))

        wiki_key = f"wiki_{commune_name}"
        mapping["sources"][wiki_key] = {
            "commune": commune_name,
            "insee_code": insee_code,
            "wiki": {
                "source_id": wiki_uuid,
                "source_uri": f"source_wiki_{commune_uri}",
                "type": "WikipediaArticle"
            }
        }

    # 3. Mapping pour les entretiens
    if os.path.exists(entretiens_path):
        with open(entretiens_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r'### Commune : ([^|]+) \| Entretien (\d+)'
        matches = re.findall(pattern, content)

        for commune_name, entretien_num in matches:
            commune_name = commune_name.strip()
            commune_uri = normalize_uri(commune_name)

            # Essayer plusieurs variantes pour trouver le code INSEE
            insee_code = insee_data.get(commune_name, "")
            if not insee_code:
                insee_code = insee_data.get(commune_name.replace(" ", "-"), "")
            if not insee_code:
                name_with_dash = commune_name.replace(" ", "-")
                simplified = name_with_dash.replace("tt", "t")
                insee_code = insee_data.get(simplified, "")

            interview_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"interview_{commune_name}_{entretien_num}"))

            interview_key = f"interview_{commune_name}_{entretien_num}"
            mapping["sources"][interview_key] = {
                "commune": commune_name,
                "insee_code": insee_code,
                "interview": {
                    "source_id": interview_uuid,
                    "source_uri": f"source_interview_{commune_uri}_{entretien_num}",
                    "type": "Interview",
                    "entretien_num": entretien_num
                }
            }

    # 4. Datasets statistiques (niveau régional)
    insee_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "insee_corse"))
    survey_agg_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "survey_aggregated_corse"))

    mapping["sources"]["insee_corse"] = {
        "commune": None,
        "insee_code": None,
        "dataset": {
            "source_id": insee_uuid,
            "source_uri": "source_insee_corse",
            "type": "StatisticalDataset"
        }
    }

    mapping["sources"]["survey_aggregated"] = {
        "commune": None,
        "insee_code": None,
        "dataset": {
            "source_id": survey_agg_uuid,
            "source_uri": "source_survey_aggregated",
            "type": "StatisticalDataset"
        }
    }

    return mapping


def load_wiki_data(csv_path: str) -> pd.DataFrame:
    """
    Charge les données Wikipedia (source principale - 360 communes).

    Returns:
        DataFrame avec toutes les communes
    """
    df = pd.read_csv(csv_path)
    return df


def load_survey_data(csv_path: str) -> Dict[str, Dict]:
    """
    Charge les données d'enquête indexées par commune.

    Returns:
        Dictionnaire {commune_name: row_data}
    """
    try:
        df = pd.read_csv(csv_path)
        survey_dict = {}
        for _, row in df.iterrows():
            commune = row['commune']
            survey_dict[commune] = row.to_dict()
        return survey_dict
    except Exception as e:
        print(f"Avertissement: Impossible de charger {csv_path}: {e}")
        return {}


def load_insee_data(csv_path: str) -> Dict[str, str]:
    """
    Charge les codes INSEE depuis data_comp_finalede2604_4_cleaned.csv.

    Returns:
        Dictionnaire {commune_name: insee_code}
    """
    try:
        df = pd.read_csv(csv_path)
        insee_dict = {}
        for _, row in df.iterrows():
            commune = row['libell_x']  # Nom de la commune
            insee = row['insee']       # Code INSEE
            insee_dict[commune] = str(insee)
        return insee_dict
    except Exception as e:
        print(f"Avertissement: Impossible de charger {csv_path}: {e}")
        return {}


def update_instances_file(
    instances_path: str,
    communes_ttl: str,
    departments_ttl: str,
    dimensions_ttl: str,
    indicators_ttl: str,
    datasources_ttl: str = ""
) -> None:
    """
    Met à jour le fichier d'instances avec les nouvelles communes.

    Args:
        instances_path: Chemin vers onto_be_instances.ttl
        communes_ttl: TTL des communes à ajouter
        departments_ttl: TTL des départements/régions
        dimensions_ttl: TTL des dimensions du bien-être
        indicators_ttl: TTL des indicateurs subjectifs et objectifs
        datasources_ttl: TTL des sources de données (optionnel)
    """
    # Lire le fichier existant
    with open(instances_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Chercher le marqueur pour les individus
    marker = "# --- tes individus à partir d'ici ---"

    if marker in content:
        # Insérer après le marqueur
        parts = content.split(marker)
        new_content = parts[0] + marker + "\n\n"
        new_content += f"# === INSTANCES GÉNÉRÉES AUTOMATIQUEMENT ===\n"
        new_content += f"# Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        new_content += f"# Source principale: communes_corse_wikipedia.csv (360 communes)\n"
        new_content += f"# Enrichissement: df_mean_by_commune.csv (données d'enquête)\n\n"
        new_content += dimensions_ttl + "\n"
        new_content += indicators_ttl + "\n"
        new_content += departments_ttl + "\n\n"
        new_content += "# === COMMUNES ===\n\n"
        new_content += communes_ttl
        if datasources_ttl:
            new_content += "\n\n" + datasources_ttl

        # Ne pas conserver d'anciennes instances - tout est régénéré
        # Les sources de données, communes, etc. sont toutes régénérées à chaque exécution
    else:
        # Ajouter à la fin
        new_content = content.rstrip() + "\n\n"
        new_content += f"# === COMMUNES GÉNÉRÉES AUTOMATIQUEMENT ===\n"
        new_content += f"# Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        new_content += departments_ttl + "\n\n"
        new_content += "# === COMMUNES ===\n\n"
        new_content += communes_ttl

    # Écrire le fichier
    with open(instances_path, 'w', encoding='utf-8') as f:
        f.write(new_content)


def main():
    """Fonction principale."""
    print("=" * 60)
    print("PEUPLEMENT DE L'ONTOLOGIE AVEC LES COMMUNES CORSES")
    print("=" * 60)

    # Chemins des fichiers
    wiki_csv = "communes_corse_wikipedia.csv"
    survey_csv = "df_mean_by_commune.csv"
    insee_csv = "data_comp_finalede2604_4_cleaned.csv"
    instances_path = "onto_be_instances.ttl"
    rag_be_path = "rag_be"
    entretiens_path = "entretiens_lea.txt"

    # Charger les données
    print("\n1. Chargement des données...")

    # Source principale: Wikipedia (360 communes)
    wiki_df = load_wiki_data(wiki_csv)
    print(f"   - {len(wiki_df)} communes trouvées dans {wiki_csv} (source principale)")

    # Source secondaire: données d'enquête (68 communes avec répondants)
    survey_data = load_survey_data(survey_csv)
    print(f"   - {len(survey_data)} communes avec données d'enquête dans {survey_csv}")

    # Codes INSEE
    insee_data = load_insee_data(insee_csv)
    print(f"   - {len(insee_data)} codes INSEE chargés depuis {insee_csv}")

    # Générer le TTL pour les dimensions
    print("\n2. Génération des instances de dimensions...")
    dimensions_ttl = generate_dimension_instances()
    print("   - 11 dimensions du bien-être générées")

    # Générer le TTL pour les indicateurs
    print("\n3. Génération des instances d'indicateurs...")
    indicators_ttl = generate_indicator_instances()
    print("   - 9 indicateurs subjectifs générés")
    print("   - 10 indicateurs objectifs générés")

    # Générer le TTL pour les départements
    print("\n4. Génération des instances région/départements...")
    departments_ttl = generate_department_instances()

    # Générer le TTL pour CHAQUE commune du fichier Wikipedia
    print("\n5. Génération des instances de communes...")
    communes_ttl_list = []
    count_with_survey = 0
    count_with_insee = 0
    count_corse_du_sud = 0
    count_haute_corse = 0

    for _, row in wiki_df.iterrows():
        commune_name = row['commune']
        wiki_data = row.to_dict()

        # Chercher les données d'enquête si disponibles
        survey = survey_data.get(commune_name, None)
        if survey:
            count_with_survey += 1

        # Chercher le code INSEE
        insee_code = insee_data.get(commune_name, None)
        if insee_code:
            count_with_insee += 1

        ttl = generate_commune_ttl(commune_name, wiki_data, survey, insee_code)
        communes_ttl_list.append(ttl)

        # Compter par département
        if ':CorseDuSud' in ttl:
            count_corse_du_sud += 1
        else:
            count_haute_corse += 1

        # Afficher uniquement toutes les 50 communes pour ne pas surcharger
        if len(communes_ttl_list) % 50 == 0:
            print(f"   ... {len(communes_ttl_list)} communes traitées")

    print(f"   - {len(communes_ttl_list)} communes générées au total")
    print(f"   - {count_with_insee} avec code INSEE")
    print(f"   - {count_with_survey} avec données d'enquête")
    print(f"   - {count_corse_du_sud} en Corse-du-Sud, {count_haute_corse} en Haute-Corse")

    communes_ttl = "\n\n".join(communes_ttl_list)

    # Générer le TTL pour les sources de données
    print("\n6. Génération des instances de sources de données...")

    # 6.1 Enquêtes quantitatives et verbatims (depuis rag_be)
    datasources_ttl = generate_datasource_instances(rag_be_path, insee_data)
    import os
    num_survey_sources = len([f for f in os.listdir(rag_be_path) if f.endswith('.txt')]) if os.path.exists(rag_be_path) else 0
    print(f"   - {num_survey_sources} enquêtes quantitatives générées")
    print(f"   - {num_survey_sources} sources de verbatims générées")

    # 6.2 Articles Wikipedia (un par commune)
    wikipedia_ttl = generate_wikipedia_instances(wiki_df, insee_data)
    print(f"   - {len(wiki_df)} articles Wikipedia générés")

    # 6.3 Entretiens qualitatifs
    interviews_ttl = generate_interview_instances(entretiens_path, insee_data)
    import re
    if os.path.exists(entretiens_path):
        with open(entretiens_path, 'r', encoding='utf-8') as f:
            num_interviews = len(re.findall(r'### Commune :', f.read()))
    else:
        num_interviews = 0
    print(f"   - {num_interviews} entretiens générés")

    # 6.4 Jeux de données statistiques
    statistical_ttl = generate_statistical_dataset_instances()
    print(f"   - 2 jeux de données statistiques générés")

    # Combiner toutes les sources de données
    all_datasources_ttl = "\n\n".join([
        datasources_ttl,
        wikipedia_ttl,
        interviews_ttl,
        statistical_ttl
    ])

    total_sources = num_survey_sources * 2 + len(wiki_df) + num_interviews + 2

    # Mettre à jour le fichier d'instances
    print(f"\n7. Mise à jour de {instances_path}...")
    update_instances_file(instances_path, communes_ttl, departments_ttl, dimensions_ttl, indicators_ttl, all_datasources_ttl)

    # Générer le fichier de mapping pour ChromaDB
    print("\n8. Génération du fichier de mapping ChromaDB-Ontologie...")
    mapping = generate_source_mapping(rag_be_path, wiki_df, entretiens_path, insee_data)

    import json
    mapping_path = "source_ontology_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"   - {len(mapping['sources'])} entrées de mapping générées")
    print(f"   - Fichier sauvegardé: {mapping_path}")

    print("\n" + "=" * 60)
    print(f"TERMINÉ! 11 dimensions + 19 indicateurs + 4 entités géo + {len(communes_ttl_list)} communes + {total_sources} sources ajoutées à {instances_path}")
    print(f"         + Mapping ChromaDB-Ontologie: {mapping_path}")
    print("=" * 60)

    # Afficher un aperçu
    print("\nAperçu du TTL généré (3 premières communes):")
    print("-" * 40)
    first_three = "\n\n".join(communes_ttl_list[:3])
    print(first_three)


if __name__ == "__main__":
    main()
