// Chemin de base pour les fichiers - détecte automatiquement selon l'environnement
// Détection intelligente basée sur l'URL actuelle
let BASE_PATH;
if (window.location.hostname.includes('github.io')) {
  // GitHub Pages : index.html à la racine
  BASE_PATH = 'stage_ambroise/Code/WEB/';
} else if (window.location.pathname.includes('/Code/WEB/')) {
  // Local : index.html dans Code/WEB/
  BASE_PATH = '';
} else {
  // Local : index.html à la racine du projet
  BASE_PATH = 'stage_ambroise/Code/WEB/';
}
console.log('🔍 BASE_PATH détecté:', BASE_PATH, '| URL:', window.location.href);

let data_indicateursOriginaux = {}
    let indiceFinale = {}
    let scoresParCommune = {}
    let indicateursCommune = {}
    let communeJson = {}
    let modeCalculPk = 'betti'; // 'egal' | 'betti'
    let scoresParCommuneRaw01 = {}; // scores 0-1 (pour calcul p_k Betti)
    let clustersLISA5pct = {}  // Clusters LISA 5% chargés depuis JSON
    let clustersLISA1pct = {}  // Clusters LISA 1% chargés depuis JSON
    let seuilsJenksCharges = {}  // Seuils Jenks chargés depuis seuils_jenks.json
    let cahCarteInitialisee = false  // Flag pour l'initialisation lazy de la carte CAH
    let routesGeojson = {
        nationales: null,
        departementales: null,
        communales: null,
        toutes: null
    }  // Données GeoJSON du réseau routier par type
    let routesLayers = {}  // Couches de routes pour chaque carte et type
    let langueFrancais = false  // Langue par défaut : anglais (checkbox cochée)

// Traductions
const traductions = {
    fr: {
        legendeTitre: "Légende",
        limitesCommunes: "Limites des communes",
        routesPrincipales: "Routes principales",
        onglets: {
            oppchovec: "OppChoVec",
            opp: "Opp",
            cho: "Cho",
            vec: "Vec"
        },
        titresCartes: {
            oppchovec: "Score OppChoVec",
            opp: "Score Opp",
            cho: "Score Cho",
            vec: "Score Vec"
        },
        ui: {
            "commune-title": "2. Sélection commune",
            "commune-select": "-- Sélectionner une commune --",
            "commune-validate": "Valider",
            "commune-info": "Informations de la commune sélectionnée :",
            "routes-title": "3. Affichage des routes",
            "routes-national": "Routes nationales (647)",
            "routes-departmental": "Routes départementales (2654)",
            "routes-municipal": "Routes communales (3686)",
            "routes-all": "Toutes les routes (7002)",
            "language-title": "4. Langue / Language",
            "language-english": "Libellés en anglais"
        },
        descriptions: {
            Indicateur_Opp1: "Avoir une bonne éducation. Se traduit par le niveau de diplôme de la population sur une échelle de 1 à 7.",
            Indicateur_Opp2: "Représente l'indice de Theil qui mesure les inégalités et les proportions des catégories socioprofessionnelles.",
            Indicateur_Opp3: "Avoir les moyens de mobilité. Score basé sur la proportion de ménages avec voiture et l'accès aux transports.",
            Indicateur_Opp4: "Avoir accès aux TIC. Moyenne de la couverture 4G, Internet haut débit et fibre.",
            Indicateur_Cho1: "Ne pas être discriminé. Calculé avec exp(-pourcentage_population_quartiers_prioritaires).",
            Indicateur_Cho2: "Avoir les moyens d'influencer les décisions politiques. Proportion de personnes possédant le droit de vote dans la commune.",
            Indicateur_Vec1: "Avoir un revenu décent. Revenu fiscal médian de la commune.",
            Indicateur_Vec2: "Avoir un logement décent. Score basé sur le confort, la densité d'occupation et le type de logement.",
            Indicateur_Vec3: "Stabilité de l'emploi. Score basé sur la répartition des types de contrats et statuts d'emploi.",
            Indicateur_Vec4: "Être proche des services. Nombre de services de vie courante accessibles en moins de 20 minutes en voiture."
        }
    },
    en: {
        legendeTitre: "Legend",
        limitesCommunes: "Municipal boundaries",
        routesPrincipales: "Main roads",
        onglets: {
            oppchovec: "OppChoLiv",
            opp: "Opp",
            cho: "Cho",
            vec: "Liv"
        },
        titresCartes: {
            oppchovec: "OppChoLiv score",
            opp: "Opp score",
            cho: "Cho score",
            vec: "Liv score"
        },
        ui: {
            "commune-title": "2. Municipality selection",
            "commune-select": "-- Select a municipality --",
            "commune-validate": "Validate",
            "commune-info": "Selected municipality information:",
            "routes-title": "3. Roads display",
            "routes-national": "National roads (647)",
            "routes-departmental": "Departmental roads (2654)",
            "routes-municipal": "Municipal roads (3686)",
            "routes-all": "All roads (7002)",
            "language-title": "4. Language / Langue",
            "language-english": "English labels"
        },
        descriptions: {
            Indicateur_Opp1: "Having a good education. Measured by the education level of the population on a scale of 1 to 7.",
            Indicateur_Opp2: "Represents the Theil index which measures inequalities and proportions of socio-professional categories.",
            Indicateur_Opp3: "Having the means of mobility. Score based on the proportion of households with a car and access to transport.",
            Indicateur_Opp4: "Having access to ICT. Average of 4G coverage, high-speed Internet and fiber.",
            Indicateur_Cho1: "Not being discriminated against. Calculated with exp(-percentage_population_priority_areas).",
            Indicateur_Cho2: "Having the means to influence political decisions. Proportion of people with voting rights in the municipality.",
            Indicateur_Vec1: "Having a decent income. Median tax income of the municipality.",
            Indicateur_Vec2: "Having decent housing. Score based on comfort, occupancy density and housing type.",
            Indicateur_Vec3: "Job stability. Score based on the distribution of contract types and employment status.",
            Indicateur_Vec4: "Being close to services. Number of everyday services accessible within 20 minutes by car."
        }
    }
};

    // 5 cartes différentes + 2 cartes LISA + 2 cartes CAH
    let cartes = {
        oppchovec: null,
        opp: null,
        cho: null,
        vec: null,
        'lisa-5pct': null,
        'lisa-1pct': null,
        'cah-3': null,
        'cah-5': null
    };
    let geojsonLayers = {
        oppchovec: null,
        opp: null,
        cho: null,
        vec: null,
        'lisa-5pct': null,
        'lisa-1pct': null,
        'cah-3': null,
        'cah-5': null
    };
    let legendControls = {
        oppchovec: null,
        opp: null,
        cho: null,
        vec: null,
        'lisa-5pct': null,
        'lisa-1pct': null,
        'cah-3': null,
        'cah-5': null
    };
    let communeLayers = {};  // Format: { 'mapType': { 'CommuneName': layer } }
    let comparaisonEnCours = null;
    let Dejasurligner = [];
    let lisaCartesInitialisees = false;  // Flag pour l'initialisation lazy des cartes LISA
    let cahCartesInitialisees = false;  // Flag pour l'initialisation lazy des cartes CAH
    let isSyncing = false;  // Flag pour éviter les boucles infinies de synchronisation


// Seuils de Jenks - seront chargés dynamiquement depuis seuils_jenks.json
let seuilsJenks = {
    oppchovec: [0, 2.29, 3.91, 5.08, 7.26, 10],  // 5 classes - valeurs par défaut basées sur Jenks
    opp: [0, 2.44, 3.75, 4.95, 6.49, 10],
    cho: [0, 2.35, 6.11, 8.20, 9.30, 10],
    vec: [0, 1.78, 3.05, 4.14, 6.14, 10]
};

// Palette de couleurs (5 classes) - Bleu clair vers Violet (inversé pour valeurs croissantes)
const colorsJenks = ["#bbdefb", "#64b5f6", "#9c27b0", "#7b1fa2", "#4a148c"];

// Coordonnées des villes principales de Corse avec positions des labels
const villesPrincipales = [
    { nom: "Ajaccio", lat: 41.9267, lng: 8.7369, labelOffset: { lat: -0.15, lng: -0.25 } },
    { nom: "Bastia", lat: 42.7028, lng: 9.4503, labelOffset: { lat: 0.15, lng: 0.20 } },
    { nom: "Corte", lat: 42.3063, lng: 9.1508, labelOffset: { lat: 0.0, lng: 0.70 } },
    { nom: "Porto-Vecchio", lat: 41.5914, lng: 9.2795, labelOffset: { lat: 0.0, lng: 0.30 } },
    { nom: "Calvi", lat: 42.5677, lng: 8.7575, labelOffset: { lat: 0.15, lng: -0.25 } }
];

// Fonction pour ajouter les marqueurs des villes principales avec lignes de repère
function ajouterVillesPrincipales(carte) {
    // Créer un pane pour les lignes pointillées (arrière-plan)
    const lignesPaneName = 'villesLignesPane';
    if (!carte.getPane(lignesPaneName)) {
        const pane = carte.createPane(lignesPaneName);
        pane.style.zIndex = 490; // Entre les routes (450) et les labels de villes (500)
    }

    // Créer un pane pour les villes avec z-index élevé (au-dessus des routes et des lignes)
    const villesPaneName = 'villesPane';
    if (!carte.getPane(villesPaneName)) {
        const pane = carte.createPane(villesPaneName);
        pane.style.zIndex = 500; // Au-dessus des routes (450) et des lignes (490)
    }

    villesPrincipales.forEach(ville => {
        const posVille = [ville.lat, ville.lng];
        const posLabel = [ville.lat + ville.labelOffset.lat, ville.lng + ville.labelOffset.lng];

        // Créer une ligne de repère (leader line) entre le point et le label
        // Utiliser le pane des lignes pour qu'elles soient en arrière-plan
        L.polyline([posVille, posLabel], {
            color: '#000000',
            weight: 1,
            opacity: 0.6,
            dashArray: '3, 3',  // Ligne pointillée
            pane: lignesPaneName
        }).addTo(carte);

        // Créer un marqueur personnalisé (point noir)
        L.circleMarker(posVille, {
            radius: 5,
            fillColor: "#000000",
            color: "#ffffff",
            weight: 2,
            opacity: 1,
            fillOpacity: 1,
            pane: villesPaneName
        }).addTo(carte);

        // Ajouter le label du nom de la ville à la position décalée
        L.marker(posLabel, {
            icon: L.divIcon({
                className: 'ville-label',
                html: `<div style="
                    font-weight: bold;
                    font-size: 13px;
                    color: #000;
                    background-color: rgba(255, 255, 255, 0.85);
                    padding: 3px 8px;
                    border: 1px solid #000;
                    border-radius: 3px;
                    white-space: nowrap;
                ">${ville.nom}</div>`,
                iconSize: [100, 20],
                iconAnchor: [50, 10]  // Centrer le label
            }),
            pane: villesPaneName
        }).addTo(carte);
    });
}

// Fonction pour ajouter une rose des vents
function ajouterRoseDesVents(carte) {
    // Vérifier si la rose des vents existe déjà
    const mapContainer = carte.getContainer();
    if (mapContainer.querySelector('.rose-des-vents')) {
        console.log('Rose des vents déjà présente, pas de duplication');
        return;
    }

    const roseControl = L.control({ position: 'topleft' });

    roseControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'rose-des-vents');
        div.innerHTML = `
            <svg width="80" height="80" viewBox="0 0 80 80" style="background: rgba(255,255,255,0.9); border-radius: 50%; padding: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                <!-- Cercle extérieur -->
                <circle cx="40" cy="40" r="35" fill="none" stroke="#333" stroke-width="1"/>
                <!-- Flèche Nord (rouge) -->
                <polygon points="40,10 45,35 40,30 35,35" fill="#d73027" stroke="#000" stroke-width="0.5"/>
                <!-- Flèche Sud -->
                <polygon points="40,70 35,45 40,50 45,45" fill="#333" stroke="#000" stroke-width="0.5"/>
                <!-- Flèche Est -->
                <polygon points="70,40 45,35 50,40 45,45" fill="#666" stroke="#000" stroke-width="0.5"/>
                <!-- Flèche Ouest -->
                <polygon points="10,40 35,45 30,40 35,35" fill="#666" stroke="#000" stroke-width="0.5"/>
                <!-- Lettres N, S, E, O -->
                <text x="40" y="8" text-anchor="middle" font-size="10" font-weight="bold" fill="#d73027">N</text>
                <text x="40" y="76" text-anchor="middle" font-size="8" font-weight="bold" fill="#333">S</text>
                <text x="73" y="43" text-anchor="middle" font-size="8" font-weight="bold" fill="#333">E</text>
                <text x="7" y="43" text-anchor="middle" font-size="8" font-weight="bold" fill="#333">O</text>
            </svg>
        `;
        return div;
    };

    roseControl.addTo(carte);
}

// Barre d'échelle fixe à 50 km — style identique au contrôle Leaflet natif
function ajouterEchelle50km(carte) {
    const ctrl = L.control({ position: 'bottomleft' });
    ctrl.onAdd = function(map) {
        const container = L.DomUtil.create('div', 'leaflet-control-scale');
        const line = L.DomUtil.create('div', 'leaflet-control-scale-line', container);
        function update() {
            const c = map.getCenter();
            const p1 = map.latLngToContainerPoint(L.latLng(c.lat, c.lng - 0.5));
            const p2 = map.latLngToContainerPoint(L.latLng(c.lat, c.lng + 0.5));
            const pxPerDeg = Math.abs(p2.x - p1.x);
            const kmPerDeg = 111.32 * Math.cos(c.lat * Math.PI / 180);
            line.style.width = Math.max(20, Math.round(50 * pxPerDeg / kmPerDeg)) + 'px';
            line.innerHTML = '50 km';
        }
        map.on('zoomend moveend', update);
        setTimeout(update, 50);
        L.DomEvent.disableClickPropagation(container);
        return container;
    };
    ctrl.addTo(carte);
}

// Fonction pour ajouter le copyright
function ajouterCopyright(carte) {
    console.log('🔍 ajouterCopyright appelé pour carte:', carte);

    // Vérifier si le copyright existe déjà
    const mapContainer = carte.getContainer();
    const existingCopyright = mapContainer.querySelector('.copyright-control');
    console.log('⚠️ Copyright existant?', existingCopyright);

    if (existingCopyright) {
        console.log('Copyright déjà présent, pas de duplication');
        return;
    }

    const copyrightControl = L.control({ position: 'bottomleft' });

    copyrightControl.onAdd = function() {
        console.log('✅ onAdd du copyright appelé - création du div');
        const div = L.DomUtil.create('div', 'copyright-control');
        div.style.cssText = `
            background: rgba(255, 255, 255, 0.9);
            padding: 6px 10px;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            font-size: 9px;
            color: #666;
            font-family: Arial, sans-serif;
            line-height: 1.3;
            text-align: center;
        `;
        div.innerHTML = '© Ghinevra COMITI, Lise BOURDEAU-LEPAGE 2025 — Tous droits réservés';
        console.log('📝 Div copyright créé:', div);
        return div;
    };

    copyrightControl.addTo(carte);

    // Repositionner uniquement le copyright au centre après ajout
    setTimeout(() => {
        const copyrightDiv = mapContainer.querySelector('.copyright-control');
        if (copyrightDiv) {
            // Trouver le conteneur parent leaflet-bottom leaflet-left
            const bottomLeftContainer = copyrightDiv.closest('.leaflet-bottom.leaflet-left');
            if (bottomLeftContainer) {
                // Créer un nouveau conteneur pour le copyright centré (décalé de 60px vers la gauche)
                const centerContainer = document.createElement('div');
                centerContainer.className = 'leaflet-bottom leaflet-center';
                centerContainer.style.cssText = `
                    position: absolute;
                    left: 50%;
                    transform: translateX(calc(-50% - 60px));
                    bottom: 0;
                    pointer-events: none;
                `;

                // Déplacer le copyright dans ce nouveau conteneur
                centerContainer.appendChild(copyrightDiv);
                copyrightDiv.style.pointerEvents = 'auto';

                // Ajouter le conteneur centré à la carte
                mapContainer.querySelector('.leaflet-control-container').appendChild(centerContainer);
            }
        }
    }, 100);

    console.log('✔️ Copyright control ajouté à la carte');
}

// Fonction pour ajouter une légende des traits (limites et routes)
function ajouterLegendeTraits(carte) {
    const legendeControl = L.control({ position: 'bottomright' });

    legendeControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'legende-traits');
        const lang = langueFrancais ? 'fr' : 'en';
        div.innerHTML = `
            <div style="
                background: rgba(255,255,255,0.95);
                padding: 10px 12px;
                border: 2px solid #333;
                border-radius: 5px;
                font-family: Arial, sans-serif;
                font-size: 12px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            ">
                <div class="legende-titre" style="font-weight: bold; margin-bottom: 8px; font-size: 13px;">${traductions[lang].legendeTitre}</div>

                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <svg width="30" height="3" style="margin-right: 8px;">
                        <line x1="0" y1="1.5" x2="30" y2="1.5" stroke="#ff0000" stroke-width="2" opacity="0.7" />
                    </svg>
                    <span class="legende-routes">${traductions[lang].routesPrincipales}</span>
                </div>

                <div style="display: flex; align-items: center;">
                    <svg width="30" height="2" style="margin-right: 8px;">
                        <line x1="0" y1="1" x2="30" y2="1" stroke="#333" stroke-width="1.5" />
                    </svg>
                    <span class="legende-limites">${traductions[lang].limitesCommunes}</span>
                </div>
            </div>
        `;
        return div;
    };

    legendeControl.addTo(carte);
}

// Fonction pour mettre à jour toutes les légendes
function mettreAJourLegendes() {
    const lang = langueFrancais ? 'fr' : 'en';

    // Définir les catégories LISA pour la traduction
    const categoriesLISA = [
        { color: '#8B4513', labelFr: 'Pôle de bien-être', labelEn: 'Wealth cluster' },
        { color: '#4575b4', labelFr: 'Pôle de mal-être', labelEn: 'Poverty cluster' },
        { color: '#fdae61', labelFr: 'Oasis de bien-être', labelEn: 'Wealth oasis' },
        { color: '#abd9e9', labelFr: 'Poche de mal-être', labelEn: 'Poverty pocket' },
        { color: '#f3f3f3', labelFr: 'Association non significative', labelEn: 'Insignificant association' }
    ];

    // Mettre à jour les légendes LISA
    document.querySelectorAll('.legende-lisa').forEach(legendeDiv => {
        const titreEl = legendeDiv.querySelector('.legende-titre');
        const parentDiv = legendeDiv.closest('.info.legend');

        // Mettre à jour le titre
        if (titreEl) {
            if (parentDiv && parentDiv.textContent.includes('5%')) {
                titreEl.textContent = lang === 'fr' ? 'Clusters LISA (Seuil 5%)' : 'LISA Clusters (Threshold 5%)';
            } else if (parentDiv && parentDiv.textContent.includes('1%')) {
                titreEl.textContent = lang === 'fr' ? 'Clusters LISA (Seuil 1%)' : 'LISA Clusters (Threshold 1%)';
            }
        }

        // Mettre à jour les labels des catégories
        const categoryDivs = legendeDiv.querySelectorAll('div[style*="margin: 4px 0"]');
        categoryDivs.forEach((div, index) => {
            if (index < categoriesLISA.length) {
                const strongEl = div.querySelector('strong');
                if (strongEl) {
                    strongEl.textContent = lang === 'fr' ? categoriesLISA[index].labelFr : categoriesLISA[index].labelEn;
                }
            }
        });
    });

    // Mettre à jour les légendes CAH
    document.querySelectorAll('.legende-cah').forEach(legendeDiv => {
        const titreEl = legendeDiv.querySelector('.legende-titre');
        const parentDiv = legendeDiv.closest('.info.legend');

        if (titreEl && parentDiv) {
            const nClusters = (parentDiv.textContent.match(/(\d+)\s+Cluster/i) || [])[1] || '3';
            titreEl.textContent = lang === 'fr' ? `CAH - ${nClusters} Clusters` : `HAC - ${nClusters} Clusters`;
        }
    });

    // Mettre à jour les titres des cartes choroplèthes (OppChoVec, Opp, Cho, Vec)
    document.querySelectorAll('.info.legend[data-carte-type]').forEach(legendeDiv => {
        // Récupérer le type de carte depuis l'attribut data
        const type = legendeDiv.getAttribute('data-carte-type');

        if (type && traductions[lang].titresCartes[type]) {
            // Trouver tous les éléments strong
            const strongElements = legendeDiv.querySelectorAll('strong');
            // Le titre de la carte est le DEUXIÈME strong (le premier est "Légende")
            if (strongElements.length >= 2) {
                const titreCarte = strongElements[1];
                titreCarte.textContent = traductions[lang].titresCartes[type];
            }

            // Mettre à jour le sous-titre (échelle)
            const smallEl = legendeDiv.querySelector('small[style*="color: #666"]');
            if (smallEl) {
                smallEl.textContent = lang === 'fr' ? 'Échelle de 0 à 10' : '0–10 scale';
            }
        }
    });

    // Mettre à jour le titre principal "Légende" en haut des légendes
    document.querySelectorAll('.legende-titre-principal').forEach(el => {
        el.textContent = traductions[lang].legendeTitre;
    });

    // Mettre à jour les autres légendes
    document.querySelectorAll('.legende-titre').forEach(el => {
        if (!el.closest('.legende-lisa') && !el.closest('.legende-cah')) {
            el.textContent = traductions[lang].legendeTitre;
        }
    });

    document.querySelectorAll('.legende-limites').forEach(el => {
        el.textContent = traductions[lang].limitesCommunes;
    });
    document.querySelectorAll('.legende-routes').forEach(el => {
        el.textContent = traductions[lang].routesPrincipales;
    });

    // Mettre à jour les titres des onglets
    const tabOppChovec = document.getElementById('tab-oppchovec');
    if (tabOppChovec) {
        tabOppChovec.textContent = traductions[lang].onglets.oppchovec;
    }
    const tabOpp = document.getElementById('tab-opp');
    if (tabOpp) {
        tabOpp.textContent = traductions[lang].onglets.opp;
    }
    const tabCho = document.getElementById('tab-cho');
    if (tabCho) {
        tabCho.textContent = traductions[lang].onglets.cho;
    }
    const tabVec = document.getElementById('tab-vec');
    if (tabVec) {
        tabVec.textContent = traductions[lang].onglets.vec;
    }

    // Mettre à jour les descriptions des indicateurs
    document.querySelectorAll('[data-indicator-desc]').forEach(el => {
        const indicator = el.getAttribute('data-indicator-desc');
        // Retirer le suffixe _comp si présent
        const indicatorKey = indicator.replace('_comp', '');
        if (traductions[lang].descriptions[indicatorKey]) {
            el.textContent = traductions[lang].descriptions[indicatorKey];
        }
    });
}

// Fonctions pour gérer l'affichage des descriptions d'indicateurs
function toggleIndicatorInfo(event, indicatorId) {
    event.preventDefault();
    event.stopPropagation();

    // Pour les indicateurs dans la comparaison, on utilise une ligne de tableau
    if (indicatorId.includes('_comp')) {
        const infoRow = document.getElementById(`info-row-${indicatorId}`);
        if (infoRow) {
            const isHidden = infoRow.style.display === 'none';
            infoRow.style.display = isHidden ? 'table-row' : 'none';
        }
    } else {
        // Pour les indicateurs normaux, on utilise un div
        const infoBox = document.getElementById(`info-${indicatorId}`);
        if (infoBox) {
            const isHidden = infoBox.style.display === 'none';
            infoBox.style.display = isHidden ? 'block' : 'none';
        }
    }
}

function closeIndicatorInfo(indicatorId) {
    // Pour les indicateurs dans la comparaison
    if (indicatorId.includes('_comp')) {
        const infoRow = document.getElementById(`info-row-${indicatorId}`);
        if (infoRow) {
            infoRow.style.display = 'none';
        }
    } else {
        // Pour les indicateurs normaux
        const infoBox = document.getElementById(`info-${indicatorId}`);
        if (infoBox) {
            infoBox.style.display = 'none';
        }
    }
}

// Rendre une div Leaflet draggable (légende déplaçable)
function _makeDraggable(div, map) {
    div.style.cursor = 'grab';
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);
    let dragging = false, startX, startY;
    const onMove = (e) => {
        if (!dragging) return;
        const mapEl = map.getContainer();
        const newLeft = Math.max(0, Math.min(e.clientX - startX, mapEl.offsetWidth  - div.offsetWidth));
        const newTop  = Math.max(0, Math.min(e.clientY - startY, mapEl.offsetHeight - div.offsetHeight));
        div.style.left = newLeft + 'px';
        div.style.top  = newTop  + 'px';
    };
    const onUp = () => { dragging = false; div.style.cursor = 'grab'; };
    div.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        const mapEl  = map.getContainer();
        const mapRect = mapEl.getBoundingClientRect();
        const divRect = div.getBoundingClientRect();
        div.style.position = 'absolute';
        div.style.margin = '0'; div.style.right = 'auto'; div.style.bottom = 'auto';
        div.style.left = (divRect.left - mapRect.left) + 'px';
        div.style.top  = (divRect.top  - mapRect.top)  + 'px';
        div.style.zIndex = '1000';
        mapEl.appendChild(div);
        startX = e.clientX - div.offsetLeft;
        startY = e.clientY - div.offsetTop;
        dragging = true; div.style.cursor = 'grabbing';
        e.preventDefault();
    });
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
}

// Fonction pour ajouter un bouton de téléchargement d'image
function ajouterBoutonTelechargement(carte, mapType) {
    // Vérifier si le bouton de téléchargement existe déjà
    const mapContainer = carte.getContainer();
    const existingButton = Array.from(mapContainer.querySelectorAll('.leaflet-control-container .leaflet-top.leaflet-right .leaflet-bar')).find(el => el.textContent.includes('📷'));
    if (existingButton) {
        console.log('Bouton de téléchargement déjà présent, pas de duplication');
        return;
    }

    const downloadControl = L.control({ position: 'topright' });

    const jenksTypes = ['oppchovec', 'opp', 'cho', 'vec'];

    downloadControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'leaflet-bar leaflet-control download-button-control');

        // Bouton PNG
        const btnPng = L.DomUtil.create('a', '', div);
        btnPng.innerHTML = '📷';
        btnPng.href = '#';
        btnPng.title = 'Télécharger la carte en PNG';
        btnPng.style.cssText = 'width:30px;height:30px;line-height:30px;text-align:center;text-decoration:none;font-size:18px;background:white;cursor:pointer;display:block;';
        L.DomEvent.on(btnPng, 'click', function(e) {
            L.DomEvent.preventDefault(e);
            telechargerCarte(carte, mapType);
        });

        // Bouton GeoJSON (uniquement pour les cartes Jenks)
        if (jenksTypes.includes(mapType)) {
            const btnGeo = L.DomUtil.create('a', '', div);
            btnGeo.innerHTML = '⬇️';
            btnGeo.href = '#';
            btnGeo.title = 'Exporter en GeoJSON (avec classes Jenks)';
            btnGeo.style.cssText = 'width:30px;height:30px;line-height:30px;text-align:center;text-decoration:none;font-size:16px;background:white;cursor:pointer;display:block;border-top:1px solid #ccc;';
            L.DomEvent.on(btnGeo, 'click', function(e) {
                L.DomEvent.preventDefault(e);
                exporterGeoJSONAvecJenks(mapType);
            });
        }

        return div;
    };

    downloadControl.addTo(carte);
}

// Exporter le GeoJSON enrichi avec les classes Jenks
function exporterGeoJSONAvecJenks(type) {
    if (!communeJson || !communeJson.features) {
        alert("GeoJSON des communes non chargé.");
        return;
    }

    const titres = { oppchovec: 'OppChoVec', opp: 'Score_Opp', cho: 'Score_Cho', vec: 'Score_Vec' };
    const seuils = seuilsJenks[type];
    const labels = genererLabelsJenks(seuils);

    // Récupérer le bon dictionnaire de valeurs
    const getValeur = (commune) => {
        if (type === 'oppchovec') return indiceFinale[commune];
        const key = type === 'opp' ? 'Score_Opp' : type === 'cho' ? 'Score_Cho' : 'Score_Vec';
        return scoresParCommune[commune] ? scoresParCommune[commune][key] : undefined;
    };

    // Fonction couleur (identique à getColor dans afficherCarteUnique)
    const getCouleur = (val) => {
        if (val === undefined || val === null) return '#ccc';
        for (let i = 1; i < seuils.length; i++) {
            if (val <= seuils[i]) return colorsJenks[i - 1];
        }
        return colorsJenks[seuils.length - 2];
    };

    // Enrichir les features
    const features = communeJson.features.map(f => {
        const nom = f.properties.nom;
        const valeur = getValeur(nom);
        let classeJenks = null, labelJenks = null, couleur = '#ccc';
        if (valeur !== undefined && valeur !== null) {
            for (let i = 1; i < seuils.length; i++) {
                if (valeur <= seuils[i]) { classeJenks = i; break; }
            }
            if (!classeJenks) classeJenks = seuils.length - 1;
            labelJenks = labels[classeJenks - 1];
            couleur = colorsJenks[classeJenks - 1];
        }
        return {
            ...f,
            properties: {
                ...f.properties,
                valeur: valeur !== undefined ? parseFloat(valeur.toFixed(4)) : null,
                classe_jenks: classeJenks,
                label_jenks: labelJenks,
                couleur_jenks: couleur
            }
        };
    });

    const geojson = {
        type: 'FeatureCollection',
        metadata: {
            indicateur: titres[type],
            date_export: new Date().toISOString().slice(0, 10),
            methode_classification: 'Jenks Natural Breaks',
            seuils_jenks: seuils.map(s => parseFloat(s.toFixed(4))),
            classes: colorsJenks.map((couleur, i) => ({
                classe: i + 1,
                label: labels[i],
                couleur
            }))
        },
        features
    };

    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/geo+json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `carte_${type}_jenks_${new Date().toISOString().slice(0, 10)}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
}

// Fonction pour recadrer une image en supprimant les bords blancs
async function recadrerImageBlancs(dataUrl, useJpeg = false) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = function() {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            // Dessiner l'image sur un canvas temporaire
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // Obtenir les données de pixels
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;

            // Détecter les limites non-blanches (avec marge de tolérance)
            let top = 0, bottom = canvas.height, left = 0, right = canvas.width;
            const threshold = 250; // Seuil pour considérer un pixel comme "blanc"

            // Trouver le bord supérieur
            for (let y = 0; y < canvas.height; y++) {
                let hasContent = false;
                for (let x = 0; x < canvas.width; x++) {
                    const idx = (y * canvas.width + x) * 4;
                    if (data[idx] < threshold || data[idx+1] < threshold || data[idx+2] < threshold) {
                        hasContent = true;
                        break;
                    }
                }
                if (hasContent) {
                    top = Math.max(0, y - 20); // Marge de 20px
                    break;
                }
            }

            // Trouver le bord inférieur
            for (let y = canvas.height - 1; y >= 0; y--) {
                let hasContent = false;
                for (let x = 0; x < canvas.width; x++) {
                    const idx = (y * canvas.width + x) * 4;
                    if (data[idx] < threshold || data[idx+1] < threshold || data[idx+2] < threshold) {
                        hasContent = true;
                        break;
                    }
                }
                if (hasContent) {
                    bottom = Math.min(canvas.height, y + 20); // Marge de 20px
                    break;
                }
            }

            // Trouver le bord gauche
            for (let x = 0; x < canvas.width; x++) {
                let hasContent = false;
                for (let y = 0; y < canvas.height; y++) {
                    const idx = (y * canvas.width + x) * 4;
                    if (data[idx] < threshold || data[idx+1] < threshold || data[idx+2] < threshold) {
                        hasContent = true;
                        break;
                    }
                }
                if (hasContent) {
                    left = Math.max(0, x - 20); // Marge de 20px
                    break;
                }
            }

            // Trouver le bord droit
            for (let x = canvas.width - 1; x >= 0; x--) {
                let hasContent = false;
                for (let y = 0; y < canvas.height; y++) {
                    const idx = (y * canvas.width + x) * 4;
                    if (data[idx] < threshold || data[idx+1] < threshold || data[idx+2] < threshold) {
                        hasContent = true;
                        break;
                    }
                }
                if (hasContent) {
                    right = Math.min(canvas.width, x + 20); // Marge de 20px
                    break;
                }
            }

            // Créer un nouveau canvas avec les dimensions recadrées
            const croppedWidth = right - left;
            const croppedHeight = bottom - top;
            const croppedCanvas = document.createElement('canvas');
            croppedCanvas.width = croppedWidth;
            croppedCanvas.height = croppedHeight;
            const croppedCtx = croppedCanvas.getContext('2d');

            // Remplir avec un fond blanc pour JPEG (JPEG ne supporte pas la transparence)
            if (useJpeg) {
                croppedCtx.fillStyle = '#ffffff';
                croppedCtx.fillRect(0, 0, croppedWidth, croppedHeight);
            }

            // Copier la zone recadrée
            croppedCtx.drawImage(canvas, left, top, croppedWidth, croppedHeight, 0, 0, croppedWidth, croppedHeight);

            // Retourner la nouvelle dataURL en JPEG haute qualité (0.95) ou PNG
            if (useJpeg) {
                resolve(croppedCanvas.toDataURL('image/jpeg', 0.95)); // 0.95 = 95% qualité (très haute qualité)
            } else {
                resolve(croppedCanvas.toDataURL('image/png'));
            }
        };
        img.onerror = reject;
        img.src = dataUrl;
    });
}

// Fonction pour télécharger la carte en PNG
async function telechargerCarte(carte, mapType) {
    try {
        // Masquer temporairement les contrôles de zoom et download
        const zoomControl = carte.getContainer().querySelector('.leaflet-control-zoom');
        const downloadControl = carte.getContainer().querySelector('.leaflet-bar a[title="Télécharger la carte en PNG"]')?.parentElement;

        if (zoomControl) zoomControl.style.display = 'none';
        if (downloadControl) downloadControl.style.display = 'none';

        // Sauvegarder et déplacer temporairement les contrôles vers le centre
        const mapContainer = carte.getContainer();

        // Rose des vents (compass) - déplacer vers la droite uniquement
        const compass = mapContainer.querySelector('.rose-des-vents');
        const compassOriginalLeft = compass?.style.left || '';
        const compassOriginalTop = compass?.style.top || '';
        if (compass) {
            compass.style.left = '120px';  // Translation vers la droite
            // On garde le top original (pas de déplacement vertical)
        }

        // Échelle (scale) - déplacer vers la droite uniquement (garder en bas)
        const scale = mapContainer.querySelector('.leaflet-control-scale');
        const scaleOriginalLeft = scale?.style.left || '';
        const scaleOriginalBottom = scale?.style.bottom || '';
        if (scale) {
            scale.style.left = '120px';  // Translation vers la droite
            // On garde le bottom original (reste en bas)
        }

        // Légendes (bottom-right)
        const legends = mapContainer.querySelectorAll('.leaflet-bottom.leaflet-right .leaflet-control');
        const legendsOriginalPositions = [];
        legends.forEach(legend => {
            legendsOriginalPositions.push({
                element: legend,
                right: legend.style.right || '',
                bottom: legend.style.bottom || ''
            });
            legend.style.right = '80px';
        });

        // Attendre un peu pour que les changements soient appliqués
        await new Promise(resolve => setTimeout(resolve, 300));

        // Capturer la carte en haute résolution (300 DPI pour publication)
        // Multiplier par un facteur de 3.125 pour obtenir 300 DPI (96 DPI de base × 3.125 ≈ 300 DPI)
        const scaleFactor = 3.125;
        const dataUrl = await domtoimage.toPng(mapContainer, {
            quality: 1,
            bgcolor: '#ffffff',
            width: mapContainer.offsetWidth * scaleFactor,
            height: mapContainer.offsetHeight * scaleFactor,
            style: {
                transform: `scale(${scaleFactor})`,
                transformOrigin: 'top left',
                width: mapContainer.offsetWidth + 'px',
                height: mapContainer.offsetHeight + 'px'
            }
        });

        // Restaurer les positions originales des contrôles
        if (compass) {
            compass.style.left = compassOriginalLeft;
            compass.style.top = compassOriginalTop;
        }
        if (scale) {
            scale.style.left = scaleOriginalLeft;
            scale.style.bottom = scaleOriginalBottom;
        }
        legendsOriginalPositions.forEach(({ element, right, bottom }) => {
            element.style.right = right;
            element.style.bottom = bottom;
        });

        // Restaurer les contrôles de zoom et download
        if (zoomControl) zoomControl.style.display = '';
        if (downloadControl) downloadControl.style.display = '';

        // Recadrer l'image et convertir en JPEG haute qualité
        const croppedDataUrl = await recadrerImageBlancs(dataUrl, true); // true = JPEG

        // Télécharger l'image recadrée en JPEG
        const link = document.createElement('a');
        link.download = `carte_${mapType}_${new Date().toISOString().slice(0,10)}.jpg`;
        link.href = croppedDataUrl;
        link.click();

        console.log(`✅ Carte ${mapType} téléchargée`);
    } catch (error) {
        console.error('❌ Erreur lors du téléchargement de la carte:', error);
        alert('Erreur lors de la génération de l\'image. Veuillez réessayer.');

        // Restaurer les contrôles en cas d'erreur
        const zoomControl = carte.getContainer().querySelector('.leaflet-control-zoom');
        const downloadControl = carte.getContainer().querySelector('.leaflet-bar a[title="Télécharger la carte en PNG"]')?.parentElement;
        if (zoomControl) zoomControl.style.display = '';
        if (downloadControl) downloadControl.style.display = '';
    }
}

// Fonction pour charger le réseau routier depuis les fichiers GeoJSON
async function chargerReseauRoutier() {
    // Vérifier si déjà chargé
    if (routesGeojson.nationales && routesGeojson.departementales &&
        routesGeojson.communales && routesGeojson.toutes) {
        console.log('ℹ️ Réseau routier déjà en cache');
        return routesGeojson;
    }

    console.log('📡 Chargement des réseaux routiers...');

    const fichiers = {
        nationales: BASE_PATH + 'routes_nationales.geojson',
        departementales: BASE_PATH + 'routes_departementales.geojson',
        communales: BASE_PATH + 'routes_communales.geojson',
        toutes: BASE_PATH + 'routes_toutes.geojson'
    };

    try {
        const promises = Object.entries(fichiers).map(async ([type, fichier]) => {
            const response = await fetch(fichier);
            if (!response.ok) {
                throw new Error(`Erreur HTTP pour ${fichier}: ${response.status}`);
            }
            const data = await response.json();
            routesGeojson[type] = data;
            console.log(`✅ Routes ${type} chargées: ${data.features.length} routes`);
        });

        await Promise.all(promises);
        console.log('✅ Tous les réseaux routiers chargés');
        return routesGeojson;
    } catch (error) {
        console.error('❌ Erreur lors du chargement du réseau routier:', error);
        return null;
    }
}

// Fonction pour ajouter le réseau routier sur une carte
function ajouterReseauRoutier(carte, mapType) {
    if (!routesGeojson.nationales || !routesGeojson.departementales ||
        !routesGeojson.communales || !routesGeojson.toutes) {
        console.warn('Réseau routier non chargé');
        return;
    }

    // Initialiser les layers pour cette carte si nécessaire
    if (!routesLayers[mapType]) {
        routesLayers[mapType] = {};
    }

    // Créer un pane personnalisé pour les routes avec un z-index élevé
    const paneName = 'routesPane';
    if (!carte.getPane(paneName)) {
        const pane = carte.createPane(paneName);
        pane.style.zIndex = 450; // Au-dessus de overlayPane (400) mais sous les markers (600)
        pane.style.pointerEvents = 'auto';
    }

    // Fonction helper pour créer une couche de routes
    const creerCoucheRoute = (geojsonData) => {
        if (!geojsonData) return null;

        return L.geoJSON(geojsonData, {
            pane: paneName,
            style: {
                color: '#ff0000',  // Rouge pour les routes
                weight: 2,
                opacity: 0.7
            },
            onEachFeature: (feature, layer) => {
                if (feature.properties) {
                    let popupContent = '<div style="font-family: Arial, sans-serif;">';

                    if (feature.properties.num_route) {
                        popupContent += `<strong>Route:</strong> ${feature.properties.num_route}<br>`;
                    }
                    if (feature.properties.class_adm) {
                        popupContent += `<strong>Classification:</strong> ${feature.properties.class_adm}<br>`;
                    }
                    if (feature.properties.toponyme) {
                        popupContent += `<strong>Nom:</strong> ${feature.properties.toponyme}<br>`;
                    }

                    popupContent += '</div>';
                    layer.bindPopup(popupContent);
                }
            }
        });
    };

    // Créer les couches pour chaque type
    routesLayers[mapType] = {
        nationales: creerCoucheRoute(routesGeojson.nationales),
        departementales: creerCoucheRoute(routesGeojson.departementales),
        communales: creerCoucheRoute(routesGeojson.communales),
        toutes: creerCoucheRoute(routesGeojson.toutes)
    };

    // Ajouter les couches cochées par défaut (nationales et départementales)
    mettreAJourAffichageRoutes(carte, mapType);

    console.log(`✅ Réseau routier ajouté sur la carte ${mapType}`);
}

// Fonction pour mettre à jour l'affichage des routes selon les checkboxes
function mettreAJourAffichageRoutes(carte, mapType) {
    if (!routesLayers[mapType]) return;

    const types = ['nationales', 'departementales', 'communales', 'toutes'];

    types.forEach(type => {
        const checkbox = document.getElementById(`checkbox-${type}`);
        const layer = routesLayers[mapType][type];

        if (!layer) return;

        // Retirer la couche si elle existe
        if (carte.hasLayer(layer)) {
            carte.removeLayer(layer);
        }

        // L'ajouter si la checkbox est cochée
        if (checkbox && checkbox.checked) {
            layer.addTo(carte);
        }
    });
}

// Fonction pour générer dynamiquement les labels depuis les seuils
function genererLabelsJenks(seuils) {
    const nbClasses = seuils.length - 1;

    if (nbClasses < 3) {
        console.warn("Nombre de classes insuffisant:", nbClasses);
        return Array(nbClasses).fill(0).map((_, i) => `Classe ${i + 1}`);
    }

    const labels = [];

    // Première classe: ≤ seuil[1]
    labels.push(`≤ ${seuils[1].toFixed(2)}`);

    // Classes intermédiaires: seuil[i] - seuil[i+1]
    for (let i = 1; i < seuils.length - 2; i++) {
        labels.push(`${seuils[i].toFixed(2)} - ${seuils[i + 1].toFixed(2)}`);
    }

    // Dernière classe: > seuil[n-2]
    labels.push(`> ${seuils[seuils.length - 2].toFixed(2)}`);

    return labels;
}

// Fonction pour charger les seuils Jenks depuis le fichier JSON
async function chargerSeuilsJenks() {
    try {
        const response = await fetch(BASE_PATH + 'seuils_jenks_optimal_gvf.json');
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }

        const data = await response.json();
        seuilsJenksCharges = data;

        // Extraire les breaks (seuils) depuis le format GVF
        // Format: {breaks: [s1, s2, ...], gvf: 0.xx, nb_classes: N}
        // Ajouter 0 au début et max à la fin pour avoir [min, s1, s2, ..., max]

        const addMinMax = (breaks, maxVal = 10) => {
            if (!breaks || !Array.isArray(breaks)) return null;
            return [0, ...breaks, maxVal];
        };

        seuilsJenks = {
            oppchovec: addMinMax(data.OppChoVec_0_10?.breaks, 10),
            opp: addMinMax(data.Score_Opp_0_10?.breaks, 10),
            cho: addMinMax(data.Score_Cho_0_10?.breaks, 10),
            vec: addMinMax(data.Score_Vec_0_10?.breaks, 10)
        };

        console.log("✅ Seuils Jenks optimaux (GVF) chargés");
        console.log(`  OppChoVec: ${seuilsJenks.oppchovec?.length - 1 || 0} classes (GVF=${data.OppChoVec_0_10?.gvf?.toFixed(3)})`);
        console.log(`  Score_Opp: ${seuilsJenks.opp?.length - 1 || 0} classes (GVF=${data.Score_Opp_0_10?.gvf?.toFixed(3)})`);
        console.log(`  Score_Cho: ${seuilsJenks.cho?.length - 1 || 0} classes (GVF=${data.Score_Cho_0_10?.gvf?.toFixed(3)})`);
        console.log(`  Score_Vec: ${seuilsJenks.vec?.length - 1 || 0} classes (GVF=${data.Score_Vec_0_10?.gvf?.toFixed(3)})`);

        return true;
    } catch (error) {
        console.error("Erreur lors du chargement de seuils_jenks_optimal_gvf.json:", error);
        console.warn("⚠ Utilisation des seuils par défaut (7 classes)");
        return false;
    }
}

// Fonction pour créer/mettre à jour une carte spécifique
function afficherCarteUnique(mapId, type, geojsonData, indicateursDict, titre) {
    // Initialiser la carte si elle n'existe pas
    if (!cartes[type]) {
        cartes[type] = L.map(mapId, {
            center: [42.0396, 9.0129],
            zoom: 8,
            zoomControl: true,
            attributionControl: false,
            zoomSnap: 0.1,       // Permet des zooms très fins par dixièmes
            zoomDelta: 0.1       // Incrément de zoom très fin pour les boutons +/-
        });

        // Fond blanc au lieu de la carte OpenStreetMap
        cartes[type].getContainer().style.backgroundColor = '#ffffff';

        // Synchroniser le zoom avec toutes les autres cartes
        cartes[type].on('zoomend moveend', function() {
            if (isSyncing) {
                console.log(`[${type}] Synchronisation ignorée (isSyncing = true)`);
                return;  // Éviter les boucles infinies
            }

            console.log(`[${type}] Début synchronisation - zoom: ${cartes[type].getZoom()}, center:`, cartes[type].getCenter());
            isSyncing = true;
            const currentZoom = cartes[type].getZoom();
            const currentCenter = cartes[type].getCenter();

            // Mettre à jour toutes les autres cartes avec le même zoom et centre
            let cartesSync = 0;
            for (const mapKey in cartes) {
                if (cartes[mapKey] && mapKey !== type) {
                    console.log(`  -> Synchronisation de ${mapKey}`);
                    cartes[mapKey].setView(currentCenter, currentZoom, { animate: false });
                    cartesSync++;
                }
            }
            console.log(`[${type}] ${cartesSync} cartes synchronisées`);

            setTimeout(() => {
                isSyncing = false;
                console.log(`[${type}] isSyncing remis à false`);
            }, 100);
        });

        // Barre d'échelle fixe à 50 km
        ajouterEchelle50km(cartes[type]);

        // Améliorer le style de l'échelle pour la rendre plus visible
        setTimeout(() => {
            const scaleElement = cartes[type].getContainer().querySelector('.leaflet-control-scale');
            if (scaleElement) {
                scaleElement.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
                scaleElement.style.padding = '4px 8px';
                scaleElement.style.borderRadius = '4px';
                scaleElement.style.border = '2px solid #333';
                scaleElement.style.fontWeight = 'bold';
                scaleElement.style.fontSize = '13px';
                scaleElement.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
            }
        }, 100);
    }

    // Utiliser les seuils de Jenks pour ce type de carte (8 seuils = 7 classes)
    const seuils = seuilsJenks[type] || [0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10];

    const getColor = (value) => {
        if (value === undefined || value === null || isNaN(value)) return "#cccccc";

        // Attribution dynamique de couleur en fonction des seuils
        for (let i = 1; i < seuils.length - 1; i++) {
            if (value <= seuils[i]) {
                return colorsJenks[i - 1];
            }
        }
        // Dernière classe (valeurs > avant-dernier seuil)
        return colorsJenks[seuils.length - 2];
    };

    // Supprimer anciennes couches si existantes
    if (geojsonLayers[type]) {
        cartes[type].removeLayer(geojsonLayers[type]);
    }
    if (legendControls[type]) {
        cartes[type].removeControl(legendControls[type]);
    }

    // Créer la couche GeoJSON
    geojsonLayers[type] = L.geoJSON(geojsonData, {
        style: feature => {
            const name = feature.properties.nom;
            const val = indicateursDict[name];
            return {
                fillColor: val !== undefined ? getColor(val) : "#ccc",
                color: "#000000",  // Contours noirs
                weight: 1,
                fillOpacity: 0.7
            };
        },
        onEachFeature: (feature, layer) => {
            const name = feature.properties.nom;
            const val = indicateursDict[name];
            // Sauvegarder chaque couche par type de carte et nom de commune
            if (!communeLayers[type]) {
                communeLayers[type] = {};
            }
            communeLayers[type][name] = layer;
            layer.bindPopup(`<strong>${name}</strong><br>${titre}: ${val !== undefined ? val.toFixed(2) : 'N/A'}/10`);
        }
    }).addTo(cartes[type]);

    // Ajouter une légende avec seuils de Jenks
    legendControls[type] = L.control({ position: 'bottomright' });

    legendControls[type].onAdd = function () {
        const div = L.DomUtil.create('div', 'info legend');
        // Ajouter un attribut pour identifier le type de carte
        div.setAttribute('data-carte-type', type);

        // Générer les labels dynamiquement depuis les seuils
        const labels = genererLabelsJenks(seuils);

        // Utiliser les traductions pour le titre selon la langue
        const lang = langueFrancais ? 'fr' : 'en';
        const titreFinal = traductions[lang].titresCartes[type] || titre;
        const sousTitre = lang === 'fr' ? 'Échelle de 0 à 10' : '0–10 scale';

        // Titre principal "Légende" en haut
        div.innerHTML += `<strong class="legende-titre-principal" style="font-size: 14px;">${traductions[lang].legendeTitre}</strong><br>`;
        div.innerHTML += `<hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">`;

        // Sous-titre de la carte
        div.innerHTML += `<strong>${titreFinal}</strong><br>`;
        div.innerHTML += `<small style="color: #666;">${sousTitre}</small><br><br>`;

        for (let i = 0; i < colorsJenks.length; i++) {
            div.innerHTML +=
                `<i style="background:${colorsJenks[i]}; width:18px; height:18px; display:inline-block; margin-right:5px;"></i> ` +
                `${labels[i]}<br>`;
        }

        // Ajouter un séparateur pour les traits
        div.innerHTML += `<hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">`;

        // Routes principales
        div.innerHTML += `
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <svg width="30" height="3" style="margin-right: 8px;">
                    <line x1="0" y1="1.5" x2="30" y2="1.5" stroke="#ff0000" stroke-width="2" opacity="0.7" />
                </svg>
                <span class="legende-routes" style="font-size: 11px;">${traductions[lang].routesPrincipales}</span>
            </div>
        `;

        // Limites communales
        div.innerHTML += `
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <svg width="30" height="2" style="margin-right: 8px;">
                    <line x1="0" y1="1" x2="30" y2="1" stroke="#333" stroke-width="1.5" />
                </svg>
                <span class="legende-limites" style="font-size: 11px;">${traductions[lang].limitesCommunes}</span>
            </div>
        `;

        return div;
    };

    legendControls[type].addTo(cartes[type]);

    // Ajouter le réseau routier, les villes principales, la rose des vents, le copyright et le bouton de téléchargement
    ajouterReseauRoutier(cartes[type], type);
    ajouterVillesPrincipales(cartes[type]);
    ajouterRoseDesVents(cartes[type]);
    ajouterCopyright(cartes[type]);
    ajouterBoutonTelechargement(cartes[type], type);
}

// Fonction principale pour afficher toutes les cartes
function afficherToutesLesCartes(geojsonData, indiceFinal, scores) {
    // Extraire les scores Opp, Cho, Vec (normalisés 0-10)
    const scoresOpp = {};
    const scoresCho = {};
    const scoresVec = {};

    for (const commune in scores) {
        scoresOpp[commune] = scores[commune].Score_Opp;
        scoresCho[commune] = scores[commune].Score_Cho;
        scoresVec[commune] = scores[commune].Score_Vec;
    }

    // Afficher chaque carte (sauf LISA qui sera chargé au clic)
    afficherCarteUnique('map-oppchovec', 'oppchovec', geojsonData, indiceFinal, 'OppChoVec');
    afficherCarteUnique('map-opp', 'opp', geojsonData, scoresOpp, 'Score Opp');
    afficherCarteUnique('map-cho', 'cho', geojsonData, scoresCho, 'Score Cho');
    afficherCarteUnique('map-vec', 'vec', geojsonData, scoresVec, 'Score Vec');

    // Si les cartes LISA ont déjà été initialisées, les mettre à jour
    if (lisaCartesInitialisees) {
        afficherCarteLISA('map-lisa-5pct', 'lisa-5pct', geojsonData, indiceFinal, clustersLISA5pct, '5%');
        afficherCarteLISA('map-lisa-1pct', 'lisa-1pct', geojsonData, indiceFinal, clustersLISA1pct, '1%');
    }
}

// Fonction pour charger les clusters LISA depuis les données intégrées
function chargerClustersLISA() {
    try {
        console.log("Chargement des clusters LISA...");

        // Vérifier si LISA_DATA (5%) est disponible
        if (typeof LISA_DATA === 'undefined') {
            throw new Error("LISA_DATA non disponible - fichier lisa_data.js manquant ?");
        }

        // Vérifier si LISA_DATA_1PCT (1%) est disponible
        if (typeof LISA_DATA_1PCT === 'undefined') {
            throw new Error("LISA_DATA_1PCT non disponible - fichier lisa_data_1pct.js manquant ?");
        }

        // Extraire les clusters depuis LISA_DATA (5%)
        const clusters5pct = {};
        for (const [commune, info] of Object.entries(LISA_DATA.clusters)) {
            clusters5pct[commune] = info.cluster;
        }

        // Extraire les clusters depuis LISA_DATA_1PCT (1%)
        const clusters1pct = {};
        for (const [commune, info] of Object.entries(LISA_DATA_1PCT.clusters)) {
            clusters1pct[commune] = info.cluster;
        }

        console.log(`✓ ${Object.keys(clusters5pct).length} clusters LISA 5% chargés`);
        console.log(`  Moran I global: ${LISA_DATA.metadata.moran_global_I.toFixed(4)}`);
        console.log(`  Communes significatives (5%): ${LISA_DATA.metadata.nb_significatives} (${LISA_DATA.metadata.pourcent_significatives.toFixed(1)}%)`);
        console.log("  Répartition (5%):", LISA_DATA.statistiques);

        console.log(`✓ ${Object.keys(clusters1pct).length} clusters LISA 1% chargés`);
        console.log(`  Communes significatives (1%): ${LISA_DATA_1PCT.metadata.nb_significatives} (${LISA_DATA_1PCT.metadata.pourcent_significatives.toFixed(1)}%)`);
        console.log("  Répartition (1%):", LISA_DATA_1PCT.statistiques);

        return { clusters5pct, clusters1pct };
    } catch (error) {
        console.error("Erreur lors du chargement des clusters LISA:", error);
        console.warn("⚠ Utilisation de clusters par défaut (Non significatif)");
        // Retourner des objets vides en cas d'erreur
        return { clusters5pct: {}, clusters1pct: {} };
    }
}

// Fonction pour initialiser les deux cartes LISA (appelée au premier clic sur l'onglet LISA)
function initialiserCartesLISA() {
    if (!lisaCartesInitialisees) {
        console.log("=== Initialisation des cartes LISA (lazy loading) ===");
        afficherCarteLISA('map-lisa-5pct', 'lisa-5pct', communeJson, indiceFinale, clustersLISA5pct, '5%');
        afficherCarteLISA('map-lisa-1pct', 'lisa-1pct', communeJson, indiceFinale, clustersLISA1pct, '1%');
        lisaCartesInitialisees = true;
        console.log("✅ Cartes LISA initialisées");

        // Invalider la taille de la carte active (LISA 5% par défaut)
        setTimeout(() => {
            if (cartes['lisa-5pct']) {
                cartes['lisa-5pct'].invalidateSize();
            }
        }, 100);
    }
}

// Fonction pour afficher une carte LISA (5% ou 1%)
function afficherCarteLISA(mapId, mapType, geojsonData, indiceFinal, clustersLISA, seuil) {
    console.log(`=== Affichage carte LISA ${seuil} ===`);
    console.log("Nombre de clusters disponibles:", Object.keys(clustersLISA).length);
    console.log("Premiers clusters:", Object.keys(clustersLISA).slice(0, 5));

    // Initialiser la carte si elle n'existe pas
    if (!cartes[mapType]) {
        cartes[mapType] = L.map(mapId, {
            center: [42.0396, 9.0129],
            zoom: 8,
            zoomControl: true,
            attributionControl: false,
            zoomSnap: 0.1,       // Permet des zooms très fins par dixièmes
            zoomDelta: 0.1       // Incrément de zoom très fin pour les boutons +/-
        });

        cartes[mapType].getContainer().style.backgroundColor = '#ffffff';

        // Synchroniser le zoom avec toutes les autres cartes
        cartes[mapType].on('zoomend moveend', function() {
            if (isSyncing) {
                console.log(`[${mapType}] Synchronisation ignorée (isSyncing = true)`);
                return;  // Éviter les boucles infinies
            }

            console.log(`[${mapType}] Début synchronisation - zoom: ${cartes[mapType].getZoom()}, center:`, cartes[mapType].getCenter());
            isSyncing = true;
            const currentZoom = cartes[mapType].getZoom();
            const currentCenter = cartes[mapType].getCenter();

            // Mettre à jour toutes les autres cartes avec le même zoom et centre
            let cartesSync = 0;
            for (const mapKey in cartes) {
                if (cartes[mapKey] && mapKey !== mapType) {
                    console.log(`  -> Synchronisation de ${mapKey}`);
                    cartes[mapKey].setView(currentCenter, currentZoom, { animate: false });
                    cartesSync++;
                }
            }
            console.log(`[${mapType}] ${cartesSync} cartes synchronisées`);

            setTimeout(() => {
                isSyncing = false;
                console.log(`[${mapType}] isSyncing remis à false`);
            }, 100);
        });

        // Barre d'échelle fixe à 50 km
        ajouterEchelle50km(cartes[mapType]);

        // Améliorer le style de l'échelle pour la rendre plus visible
        setTimeout(() => {
            const scaleElement = cartes[mapType].getContainer().querySelector('.leaflet-control-scale');
            if (scaleElement) {
                scaleElement.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
                scaleElement.style.padding = '4px 8px';
                scaleElement.style.borderRadius = '4px';
                scaleElement.style.border = '2px solid #333';
                scaleElement.style.fontWeight = 'bold';
                scaleElement.style.fontSize = '13px';
                scaleElement.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
            }
        }, 100);
    }

    // Palette de couleurs LISA
    const colorsLISA = {
        'Non significatif': '#f3f3f3',
        'HH (High-High)': '#8B4513',
        'LL (Low-Low)': '#4575b4',
        'LH (Low-High)': '#abd9e9',
        'HL (High-Low)': '#fdae61'
    };

    // Supprimer anciennes couches si existantes
    if (geojsonLayers[mapType]) {
        cartes[mapType].removeLayer(geojsonLayers[mapType]);
    }

    // Statistiques de debug
    let clustersUtilises = {};
    let communesNonTrouvees = [];

    // Créer la couche GeoJSON
    geojsonLayers[mapType] = L.geoJSON(geojsonData, {
        style: feature => {
            const name = feature.properties.nom;
            const cluster = clustersLISA[name] || 'Non significatif';

            // Debug
            if (!clustersLISA[name]) {
                communesNonTrouvees.push(name);
            }
            clustersUtilises[cluster] = (clustersUtilises[cluster] || 0) + 1;

            return {
                fillColor: colorsLISA[cluster],
                color: "#000000",
                weight: 1,
                fillOpacity: 0.7
            };
        },
        onEachFeature: (feature, layer) => {
            const name = feature.properties.nom;
            const cluster = clustersLISA[name] || 'Non significatif';
            const val = indiceFinal[name];

            // Sauvegarder chaque couche par type de carte et nom de commune
            if (!communeLayers[mapType]) {
                communeLayers[mapType] = {};
            }
            communeLayers[mapType][name] = layer;

            // Créer un popup informatif
            let popupHTML = `<strong>${name}</strong><br>`;
            popupHTML += `<strong>Cluster LISA (${seuil}):</strong> ${cluster}<br>`;
            popupHTML += `<strong>OppChoVec:</strong> ${val !== undefined ? val.toFixed(2) : 'N/A'}/10`;

            layer.bindPopup(popupHTML);
        }
    }).addTo(cartes[mapType]);

    // Afficher les statistiques
    console.log(`Clusters utilisés (${seuil}):`, clustersUtilises);
    if (communesNonTrouvees.length > 0) {
        console.warn(`Communes non trouvées dans LISA ${seuil}:`, communesNonTrouvees.slice(0, 10));
    }

    // Ajouter le réseau routier, les villes principales, la rose des vents, le copyright, la légende et le bouton de téléchargement
    ajouterReseauRoutier(cartes[mapType], mapType);
    ajouterVillesPrincipales(cartes[mapType]);
    ajouterRoseDesVents(cartes[mapType]);
    ajouterCopyright(cartes[mapType]);
    ajouterLegendeLISA(cartes[mapType], seuil);
    ajouterBoutonTelechargement(cartes[mapType], mapType);
}

// Fonction pour ajouter une légende LISA conventionnelle (style Leaflet)
function ajouterLegendeLISA(carte, seuil) {
    const legendeControl = L.control({ position: 'bottomright' });

    legendeControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'info legend legende-lisa');
        const lang = langueFrancais ? 'fr' : 'en';

        const categories = [
            { color: '#8B4513', labelFr: 'Pôle de bien-être', labelEn: 'Wealth cluster' },
            { color: '#4575b4', labelFr: 'Pôle de mal-être', labelEn: 'Poverty cluster' },
            { color: '#fdae61', labelFr: 'Oasis de bien-être', labelEn: 'Wealth oasis' },
            { color: '#abd9e9', labelFr: 'Poche de mal-être', labelEn: 'Poverty pocket' },
            { color: '#f3f3f3', labelFr: 'Association non significative', labelEn: 'Insignificant association' }
        ];

        div.innerHTML = `
            <div style="background: rgba(255,255,255,0.95); padding: 8px 10px; border: 2px solid #333; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                <div class="legende-titre" style="margin-bottom: 6px; font-size: 12px;">
                    ${lang === 'fr' ? `Clusters LISA (Seuil ${seuil})` : `LISA Clusters (Threshold ${seuil})`}
                </div>
                ${categories.map(cat => `
                    <div style="margin: 3px 0; display: flex; align-items: center;">
                        <span style="display: inline-block; width: 16px; height: 16px; background-color: ${cat.color}; border: 1px solid #333; margin-right: 6px;"></span>
                        <span style="font-size: 11px;">${lang === 'fr' ? cat.labelFr : cat.labelEn}</span>
                    </div>
                `).join('')}

                <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                <div style="margin-bottom: 6px; font-size: 11px;">${traductions[lang].legendeTitre}</div>

                <div style="display: flex; align-items: center; margin: 4px 0;">
                    <svg width="26" height="3" style="margin-right: 6px;">
                        <line x1="0" y1="1.5" x2="26" y2="1.5" stroke="#ff0000" stroke-width="2" opacity="0.7" />
                    </svg>
                    <span class="legende-routes" style="font-size: 10px;">${traductions[lang].routesPrincipales}</span>
                </div>

                <div style="display: flex; align-items: center; margin: 4px 0;">
                    <svg width="26" height="2" style="margin-right: 6px;">
                        <line x1="0" y1="1" x2="26" y2="1" stroke="#333" stroke-width="1.5" />
                    </svg>
                    <span class="legende-limites" style="font-size: 10px;">${traductions[lang].limitesCommunes}</span>
                </div>
            </div>
        `;

        // Empêcher les clics sur la légende de se propager à la carte
        L.DomEvent.disableClickPropagation(div);

        return div;
    };

    legendeControl.addTo(carte);
}

// fonction pour surligner les contours d'une commune sur toutes les cartes
function surlignerCommune(nomCommune, couleur = "orange") {
    // Parcourir tous les types de cartes
    for (const mapType in communeLayers) {
        if (communeLayers[mapType] && communeLayers[mapType][nomCommune]) {
            const layer = communeLayers[mapType][nomCommune];
            layer.setStyle({
                color: couleur,
                weight: 3,
                fillOpacity: 0.7
            });
            layer.bringToFront();
        }
    }
    Dejasurligner.push(nomCommune)
    console.log("Commune surlignée:", nomCommune, "Couleur:", couleur)
}

// fonction pour réinitialiser le surlignement d'une commune sur toutes les cartes
function reinitialiserStyleCommune(nomCommune, indicateursDict) {
    // Parcourir tous les types de cartes
    for (const mapType in communeLayers) {
        if (communeLayers[mapType] && communeLayers[mapType][nomCommune]) {
            const layer = communeLayers[mapType][nomCommune];
            layer.setStyle({
                color: "#000000",  // Noir comme défini dans le style par défaut
                weight: 1,
                fillOpacity: 0.7
            });
        }
    }
    console.log("Style réinitialisé pour:", nomCommune)
}

// ============================================
// FONCTIONS CAH (Classification Hiérarchique Ascendante)
// ============================================

// Fonction pour initialiser les cartes CAH (appelée au premier clic sur l'onglet CAH)
function initialiserCartesCAH() {
    if (!cahCartesInitialisees) {
        console.log("=== Initialisation des cartes CAH (lazy loading) ===");
        afficherCarteCAH('map-cah-3', 'cah-3', communeJson, CAH_DATA_3, 3);
        afficherCarteCAH('map-cah-5', 'cah-5', communeJson, CAH_DATA_5, 5);
        cahCartesInitialisees = true;
        console.log("✅ Cartes CAH initialisées");

        // Invalider la taille de la carte active (CAH 3 par défaut)
        setTimeout(() => {
            if (cartes['cah-3']) {
                cartes['cah-3'].invalidateSize();
            }
        }, 100);
    }
}

// Fonction pour afficher une carte CAH (3 ou 5 clusters)
function afficherCarteCAH(mapId, mapType, geojsonData, cahData, nClusters) {
    console.log(`=== Affichage carte CAH ${nClusters} clusters ===`);

    // Vérifier si cahData est disponible
    if (!cahData || !cahData.clusters) {
        console.error(`CAH_DATA_${nClusters} n'est pas défini. Assurez-vous que cah_data.js est chargé.`);
        return;
    }

    console.log("Nombre de communes avec clusters:", Object.keys(cahData.clusters || {}).length);

    // Initialiser la carte si elle n'existe pas
    if (!cartes[mapType]) {
        cartes[mapType] = L.map(mapId, {
            center: [42.0396, 9.0129],
            zoom: 8,
            zoomControl: true,
            attributionControl: false,
            zoomSnap: 0.1,       // Permet des zooms très fins par dixièmes
            zoomDelta: 0.1       // Incrément de zoom très fin pour les boutons +/-
        });

        cartes[mapType].getContainer().style.backgroundColor = '#ffffff';

        // Synchroniser le zoom avec toutes les autres cartes
        cartes[mapType].on('zoomend moveend', function() {
            if (isSyncing) {
                console.log(`[${mapType}] Synchronisation ignorée (isSyncing = true)`);
                return;  // Éviter les boucles infinies
            }

            console.log(`[${mapType}] Début synchronisation - zoom: ${cartes[mapType].getZoom()}, center:`, cartes[mapType].getCenter());
            isSyncing = true;
            const currentZoom = cartes[mapType].getZoom();
            const currentCenter = cartes[mapType].getCenter();

            // Mettre à jour toutes les autres cartes avec le même zoom et centre
            let cartesSync = 0;
            for (const mapKey in cartes) {
                if (cartes[mapKey] && mapKey !== mapType) {
                    console.log(`  -> Synchronisation de ${mapKey}`);
                    cartes[mapKey].setView(currentCenter, currentZoom, { animate: false });
                    cartesSync++;
                }
            }
            console.log(`[${mapType}] ${cartesSync} cartes synchronisées`);

            setTimeout(() => {
                isSyncing = false;
                console.log(`[${mapType}] isSyncing remis à false`);
            }, 100);
        });

        // Barre d'échelle fixe à 50 km
        ajouterEchelle50km(cartes[mapType]);

        // Améliorer le style de l'échelle pour la rendre plus visible
        setTimeout(() => {
            const scaleElement = cartes[mapType].getContainer().querySelector('.leaflet-control-scale');
            if (scaleElement) {
                scaleElement.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
                scaleElement.style.padding = '4px 8px';
                scaleElement.style.borderRadius = '4px';
                scaleElement.style.border = '2px solid #333';
                scaleElement.style.fontWeight = 'bold';
                scaleElement.style.fontSize = '13px';
                scaleElement.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
            }
        }, 100);
    }

    // Palette de couleurs CAH (jusqu'à 5 clusters)
    const colorsCAH = {
        1: '#917648',  // Marron - Cluster 1
        2: '#9e9e9e',  // Gris - Cluster 2
        3: '#61e75c',  // Vert - Cluster 3
        4: '#de7eed',  // Violet - Cluster 4
        5: '#f4b474'   // Orange - Cluster 5
    };

    // Supprimer anciennes couches si existantes
    if (geojsonLayers[mapType]) {
        cartes[mapType].removeLayer(geojsonLayers[mapType]);
    }

    // Fonction de style pour chaque commune
    function styleCommune(feature) {
        const nomCommune = feature.properties.nom?.trim();
        const clusterInfo = cahData.clusters?.[nomCommune];

        if (!clusterInfo) {
            console.warn("Commune sans cluster:", nomCommune);
            return {
                fillColor: '#cccccc',
                fillOpacity: 0.6,
                color: '#000000',
                weight: 1
            };
        }

        const cluster = clusterInfo.cluster;
        const fillColor = colorsCAH[cluster] || '#cccccc';

        return {
            fillColor: fillColor,
            fillOpacity: 0.7,
            color: '#000000',
            weight: 1
        };
    }

    // Ajouter la couche GeoJSON
    geojsonLayers[mapType] = L.geoJSON(geojsonData, {
        style: styleCommune,
        onEachFeature: function(feature, layer) {
            const nomCommune = feature.properties.nom?.trim();
            const clusterInfo = cahData.clusters?.[nomCommune];

            // Enregistrer la couche par type de carte et nom de commune
            if (!communeLayers[mapType]) {
                communeLayers[mapType] = {};
            }
            communeLayers[mapType][nomCommune] = layer;

            // Contenu du popup
            let popupContent = `<div style="font-family: Arial, sans-serif;">
                <h3 style="margin: 0 0 10px 0; color: #333;">${nomCommune}</h3>`;

            if (clusterInfo) {
                const cluster = clusterInfo.cluster;
                const colorCluster = colorsCAH[cluster];

                popupContent += `
                    <div style="margin-bottom: 8px;">
                        <span style="display: inline-block; width: 20px; height: 20px; background-color: ${colorCluster}; border: 1px solid #333; margin-right: 8px; vertical-align: middle;"></span>
                        <strong>Cluster ${cluster}</strong> (${nClusters} clusters)
                    </div>
                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
                    <p style="margin: 5px 0;"><strong>Score Opportunités:</strong> ${clusterInfo.Score_Opp.toFixed(2)}</p>
                    <p style="margin: 5px 0;"><strong>Score Choix:</strong> ${clusterInfo.Score_Cho.toFixed(2)}</p>
                    <p style="margin: 5px 0;"><strong>Score Vécu:</strong> ${clusterInfo.Score_Vec.toFixed(2)}</p>
                    <p style="margin: 5px 0;"><strong>OppChoVec:</strong> ${clusterInfo.OppChoVec.toFixed(2)}</p>
                `;
            } else {
                popupContent += `<p style="color: #999;">Données de cluster non disponibles</p>`;
            }

            popupContent += `</div>`;

            layer.bindPopup(popupContent);

            // Événements de survol
            layer.on('mouseover', function() {
                this.setStyle({
                    weight: 3,
                    color: '#333',
                    fillOpacity: 0.9
                });
                this.bringToFront();
            });

            layer.on('mouseout', function() {
                geojsonLayers[mapType].resetStyle(this);
            });
        }
    }).addTo(cartes[mapType]);

    // Ajouter le réseau routier, les villes principales, la rose des vents, le copyright, la légende et le bouton de téléchargement
    ajouterReseauRoutier(cartes[mapType], mapType);
    ajouterVillesPrincipales(cartes[mapType]);
    ajouterRoseDesVents(cartes[mapType]);
    ajouterCopyright(cartes[mapType]);
    ajouterLegendeCAH(cartes[mapType], nClusters, cahData);
    ajouterBoutonTelechargement(cartes[mapType], mapType);

    console.log(`✅ Carte CAH ${nClusters} clusters affichée avec succès`);
}

// Fonction pour ajouter une légende CAH conventionnelle (style Leaflet)
function ajouterLegendeCAH(carte, nClusters, cahData) {
    const legendeControl = L.control({ position: 'bottomright' });

    legendeControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'info legend legende-cah');
        const lang = langueFrancais ? 'fr' : 'en';

        // Palette de couleurs CAH
        const colorsCAH = {
            1: '#917648',  // Marron
            2: '#9e9e9e',  // Gris
            3: '#61e75c',  // Vert
            4: '#de7eed',  // Violet
            5: '#f4b474'   // Orange
        };

        // Calculer le nombre de communes par cluster
        const clustersCount = {};
        if (cahData && cahData.clusters) {
            for (const commune of Object.values(cahData.clusters)) {
                clustersCount[commune.cluster] = (clustersCount[commune.cluster] || 0) + 1;
            }
        }

        div.innerHTML = `
            <div style="background: rgba(255,255,255,0.95); padding: 10px 12px; border: 2px solid #333; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                <div class="legende-titre" style="font-weight: bold; margin-bottom: 8px; font-size: 14px;">
                    ${lang === 'fr' ? `CAH - ${nClusters} Clusters` : `HAC - ${nClusters} Clusters`}
                </div>
                ${[1, 2, 3, 4, 5].slice(0, nClusters).map(cluster => `
                    <div style="margin: 4px 0; display: flex; align-items: center;">
                        <span style="display: inline-block; width: 18px; height: 18px; background-color: ${colorsCAH[cluster]}; border: 1px solid #333; margin-right: 8px;"></span>
                        <span style="font-size: 12px;">
                            <strong>Cluster ${cluster}</strong>
                            ${clustersCount[cluster] ? `<span style="font-size: 11px; color: #555;"> (${clustersCount[cluster]} communes)</span>` : ''}
                        </span>
                    </div>
                `).join('')}

                <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: 12px;">${traductions[lang].legendeTitre}</div>

                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <svg width="30" height="3" style="margin-right: 8px;">
                        <line x1="0" y1="1.5" x2="30" y2="1.5" stroke="#ff0000" stroke-width="2" opacity="0.7" />
                    </svg>
                    <span class="legende-routes" style="font-size: 11px;">${traductions[lang].routesPrincipales}</span>
                </div>

                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <svg width="30" height="2" style="margin-right: 8px;">
                        <line x1="0" y1="1" x2="30" y2="1" stroke="#333" stroke-width="1.5" />
                    </svg>
                    <span class="legende-limites" style="font-size: 11px;">${traductions[lang].limitesCommunes}</span>
                </div>
            </div>
        `;

        // Empêcher les clics sur la légende de se propager à la carte
        L.DomEvent.disableClickPropagation(div);

        return div;
    };

    legendeControl.addTo(carte);
}


// 1. Utilitaire qui transforme un FileReader en Promise de texte (conservé pour compatibilité)
function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Erreur de lecture du fichier."));
    reader.onload  = () => resolve(reader.result);
    reader.readAsText(file);
  });
}

// Fonction de chargement automatique des fichiers
// Fonction pour générer les visualisations de données (top 10 et histogramme)
function genererDataVisualisations(indiceFinale) {
    console.log("📊 Génération des visualisations de données...");

    // Créer un tableau de [commune, score] et trier par score décroissant
    const communesScores = Object.entries(indiceFinale).map(([commune, score]) => ({
        commune: commune,
        score: score
    })).sort((a, b) => b.score - a.score);

    // Remplir le tableau du top 10
    const tbody = document.getElementById('top10-tbody');
    if (tbody) {
        tbody.innerHTML = '';
        communesScores.slice(0, 10).forEach((item, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${item.commune}</td>
                <td>${item.score.toFixed(2)}</td>
            `;
            tbody.appendChild(row);
        });
    }

    // Créer l'histogramme avec les seuils de Jenks
    const canvas = document.getElementById('histogram-chart');
    if (canvas) {
        const ctx = canvas.getContext('2d');

        // Utiliser les seuils de Jenks pour OppChoVec
        const seuils = seuilsJenks.oppchovec || [0, 2.29, 3.91, 5.08, 7.26, 10];
        const bins = new Array(seuils.length - 1).fill(0);
        const binLabels = [];
        const backgroundColors = [];

        // Générer les labels et couleurs selon les seuils Jenks
        for (let i = 0; i < seuils.length - 1; i++) {
            binLabels.push(`${seuils[i].toFixed(2)} - ${seuils[i + 1].toFixed(2)}`);
            backgroundColors.push(colorsJenks[i]);
        }

        // Compter les communes dans chaque classe Jenks
        communesScores.forEach(item => {
            for (let i = 0; i < seuils.length - 1; i++) {
                if (item.score >= seuils[i] && item.score <= seuils[i + 1]) {
                    bins[i]++;
                    break;
                }
            }
        });

        // Détruire le graphique existant s'il existe
        if (window.oppchovec_histogram) {
            window.oppchovec_histogram.destroy();
        }

        // Créer le nouveau graphique
        window.oppchovec_histogram = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    data: bins,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1,
                            font: {
                                size: 12
                            }
                        },
                        title: {
                            display: true,
                            text: 'Nombre de communes',
                            font: {
                                size: 13,
                                weight: 'bold'
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Classes de score OppChoVec (Jenks)',
                            font: {
                                size: 13,
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            font: {
                                size: 11
                            }
                        }
                    }
                }
            }
        });
    }

    console.log("✅ Visualisations de données générées");
}

async function chargerFichiersAutomatiquement() {
  try {
    console.log("🔄 Chargement automatique des fichiers...");

    // 1. Charger les seuils Jenks et le réseau routier en parallèle
    await Promise.all([
      chargerSeuilsJenks(),
      chargerReseauRoutier()
    ]);

    // 2. Chargement en parallèle des fichiers JSON et GeoJSON
    const geojsonPath = BASE_PATH.replace('Code/WEB/', 'Données/');
    const [responseJson, responseGeo] = await Promise.all([
      fetch(BASE_PATH + 'data_scores_0_10.json'),
      fetch(geojsonPath + 'Commune_Corse.geojson')
    ]);

    if (!responseJson.ok) {
      throw new Error(`Erreur lors du chargement de data_scores_0_10.json: ${responseJson.status}`);
    }
    if (!responseGeo.ok) {
      throw new Error(`Erreur lors du chargement de Commune_Corse.geojson: ${responseGeo.status}`);
    }

    // 3. Parsing du JSON indicateurs
    const dataIndicateurs = await responseJson.json();
    data_indicateursOriginaux = dataIndicateurs;
    console.log("✅ Fichier indicateurs chargé automatiquement:", dataIndicateurs);

    // 4. Parsing et validation du GeoJSON
    const geojsonData = await responseGeo.json();
    if (geojsonData.type !== "FeatureCollection" || !Array.isArray(geojsonData.features)) {
      throw new Error("Invalid GeoJSON : attendu un FeatureCollection avec un tableau 'features'.");
    }
    communeJson = geojsonData;
    console.log("✅ GeoJSON communes chargé automatiquement:", geojsonData);

    // 5. Charger les clusters LISA (5% et 1%)
    const { clusters5pct, clusters1pct } = chargerClustersLISA();
    clustersLISA5pct = clusters5pct;
    clustersLISA1pct = clusters1pct;
    console.log("✅ Clusters LISA 5% et 1% chargés");

    // 6. Suite du traitement
    const data_indicateurs_dict = calculerIndicateurs(dataIndicateurs);
    populateCommuneSelect(data_indicateurs_dict);

    // 7. Appliquer le mode Betti par défaut
    recalculerCarteOppChoVec();

    // 8. Générer les visualisations de données
    if (indiceFinale && Object.keys(indiceFinale).length > 0) {
        genererDataVisualisations(indiceFinale);
    }

    console.log("✅ Chargement automatique terminé avec succès !");

  } catch (err) {
    console.error("❌ Erreur lors du chargement automatique:", err);
    alert("Erreur lors du chargement automatique des fichiers : " + err.message);
  }
}

// Lancer le chargement automatique au démarrage
window.addEventListener('DOMContentLoaded', chargerFichiersAutomatiquement);

// Conserver le bouton de validation manuel pour compatibilité (optionnel)
const validateBtn = document.getElementById("validateBtn");
if (validateBtn) {
  validateBtn.addEventListener("click", async () => {
    const fileJson    = document.getElementById("file").files[0];
    const fileGeoJson = document.getElementById("file_geojson").files[0];

    if (!fileJson || !fileGeoJson) {
      alert("Veuillez sélectionner à la fois un fichier JSON et un GeoJSON.");
    return;
  }

  try {
    console.log("Chargement manuel des fichiers...");

    // 1. Charger les seuils Jenks et le réseau routier en parallèle
    await Promise.all([
      chargerSeuilsJenks(),
      chargerReseauRoutier()
    ]);

    // 2. Lecture en parallèle des fichiers
    const [textJson, textGeo] = await Promise.all([
      readFileAsText(fileJson),
      readFileAsText(fileGeoJson)
    ]);

    // 3. Parsing du JSON indicateurs
    const dataIndicateurs = JSON.parse(textJson);
    data_indicateursOriginaux = dataIndicateurs;
    console.log("✅ Fichier indicateurs chargé :", dataIndicateurs);

    // 4. Parsing et validation du GeoJSON
    const geojsonData = JSON.parse(textGeo);
    if (geojsonData.type !== "FeatureCollection" || !Array.isArray(geojsonData.features)) {
      throw new Error("Invalid GeoJSON : attendu un FeatureCollection avec un tableau 'features'.");
    }
    communeJson = geojsonData;
    console.log("✅ GeoJSON communes chargé :", geojsonData);

    // 5. Charger les clusters LISA (5% et 1%)
    const { clusters5pct, clusters1pct } = chargerClustersLISA();
    clustersLISA5pct = clusters5pct;
    clustersLISA1pct = clusters1pct;
    console.log("✅ Clusters LISA 5% et 1% chargés");

    // 6. Suite du traitement
    const data_indicateurs_dict = calculerIndicateurs(dataIndicateurs);
    populateCommuneSelect(data_indicateurs_dict);
    alert("Validation réussie ! Veuillez sélectionner une commune.");

  } catch (err) {
    alert("Erreur lors de la lecture ou du traitement : " + err.message);
  }

  });
}


  document.getElementById("validerCommune").addEventListener("click", () => {
  const selectedCommune = document.getElementById("communeSelect").value;
  if (!selectedCommune) {
    document.getElementById("resultCommune").innerHTML = "<p style='color:red;'>❌ Veuillez sélectionner une commune.</p>";
    return;
  }
  afficherCommune(selectedCommune);
});

// fonction pour afficher le detail d'une commune
function afficherCommune(communeNom) {

  for (const commune of Dejasurligner){
       reinitialiserStyleCommune(commune, indicateursCommune);
       Dejasurligner.pop(commune)
       console.log(Dejasurligner)
  }
  surlignerCommune(communeNom, "red");

  const resultDiv = document.getElementById("resultCommune");

  const valeurIndice = indiceFinale[communeNom];
  const commune = indicateursCommune[communeNom];
  if (valeurIndice === undefined || !commune) {
    resultDiv.innerHTML = "<p style='color:red;'>❌ Données introuvables pour cette commune.</p>";
    return;
  }

  let indicateursHTML = `
    <p><strong>Commune :</strong> ${communeNom}</p>
    <p><strong>OppChoVec :</strong> ${valeurIndice.toFixed(2)}/10</p>
    <h4>Indicateurs :</h4>
    <ul>
  `;

  const stepsParIndicateur = {
    Indicateur_Opp1: 0.1,
    Indicateur_Opp2: 0.01,
    Indicateur_Opp3: 1,
    Indicateur_Opp4: 1,
    Indicateur_Cho1: 0.01,
    Indicateur_Cho2: 1,
    Indicateur_Vec1: 100,
    Indicateur_Vec2: 0.01,
    Indicateur_Vec3: 0.01,
    Indicateur_Vec4: 1
  };

const bornesParIndicateur = {
  Indicateur_Opp1: { min: 1, max: 7 },
  Indicateur_Opp2: { min: 0, max: 1 },
  Indicateur_Opp3: { min: 0, max: 300 },
  Indicateur_Opp4: { min: 0, max: 100 },
  Indicateur_Cho1: { min: 0, max: 1 },
  Indicateur_Cho2: { min: 0, max: 100 },
  Indicateur_Vec1: { min: 15000, max: 30000 },
  Indicateur_Vec2: { min: 0, max: 1 },
  Indicateur_Vec3: { min: 0, max: 1 },
  Indicateur_Vec4: { min: 0, max: 20 }
};



  for (const [nomIndicateur, valeur] of Object.entries(commune)) {
    const nombre = typeof valeur === 'number' ? Number(valeur).toFixed(2) : valeur;

    const step = stepsParIndicateur[nomIndicateur] || 1;
    const lang = langueFrancais ? 'fr' : 'en';
    const description = traductions[lang].descriptions[nomIndicateur] || descriptionsIndicateurs[nomIndicateur] || "No description available.";

    indicateursHTML += `
      <li class="indicateur-row">
        <div class="indicateur-row-header">
          <strong>${nomIndicateur}</strong>
          <button class="info-button" data-indicator="${nomIndicateur}" onclick="toggleIndicatorInfo(event, '${nomIndicateur}')" aria-label="Information">
            🛈
          </button>
          <span id="${nomIndicateur}_val">${nombre}</span>
        </div>
        <div id="info-${nomIndicateur}" class="indicator-info-box" style="display: none;">
          <button class="info-close-button" onclick="closeIndicatorInfo('${nomIndicateur}')" aria-label="Close">✕</button>
          <p data-indicator-desc="${nomIndicateur}">${description}</p>
        </div>
        <input type="range"
               id="${nomIndicateur}"
               value="${nombre}"
               step="${step}"
               min="${bornesParIndicateur[nomIndicateur]?.min ?? 0}"
               max="${bornesParIndicateur[nomIndicateur]?.max ?? 100}"
               oninput="document.getElementById('${nomIndicateur}_val').innerText = parseFloat(this.value).toFixed(2)" />
      </li>`;
  }

  indicateursHTML += `
  </ul>
  <div style="display: flex; justify-content: space-between; margin-top: 10px;">
    <button onclick="recalculerIndice('${communeNom}')">🔁 Recalculer Indice</button>
    <button onclick="reinitialiserValeurs('${communeNom}')">🔄 Réinitialiser</button>
  </div>

   <div style="text-align: center; margin-top: 1em;">
    <button onclick="afficherComparaison('${communeNom}')">📊 Comparer</button>
  </div>

  <div id="comparaisonCommune" style="display: none; margin-top: 1em;">
    <h2>3. Sélection commune à comparer</h2>
    <select id="communeSelectComparaison" style="width:100%; padding:0.5em;">
      <option value="">-- Sélectionner une commune --</option>
    </select>
    <button id="validerComparaison" style="margin-top: 0.5em;">Valider la comparaison</button>
  </div>
`;

  resultDiv.innerHTML = indicateursHTML;

  // Debug: vérifier que les boutons sont bien dans le DOM
  console.log('🔍 Nombre de boutons .info-button créés:', document.querySelectorAll('.info-button').length);
  const buttons = document.querySelectorAll('.info-button');
  buttons.forEach((btn, index) => {
    console.log(`Bouton ${index}:`, btn, 'Visible?', btn.offsetWidth > 0 && btn.offsetHeight > 0);
  });
}


// fonction pour selectionner une commune de comparaison
  function afficherComparaison(communeNom1) {
  alert("Veuillez sélectionner une commune pour la comparaison.");
  const comparaisonDiv = document.getElementById("comparaisonCommune");
  comparaisonDiv.style.display = "block";

  const select = document.getElementById("communeSelectComparaison");
  select.innerHTML = '<option value="">-- Sélectionner une commune --</option>';

  // Tri par ordre alphabetique
  const communesTriees = Object.keys(indicateursCommune).sort((a, b) => a.localeCompare(b, 'fr'))
  console.log(communesTriees)
  for (const nom of communesTriees) {
    if (nom !== communeNom1) {
      const option = document.createElement("option");
      option.value = nom;
      option.textContent = nom;
      select.appendChild(option);
    }
  }

  document.getElementById("validerComparaison").onclick = () => {
    const communeNom2 = select.value;
    if (!communeNom2) {
      alert("Veuillez sélectionner une commune pour la comparaison.");
      return;
    }
    afficherResultatComparaison(communeNom1, communeNom2);
  };
}

// fonction pour afficher le resultat de comparaison
  function afficherResultatComparaison(commune1, commune2) {

  for (const commune of [...Dejasurligner]){
      console.log(Dejasurligner)
      reinitialiserStyleCommune(commune, indicateursCommune);
      const index = Dejasurligner.indexOf(commune);
      if (index !== -1) {
        Dejasurligner.splice(index, 1); // supprime l'élément à l'index trouvé
        console.log(`${commune} a été supprimée.`);
      } else {
        console.log(`${nomCommune} n'existe pas dans le tableau.`);
      }
       console.log(commune)
       console.log(Dejasurligner)
  }
  console.log(Dejasurligner)
  surlignerCommune(commune1, "red");
  surlignerCommune(commune2, "red");

  comparaisonEnCours = { commune1, commune2 }; // 👈 Sauvegarde
  const data1 = indicateursCommune[commune1];
  const data2 = indicateursCommune[commune2];

  // 🧽 Nettoyer le contenu précédent
  const comparaisonDiv = document.getElementById("comparaisonCommune");
  // Supprime tout sauf le formulaire de sélection
  while (comparaisonDiv.children.length > 3) {
    comparaisonDiv.removeChild(comparaisonDiv.lastChild);
  }

  let html = `
  <h3>Comparaison entre <strong>${commune1}</strong> et <strong>${commune2}</strong></h3>
  <table border="1" style="width:100%; border-collapse: collapse; text-align: center;">
    <thead>
      <tr>
        <th>Indicateur</th>
        <th>${commune1}</th>
        <th>${commune2}</th>
      </tr>
    </thead>
    <tbody>
`;

 const valeurIndice1 = indiceFinale[commune1];
 const valeurIndice2 = indiceFinale[commune2];

for (const indicateur in data1) {
  const lang = langueFrancais ? 'fr' : 'en';
  const description = traductions[lang].descriptions[indicateur] || descriptionsIndicateurs[indicateur] || "No description available.";
  if (data2[indicateur] !== undefined) {
    html += `
      <tr>
        <td style="position: relative;">
          ${indicateur}
          <button class="info-button" data-indicator="${indicateur}" onclick="toggleIndicatorInfo(event, '${indicateur}_comp')" aria-label="Information" style="float: right; margin-left: 8px;">
            🛈
          </button>
        </td>
        <td>${data1[indicateur].toFixed(2)}</td>
        <td>${data2[indicateur].toFixed(2)}</td>
      </tr>
      <tr id="info-row-${indicateur}_comp" style="display: none;">
        <td colspan="3" style="background: #f0f8ff; padding: 10px; border-left: 3px solid #4a90e2;">
          <button class="info-close-button" onclick="closeIndicatorInfo('${indicateur}_comp')" aria-label="Close">✕</button>
          <p data-indicator-desc="${indicateur}_comp">${description}</p>
        </td>
      </tr>
    `;
  }
}

html += `
        <td>${"OppChoVec"}</td>
        <td>${valeurIndice1.toFixed(2)}/10</td>
        <td>${valeurIndice2.toFixed(2)}/10</td>
    </tbody>
  </table>
`;

  const resultDiv = document.createElement("div");
  resultDiv.innerHTML = html;
  comparaisonDiv.appendChild(resultDiv);
}




// fonction de réinitialisation
  function reinitialiserValeurs(commune) {
    calculerIndicateurs(data_indicateursOriginaux)
    afficherCommune(commune);
    alert("Valeurs réinitialisées avec succès");
}

// fonction pour recalcules l'indice après modification des valeurs d'indicateur
  function recalculerIndice(selectedCommune) {
  const communeData = indicateursCommune[selectedCommune];
  if (!communeData) return;

  // Étape 1 : mise à jour des valeurs depuis les champs input
  for (const indicateur in communeData) {
    const input = document.getElementById(indicateur);
    if (input) {
      communeData[indicateur] = parseFloat(input.value);
    }
  }

  // Étape 2 : recalculer à partir des valeurs mises à jour
  indicateursCommune[selectedCommune] = communeData
  const min_vals = (minmax(indicateursCommune)).min
  const max_vals = (minmax(indicateursCommune)).max

  const donneesNormalisees = normaliserDonnees(indicateursCommune, min_vals, max_vals);

  const data_dimensions_scores_dict = calculerScoresParCommune(donneesNormalisees)

  const pkValues = modeCalculPk === 'betti'
      ? calculerPkBetti(data_dimensions_scores_dict)
      : [1, 1, 1];
  const indiceFinalBrut = calculerIndiceBienEtre(data_dimensions_scores_dict, pkValues);

  // Normaliser l'indice final sur 0-10
  const valeursIndice = Object.values(indiceFinalBrut);
  const minIndice = Math.min(...valeursIndice);
  const maxIndice = Math.max(...valeursIndice);

  const indiceFinal = {};
  for (const commune in indiceFinalBrut) {
    if (maxIndice === minIndice) {
      indiceFinal[commune] = 5; // Valeur par défaut si tous égaux
    } else {
      indiceFinal[commune] = ((indiceFinalBrut[commune] - minIndice) / (maxIndice - minIndice)) * 10;
    }
  }

  // Normaliser les scores par dimension sur 0-10
  const scoresNormalises = {};
  for (const commune in data_dimensions_scores_dict) {
    scoresNormalises[commune] = {
      Score_Opp: data_dimensions_scores_dict[commune].Score_Opp * 10,
      Score_Cho: data_dimensions_scores_dict[commune].Score_Cho * 10,
      Score_Vec: data_dimensions_scores_dict[commune].Score_Vec * 10
    };
  }

  indiceFinale = indiceFinal
  scoresParCommune = scoresNormalises

  afficherToutesLesCartes(communeJson, indiceFinale, scoresNormalises);

  // Mettre à jour les visualisations de données
  genererDataVisualisations(indiceFinale);

  valeurIndice = indiceFinale[selectedCommune]
  if (valeurIndice === undefined) {
    const resultDiv = document.getElementById("resultCommune");
    resultDiv.innerHTML = "<p style='color:red;'>❌ Données introuvables pour cette commune.</p>";
    return;
  }

  // Étape 3 : mettre à jour l'affichage
  const resultDiv = document.getElementById("resultCommune");
  let outputHTML = `
    <p><strong>OppChoVec :</strong> ${valeurIndice.toFixed(2)}/10</p>
  `;
  resultDiv.querySelector("p:nth-child(2)").innerHTML = outputHTML;

  alert("Indice recalculé avec succès");

  // Puis si une comparaison est en cours, on met à jour si nécessaire
  if (
    comparaisonEnCours &&
    (comparaisonEnCours.commune1 === selectedCommune || comparaisonEnCours.commune2 === selectedCommune)
  ) {
    afficherResultatComparaison(comparaisonEnCours.commune1, comparaisonEnCours.commune2);
  }
}

// fonction pour ajuster les valeurs d'indicateur lors de la modification
function ajusterValeur(indicateur, delta) {
  const input = document.getElementById(indicateur);
  const display = document.getElementById(indicateur + "_val");

  if (input && display) {
    let nouvelleValeur = parseFloat(input.value) + delta;

    // Clamp la valeur dans les bornes min/max
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    nouvelleValeur = Math.max(min, Math.min(max, nouvelleValeur));

    input.value = nouvelleValeur.toFixed(2);
    display.innerText = nouvelleValeur.toFixed(2);
  }
}


// fonction de rangement et de tri des communes pour la selection
  function populateCommuneSelect(data) {
    const select = document.getElementById("communeSelect");
    const lang = langueFrancais ? 'fr' : 'en';
    select.innerHTML = `<option value="" data-translate="commune-select">${traductions[lang].ui['commune-select']}</option>`; // reset with dynamic translation

    // 🔤 Trier les noms de communes
    const communesTriees = Object.keys(data).sort((a, b) => a.localeCompare(b, 'fr'));

    for (const commune of communesTriees) {
      const option = document.createElement("option");
      option.value = commune;
      option.textContent = commune;
      select.appendChild(option);
    }

    console.log("✅ Communes chargées et triées par ordre alphabétique");
}



// fonctions de calculs des indicateurs

    function calc_opp1(e) {
        return e;
    }

    function calc_opp2(indice) {
        return indice;
    }

    function calc_opp3(voiture, transport) {
        return (voiture + transport) / 2;
    }

    function calc_opp4(internet, debit) {
        return (internet + debit) / 2;
    }

    function calc_cho1(quartier) {
        console.log(Math.exp(-quartier))
        return Math.exp(-quartier);
    }

    function calc_cho2(proportion) {
        return proportion;
    }

    function calc_vec1(revenu) {
        return revenu;
    }

    function calc_vec2(piece, logement, individuel) {
        const vec21 = Math.exp(-piece);
        const vec22 = logement;
        const vec23 = individuel;
        return (vec21 + vec22 + vec23) / 3;
    }

    function calc_vec3(p_vec, valeur) {
        let sum = 0;
        for (let i = 0; i < p_vec.length; i++) {
            sum += p_vec[i] * valeur[i];
        }
        return sum / 100;
    }

    function calc_vec4(n_etablissements) {
        return n_etablissements;
    }


    function calculerIndicateurs(data_indicateurs) {
      // Les données sont déjà calculées dans le JSON avec valeurs normalisées 0-10
      console.log("Chargement des données précalculées (normalisées 0-10)...");

      const data_indicateurs_dict = {};
      const data_dimensions_scores_dict = {};
      const indiceFinal = {};

      // Détecter le format du JSON
      const premiereEntree = Object.entries(data_indicateurs)[0];
      const formatAvecZone = premiereEntree && premiereEntree[1].hasOwnProperty("Zone");

      console.log(`Format JSON détecté: ${formatAvecZone ? 'avec champ Zone' : 'clés directes (communes)'}`);

      for (const [key, valeurs] of Object.entries(data_indicateurs)) {
        // Récupérer le nom de commune selon le format
        let commune;
        if (formatAvecZone) {
          // Format ancien: {"0": {"Zone": "Afa", ...}}
          commune = valeurs["Zone"];
          if (!commune) continue;
        } else {
          // Format nouveau: {"Afa": {...}}
          commune = key;
        }

        // Extraire les indicateurs bruts
        data_indicateurs_dict[commune] = {
          "Indicateur_Opp1": valeurs["Opp1"],
          "Indicateur_Opp2": valeurs["Opp2"],
          "Indicateur_Opp3": valeurs["Opp3"],
          "Indicateur_Opp4": valeurs["Opp4"],
          "Indicateur_Cho1": valeurs["Cho1"],
          "Indicateur_Cho2": valeurs["Cho2"],
          "Indicateur_Vec1": valeurs["Vec1"],
          "Indicateur_Vec2": valeurs["Vec2"],
          "Indicateur_Vec3": valeurs["Vec3"],
          "Indicateur_Vec4": valeurs["Vec4"]
        };

        // Utiliser les scores NORMALISES 0-10
        data_dimensions_scores_dict[commune] = {
          "Score_Opp": valeurs["Score_Opp_0_10"] || valeurs["Score_Opp"],
          "Score_Cho": valeurs["Score_Cho_0_10"] || valeurs["Score_Cho"],
          "Score_Vec": valeurs["Score_Vec_0_10"] || valeurs["Score_Vec"]
        };
        // Stocker aussi les scores bruts 0-1 pour le calcul p_k Betti (CV scale-invariant 0-1)
        scoresParCommuneRaw01[commune] = {
          "Score_Opp": valeurs["Score_Opp"],
          "Score_Cho": valeurs["Score_Cho"],
          "Score_Vec": valeurs["Score_Vec"]
        };

        // Utiliser OppChoVec normalisé 0-10
        indiceFinal[commune] = valeurs["OppChoVec_0_10"] || valeurs["OppChoVec"];
      }

      console.log(`✅ ${Object.keys(indiceFinal).length} communes chargées`);
      console.log("Exemples de communes:", Object.keys(indiceFinal).slice(0, 5));

      // Sauvegarde dans les variables globales
      indicateursCommune = data_indicateurs_dict;
      indiceFinale = indiceFinal;
      scoresParCommune = data_dimensions_scores_dict;

      afficherToutesLesCartes(communeJson, indiceFinale, scoresParCommune);

      // Générer les visualisations de données
      genererDataVisualisations(indiceFinale);

      return indiceFinale;
    }

// fonction de calcul de l'indicateur de vien-être
function calculerIndiceBienEtre(scoresParCommune, pkValues = [1, 1, 1]) {
  const alpha = 2.5;
  const beta = 1.5;

  const bienEtreParCommune = {};

  for (const commune in scoresParCommune) {
    const scores = scoresParCommune[commune];

    // Extraire et élever les scores à la puissance beta
    const dik = [
      scores.Score_Opp,
      scores.Score_Cho,
      scores.Score_Vec
    ].map(score => Math.pow(score, beta));

    // Somme pondérée
    let sommePonderee = 0;
    for (let i = 0; i < pkValues.length; i++) {
      sommePonderee += pkValues[i] * dik[i];
    }

    // Calcul de l'indice final
    const indice = (1 / 3) * Math.pow(sommePonderee, alpha / beta);

    bienEtreParCommune[commune] = indice;
  }

  return bienEtreParCommune;
}


// === Calcul du vrai p_k selon Betti et al. (2008) ===
// Utilise les scores 0-1 (bruts) pour le CV — invariance d'échelle correcte
function calculerPkBetti(scoresParCommune) {
    const communes = Object.keys(scoresParCommune);
    const dims = ['Score_Opp', 'Score_Cho', 'Score_Vec'];
    // Utiliser les scores 0-1 si disponibles, sinon fallback sur les 0-10
    const src = (Object.keys(scoresParCommuneRaw01).length > 0) ? scoresParCommuneRaw01 : scoresParCommune;
    const data = dims.map(dim => communes.map(c => src[c][dim]));

    // p¹_k = cv_k = std / mean (coefficient de variation)
    const p1 = data.map(arr => {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length;
        return mean === 0 ? 0 : Math.sqrt(variance) / mean;
    });

    // Corrélation de Pearson entre deux vecteurs
    function pearsonCorr(a, b) {
        const n = a.length;
        const meanA = a.reduce((s, x) => s + x, 0) / n;
        const meanB = b.reduce((s, x) => s + x, 0) / n;
        const num = a.reduce((s, x, i) => s + (x - meanA) * (b[i] - meanB), 0);
        const denA = Math.sqrt(a.reduce((s, x) => s + (x - meanA) ** 2, 0));
        const denB = Math.sqrt(b.reduce((s, x) => s + (x - meanB) ** 2, 0));
        return (denA * denB === 0) ? 0 : num / (denA * denB);
    }

    // p²_k = 1 / mean(|ρ_{k,k'}|) pour tous k' (incl. k lui-même, ρ_{k,k}=1)
    const p2 = data.map((a) => {
        const avgCorr = data.reduce((s, b) => s + Math.abs(pearsonCorr(a, b)), 0) / data.length;
        return avgCorr === 0 ? 0 : 1 / avgCorr;
    });

    // p_k = p¹_k × p²_k, normalisé pour que Σp_k = 1
    const pkRaw = p1.map((v, i) => v * p2[i]);
    const sum = pkRaw.reduce((a, b) => a + b, 0);
    return sum === 0 ? [1/3, 1/3, 1/3] : pkRaw.map(v => v / sum);
}

// === Jenks Natural Breaks (programmation dynamique) ===
function calculerJenksBreaks(values, nClasses) {
    const sorted = [...values].filter(v => isFinite(v)).sort((a, b) => a - b);
    const n = sorted.length;
    if (n <= nClasses) return sorted.slice(1);

    // Sommes préfixes pour calculer SSQ(i,j) en O(1)
    const prefSum = new Float64Array(n + 1);
    const prefSumSq = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
        prefSum[i + 1] = prefSum[i] + sorted[i];
        prefSumSq[i + 1] = prefSumSq[i] + sorted[i] * sorted[i];
    }
    // SSQ des éléments sorted[i..j] (0-indexés, inclusifs)
    function ssd(i, j) {
        const cnt = j - i + 1;
        const sum = prefSum[j + 1] - prefSum[i];
        const sumSq = prefSumSq[j + 1] - prefSumSq[i];
        return sumSq - (sum * sum) / cnt;
    }

    // dp[i][k] = min SSQ pour les i premiers éléments en k classes
    const dp = Array.from({length: n + 1}, () => new Float64Array(nClasses + 1).fill(Infinity));
    const prev = Array.from({length: n + 1}, () => new Int32Array(nClasses + 1));
    dp[0][0] = 0;
    for (let i = 1; i <= n; i++) {
        dp[i][1] = ssd(0, i - 1); // variance réelle des i premiers éléments en 1 classe
        prev[i][1] = 0;
    }

    for (let k = 2; k <= nClasses; k++) {
        for (let i = k; i <= n; i++) {
            for (let m = k - 1; m < i; m++) {
                const cost = dp[m][k - 1] + ssd(m, i - 1);
                if (cost < dp[i][k]) {
                    dp[i][k] = cost;
                    prev[i][k] = m;
                }
            }
        }
    }

    // Backtracking : sorted[m-1] = dernier élément de la classe gauche (convention jenkspy)
    const breaks = [];
    let k = nClasses;
    let i = n;
    while (k > 1) {
        const m = prev[i][k];
        breaks.unshift(sorted[m - 1]);
        i = m;
        k--;
    }
    return breaks; // nClasses-1 seuils internes
}

// === Bascule mode p_k ===
function toggleModePk() {
    modeCalculPk = (modeCalculPk === 'egal') ? 'betti' : 'egal';
    recalculerCarteOppChoVec();
}

function recalculerCarteOppChoVec() {
    if (!scoresParCommune || Object.keys(scoresParCommune).length === 0) {
        console.warn('scoresParCommune non disponible');
        return;
    }

    const pkValues = modeCalculPk === 'betti'
        ? calculerPkBetti(scoresParCommune)
        : [1, 1, 1];

    console.log(`[p_k mode=${modeCalculPk}] Opp=${pkValues[0].toFixed(4)} Cho=${pkValues[1].toFixed(4)} Vec=${pkValues[2].toFixed(4)} (somme=${pkValues.reduce((a,b)=>a+b,0).toFixed(4)})`);

    // Recalculer OppChoVec brut avec le bon p_k
    // Utiliser les scores 0-1 (cohérent avec le Python/Excel) si disponibles
    const srcScores = (Object.keys(scoresParCommuneRaw01).length > 0) ? scoresParCommuneRaw01 : scoresParCommune;
    const indiceBrut = calculerIndiceBienEtre(srcScores, pkValues);

    // Renormaliser 0-10
    const valeurs = Object.values(indiceBrut);
    const minVal = Math.min(...valeurs);
    const maxVal = Math.max(...valeurs);
    const indiceNorm = {};
    for (const commune in indiceBrut) {
        indiceNorm[commune] = (maxVal === minVal) ? 5
            : ((indiceBrut[commune] - minVal) / (maxVal - minVal)) * 10;
    }

    // Recalculer Jenks sur les nouvelles valeurs
    const breaks = calculerJenksBreaks(Object.values(indiceNorm), 5);
    seuilsJenks['oppchovec'] = [0, ...breaks, 10];
    console.log('[Jenks oppchovec]', seuilsJenks['oppchovec'].map(v => v.toFixed(3)).join(' | '));

    // Mettre à jour la carte (afficherCarteUnique gère le rechargement des layers)
    afficherCarteUnique('map-oppchovec', 'oppchovec', communeJson, indiceNorm, 'OppChoLiv');

    // Mettre à jour indiceFinale pour que les popups LISA montrent les bonnes valeurs
    indiceFinale = indiceNorm;

    // Switcher les clusters LISA selon le mode
    if (modeCalculPk === 'betti' && typeof LISA_DATA_BETTI !== 'undefined' && typeof LISA_DATA_BETTI_1PCT !== 'undefined') {
        const c5 = {}, c1 = {};
        for (const [k, v] of Object.entries(LISA_DATA_BETTI.clusters))     c5[k] = v.cluster;
        for (const [k, v] of Object.entries(LISA_DATA_BETTI_1PCT.clusters)) c1[k] = v.cluster;
        clustersLISA5pct = c5;
        clustersLISA1pct = c1;
        console.log(`[LISA] Mode Betti — I=${LISA_DATA_BETTI.metadata.moran_global_I.toFixed(4)}, sig5%=${LISA_DATA_BETTI.metadata.nb_significatives}, sig1%=${LISA_DATA_BETTI_1PCT.metadata.nb_significatives}`);
    } else {
        const c5 = {}, c1 = {};
        for (const [k, v] of Object.entries(LISA_DATA.clusters))     c5[k] = v.cluster;
        for (const [k, v] of Object.entries(LISA_DATA_1PCT.clusters)) c1[k] = v.cluster;
        clustersLISA5pct = c5;
        clustersLISA1pct = c1;
        console.log(`[LISA] Mode Égal — I=${LISA_DATA.metadata.moran_global_I.toFixed(4)}, sig5%=${LISA_DATA.metadata.nb_significatives}, sig1%=${LISA_DATA_1PCT.metadata.nb_significatives}`);
    }

    // Mettre à jour les cartes LISA si déjà initialisées
    if (lisaCartesInitialisees) {
        afficherCarteLISA('map-lisa-5pct', 'lisa-5pct', communeJson, indiceNorm, clustersLISA5pct, '5%');
        afficherCarteLISA('map-lisa-1pct', 'lisa-1pct', communeJson, indiceNorm, clustersLISA1pct, '1%');
    }

    majAffichagePk(pkValues);

    // Mettre à jour le tableau top 10 et l'histogramme
    genererDataVisualisations(indiceNorm);

    // Mettre à jour l'affichage de la commune sélectionnée si présent
    const communeSelectEl = document.getElementById('communeSelect');
    if (communeSelectEl && communeSelectEl.value) {
        afficherCommune(communeSelectEl.value);
    }

    // Mettre à jour les parangons si déjà initialisés
    if (parangonsInitialise) {
        parangonsInitialise = false;
        initialiserOngletParangons();
    }

    // Mettre à jour la corrélation si déjà initialisée
    if (correlationInitialise) {
        correlationInitialise = false;
        if (cartes['correlation']) {
            cartes['correlation'].remove();
            cartes['correlation'] = null;
            document.getElementById('map-correlation').innerHTML = '';
        }
        initialiserOngletCorrelation();
    }
    if (corrInverseInitialise) {
        corrInverseInitialise = false;
        if (cartes['correlation2']) {
            cartes['correlation2'].remove();
            cartes['correlation2'] = null;
            document.getElementById('map-correlation2').innerHTML = '';
        }
    }
    if (corrSubjectifInitialise) {
        corrSubjectifInitialise = false;
        if (cartes['subjectif']) {
            cartes['subjectif'].remove();
            cartes['subjectif'] = null;
            document.getElementById('map-subjectif').innerHTML = '';
        }
    }
    if (corrSubjectifDensInitialise) {
        corrSubjectifDensInitialise = false;
        if (cartes['subjectif-dens']) {
            cartes['subjectif-dens'].remove();
            cartes['subjectif-dens'] = null;
            document.getElementById('map-subjectif-dens').innerHTML = '';
        }
    }
}

function majAffichagePk(pkValues) {
    const btn = document.getElementById('btn-toggle-pk');
    const info = document.getElementById('pk-values-display');
    if (!btn || !info) return;
    if (modeCalculPk === 'betti') {
        btn.textContent = 'p_k : Betti et al. ✓';
        btn.classList.add('active');
        info.textContent = `Opp=${pkValues[0].toFixed(3)} | Cho=${pkValues[1].toFixed(3)} | Vec=${pkValues[2].toFixed(3)}`;
    } else {
        btn.textContent = 'p_k : égal [1,1,1]';
        btn.classList.remove('active');
        info.textContent = 'Pondérations égales (p_k = 1/3 chacun)';
    }
}

// fonction de calcul des valeurs des dimensions de OppChoVec
function calculerScoresParCommune(dataNormalise) {
  const oppPonderation = {
    "Opp1": 0.25, "Opp2": 0.25, "Opp3": 0.25, "Opp4": 0.25
  };

  const choPonderation = {
    "Cho1": 0.50, "Cho2": 0.50
  };

  const vecPonderation = {
    "Vec1": 0.25, "Vec2": 0.25, "Vec3": 0.25, "Vec4": 0.25
  };
  console.log(dataNormalise)

  function calcDik(indicateurs, ponderations) {
    const valeurs = [];
    const poids = [];

    for (const cle in ponderations) {
      if (cle in indicateurs) {
        valeurs.push(indicateurs[cle]);
        poids.push(ponderations[cle]);
      }
    }

    // Vérifier que nous avons des valeurs
    if (!valeurs || valeurs.length === 0 || !poids || poids.length === 0) {
      console.warn("Aucune valeur trouvée pour les pondérations:", ponderations);
      return 0;
    }

    const sommePoids = poids.reduce((a, b) => a + b, 0);
    if (sommePoids === 0) return 0;

    const produit = valeurs.map((v, i) => v * poids[i]);
    const sommeProduit = produit.reduce((a, b) => a + b, 0);

    return sommeProduit / sommePoids;
  }

  const scores = {};

  for (const commune in dataNormalise) {
    const indicateursBruts = dataNormalise[commune];

    const indicateursSimplifies = {};
    for (const k in indicateursBruts) {
      const nom = k.replace("Indicateur_", "").trim();
      indicateursSimplifies[nom] = parseFloat(indicateursBruts[k]);
    }

    const scoreOpp = calcDik(indicateursSimplifies, oppPonderation);
    const scoreCho = calcDik(indicateursSimplifies, choPonderation);
    const scoreVec = calcDik(indicateursSimplifies, vecPonderation);

    scores[commune] = {
      Score_Opp: scoreOpp,
      Score_Cho: scoreCho,
      Score_Vec: scoreVec
    };
  }
  console.log(scores)
  return scores;
}


// fonction de normalisation des données dans la méthode OppChoVec
function normaliserDonnees(data, minVals, maxVals) {
  const dataNormalise = {};

  for (const commune in data) {
    const indicateurs = data[commune];
    const communeData = {};

    for (const indicateur in indicateurs) {
      const valeur = indicateurs[indicateur];
      const minX = minVals[indicateur] ?? 0;
      const maxX = maxVals[indicateur] ?? 1;

      let normVal;
      if (maxX === minX) {
        normVal = 0;
      } else {
        normVal = (valeur - minX) / (maxX - minX);
      }

      communeData[indicateur] = normVal;
    }

    dataNormalise[commune] = communeData;
  }

  return dataNormalise;
}

// function min max pour la normalisation des données. Cette fonction nous permet de retenir le min et max parmi les valeurs d'indicateur
function minmax(data) {
  const data_indicateurs_min_dict = {};
  const data_indicateurs_max_dict = {};

  // Récupère toutes les clés des indicateurs depuis la première commune
  const keys = Object.keys(Object.values(data)[0]);

  keys.forEach((key) => {
    let valeurs = [];

    for (const commune in data) {
      if (data[commune][key] !== undefined) {
        valeurs.push(data[commune][key]);
      }
    }

    data_indicateurs_min_dict[key] = Math.min(...valeurs);
    data_indicateurs_max_dict[key] = Math.max(...valeurs);
  });

  return {
    min: data_indicateurs_min_dict,
    max: data_indicateurs_max_dict
  };
}

// ==============================================================================
// ==============================================================================
// ONGLET CORRÉLATION OppChoVec × Densité entrepreneuriale
// ==============================================================================

// mapId ex: 'map-correlation', 'map-subjectif', 'map-correlation2', 'map-subjectif-dens'
// clé dans cartes[] : 'correlation', 'subjectif', 'correlation2', 'subjectif-dens'
function toggleCorrMap(mapId, btn) {
    const panel = document.getElementById(mapId + '-panel');
    if (!panel) return;
    const isOpen = panel.style.display !== 'none';
    panel.style.display = isOpen ? 'none' : 'block';
    btn.classList.toggle('active', !isOpen);
    if (!isOpen) {
        btn.textContent = btn.textContent.replace('📍 Voir', 'Masquer');
        const carteKey = mapId.replace(/^map-/, '');
        setTimeout(() => { if (cartes[carteKey]) cartes[carteKey].invalidateSize(); }, 150);
    } else {
        btn.textContent = btn.textContent.replace('Masquer', '📍 Voir');
    }
}

// Communes à étiqueter : nom → décalage [dx, dy] en pixels
const COMMUNES_LABELS = {
    'Ajaccio':           [9, -14],
    'Bastia':            [9,  14],
    'Corte':             [9,   0],
    'Furiani':           [9,   0],
    'Lucciana':          [9,   0],
    'Calvi':             [9,   0],
    'Biguglia':          [9,   0],
    'Linguizzetta':      [9, -14],
    'Penta-di-Casinca':  [9,   0],
    'Porto-Vecchio':     [-105, -14],
};

// Plugin Chart.js : affiche les noms des communes ciblées à côté de leurs points
const communeLabelsPlugin = {
    id: 'communeLabels',
    afterDatasetsDraw(chart) {
        const dataset = chart.data.datasets[0];
        if (!dataset || !dataset.data) return;
        const meta = chart.getDatasetMeta(0);
        if (!meta || !meta.data) return;
        const ctx = chart.ctx;
        ctx.save();
        ctx.font = 'bold 11px sans-serif';
        ctx.textBaseline = 'middle';
        dataset.data.forEach((point, i) => {
            if (!point.nom || !(point.nom in COMMUNES_LABELS)) return;
            const el = meta.data[i];
            if (!el) return;
            const [dx, dy] = COMMUNES_LABELS[point.nom];
            const x = el.x + dx, y = el.y + dy;
            const label = point.nom;
            const w = ctx.measureText(label).width;
            ctx.fillStyle = 'rgba(255,255,255,0.78)';
            ctx.fillRect(x - 1, y - 8, w + 4, 16);
            ctx.fillStyle = '#1a1a2e';
            ctx.fillText(label, x + 1, y);
        });
        ctx.restore();
    }
};
Chart.register(communeLabelsPlugin);

function exporterScatterPNG(chart, filename) {
    if (!chart) return;
    const EXPORT_W = 1200;
    const EXPORT_H = 800;

    // Données du 1er dataset (points scatter, pas la droite de tendance)
    const scatterData = chart.data.datasets[0].data;
    const xs = scatterData.map(p => p.x);
    const ys = scatterData.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin) * 0.08 || 0.5;
    const yPad = (yMax - yMin) * 0.12 || 0.2;

    const sx = chart.options.scales.x;
    const sy = chart.options.scales.y;

    // Sauvegarder les valeurs courantes
    const saved = {
        xMin: sx.min, xMax: sx.max,
        yMin: sy.min, yMax: sy.max,
        xTitleFont: sx.title?.font,
        yTitleFont: sy.title?.font,
        xTicksFont: sx.ticks?.font,
        yTicksFont: sy.ticks?.font,
        padding: chart.options.layout?.padding,
    };

    // Axe X : serré autour des données
    sx.min = xMin - xPad;
    sx.max = xMax + xPad;
    // Axe Y : idem (ignorer le min:1/max:5 fixe pour l'export)
    sy.min = yMin - yPad;
    sy.max = yMax + yPad;

    // Grossir les polices pour l'export
    if (!sx.title) sx.title = { display: true };
    if (!sy.title) sy.title = { display: true };
    if (!sx.ticks) sx.ticks = {};
    if (!sy.ticks) sy.ticks = {};
    sx.title.font = { size: 20, weight: 'bold' };
    sy.title.font = { size: 20, weight: 'bold' };
    sx.ticks.font = { size: 15 };
    sy.ticks.font = { size: 15 };
    if (!chart.options.layout) chart.options.layout = {};
    chart.options.layout.padding = { top: 20, right: 30, bottom: 10, left: 10 };

    chart.update('none');
    chart.resize(EXPORT_W, EXPORT_H);

    setTimeout(() => {
        // Fond blanc (Chart.js exporte en transparent sinon)
        const offscreen = document.createElement('canvas');
        offscreen.width = EXPORT_W;
        offscreen.height = EXPORT_H;
        const ctx = offscreen.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, EXPORT_W, EXPORT_H);
        ctx.drawImage(chart.canvas, 0, 0);

        const link = document.createElement('a');
        link.download = filename + '_1200x800.png';
        link.href = offscreen.toDataURL('image/png');
        link.click();

        // Restaurer les options et taille d'affichage
        sx.min = saved.xMin;
        sx.max = saved.xMax;
        sy.min = saved.yMin;
        sy.max = saved.yMax;
        sx.title.font = saved.xTitleFont;
        sy.title.font = saved.yTitleFont;
        sx.ticks.font = saved.xTicksFont;
        sy.ticks.font = saved.yTicksFont;
        chart.options.layout.padding = saved.padding;
        chart.update('none');
        chart.resize();
    }, 200);
}

let correlationInitialise = false;
let _scatterChart = null;

async function initialiserOngletCorrelation() {
    if (correlationInitialise) {
        setTimeout(() => { if (cartes['correlation']) cartes['correlation'].invalidateSize(); }, 100);
        return;
    }
    if (!communeJson || !communeJson.features || !indiceFinale || Object.keys(indiceFinale).length === 0) {
        console.warn('Corrélation: données non prêtes'); return;
    }

    // Charger densité si pas encore fait
    if (!_densiteEntrepreneuriale) {
        try {
            const resp = await fetch(BASE_PATH + 'densite_entrepreneuriale.json');
            _densiteEntrepreneuriale = await resp.json();
        } catch(e) { console.error('Erreur chargement densité', e); return; }
    }

    correlationInitialise = true;

    // ---- Données communes appariées ----
    const points = [];
    for (const [nom, occ] of Object.entries(indiceFinale)) {
        const dens = _densiteEntrepreneuriale[nom];
        if (dens !== undefined && dens !== null) points.push({ nom, occ, dens });
    }

    // ---- Régression linéaire ----
    const n  = points.length;
    const mx = points.reduce((s, p) => s + p.dens, 0) / n;
    const my = points.reduce((s, p) => s + p.occ,  0) / n;
    const ss_xy = points.reduce((s, p) => s + (p.dens - mx) * (p.occ - my), 0);
    const ss_xx = points.reduce((s, p) => s + (p.dens - mx) ** 2, 0);
    const ss_yy = points.reduce((s, p) => s + (p.occ  - my) ** 2, 0);
    const slope = ss_xy / ss_xx;
    const intercept = my - slope * mx;
    const r = ss_xy / Math.sqrt(ss_xx * ss_yy);

    // Résidu par commune
    points.forEach(p => {
        p.predicted = slope * p.dens + intercept;
        p.residual  = p.occ - p.predicted;
    });
    const maxAbsRes = Math.max(...points.map(p => Math.abs(p.residual)));

    // ---- Corrélation header ----
    const headerEl = document.getElementById('correlation-r');
    if (headerEl) {
        const force = Math.abs(r) >= 0.7 ? 'forte' : Math.abs(r) >= 0.4 ? 'modérée' : 'faible';
        const signe = r >= 0 ? 'positive' : 'négative';
        headerEl.innerHTML = `
            <strong>Corrélation OppChoVec × Densité entrepreneuriale</strong><br>
            <span style="font-size:20px;font-weight:700;color:${r>=0?'#1b5e20':'#b71c1c'};">r = ${r.toFixed(3)}</span>
            &nbsp;·&nbsp; Corrélation ${force} ${signe} (n = ${n} communes)`;
    }

    // ---- Scatter Chart.js ----
    const minDens = Math.min(...points.map(p => p.dens));
    const maxDens = Math.max(...points.map(p => p.dens));
    const canvas = document.getElementById('scatter-chart');
    if (canvas) {
        if (_scatterChart) { _scatterChart.destroy(); _scatterChart = null; }
        const ctx = canvas.getContext('2d');
        _scatterChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Communes',
                        data: points.map(p => ({ x: p.dens, y: p.occ, nom: p.nom })),
                        backgroundColor: points.map(p => {
                            const t = (p.residual + maxAbsRes) / (2 * maxAbsRes);
                            const r2 = Math.round(220 * (1 - t)), g2 = Math.round(50 + 150 * t), b2 = 80;
                            return `rgba(${r2},${g2},${b2},0.65)`;
                        }),
                        pointRadius: 4,
                        pointHoverRadius: 7,
                    },
                    {
                        label: 'Tendance',
                        data: [
                            { x: minDens, y: slope * minDens + intercept },
                            { x: maxDens, y: slope * maxDens + intercept }
                        ],
                        type: 'line',
                        borderColor: '#555',
                        borderWidth: 1.5,
                        borderDash: [5, 4],
                        pointRadius: 0,
                        fill: false,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const d = ctx.raw;
                                return `${d.nom} — OppChoVec: ${d.y.toFixed(2)}, Densité: ${d.x.toFixed(1)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Densité entrepreneuriale (entr./1000 hab.)' } },
                    y: { title: { display: true, text: 'OppChoVec (0–10)' } }
                }
            }
        });
    }

    // ---- Carte des résidus ----
    const panelCorr1 = document.getElementById('map-correlation-panel');
    if (panelCorr1) panelCorr1.style.display = 'block';
    const carte = L.map(document.getElementById('map-correlation'), {
        center: [42.0, 9.0], zoom: 8, zoomSnap: 0.1, zoomDelta: 0.1
    });
    if (panelCorr1) panelCorr1.style.display = 'none';
    cartes['correlation'] = carte;
    carte.getContainer().style.backgroundColor = '#ffffff';

    const residualMap = Object.fromEntries(points.map(p => [p.nom, p.residual]));

    const getColorRes = (res) => {
        if (res === undefined) return '#ccc';
        const t = Math.max(0, Math.min(1, (res + maxAbsRes) / (2 * maxAbsRes)));
        if (t >= 0.5) {
            const intensity = (t - 0.5) * 2;
            const g = Math.round(100 + 100 * intensity);
            return `rgb(${Math.round(220*(1-intensity))},${g},${Math.round(50*(1-intensity))})`;
        } else {
            const intensity = (0.5 - t) * 2;
            return `rgb(${Math.round(180+40*intensity)},${Math.round(80*(1-intensity))},${Math.round(80*(1-intensity))})`;
        }
    };

    L.geoJSON(communeJson, {
        style: feature => ({
            fillColor: getColorRes(residualMap[feature.properties.nom]),
            color: '#333', weight: 0.8, fillOpacity: 0.8
        }),
        onEachFeature: (feature, layer) => {
            const nom = feature.properties.nom;
            const res = residualMap[nom];
            const occ = indiceFinale[nom];
            const dens = _densiteEntrepreneuriale[nom];
            layer.bindPopup(`<strong>${nom}</strong><br>
                OppChoVec : ${occ !== undefined ? occ.toFixed(2) : 'N/A'}<br>
                Densité : ${dens !== undefined ? dens.toFixed(1) + ' entr./1000 hab.' : 'N/A'}<br>
                Résidu : <strong>${res !== undefined ? (res > 0 ? '+' : '') + res.toFixed(2) : 'N/A'}</strong>`);
        }
    }).addTo(carte);

    // Légende résidus
    const legendRes = L.control({ position: 'bottomright' });
    legendRes.onAdd = function() {
        const div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = `<strong style="font-size:14px;">Légende</strong><br>
            <hr style="margin:8px 0;border:none;border-top:1px solid #ddd;">
            <strong>Résidu OppChoVec</strong><br>
            <small style="color:#666;">OppChoVec observé − attendu</small><br><br>
            <i style="background:#1e7c1e;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>OppChoVec &gt; attendu<br>
            <i style="background:#aaa;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Proche tendance<br>
            <i style="background:#c05050;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>OppChoVec &lt; attendu<br>`;
        _makeDraggable(div, carte);
        return div;
    };
    legendRes.addTo(carte);

    ajouterVillesPrincipales(carte);
    ajouterCopyright(carte);
    ajouterBoutonTelechargement(carte, 'correlation');
    setTimeout(() => carte.invalidateSize(), 150);
}

// ==============================================================================
// CORRÉLATION INVERSE : OppChoVec → Densité entrepreneuriale
// ==============================================================================

let corrInverseInitialise = false;
let _scatterChart2 = null;

async function initialiserCorrInverse() {
    if (corrInverseInitialise) {
        setTimeout(() => { if (cartes['correlation2']) cartes['correlation2'].invalidateSize(); }, 100);
        return;
    }
    if (!communeJson || !communeJson.features || !indiceFinale || Object.keys(indiceFinale).length === 0) {
        console.warn('CorrInverse: données non prêtes'); return;
    }
    if (!_densiteEntrepreneuriale) {
        try {
            const resp = await fetch(BASE_PATH + 'densite_entrepreneuriale.json');
            _densiteEntrepreneuriale = await resp.json();
        } catch(e) { console.error('Erreur chargement densité', e); return; }
    }

    corrInverseInitialise = true;

    const points = [];
    for (const [nom, occ] of Object.entries(indiceFinale)) {
        const dens = _densiteEntrepreneuriale[nom];
        if (dens !== undefined && dens !== null) points.push({ nom, occ, dens });
    }

    // Régression : OppChoVec (X) → Densité (Y)
    const n  = points.length;
    const mx = points.reduce((s, p) => s + p.occ,  0) / n;
    const my = points.reduce((s, p) => s + p.dens, 0) / n;
    const ss_xy = points.reduce((s, p) => s + (p.occ - mx) * (p.dens - my), 0);
    const ss_xx = points.reduce((s, p) => s + (p.occ - mx) ** 2, 0);
    const ss_yy = points.reduce((s, p) => s + (p.dens - my) ** 2, 0);
    const slope = ss_xy / ss_xx;
    const intercept = my - slope * mx;
    const r = ss_xy / Math.sqrt(ss_xx * ss_yy);

    points.forEach(p => {
        p.predicted = slope * p.occ + intercept;
        p.residual  = p.dens - p.predicted;
    });
    const maxAbsRes = Math.max(...points.map(p => Math.abs(p.residual)));

    // Header
    const headerEl2 = document.getElementById('correlation-r2');
    if (headerEl2) {
        const force = Math.abs(r) >= 0.7 ? 'forte' : Math.abs(r) >= 0.4 ? 'modérée' : 'faible';
        const signe = r >= 0 ? 'positive' : 'négative';
        headerEl2.innerHTML = `
            <strong>Densité attendue selon OppChoVec</strong><br>
            <span style="font-size:20px;font-weight:700;color:${r>=0?'#1b5e20':'#b71c1c'};">r = ${r.toFixed(3)}</span>
            &nbsp;·&nbsp; Corrélation ${force} ${signe} (n = ${n} communes)`;
    }

    // Scatter : X = OppChoVec, Y = Densité
    const minOcc = Math.min(...points.map(p => p.occ));
    const maxOcc = Math.max(...points.map(p => p.occ));
    const canvas2 = document.getElementById('scatter-chart2');
    if (canvas2) {
        if (_scatterChart2) { _scatterChart2.destroy(); _scatterChart2 = null; }
        const ctx2 = canvas2.getContext('2d');
        _scatterChart2 = new Chart(ctx2, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Communes',
                        data: points.map(p => ({ x: p.occ, y: p.dens, nom: p.nom })),
                        backgroundColor: points.map(p => {
                            const t = (p.residual + maxAbsRes) / (2 * maxAbsRes);
                            const r2 = Math.round(220 * (1 - t)), g2 = Math.round(50 + 150 * t), b2 = 80;
                            return `rgba(${r2},${g2},${b2},0.65)`;
                        }),
                        pointRadius: 4, pointHoverRadius: 7,
                    },
                    {
                        label: 'Tendance',
                        data: [
                            { x: minOcc, y: slope * minOcc + intercept },
                            { x: maxOcc, y: slope * maxOcc + intercept }
                        ],
                        type: 'line', borderColor: '#555', borderWidth: 1.5,
                        borderDash: [5, 4], pointRadius: 0, fill: false,
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => {
                        const d = ctx.raw;
                        return `${d.nom} — OppChoVec: ${d.x.toFixed(2)}, Densité: ${d.y.toFixed(1)}`;
                    }}}
                },
                scales: {
                    x: { title: { display: true, text: 'OppChoVec (0–10)' } },
                    y: { title: { display: true, text: 'Densité entrepreneuriale (entr./1000 hab.)' } }
                }
            }
        });
    }

    // Carte des résidus (densité observée − densité attendue)
    const panelCorr2 = document.getElementById('map-correlation2-panel');
    if (panelCorr2) panelCorr2.style.display = 'block';
    const carte2 = L.map(document.getElementById('map-correlation2'), {
        center: [42.0, 9.0], zoom: 8, zoomSnap: 0.1, zoomDelta: 0.1
    });
    if (panelCorr2) panelCorr2.style.display = 'none';
    cartes['correlation2'] = carte2;
    carte2.getContainer().style.backgroundColor = '#ffffff';

    const residualMap2 = Object.fromEntries(points.map(p => [p.nom, p.residual]));

    const getColorRes2 = (res) => {
        if (res === undefined) return '#ccc';
        const t = Math.max(0, Math.min(1, (res + maxAbsRes) / (2 * maxAbsRes)));
        if (t >= 0.5) {
            const intensity = (t - 0.5) * 2;
            const g = Math.round(100 + 100 * intensity);
            return `rgb(${Math.round(220*(1-intensity))},${g},${Math.round(50*(1-intensity))})`;
        } else {
            const intensity = (0.5 - t) * 2;
            return `rgb(${Math.round(180+40*intensity)},${Math.round(80*(1-intensity))},${Math.round(80*(1-intensity))})`;
        }
    };

    L.geoJSON(communeJson, {
        style: feature => ({
            fillColor: getColorRes2(residualMap2[feature.properties.nom]),
            color: '#333', weight: 0.8, fillOpacity: 0.8
        }),
        onEachFeature: (feature, layer) => {
            const nom = feature.properties.nom;
            const res = residualMap2[nom];
            const occ = indiceFinale[nom];
            const dens = _densiteEntrepreneuriale[nom];
            layer.bindPopup(`<strong>${nom}</strong><br>
                OppChoVec : ${occ !== undefined ? occ.toFixed(2) : 'N/A'}<br>
                Densité observée : ${dens !== undefined ? dens.toFixed(1) + ' entr./1000 hab.' : 'N/A'}<br>
                Densité attendue : ${occ !== undefined ? (slope * occ + intercept).toFixed(1) + ' entr./1000 hab.' : 'N/A'}<br>
                Résidu : <strong>${res !== undefined ? (res > 0 ? '+' : '') + res.toFixed(2) : 'N/A'}</strong>`);
        }
    }).addTo(carte2);

    const legendRes2 = L.control({ position: 'bottomright' });
    legendRes2.onAdd = function() {
        const div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = `<strong style="font-size:14px;">Légende</strong><br>
            <hr style="margin:8px 0;border:none;border-top:1px solid #ddd;">
            <strong>Résidu densité entrepreneuriale</strong><br>
            <small style="color:#666;">Densité observée − attendue selon OppChoVec</small><br><br>
            <i style="background:#1e7c1e;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Plus dynamique qu'attendu<br>
            <i style="background:#aaa;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Proche tendance<br>
            <i style="background:#c05050;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Moins dynamique qu'attendu<br>`;
        _makeDraggable(div, carte2);
        return div;
    };
    legendRes2.addTo(carte2);

    ajouterVillesPrincipales(carte2);
    ajouterCopyright(carte2);
    ajouterBoutonTelechargement(carte2, 'correlation2');
    setTimeout(() => carte2.invalidateSize(), 150);
}

// ==============================================================================
// CORRÉLATION BIEN-ÊTRE SUBJECTIF × OppChoVec
// ==============================================================================

let corrSubjectifInitialise = false;
let _scatterChart3 = null;
let _bienEtreSubjectif = null;

async function initialiserCorrSubjectif() {
    if (corrSubjectifInitialise) {
        setTimeout(() => { if (cartes['subjectif']) cartes['subjectif'].invalidateSize(); }, 100);
        return;
    }
    if (!communeJson || !communeJson.features || !indiceFinale || Object.keys(indiceFinale).length === 0) {
        console.warn('CorrSubjectif: données non prêtes'); return;
    }

    if (!_bienEtreSubjectif) {
        try {
            const resp = await fetch(BASE_PATH + 'bien_etre_subjectif.json');
            _bienEtreSubjectif = await resp.json();
        } catch(e) { console.error('Erreur chargement bien_etre_subjectif.json', e); return; }
    }

    corrSubjectifInitialise = true;

    // Points appariés (communes avec les deux données)
    const points = [];
    for (const [nom, data] of Object.entries(_bienEtreSubjectif)) {
        const occ = indiceFinale[nom];
        if (occ !== undefined && occ !== null) {
            points.push({ nom, occ, subjScore: data.score, n: data.n });
        }
    }

    // Régression et corrélation si assez de points
    let r = null, slope = null, intercept = null;
    if (points.length >= 3) {
        const n  = points.length;
        const mx = points.reduce((s, p) => s + p.occ,       0) / n;
        const my = points.reduce((s, p) => s + p.subjScore, 0) / n;
        const ss_xy = points.reduce((s, p) => s + (p.occ - mx) * (p.subjScore - my), 0);
        const ss_xx = points.reduce((s, p) => s + (p.occ - mx) ** 2, 0);
        const ss_yy = points.reduce((s, p) => s + (p.subjScore - my) ** 2, 0);
        slope = ss_xy / ss_xx;
        intercept = my - slope * mx;
        r = ss_xy / Math.sqrt(ss_xx * ss_yy);
    }

    // Header
    const headerEl3 = document.getElementById('correlation-r3');
    if (headerEl3) {
        if (r !== null) {
            const force = Math.abs(r) >= 0.7 ? 'forte' : Math.abs(r) >= 0.4 ? 'modérée' : 'faible';
            const signe = r >= 0 ? 'positive' : 'négative';
            headerEl3.innerHTML = `
                <strong>Bien-être subjectif × OppChoVec</strong><br>
                <span style="font-size:20px;font-weight:700;color:${r>=0?'#1b5e20':'#b71c1c'};">r = ${r.toFixed(3)}</span>
                &nbsp;·&nbsp; Corrélation ${force} ${signe} (n = ${points.length} communes avec données)`;
        } else {
            headerEl3.innerHTML = `<strong>Bien-être subjectif × OppChoVec</strong><br>
                <em style="color:#888;">Données insuffisantes pour calculer la corrélation</em>`;
        }
    }

    // Scatter : X = OppChoVec, Y = score subjectif
    const canvas3 = document.getElementById('scatter-chart3');
    if (canvas3) {
        if (_scatterChart3) { _scatterChart3.destroy(); _scatterChart3 = null; }
        const ctx3 = canvas3.getContext('2d');
        const datasets = [{
            label: 'Communes',
            data: points.map(p => ({ x: p.occ, y: p.subjScore, nom: p.nom, n: p.n })),
            backgroundColor: 'rgba(70, 130, 180, 0.7)',
            pointRadius: points.map(p => Math.min(4 + p.n * 0.15, 10)),
            pointHoverRadius: 8,
        }];
        if (r !== null) {
            const minOcc = Math.min(...points.map(p => p.occ));
            const maxOcc = Math.max(...points.map(p => p.occ));
            datasets.push({
                label: 'Tendance',
                data: [
                    { x: minOcc, y: slope * minOcc + intercept },
                    { x: maxOcc, y: slope * maxOcc + intercept }
                ],
                type: 'line', borderColor: '#555', borderWidth: 1.5,
                borderDash: [5, 4], pointRadius: 0, fill: false,
            });
        }
        _scatterChart3 = new Chart(ctx3, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => {
                        const d = ctx.raw;
                        return `${d.nom} — OppChoVec: ${d.x.toFixed(2)}, B-E subjectif: ${d.y.toFixed(3)} (n=${d.n})`;
                    }}}
                },
                scales: {
                    x: { title: { display: true, text: 'OppChoVec (0–10)' } },
                    y: { title: { display: true, text: 'Score bien-être subjectif (1–5)' }, min: 1, max: 5 }
                }
            }
        });
    }

    // Carte colorée par score subjectif (uniquement les communes avec données)
    const panelSubj = document.getElementById('map-subjectif-panel');
    if (panelSubj) panelSubj.style.display = 'block';
    const carteSubj = L.map(document.getElementById('map-subjectif'), {
        center: [42.0, 9.0], zoom: 8, zoomSnap: 0.1, zoomDelta: 0.1
    });
    if (panelSubj) panelSubj.style.display = 'none';
    cartes['subjectif'] = carteSubj;
    carteSubj.getContainer().style.backgroundColor = '#ffffff';

    const scoreMap = Object.fromEntries(points.map(p => [p.nom, p]));

    // Palette : 1=rouge, 3=jaune, 5=vert
    const getColorSubj = (score) => {
        if (score === undefined) return '#e0e0e0';
        const t = Math.max(0, Math.min(1, (score - 1) / 4));
        if (t < 0.5) {
            const i = t * 2;
            return `rgb(${Math.round(220)},${Math.round(80 + 140 * i)},${Math.round(50 * i)})`;
        } else {
            const i = (t - 0.5) * 2;
            return `rgb(${Math.round(220 * (1 - i))},${Math.round(180 + 40 * i)},${Math.round(50 * (1 - i))})`;
        }
    };

    L.geoJSON(communeJson, {
        style: feature => {
            const nom = feature.properties.nom;
            const hasData = scoreMap[nom] !== undefined;
            return {
                fillColor: hasData ? getColorSubj(scoreMap[nom].subjScore) : '#e0e0e0',
                color: hasData ? '#333' : '#bbb',
                weight: hasData ? 1 : 0.5,
                fillOpacity: hasData ? 0.85 : 0.3
            };
        },
        onEachFeature: (feature, layer) => {
            const nom = feature.properties.nom;
            const occ = indiceFinale[nom];
            const p = scoreMap[nom];
            if (p) {
                layer.bindPopup(`<strong>${nom}</strong><br>
                    OppChoVec : ${occ !== undefined ? occ.toFixed(2) : 'N/A'}<br>
                    Bien-être subjectif : <strong>${p.subjScore.toFixed(3)}</strong> (n=${p.n} répondants)`);
            } else {
                layer.bindPopup(`<strong>${nom}</strong><br>
                    OppChoVec : ${occ !== undefined ? occ.toFixed(2) : 'N/A'}<br>
                    <em style="color:#888;">Pas de données subjectives</em>`);
            }
        }
    }).addTo(carteSubj);

    // Légende
    const legendSubj = L.control({ position: 'bottomright' });
    legendSubj.onAdd = function() {
        const div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = `<strong style="font-size:14px;">Légende</strong><br>
            <hr style="margin:8px 0;border:none;border-top:1px solid #ddd;">
            <strong>Score bien-être subjectif</strong><br>
            <small style="color:#666;">Moyenne questionnaire (1–5)<br>${points.length} communes avec données</small><br><br>
            <i style="background:#1e7c1e;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Élevé (≥ 4)<br>
            <i style="background:#e8c820;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Moyen (≈ 3)<br>
            <i style="background:#c05050;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Faible (≤ 2)<br>
            <i style="background:#e0e0e0;width:18px;height:18px;display:inline-block;margin-right:5px;border:1px solid #bbb;"></i>Pas de données<br>`;
        _makeDraggable(div, carteSubj);
        return div;
    };
    legendSubj.addTo(carteSubj);

    ajouterVillesPrincipales(carteSubj);
    ajouterCopyright(carteSubj);
    ajouterBoutonTelechargement(carteSubj, 'subjectif');
    setTimeout(() => carteSubj.invalidateSize(), 150);
}

// ==============================================================================
// CORRÉLATION BIEN-ÊTRE SUBJECTIF × DENSITÉ ENTREPRENEURIALE
// ==============================================================================

let corrSubjectifDensInitialise = false;
let _scatterChart4 = null;

async function initialiserCorrSubjectifDens() {
    if (corrSubjectifDensInitialise) {
        setTimeout(() => { if (cartes['subjectif-dens']) cartes['subjectif-dens'].invalidateSize(); }, 100);
        return;
    }
    if (!communeJson || !communeJson.features) {
        console.warn('CorrSubjectifDens: données non prêtes'); return;
    }

    // Charger les deux sources si nécessaire
    if (!_bienEtreSubjectif) {
        try {
            const resp = await fetch(BASE_PATH + 'bien_etre_subjectif.json');
            _bienEtreSubjectif = await resp.json();
        } catch(e) { console.error('Erreur chargement bien_etre_subjectif.json', e); return; }
    }
    if (!_densiteEntrepreneuriale) {
        try {
            const resp = await fetch(BASE_PATH + 'densite_entrepreneuriale.json');
            _densiteEntrepreneuriale = await resp.json();
        } catch(e) { console.error('Erreur chargement densité', e); return; }
    }

    corrSubjectifDensInitialise = true;

    // Points appariés : communes avec les deux données
    const points = [];
    for (const [nom, data] of Object.entries(_bienEtreSubjectif)) {
        const dens = _densiteEntrepreneuriale[nom];
        if (dens !== undefined && dens !== null) {
            points.push({ nom, subjScore: data.score, n: data.n, dens });
        }
    }

    // Régression et corrélation : X = densité, Y = bien-être subjectif
    let r = null, slope = null, intercept = null;
    if (points.length >= 3) {
        const n  = points.length;
        const mx = points.reduce((s, p) => s + p.dens,      0) / n;
        const my = points.reduce((s, p) => s + p.subjScore, 0) / n;
        const ss_xy = points.reduce((s, p) => s + (p.dens - mx) * (p.subjScore - my), 0);
        const ss_xx = points.reduce((s, p) => s + (p.dens - mx) ** 2, 0);
        const ss_yy = points.reduce((s, p) => s + (p.subjScore - my) ** 2, 0);
        slope = ss_xy / ss_xx;
        intercept = my - slope * mx;
        r = ss_xy / Math.sqrt(ss_xx * ss_yy);
        points.forEach(p => { p.residual = p.subjScore - (slope * p.dens + intercept); });
    }

    // Header
    const headerEl4 = document.getElementById('correlation-r4');
    if (headerEl4) {
        if (r !== null) {
            const force = Math.abs(r) >= 0.7 ? 'forte' : Math.abs(r) >= 0.4 ? 'modérée' : 'faible';
            const signe = r >= 0 ? 'positive' : 'négative';
            headerEl4.innerHTML = `
                <strong>Bien-être subjectif × Densité entrepreneuriale</strong><br>
                <span style="font-size:20px;font-weight:700;color:${r>=0?'#1b5e20':'#b71c1c'};">r = ${r.toFixed(3)}</span>
                &nbsp;·&nbsp; Corrélation ${force} ${signe} (n = ${points.length} communes avec données)`;
        } else {
            headerEl4.innerHTML = `<strong>Bien-être subjectif × Densité entrepreneuriale</strong><br>
                <em style="color:#888;">Données insuffisantes pour calculer la corrélation</em>`;
        }
    }

    // Scatter : X = densité, Y = score subjectif
    const canvas4 = document.getElementById('scatter-chart4');
    if (canvas4) {
        if (_scatterChart4) { _scatterChart4.destroy(); _scatterChart4 = null; }
        const ctx4 = canvas4.getContext('2d');
        const maxAbsRes = r !== null ? Math.max(...points.map(p => Math.abs(p.residual))) : 1;
        const datasets = [{
            label: 'Communes',
            data: points.map(p => ({ x: p.dens, y: p.subjScore, nom: p.nom, n: p.n })),
            backgroundColor: r !== null
                ? points.map(p => {
                    const t = (p.residual + maxAbsRes) / (2 * maxAbsRes);
                    const r2 = Math.round(220 * (1 - t)), g2 = Math.round(50 + 150 * t), b2 = 80;
                    return `rgba(${r2},${g2},${b2},0.75)`;
                })
                : 'rgba(70, 130, 180, 0.7)',
            pointRadius: points.map(p => Math.min(4 + p.n * 0.15, 10)),
            pointHoverRadius: 8,
        }];
        if (r !== null) {
            const minDens = Math.min(...points.map(p => p.dens));
            const maxDens = Math.max(...points.map(p => p.dens));
            datasets.push({
                label: 'Tendance',
                data: [
                    { x: minDens, y: slope * minDens + intercept },
                    { x: maxDens, y: slope * maxDens + intercept }
                ],
                type: 'line', borderColor: '#555', borderWidth: 1.5,
                borderDash: [5, 4], pointRadius: 0, fill: false,
            });
        }
        _scatterChart4 = new Chart(ctx4, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => {
                        const d = ctx.raw;
                        return `${d.nom} — Densité: ${d.x.toFixed(1)}, B-E subjectif: ${d.y.toFixed(3)} (n=${d.n})`;
                    }}}
                },
                scales: {
                    x: { title: { display: true, text: 'Densité entrepreneuriale (entr./1000 hab.)' } },
                    y: { title: { display: true, text: 'Score bien-être subjectif (1–5)' }, min: 1, max: 5 }
                }
            }
        });
    }

    // Carte des résidus (B-E observé − attendu selon densité)
    const panelSubjDens = document.getElementById('map-subjectif-dens-panel');
    if (panelSubjDens) panelSubjDens.style.display = 'block';
    const carteSubjDens = L.map(document.getElementById('map-subjectif-dens'), {
        center: [42.0, 9.0], zoom: 8, zoomSnap: 0.1, zoomDelta: 0.1
    });
    if (panelSubjDens) panelSubjDens.style.display = 'none';
    cartes['subjectif-dens'] = carteSubjDens;
    carteSubjDens.getContainer().style.backgroundColor = '#ffffff';

    const scoreMap = Object.fromEntries(points.map(p => [p.nom, p]));
    const maxAbsRes = r !== null ? Math.max(...points.map(p => Math.abs(p.residual))) : 1;

    const getColorSubjDens = (p) => {
        if (!p) return '#e0e0e0';
        if (r === null) {
            // Pas de régression, colorier par score brut
            const t = Math.max(0, Math.min(1, (p.subjScore - 1) / 4));
            if (t < 0.5) {
                const i = t * 2;
                return `rgb(220,${Math.round(80 + 140 * i)},${Math.round(50 * i)})`;
            } else {
                const i = (t - 0.5) * 2;
                return `rgb(${Math.round(220 * (1 - i))},${Math.round(180 + 40 * i)},${Math.round(50 * (1 - i))})`;
            }
        }
        const t = Math.max(0, Math.min(1, (p.residual + maxAbsRes) / (2 * maxAbsRes)));
        if (t >= 0.5) {
            const intensity = (t - 0.5) * 2;
            return `rgb(${Math.round(220*(1-intensity))},${Math.round(100 + 100*intensity)},${Math.round(50*(1-intensity))})`;
        } else {
            const intensity = (0.5 - t) * 2;
            return `rgb(${Math.round(180+40*intensity)},${Math.round(80*(1-intensity))},${Math.round(80*(1-intensity))})`;
        }
    };

    L.geoJSON(communeJson, {
        style: feature => {
            const nom = feature.properties.nom;
            const p = scoreMap[nom];
            return {
                fillColor: getColorSubjDens(p),
                color: p ? '#333' : '#bbb',
                weight: p ? 1 : 0.5,
                fillOpacity: p ? 0.85 : 0.3
            };
        },
        onEachFeature: (feature, layer) => {
            const nom = feature.properties.nom;
            const p = scoreMap[nom];
            const dens = _densiteEntrepreneuriale[nom];
            if (p) {
                const attendu = r !== null ? (slope * p.dens + intercept).toFixed(3) : 'N/A';
                const residuTxt = r !== null
                    ? `Résidu : <strong>${(p.residual > 0 ? '+' : '') + p.residual.toFixed(3)}</strong><br>`
                    : '';
                layer.bindPopup(`<strong>${nom}</strong><br>
                    Densité : ${p.dens.toFixed(1)} entr./1000 hab.<br>
                    B-E subjectif : <strong>${p.subjScore.toFixed(3)}</strong> (n=${p.n})<br>
                    B-E attendu : ${attendu}<br>
                    ${residuTxt}`);
            } else {
                layer.bindPopup(`<strong>${nom}</strong><br>
                    Densité : ${dens !== undefined ? dens.toFixed(1) + ' entr./1000 hab.' : 'N/A'}<br>
                    <em style="color:#888;">Pas de données bien-être subjectif</em>`);
            }
        }
    }).addTo(carteSubjDens);

    // Légende
    const legendSubjDens = L.control({ position: 'bottomright' });
    legendSubjDens.onAdd = function() {
        const div = L.DomUtil.create('div', 'info legend');
        const legendBody = r !== null
            ? `<strong>Résidu bien-être subjectif</strong><br>
               <small style="color:#666;">B-E observé − attendu selon densité<br>${points.length} communes avec données</small><br><br>
               <i style="background:#1e7c1e;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>B-E &gt; attendu<br>
               <i style="background:#aaa;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Proche tendance<br>
               <i style="background:#c05050;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>B-E &lt; attendu<br>`
            : `<strong>Score bien-être subjectif</strong><br>
               <small style="color:#666;">${points.length} communes avec données</small><br><br>
               <i style="background:#1e7c1e;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Élevé (≥ 4)<br>
               <i style="background:#e8c820;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Moyen (≈ 3)<br>
               <i style="background:#c05050;width:18px;height:18px;display:inline-block;margin-right:5px;"></i>Faible (≤ 2)<br>`;
        div.innerHTML = `<strong style="font-size:14px;">Légende</strong><br>
            <hr style="margin:8px 0;border:none;border-top:1px solid #ddd;">
            ${legendBody}
            <i style="background:#e0e0e0;width:18px;height:18px;display:inline-block;margin-right:5px;border:1px solid #bbb;"></i>Pas de données<br>`;
        _makeDraggable(div, carteSubjDens);
        return div;
    };
    legendSubjDens.addTo(carteSubjDens);

    ajouterVillesPrincipales(carteSubjDens);
    ajouterCopyright(carteSubjDens);
    ajouterBoutonTelechargement(carteSubjDens, 'subjectif-dens');
    setTimeout(() => carteSubjDens.invalidateSize(), 150);
}

// ==============================================================================
// ONGLET ENTREPRISES
// ==============================================================================

let entreprisesInitialise = false;
let _densiteEntrepreneuriale = null; // données chargées depuis JSON

async function initialiserOngletEntreprises() {
    if (entreprisesInitialise) {
        setTimeout(() => { if (cartes['entreprises']) cartes['entreprises'].invalidateSize(); }, 100);
        return;
    }
    if (!communeJson || !communeJson.features) {
        console.warn('Entreprises: GeoJSON communes non prêt');
        return;
    }

    // Charger les données de densité entrepreneuriale
    if (!_densiteEntrepreneuriale) {
        try {
            const resp = await fetch(BASE_PATH + 'densite_entrepreneuriale.json');
            _densiteEntrepreneuriale = await resp.json();
        } catch (e) {
            console.error('Erreur chargement densite_entrepreneuriale.json', e);
            return;
        }
    }

    entreprisesInitialise = true;

    // Calcul des seuils Jenks sur les valeurs de densité
    const valeurs = Object.values(_densiteEntrepreneuriale).filter(v => v > 0);
    const breaks = calculerJenksBreaks(valeurs, 5);
    const seuilsEnt = [0, ...breaks, Math.ceil(Math.max(...valeurs))];
    const labelsEnt = genererLabelsJenks(seuilsEnt);

    // Palette dégradé vert (5 classes)
    const couleursEnt = ['#c8e6c9', '#81c784', '#388e3c', '#1b5e20', '#0a2e0f'];

    const getColorEnt = (val) => {
        if (val === undefined || val === null) return '#ccc';
        for (let i = 1; i < seuilsEnt.length; i++) {
            if (val <= seuilsEnt[i]) return couleursEnt[i - 1];
        }
        return couleursEnt[couleursEnt.length - 1];
    };

    // Initialiser la carte Leaflet
    const carte = L.map(document.getElementById('map-entreprises'), {
        center: [42.0, 9.0],
        zoom: 8,
        zoomSnap: 0.1,
        zoomDelta: 0.1
    });
    cartes['entreprises'] = carte;
    carte.getContainer().style.backgroundColor = '#ffffff';

    // Couche GeoJSON
    const layer = L.geoJSON(communeJson, {
        style: feature => ({
            fillColor: getColorEnt(_densiteEntrepreneuriale[feature.properties.nom]),
            color: '#333',
            weight: 0.8,
            fillOpacity: 0.75
        }),
        onEachFeature: (feature, l) => {
            const nom = feature.properties.nom;
            const val = _densiteEntrepreneuriale[nom];
            l.bindPopup(`<strong>${nom}</strong><br>Densité : ${val !== undefined ? val.toFixed(1) + ' entr./1000 hab.' : 'N/A'}`);
        }
    }).addTo(carte);

    // Légende identique aux autres cartes
    const legendControl = L.control({ position: 'bottomright' });
    legendControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'info legend');
        div.setAttribute('data-carte-type', 'entreprises');
        div.innerHTML += `<strong class="legende-titre-principal" style="font-size:14px;">Légende</strong><br>`;
        div.innerHTML += `<hr style="margin:8px 0;border:none;border-top:1px solid #ddd;">`;
        div.innerHTML += `<strong>Densité entrepreneuriale</strong><br>`;
        div.innerHTML += `<small style="color:#666;">Entreprises pour 1 000 hab.</small><br><br>`;
        for (let i = couleursEnt.length - 1; i >= 0; i--) {
            div.innerHTML += `<i style="background:${couleursEnt[i]};width:18px;height:18px;display:inline-block;margin-right:5px;"></i>${labelsEnt[i]}<br>`;
        }
        div.innerHTML += `<hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">`;
        div.innerHTML += `
            <div style="display:flex;align-items:center;margin:5px 0;">
                <svg width="30" height="2" style="margin-right:8px;">
                    <line x1="0" y1="1" x2="30" y2="1" stroke="#333" stroke-width="1.5"/>
                </svg>
                <span style="font-size:11px;">Limites communales</span>
            </div>`;
        _makeDraggable(div, carte);
        return div;
    };
    legendControl.addTo(carte);
    legendControls['entreprises'] = legendControl;

    // Villes, rose des vents, copyright personnalisé, bouton téléchargement PNG
    ajouterVillesPrincipales(carte);
    ajouterRoseDesVents(carte);
    const copyrightEnt = L.control({ position: 'bottomleft' });
    copyrightEnt.onAdd = function() {
        const div = L.DomUtil.create('div', 'copyright-control');
        div.style.cssText = 'background:rgba(255,255,255,0.9);padding:6px 10px;border-radius:4px;box-shadow:0 2px 5px rgba(0,0,0,0.2);font-size:9px;color:#666;font-family:Arial,sans-serif;line-height:1.3;text-align:center;';
        div.innerHTML = '© Ghinevra Comiti 2025 — Tous droits réservés';
        return div;
    };
    copyrightEnt.addTo(carte);
    ajouterBoutonTelechargement(carte, 'entreprises');

    // Sync zoom avec les autres cartes
    carte.on('zoomend moveend', function() {
        if (isSyncing) return;
        const size = carte.getSize();
        if (!size || size.x === 0 || size.y === 0) return;
        isSyncing = true;
        const z = carte.getZoom(), c = carte.getCenter();
        for (const k in cartes) {
            if (cartes[k] && k !== 'entreprises') {
                const s = cartes[k].getSize();
                if (s && s.x > 0) cartes[k].setView(c, z, { animate: false });
            }
        }
        setTimeout(() => { isSyncing = false; }, 100);
    });

    setTimeout(() => carte.invalidateSize(), 150);
}

// ONGLET PARANGONS
// ==============================================================================

let parangonsInitialise = false;
let _carteParangons = null;        // carte Leaflet parangons (hors sync)
let parangonsLayerRef = null;      // couche GeoJSON de la carte parangons
let parangonsSelectedCommune = null;
let parangonsCurrentCluster = null;

function initialiserOngletParangons() {
    if (parangonsInitialise) {
        setTimeout(() => { if (_carteParangons) _carteParangons.invalidateSize(); }, 100);
        return;
    }
    // Détruire la carte précédente si elle existe (recalcul après bascule p_k)
    if (_carteParangons) {
        _carteParangons.remove();
        _carteParangons = null;
        parangonsLayerRef = null;
        document.getElementById('map-parangons').innerHTML = '';
    }
    if (!communeJson || !communeJson.features || !indiceFinale || Object.keys(indiceFinale).length === 0) {
        console.warn('Parangons: données non prêtes');
        return;
    }

    parangonsInitialise = true;

    const seuils = seuilsJenks.oppchovec || [0, 2.29, 3.91, 5.08, 7.26, 10];
    const labels = genererLabelsJenks(seuils);
    const nbClasses = seuils.length - 1;

    // ---- 1. Construire les clusters ----
    // clusters[i] = { communes: [{nom, score}], mean, label, couleur }
    const clusters = [];
    for (let i = 0; i < nbClasses; i++) {
        clusters.push({ communes: [], mean: 0, label: labels[i], couleur: colorsJenks[i] });
    }

    for (const [nom, score] of Object.entries(indiceFinale)) {
        let classe = nbClasses - 1;
        for (let i = 1; i < seuils.length; i++) {
            if (score <= seuils[i]) { classe = i - 1; break; }
        }
        clusters[classe].communes.push({ nom, score });
    }

    // Calculer les moyennes
    clusters.forEach(cl => {
        if (cl.communes.length === 0) { cl.mean = 0; return; }
        cl.mean = cl.communes.reduce((s, c) => s + c.score, 0) / cl.communes.length;
        // Trier par distance croissante à la moyenne
        cl.communes.sort((a, b) =>
            Math.abs(a.score - cl.mean) - Math.abs(b.score - cl.mean)
        );
    });

    // ---- 2. Initialiser la carte ----
    const mapEl = document.getElementById('map-parangons');
    const carte = L.map(mapEl, {
        center: [42.0, 9.0],
        zoom: 8,
        zoomSnap: 0.1,
        zoomDelta: 0.1
    });
    _carteParangons = carte;   // NE PAS mettre dans cartes[] pour éviter le sync
    carte.getContainer().style.backgroundColor = '#ffffff';
    ajouterCopyright(carte);

    // Créer la couche GeoJSON
    const getColor = (val) => {
        if (val === undefined || val === null) return '#ccc';
        for (let i = 1; i < seuils.length; i++) {
            if (val <= seuils[i]) return colorsJenks[i - 1];
        }
        return colorsJenks[seuils.length - 2];
    };

    parangonsLayerRef = L.geoJSON(communeJson, {
        style: feature => {
            const val = indiceFinale[feature.properties.nom];
            return {
                fillColor: getColor(val),
                color: '#333',
                weight: 0.8,
                fillOpacity: 0.75
            };
        },
        onEachFeature: (feature, layer) => {
            const nom = feature.properties.nom;
            const val = indiceFinale[nom];
            layer._parangonsNom = nom;
            layer.on('click', () => {
                // Trouver le cluster de cette commune
                let classeIdx = nbClasses - 1;
                if (val !== undefined) {
                    for (let i = 1; i < seuils.length; i++) {
                        if (val <= seuils[i]) { classeIdx = i - 1; break; }
                    }
                }
                // Activer ce cluster dans le panneau
                _parangonsActiverCluster(classeIdx, clusters, carte);
                // Sélectionner cette commune dans la liste
                _parangonsSelectionnerCommune(nom, carte, parangonsLayerRef, getColor);
            });
            layer.bindTooltip(`<strong>${nom}</strong><br>OppChoVec: ${val !== undefined ? val.toFixed(2) : 'N/A'}`);
        }
    }).addTo(carte);

    // ---- 3. Construire les boutons clusters ----
    const btnsDiv = document.getElementById('parangons-cluster-btns');
    btnsDiv.innerHTML = '';
    clusters.forEach((cl, i) => {
        const btn = document.createElement('button');
        btn.className = 'parangons-cluster-btn';
        btn.style.background = cl.couleur;
        btn.textContent = `Classe ${i + 1} (${cl.communes.length})`;
        btn.dataset.cluster = i;
        btn.addEventListener('click', () => _parangonsActiverCluster(i, clusters, carte));
        btnsDiv.appendChild(btn);
    });

    // Activer la classe 1 par défaut
    _parangonsActiverCluster(0, clusters, carte);

    // ---- 4. Recherche de commune ----
    const searchInput = document.getElementById('parangons-search');
    if (searchInput) {
        searchInput.value = '';
        searchInput.oninput = function() {
            const query = this.value.trim().toLowerCase();
            if (!query) return;
            // Chercher dans tous les clusters
            for (let i = 0; i < clusters.length; i++) {
                const match = clusters[i].communes.find(c => c.nom.toLowerCase().includes(query));
                if (match) {
                    // Basculer vers ce cluster si nécessaire
                    if (parangonsCurrentCluster !== i) {
                        _parangonsActiverCluster(i, clusters, carte);
                    }
                    // Sélectionner la commune
                    _parangonsSelectionnerCommune(match.nom, carte, parangonsLayerRef,
                        (v) => {
                            const s = seuilsJenks.oppchovec || [0, 2.29, 3.91, 5.08, 7.26, 10];
                            if (v === undefined || v === null) return '#ccc';
                            for (let j = 1; j < s.length; j++) {
                                if (v <= s[j]) return colorsJenks[j - 1];
                            }
                            return colorsJenks[s.length - 2];
                        });
                    // Faire défiler jusqu'à la ligne dans le tableau
                    setTimeout(() => {
                        const row = document.querySelector(`#parangons-tbody tr[data-nom="${match.nom}"]`);
                        if (row) row.scrollIntoView({ block: 'center', behavior: 'smooth' });
                    }, 100);
                    break;
                }
            }
        };
    }

    setTimeout(() => carte.invalidateSize(), 150);
}

function _parangonsActiverCluster(idx, clusters, carte) {
    parangonsCurrentCluster = idx;
    parangonsSelectedCommune = null;

    // Mettre à jour les boutons
    document.querySelectorAll('.parangons-cluster-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.cluster) === idx);
    });

    const cl = clusters[idx];
    const listDiv = document.getElementById('parangons-list');
    listDiv.innerHTML = `
        <p class="parangons-list-header">Classe ${idx + 1} — ${cl.label}</p>
        <p class="parangons-cluster-info">
            ${cl.communes.length} communes · Moyenne : <strong>${cl.mean.toFixed(2)}</strong>
        </p>
        <table class="parangons-table">
            <thead><tr>
                <th>#</th>
                <th>Commune</th>
                <th>Score</th>
                <th>|Δ moy|</th>
            </tr></thead>
            <tbody id="parangons-tbody"></tbody>
        </table>`;

    const tbody = document.getElementById('parangons-tbody');
    cl.communes.forEach((c, rank) => {
        const dist = Math.abs(c.score - cl.mean).toFixed(3);
        const badgeClass = rank === 0 ? 'gold' : rank === 1 ? 'silver' : rank === 2 ? 'bronze' : '';
        const tr = document.createElement('tr');
        tr.dataset.nom = c.nom;
        tr.innerHTML = `
            <td><span class="parangons-rank-badge ${badgeClass}">${rank + 1}</span></td>
            <td>${c.nom}</td>
            <td>${c.score.toFixed(2)}</td>
            <td>${dist}</td>`;
        tr.addEventListener('click', () => {
            _parangonsSelectionnerCommune(c.nom, carte, parangonsLayerRef,
                (v) => {
                    const seuils2 = seuilsJenks.oppchovec || [0, 2.29, 3.91, 5.08, 7.26, 10];
                    if (v === undefined || v === null) return '#ccc';
                    for (let i = 1; i < seuils2.length; i++) {
                        if (v <= seuils2[i]) return colorsJenks[i - 1];
                    }
                    return colorsJenks[seuils2.length - 2];
                });
        });
        tbody.appendChild(tr);
    });

    // Zoomer sur l'étendue du cluster
    const communesCluster = new Set(cl.communes.map(c => c.nom));
    const bounds = [];
    parangonsLayerRef.eachLayer(layer => {
        if (communesCluster.has(layer._parangonsNom)) {
            bounds.push(layer.getBounds());
        }
    });
    if (bounds.length > 0) {
        const combined = bounds.reduce((acc, b) => acc.extend(b), bounds[0]);
        carte.fitBounds(combined, { padding: [20, 20] });
    }
}

function _parangonsSelectionnerCommune(nom, carte, layer, getColor) {
    parangonsSelectedCommune = nom;

    // Réinitialiser tous les styles
    layer.eachLayer(l => {
        const val = indiceFinale[l._parangonsNom];
        l.setStyle({
            fillColor: l._parangonsNom === nom ? '#ff0000' : getColor(val),
            color: l._parangonsNom === nom ? '#cc0000' : '#333',
            weight: l._parangonsNom === nom ? 2.5 : 0.8,
            fillOpacity: l._parangonsNom === nom ? 0.9 : 0.75
        });
        if (l._parangonsNom === nom) l.bringToFront();
    });

    // Surligner dans la liste
    document.querySelectorAll('#parangons-tbody tr').forEach(tr => {
        tr.classList.toggle('parangon-selected', tr.dataset.nom === nom);
    });

    // Centrer sur la commune sélectionnée
    layer.eachLayer(l => {
        if (l._parangonsNom === nom) {
            carte.fitBounds(l.getBounds(), { padding: [40, 40], maxZoom: 11 });
        }
    });
}

// Gestion des onglets
document.addEventListener('DOMContentLoaded', function() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    console.log('🔍 Nombre de boutons d\'onglets trouvés:', tabButtons.length);
    tabButtons.forEach((btn, index) => {
        console.log(`  Onglet ${index}:`, btn.getAttribute('data-tab'), btn.textContent);
    });

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            console.log('🖱️ Clic sur onglet:', targetTab);

            // Retirer la classe active de tous les boutons et contenus
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Ajouter la classe active au bouton cliqué et au contenu correspondant
            this.classList.add('active');
            document.getElementById(targetTab).classList.add('active');

            // Si on clique sur l'onglet LISA, initialiser les cartes LISA (lazy loading)
            if (targetTab === 'lisatab') {
                initialiserCartesLISA();
            }

            // Si on clique sur l'onglet CAH, initialiser les cartes CAH (lazy loading)
            if (targetTab === 'cahtab') {
                initialiserCartesCAH();
            }

            // Si on clique sur l'onglet Entreprises, initialiser (lazy loading)
            if (targetTab === 'entreprisestab') {
                initialiserOngletEntreprises();
            }

            // Si on clique sur l'onglet Corrélation, initialiser (lazy loading)
            if (targetTab === 'correlationtab') {
                initialiserOngletCorrelation();
            }

            // Si on clique sur l'onglet Parangons, initialiser (lazy loading)
            if (targetTab === 'parangonstab') {
                initialiserOngletParangons();
            }

            // Invalider la taille de la carte pour forcer le redimensionnement
            const mapType = targetTab.replace('tab', '');
            if (cartes[mapType]) {
                setTimeout(() => {
                    cartes[mapType].invalidateSize();
                }, 100);
            }
        });
    });

    // Gestion des sous-onglets LISA
    const lisaSubtabButtons = document.querySelectorAll('.lisa-subtab-button');
    const lisaSubtabContents = document.querySelectorAll('.lisa-subtab-content');

    lisaSubtabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetLisaTab = this.getAttribute('data-lisa-tab');

            // Retirer la classe active de tous les boutons et contenus LISA
            lisaSubtabButtons.forEach(btn => btn.classList.remove('active'));
            lisaSubtabContents.forEach(content => content.classList.remove('active'));

            // Ajouter la classe active au bouton cliqué et au contenu correspondant
            this.classList.add('active');
            document.getElementById(targetLisaTab).classList.add('active');

            // Invalider la taille de la carte LISA pour forcer le redimensionnement
            if (targetLisaTab === 'lisa5pct' && cartes['lisa-5pct']) {
                setTimeout(() => {
                    cartes['lisa-5pct'].invalidateSize();
                }, 100);
            } else if (targetLisaTab === 'lisa1pct' && cartes['lisa-1pct']) {
                setTimeout(() => {
                    cartes['lisa-1pct'].invalidateSize();
                }, 100);
            }
        });
    });

    // Gestion des sous-onglets CAH
    const cahSubtabButtons = document.querySelectorAll('.cah-subtab-button');
    const cahSubtabContents = document.querySelectorAll('.cah-subtab-content');

    cahSubtabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetCAHTab = this.getAttribute('data-cah-tab');

            // Retirer la classe active de tous les boutons et contenus CAH
            cahSubtabButtons.forEach(btn => btn.classList.remove('active'));
            cahSubtabContents.forEach(content => content.classList.remove('active'));

            // Ajouter la classe active au bouton cliqué et au contenu correspondant
            this.classList.add('active');
            document.getElementById(targetCAHTab).classList.add('active');

            // Invalider la taille de la carte CAH pour forcer le redimensionnement
            if (targetCAHTab === 'cah3clusters' && cartes['cah-3']) {
                setTimeout(() => {
                    cartes['cah-3'].invalidateSize();
                }, 100);
            } else if (targetCAHTab === 'cah5clusters' && cartes['cah-5']) {
                setTimeout(() => {
                    cartes['cah-5'].invalidateSize();
                }, 100);
            }
        });
    });

    // Gestion du bouton toggle CAH 3 clusters (carte <-> graphique)
    const toggleCAH3Btn = document.getElementById('toggleCAH3View');
    if (toggleCAH3Btn) {
        toggleCAH3Btn.addEventListener('click', function() {
            const mapView = document.getElementById('cah3-map-view');
            const graphView = document.getElementById('cah3-graph-view');

            if (mapView.style.display === 'none') {
                mapView.style.display = 'block';
                graphView.style.display = 'none';
                this.textContent = '📊 Voir les écarts standardisés';
                setTimeout(() => {
                    if (cartes['cah-3']) {
                        cartes['cah-3'].invalidateSize();
                    }
                }, 100);
            } else {
                mapView.style.display = 'none';
                graphView.style.display = 'block';
                this.textContent = '🗺️ Voir la carte';
            }
        });
    }

    // Gestion des sous-onglets Data Visualisation
    const datavisSubtabButtons = document.querySelectorAll('.datavis-subtab-button');
    datavisSubtabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetDatavisTab = this.getAttribute('data-datavis-tab');

            // Retirer la classe active de tous les boutons et contenus
            datavisSubtabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.datavis-subtab-content').forEach(content => content.classList.remove('active'));

            // Ajouter la classe active au bouton cliqué et au contenu correspondant
            this.classList.add('active');
            document.getElementById('datavis-' + targetDatavisTab).classList.add('active');
        });
    });

    // Gestion des sous-onglets Corrélation
    document.querySelectorAll('.corr-subtab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const targetCorr = this.getAttribute('data-corr');
            document.querySelectorAll('.corr-subtab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.corr-subtab-content').forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            document.getElementById(targetCorr).classList.add('active');

            if (targetCorr === 'corr-entr-occ') {
                // Sous-onglet 1 déjà initialisé, juste invalider la carte
                setTimeout(() => { if (cartes['correlation']) cartes['correlation'].invalidateSize(); }, 100);
            } else if (targetCorr === 'corr-occ-entr') {
                initialiserCorrInverse();
            } else if (targetCorr === 'corr-subjectif') {
                initialiserCorrSubjectif();
            } else if (targetCorr === 'corr-subjectif-dens') {
                initialiserCorrSubjectifDens();
            }
        });
    });

    // Gestion du bouton toggle CAH 5 clusters (carte <-> graphique)
    const toggleCAH5Btn = document.getElementById('toggleCAH5View');
    if (toggleCAH5Btn) {
        toggleCAH5Btn.addEventListener('click', function() {
            const mapView = document.getElementById('cah5-map-view');
            const graphView = document.getElementById('cah5-graph-view');

            if (mapView.style.display === 'none') {
                mapView.style.display = 'block';
                graphView.style.display = 'none';
                this.textContent = '📊 Voir les écarts standardisés';
                setTimeout(() => {
                    if (cartes['cah-5']) {
                        cartes['cah-5'].invalidateSize();
                    }
                }, 100);
            } else {
                mapView.style.display = 'none';
                graphView.style.display = 'block';
                this.textContent = '🗺️ Voir la carte';
            }
        });
    }

    // Event listeners pour les checkboxes de contrôle des routes
    const routeCheckboxes = document.querySelectorAll('.route-checkbox');
    routeCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            // Mettre à jour l'affichage des routes sur toutes les cartes
            Object.keys(cartes).forEach(mapType => {
                if (cartes[mapType]) {
                    mettreAJourAffichageRoutes(cartes[mapType], mapType);
                }
            });
        });
    });

    // Fonction pour mettre à jour les traductions de l'interface
    function mettreAJourTraductionsUI() {
        const lang = langueFrancais ? 'fr' : 'en';
        const uiTranslations = traductions[lang].ui;

        // Mettre à jour tous les éléments avec data-translate
        document.querySelectorAll('[data-translate]').forEach(element => {
            const key = element.getAttribute('data-translate');
            if (uiTranslations[key]) {
                element.textContent = uiTranslations[key];
                // Pour les options de select, forcer la mise à jour
                if (element.tagName === 'OPTION') {
                    element.text = uiTranslations[key];
                }
            }
        });

        console.log(`Interface traduite en ${lang === 'fr' ? 'français' : 'anglais'}`);
    }

    // Event listener pour la checkbox de langue
    const englishCheckbox = document.getElementById('checkbox-english');
    if (englishCheckbox) {
        englishCheckbox.addEventListener('change', function() {
            langueFrancais = !this.checked;
            mettreAJourLegendes();
            mettreAJourTraductionsUI();
        });
    }

    // Appliquer les traductions par défaut au chargement
    mettreAJourTraductionsUI();
    mettreAJourLegendes();
});

// ============================================
// CHATBOT DUMEGPT
// ============================================

// Fonction pour ajouter un message dans l'interface
function addMessageToChat(text, isUser = false) {
    const messagesContainer = document.getElementById('chatbotMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? '👤' : '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';

    // Convertir les retours à la ligne en paragraphes
    const paragraphs = text.split('\n').filter(p => p.trim() !== '');
    paragraphs.forEach(p => {
        const pElement = document.createElement('p');
        pElement.innerHTML = p;
        content.appendChild(pElement);
    });

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    messagesContainer.appendChild(messageDiv);

    // Scroll vers le bas
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Fonction pour afficher l'indicateur de saisie
function showTypingIndicator() {
    const messagesContainer = document.getElementById('chatbotMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typing-indicator-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';

    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<span></span><span></span><span></span>';

    content.appendChild(typingIndicator);
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(content);
    messagesContainer.appendChild(typingDiv);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Fonction pour retirer l'indicateur de saisie
function hideTypingIndicator() {
    const typingMessage = document.getElementById('typing-indicator-message');
    if (typingMessage) {
        typingMessage.remove();
    }
}

// Fonction pour envoyer un message (à connecter avec votre backend)
async function sendMessage() {
    const input = document.getElementById('chatbotInput');
    const sendBtn = document.getElementById('chatbotSend');
    const message = input.value.trim();

    if (message === '') return;

    // Afficher le message de l'utilisateur
    addMessageToChat(message, true);

    // Vider l'input et désactiver le bouton
    input.value = '';
    sendBtn.disabled = true;

    // Afficher l'indicateur de saisie
    showTypingIndicator();

    try {
        // TODO: Remplacer par votre appel API backend
        // const response = await fetch('/api/chat', {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify({ message: message, commune: selectedCommune })
        // });
        // const data = await response.json();
        // const botResponse = data.response;

        // Pour l'instant, réponse simulée
        await new Promise(resolve => setTimeout(resolve, 1500));
        const botResponse = "Je suis prêt à vous aider ! Cette fonctionnalité sera bientôt connectée au backend pour répondre à vos questions sur le bien-être dans les communes de Corse.";

        hideTypingIndicator();
        addMessageToChat(botResponse, false);

    } catch (error) {
        console.error('Erreur lors de l\'envoi du message:', error);
        hideTypingIndicator();
        addMessageToChat("Désolé, une erreur s'est produite. Veuillez réessayer.", false);
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

// Gestion de l'envoi par bouton
document.getElementById('chatbotSend').addEventListener('click', sendMessage);

// Gestion de l'envoi par touche Entrée (Shift+Entrée pour nouvelle ligne)
document.getElementById('chatbotInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ============================================================================
// CARTE DES COMMUNES NUMÉROTÉES
// ============================================================================

// Liste des communes à numéroter (nom normalisé pour correspondre au GeoJSON)
const NUMBERED_MUNICIPALITIES = [
    'Ajaccio', 'Porto-Vecchio', 'Sartène', 'Corte', 'Bastia',
    'Calvi', 'Ghisonaccia', 'Aghione', 'Antisanti', 'Canale-di-Verde',
    'Linguizzetta', 'San-Giuliano', 'Olmo', 'Monte', 'Asco',
    'Moltifao', 'Castifao', 'Canavaggia', 'Monacia-d\'Aullène',
    'Pianottoli-Caldarello', 'Valle-di-Mezzana', 'Villanova',
    'Grossetto-Prugna', 'Cargèse', 'Eccica-Suarella', 'Cauro'
];

let mapNumbered = null;
let numberedMarkersLayer = null;

function initMapNumbered() {
    if (!mapNumbered) {
        mapNumbered = L.map('map-numbered', {
            center: [42.0396, 9.0129],
            zoom: 8,
            zoomControl: true,
            attributionControl: false,
            zoomSnap: 0.1,
            zoomDelta: 0.1
        });

        // Ajouter à la liste des cartes pour la synchronisation
        cartes['numbered'] = mapNumbered;

        // Synchroniser avec les autres cartes
        mapNumbered.on('zoomend moveend', function() {
            if (isSyncing) return;
            isSyncing = true;

            const currentZoom = mapNumbered.getZoom();
            const currentCenter = mapNumbered.getCenter();

            Object.keys(cartes).forEach(mapKey => {
                if (cartes[mapKey] && mapKey !== 'numbered') {
                    cartes[mapKey].setView(currentCenter, currentZoom, { animate: false });
                }
            });

            setTimeout(() => { isSyncing = false; }, 100);
        });

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(mapNumbered);

        // Ajouter le bouton de téléchargement
        ajouterBoutonDownload(mapNumbered, 'numbered_municipalities_map');
    }
}

function afficherCommunesNumerotees() {
    if (!communeJson || !communeJson.features) {
        console.error('GeoJSON des communes non chargé');
        return;
    }

    initMapNumbered();

    // Supprimer l'ancienne couche si elle existe
    if (numberedMarkersLayer) {
        mapNumbered.removeLayer(numberedMarkersLayer);
    }

    // Créer une nouvelle couche de groupe
    numberedMarkersLayer = L.layerGroup().addTo(mapNumbered);

    // Ajouter le fond de carte des communes (blanc)
    L.geoJSON(communeJson, {
        style: {
            fillColor: '#ffffff',
            fillOpacity: 1,
            color: '#d0d0d0',
            weight: 1
        }
    }).addTo(numberedMarkersLayer);

    // Map pour normaliser les noms
    const normalizeName = (name) => {
        return name
            .normalize('NFD')
            .replace(/[\u0300-\u036f]/g, '')
            .toLowerCase()
            .replace(/['-]/g, '')
            .trim();
    };

    // Créer un index des communes par nom normalisé
    const communesByNormalizedName = {};
    communeJson.features.forEach(feature => {
        const nom = feature.properties.nom || feature.properties.NOM || feature.properties.name;
        if (nom) {
            const normalized = normalizeName(nom);
            communesByNormalizedName[normalized] = feature;
        }
    });

    // Parcourir la liste des communes à numéroter
    let foundCount = 0;
    const notFound = [];

    NUMBERED_MUNICIPALITIES.forEach((communeName, index) => {
        const normalizedSearch = normalizeName(communeName);
        const feature = communesByNormalizedName[normalizedSearch];

        if (feature) {
            foundCount++;
            const numero = index + 1;

            // Calculer le centroïde géométrique réel de la commune
            const geom = feature.geometry;
            let center;

            if (geom.type === 'Polygon') {
                // Pour un polygone simple, calculer le centroïde des coordonnées
                const coords = geom.coordinates[0];
                let sumLat = 0, sumLng = 0;
                coords.forEach(coord => {
                    sumLng += coord[0];
                    sumLat += coord[1];
                });
                center = L.latLng(sumLat / coords.length, sumLng / coords.length);
            } else if (geom.type === 'MultiPolygon') {
                // Pour un multipolygone, prendre le centroïde du plus grand polygone
                let largestPoly = geom.coordinates[0];
                let largestArea = 0;
                geom.coordinates.forEach(poly => {
                    const polyCoords = poly[0];
                    if (polyCoords.length > largestArea) {
                        largestArea = polyCoords.length;
                        largestPoly = poly;
                    }
                });
                const coords = largestPoly[0];
                let sumLat = 0, sumLng = 0;
                coords.forEach(coord => {
                    sumLng += coord[0];
                    sumLat += coord[1];
                });
                center = L.latLng(sumLat / coords.length, sumLng / coords.length);
            } else {
                // Fallback sur la méthode bounds
                const bounds = L.geoJSON(feature).getBounds();
                center = bounds.getCenter();
            }

            // Calculer la position du label à l'extérieur (décalage radial)
            const bounds = L.geoJSON(communeJson).getBounds();
            const mapCenter = bounds.getCenter();

            // Vecteur du centre de la carte vers le centre de la commune
            const dx = center.lng - mapCenter.lng;
            const dy = center.lat - mapCenter.lat;
            const distance = Math.sqrt(dx * dx + dy * dy);

            // Normaliser et appliquer un décalage plus important pour éviter la Corse
            const offset = 0.25; // Décalage en degrés (augmenté de 0.15 à 0.25)
            const labelLng = center.lng + (dx / distance) * offset;
            const labelLat = center.lat + (dy / distance) * offset;
            const labelPos = L.latLng(labelLat, labelLng);

            // Créer un marqueur DRAGGABLE avec un DivIcon personnalisé au bout de la ligne
            const marker = L.marker(labelPos, {
                draggable: true,
                icon: L.divIcon({
                    className: 'numbered-municipality-marker',
                    html: `<div class="marker-number" title="Cliquez et glissez pour déplacer">${numero}</div>`,
                    iconSize: [50, 50],
                    iconAnchor: [25, 25]
                })
            });

            // Popup avec les informations
            marker.bindPopup(`
                <div style="font-family: Arial, sans-serif;">
                    <h3 style="margin: 0 0 10px 0; font-size: 16px; color: #333;">
                        <strong>#${numero}</strong> - ${communeName}
                    </h3>
                    <p style="margin: 5px 0; font-size: 13px;">
                        <strong>Code:</strong> ${feature.properties.code || feature.properties.CODE || 'N/A'}
                    </p>
                    <p style="margin: 5px 0; font-size: 12px; color: #666;">
                        💡 <em>Vous pouvez déplacer ce numéro en le glissant</em>
                    </p>
                </div>
            `);

            // Créer une référence à la ligne pour la mettre à jour lors du drag
            const polyline = L.polyline([center, labelPos], {
                color: '#666666',
                weight: 1.5,
                opacity: 0.7,
                dashArray: '3, 6'
            }).addTo(numberedMarkersLayer);

            // Mettre à jour la ligne lors du déplacement du marqueur
            marker.on('drag', function(e) {
                const newPos = e.target.getLatLng();
                polyline.setLatLngs([center, newPos]);
            });

            // Ajouter un effet visuel lors du survol
            marker.on('mouseover', function() {
                polyline.setStyle({ weight: 2.5, opacity: 1 });
            });

            marker.on('mouseout', function() {
                polyline.setStyle({ weight: 1.5, opacity: 0.7 });
            });

            marker.addTo(numberedMarkersLayer);

            // Surligner la commune en rouge
            L.geoJSON(feature, {
                style: {
                    fillColor: '#dc3545',
                    fillOpacity: 0.5,
                    color: '#a71d2a',
                    weight: 2
                }
            }).addTo(numberedMarkersLayer);

        } else {
            notFound.push(communeName);
        }
    });

    console.log(`✅ Communes numérotées trouvées: ${foundCount}/${NUMBERED_MUNICIPALITIES.length}`);
    if (notFound.length > 0) {
        console.warn('⚠️ Communes non trouvées:', notFound);
    }

    // Ajuster la vue pour afficher toutes les communes
    if (numberedMarkersLayer.getLayers().length > 0) {
        mapNumbered.fitBounds(numberedMarkersLayer.getBounds(), { padding: [20, 20] });
    }
}

// Ajouter le gestionnaire d'événement pour l'onglet Numbered
document.querySelector('[data-tab="numberedtab"]').addEventListener('click', function() {
    setTimeout(() => {
        afficherCommunesNumerotees();
        if (mapNumbered) {
            mapNumbered.invalidateSize();
        }
    }, 100);
});
