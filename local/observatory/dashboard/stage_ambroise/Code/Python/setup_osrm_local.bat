@echo off
REM Script d'installation automatique d'OSRM Local pour Windows
REM Ce script automatise toutes les étapes après l'installation de Docker Desktop

echo ========================================================================
echo Installation OSRM Local - Calcul Temps de Trajet Corse
echo ========================================================================
echo.

REM Vérifier que Docker est installé
echo [1/7] Verification de Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Docker n'est pas installe ou n'est pas demarre
    echo.
    echo Veuillez installer Docker Desktop depuis:
    echo https://www.docker.com/products/docker-desktop/
    echo.
    echo Apres installation, relancez ce script.
    pause
    exit /b 1
)
echo [OK] Docker est installe et accessible
echo.

REM Créer le dossier pour les données OSRM
echo [2/7] Creation du dossier osrm_data...
if not exist "osrm_data" mkdir osrm_data
cd osrm_data
echo [OK] Dossier cree: %CD%
echo.

REM Télécharger les données OpenStreetMap Corse
echo [3/7] Telechargement des donnees OpenStreetMap Corse (~50 Mo)...
if exist "corse-latest.osm.pbf" (
    echo [INFO] Fichier corse-latest.osm.pbf deja present, telechargement ignore
) else (
    echo Telechargement en cours... Cela peut prendre 2-5 minutes selon votre connexion.
    powershell -Command "Invoke-WebRequest -Uri 'http://download.geofabrik.de/europe/france/corse-latest.osm.pbf' -OutFile 'corse-latest.osm.pbf'"
    if %errorlevel% neq 0 (
        echo [ERREUR] Echec du telechargement
        echo Essayez de telecharger manuellement depuis:
        echo http://download.geofabrik.de/europe/france/corse.html
        pause
        exit /b 1
    )
    echo [OK] Donnees OSM Corse telechargees
)
echo.

REM Télécharger l'image Docker OSRM
echo [4/7] Telechargement de l'image Docker OSRM (~300 Mo)...
echo Cela peut prendre 5-10 minutes selon votre connexion.
docker pull osrm/osrm-backend
if %errorlevel% neq 0 (
    echo [ERREUR] Echec du telechargement de l'image Docker
    pause
    exit /b 1
)
echo [OK] Image Docker OSRM telechargee
echo.

REM Extraction des données
echo [5/7] Extraction des donnees OSM (1-2 minutes)...
docker run -t -v "%CD%:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/corse-latest.osm.pbf
if %errorlevel% neq 0 (
    echo [ERREUR] Echec de l'extraction
    pause
    exit /b 1
)
echo [OK] Extraction terminee
echo.

REM Contraction du graphe
echo [6/7] Construction du graphe de routage (30-60 secondes)...
docker run -t -v "%CD%:/data" osrm/osrm-backend osrm-contract /data/corse-latest.osrm
if %errorlevel% neq 0 (
    echo [ERREUR] Echec de la contraction
    pause
    exit /b 1
)
echo [OK] Graphe de routage construit
echo.

REM Créer un script de lancement
echo [7/7] Creation du script de lancement...
cd ..
(
echo @echo off
echo echo ========================================================================
echo echo Serveur OSRM Local - Corse
echo echo ========================================================================
echo echo.
echo echo Le serveur OSRM est maintenant accessible sur: http://localhost:5000
echo echo.
echo echo Pour arreter le serveur: appuyez sur Ctrl+C
echo echo.
echo echo ========================================================================
echo echo.
echo cd osrm_data
echo docker run -t -i -p 5000:5000 -v "%%CD%%:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/corse-latest.osrm
) > start_osrm_server.bat
echo [OK] Script de lancement cree: start_osrm_server.bat
echo.

echo ========================================================================
echo INSTALLATION TERMINEE AVEC SUCCES !
echo ========================================================================
echo.
echo Pour utiliser OSRM local:
echo.
echo 1. Lancez le serveur OSRM:
echo    ^> start_osrm_server.bat
echo.
echo 2. Dans un autre terminal, lancez le script Python:
echo    ^> python services_accessibles_osrm_local.py
echo.
echo Le serveur doit rester ouvert pendant que vous utilisez le script Python.
echo.
echo Gain de performance attendu: 60-100x plus rapide que l'API publique
echo (3-5 minutes au lieu de 3-5 heures pour 360 communes)
echo.
echo ========================================================================
echo.
pause
