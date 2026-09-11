@echo off
REM Script pour télécharger Docker Desktop automatiquement

echo ========================================================================
echo TELECHARGEMENT DOCKER DESKTOP
echo ========================================================================
echo.
echo Ce script va telecharger Docker Desktop pour vous.
echo Vous devrez ensuite lancer l'installateur manuellement.
echo.
echo ========================================================================
echo.

REM Déterminer l'architecture (AMD64 ou ARM64)
if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    set "DOCKER_URL=https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe"
    set "ARCH=AMD64"
) else if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    set "DOCKER_URL=https://desktop.docker.com/win/main/arm64/Docker%%20Desktop%%20Installer.exe"
    set "ARCH=ARM64"
) else (
    set "DOCKER_URL=https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe"
    set "ARCH=AMD64 (par defaut)"
)

echo Architecture detectee : %ARCH%
echo.
echo URL de telechargement :
echo %DOCKER_URL%
echo.
echo Telechargement en cours...
echo (Fichier ~500 Mo, cela peut prendre 5-10 minutes)
echo.

REM Télécharger avec PowerShell
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%DOCKER_URL%' -OutFile 'DockerDesktopInstaller.exe' -UseBasicParsing}"

if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] Echec du telechargement
    echo.
    echo Telechargez manuellement depuis :
    echo https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo TELECHARGEMENT TERMINE !
echo ========================================================================
echo.
echo Fichier telecharge : DockerDesktopInstaller.exe
echo Emplacement : %CD%
echo.
echo PROCHAINES ETAPES :
echo.
echo 1. Double-cliquez sur DockerDesktopInstaller.exe
echo 2. Suivez l'assistant d'installation (laissez les options par defaut)
echo 3. Redemarrez votre PC si demande
echo 4. Lancez Docker Desktop depuis le menu Demarrer
echo 5. Attendez que Docker soit pret (icone dans la barre des taches)
echo 6. Relancez setup_osrm_local.bat
echo.
echo ========================================================================
echo.

REM Demander si l'utilisateur veut lancer l'installateur maintenant
choice /C ON /M "Voulez-vous lancer l'installateur maintenant"

if errorlevel 2 (
    echo.
    echo Installation reportee. Lancez DockerDesktopInstaller.exe quand vous etes pret.
    echo.
) else (
    echo.
    echo Lancement de l'installateur...
    start "" "DockerDesktopInstaller.exe"
)

pause
