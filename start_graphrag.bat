@echo off
REM Script pour lancer demo_graphrag.py dans l'environnement virtuel

echo ========================================
echo   GRAPH-RAG - Demarrage
echo ========================================
echo.

REM Activer l'environnement virtuel
call venv_graphrag\Scripts\activate.bat

REM Charger les variables d'environnement depuis .env
echo Chargement de la configuration (.env)...
python -c "from dotenv import load_dotenv; import os; load_dotenv(); [print(f'set {k}={v}') for k,v in os.environ.items() if k.startswith(('OPENAI_', 'NEO4J_'))]" > temp_env.bat
if exist temp_env.bat (
    call temp_env.bat
    del temp_env.bat
    echo Configuration chargee !
    echo.
)

REM Lancer le script
python demo_graphrag.py

REM Garder la fenetre ouverte en cas d'erreur
pause
