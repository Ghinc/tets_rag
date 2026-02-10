@echo off
echo ======================================================
echo REINDEXATION CHROMADB AVEC METADONNEES ONTOLOGIE
echo ======================================================
cd /d "%~dp0"
call venv_graphrag\Scripts\activate.bat
python reindex_with_ontology.py
pause
