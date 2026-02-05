@echo off
TITLE Market Sentiment Bot - Baslatiliyor...
cd /d "%~dp0"

echo [1/2] Bagimliliklar kontrol ediliyor...
python -m pip install -r requirements.txt --quiet

echo [2/2] Web Dashboard baslatiliyor...
python -m streamlit run app.py

pause
