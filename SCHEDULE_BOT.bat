@echo off
TITLE Market Sentiment Bot - Arkaplan Otomasyonu
cd /d "%~dp0"

echo [%date% %time%] Otomatik analiz baslatiliyor...

:: 1. Bitcoin Analizi
echo Bitcoin analizi yapiliyor...
python sentiment_bot.py --asset crypto --symbol bitcoin --count 30

:: 2. BIST100 Analizi
echo BIST100 analizi yapiliyor...
python sentiment_bot.py --asset bist --count 30

echo [%date% %time%] Analiz tamamlandi ve veritabanina kaydedildi.
exit
