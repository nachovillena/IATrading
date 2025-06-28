@echo off
echo Ejecutando menu_app.py y guardando la salida en output_log.txt...
python menu_app.py > output_log.txt 2>&1
echo Ejecución finalizada. Revisando los últimos 20 líneas del log:
echo ------------------------------------------------
type output_log.txt | findstr /v "^$" | tail -20
echo ------------------------------------------------
echo El log completo está disponible en output_log.txt
pause
