#!/bin/bash
echo "Ejecutando menu_app.py..."
python menu_app.py 2>&1 | tee output_log_bash.txt
echo "Ejecución finalizada. El log está disponible en output_log_bash.txt"
