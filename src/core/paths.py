"""
Sistema de rutas para el proyecto
================================

Este módulo gestiona las rutas del proyecto para asegurar una correcta 
configuración independientemente de cómo se ejecute el sistema.
"""

import os
import sys
from pathlib import Path

# Detectar directorio raíz del proyecto
if getattr(sys, 'frozen', False):
    # Si estamos en un ejecutable compilado
    ROOT_DIR = Path(sys.executable).parent
else:
    # En desarrollo, detectar el directorio raíz
    # (asumimos que este archivo está en src/core/paths.py)
    ROOT_DIR = Path(__file__).parent.parent.parent

# Asegurar que el directorio raíz esté en el sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Rutas principales del sistema
SRC_DIR = ROOT_DIR / 'src'
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / 'data'
DATA_CACHE_DIR = DATA_DIR / 'cache'
SIGNALS_DIR = ROOT_DIR / 'senales'
MODELS_DIR = ROOT_DIR / 'modelos'
EVALUATIONS_DIR = ROOT_DIR / 'evaluations'
PARAMETERS_DIR = ROOT_DIR / 'parameters'
LOGS_DIR = ROOT_DIR / 'logs'

# Crear directorios si no existen
def ensure_directories():
    """Crear directorios necesarios si no existen"""
    for directory in [DATA_DIR, DATA_CACHE_DIR, SIGNALS_DIR, MODELS_DIR, 
                     EVALUATIONS_DIR, PARAMETERS_DIR, LOGS_DIR]:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✅ Directorio creado: {directory}")
            except Exception as e:
                print(f"❌ Error al crear directorio {directory}: {e}")

# Obtener ruta absoluta
def get_abs_path(relative_path):
    """Convierte una ruta relativa a absoluta basada en la raíz del proyecto"""
    return ROOT_DIR / relative_path

# Verificar existencia de archivo
def check_file_exists(file_path):
    """Verifica si un archivo existe en el sistema"""
    path = get_abs_path(file_path) if not os.path.isabs(file_path) else Path(file_path)
    return path.exists() and path.is_file()

# Expone información sobre el entorno
def get_environment_info():
    """Devuelve información sobre el entorno actual"""
    return {
        "root_dir": str(ROOT_DIR),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": sys.platform,
        "is_frozen": getattr(sys, 'frozen', False)
    }

if __name__ == "__main__":
    # Si se ejecuta el archivo directamente, mostrar información
    print("🔧 Configuración de rutas del sistema")
    print("=" * 50)
    print(f"📁 Directorio raíz: {ROOT_DIR}")
    print(f"📁 Directorio src: {SRC_DIR}")
    print(f"📁 Directorio de datos: {DATA_DIR}")
    print(f"📁 Directorio de caché: {DATA_CACHE_DIR}")
    print(f"📁 Directorio de señales: {SIGNALS_DIR}")
    
    # Verificar directorios
    ensure_directories()
