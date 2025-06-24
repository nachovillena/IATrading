# Proyecto Final de Trading IA

## Características
- Ingestión de datos multi-fuente con validación de integridad (Pandera)
- Pipeline modular: optimización, validación, entrenamiento, generación de señales, ajuste de riesgo, búsqueda fina, evaluación
- Orquestación con CLI y Prefect
- Tests unitarios y CI/CD con GitHub Actions
- Contenerización con Docker

## Estructura del Proyecto
- **config/**: YAML y claves API
- **src/**: Código fuente
- **data/**: Cache y datos procesados
- **parameters/**: Grids de parámetros
- **scaler/**, **modelos/**, **senales/**, **tuning/**, **tuning_fino/**, **evaluations/**: Artefactos del pipeline
- **tests/**: Test suite
- **.github/workflows/**: CI pipeline
- **docs/**: Documentación

### Comenzando
```bash
python -m venv .venv
Linux/macOS/Git_Bash -> source .venv/bin/activate
Windows -> .venv\Scripts\activate
pip install -r requirements.txt
```
### Uso CLI
```bash
python src/cli.py optimize EMA_strategy
python src/cli.py validar_parametros
...
```
### Despliegue Docker
```bash
docker build -t trading-ia-final .
docker run trading-ia-final
```

## Menú interactivo por consola

Para usar el menú interactivo, ejecuta:
```
python src/menu.py
```
