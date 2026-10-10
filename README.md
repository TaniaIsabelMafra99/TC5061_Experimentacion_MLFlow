# TC5061_Experimentacion_MLFlow

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Pipeline modularizado de entrenamiento para predecir la tarifa de viajes en taxi usando **Scikit-Learn** (`RandomForestRegressor`) y seguimiento de experimentos con **MLflow**.

---

## Requisitos Previos

- Python 3.13 (o superior)
- Entorno virtual (`venv`)

---

## Instalación

1. Clonar el repositorio y posicionarse en la carpeta del proyecto:
   ```bash
   cd TC5061_Experimentacion_MLFlow
   ```

2. Activar el entorno virtual:
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

3. Instalar las dependencias fijadas:
   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución del Script de Entrenamiento (CLI)

El script `train.py` permite modificar los hiperparámetros del modelo y la semilla de reproducibilidad desde la línea de comandos:

```powershell
python train.py --n_estimators 100 --max_depth 12 --random_state 42
```

### Argumentos disponibles:
- `--n_estimators`: Número de árboles en el bosque (por defecto: `100`).
- `--max_depth`: Profundidad máxima de los árboles (por defecto: `10`).
- `--random_state`: Semilla de aleatoriedad para garantizar reproducibilidad (por defecto: `42`).

---

##  Visualización de Experimentos en MLflow

Para levantar la interfaz gráfica y comparar las corridas guardadas en SQLite:

```powershell
.\venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Abre en tu navegador: `http://127.0.0.1:5001`

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         tc5061_experimentacion_mlflow and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── tc5061_experimentacion_mlflow   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes tc5061_experimentacion_mlflow a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

