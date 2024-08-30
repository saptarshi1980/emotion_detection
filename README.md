# sentiment_analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

sentiment_analysis a machine learning project with DVC pipleine

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
│                         sentiment_analysis and configuration for tools like black
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
└── sentiment_analysis   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sentiment_analysis a Python module
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

DVC Pipeline

dvc repro -f
dvc dag
dvc metrics show



dvc stage add --force -n data_ingestion -d sentiment_analysis/data_ingestion.py -p data_ingestion.test_size -o data/raw python sentiment_analysis/data_ingestion.py

dvc stage add --force -n data_preprocessing -d sentiment_analysis/data_preprocessing.py -o data/raw python sentiment_analysis/data_preprocessing.py

dvc stage add --force -n feature_engineering -d sentiment_analysis/feature_engineering.py -p feature_engineering.max_features -o data/features python sentiment_analysis/feature_engineering.py

dvc stage add --force -n model_building -d sentiment_analysis/model_building.py -p model_building.n_estimators -p model_building.learning_rate -d data/features -o models/model.pkl python sentiment_analysis/model_building.py


dvc stage add --force -n model_evaluation -d sentiment_analysis/model_evaluation.py -d models/model.pkl -d data/features --metrics metrics.json python sentiment_analysis/model_evaluation.py



git branch -M main




--------

