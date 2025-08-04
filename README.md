# Phishing Email Analysis

## Aperçu
Ce projet analyse un jeu de données d’emails pour détecter les emails de phishing, en combinant **SQL** pour le nettoyage, **Python** pour l’analyse et la modélisation, et **Power BI** est en cours

## Structure
- `data/` : Données brutes (`Phishing_Email.csv`) et nettoyées (`cleaned_emails.csv`).
- `sql/` : Scripts SQL (`data_cleaning.sql`).
- `python/` : Notebook Jupyter (`phishing_emails.ipynb`) et script Python (`phishing_emails.py`).
- `python/outputs/` : Visualisations (scatterplots, WordClouds, matrice de confusion).

## Installation et exécution
### Jupyter Notebook
1. Installer les dépendances :
   ```bash
   pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud
Exécuter python/phishing_emails.ipynb dans Jupyter Notebook pour voir le processus complet (code, markdown, visualisations).


