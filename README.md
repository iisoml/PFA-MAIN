# 🏥 Medical Waiting Time Prediction

### Machine Learning | Healthcare | Data Science

## 📌 Présentation du projet

Ce projet vise à développer une solution intelligente basée sur le **Machine Learning pour analyser et prédire les délais médicaux**.

L'objectif est d'exploiter les données disponibles afin d'estimer les délais, d'identifier les facteurs qui influencent l'attente et de fournir des informations utiles à l'amélioration de la gestion des services de santé.

La solution s'appuie sur plusieurs modèles de Machine Learning et une architecture permettant de servir les prédictions à travers une API et une interface utilisateur.

## 🎯 Objectifs

- Analyser et exploiter les données liées aux délais médicaux.
- Identifier les variables ayant un impact sur les délais.
- Préparer les données pour l'entraînement des modèles.
- Développer et comparer des modèles prédictifs.
- Mettre les modèles à disposition via une API.
- Faciliter l'accès aux prédictions grâce à une interface utilisateur.

## 🧠 Approche Machine Learning

Le projet intègre notamment deux approches :

- **XGBoost Regression :** modèle de régression pour l'estimation de valeurs numériques.
- **Multi-Layer Perceptron (MLP) :** réseau de neurones utilisé pour une tâche de classification.

Les modèles entraînés sont sauvegardés sous forme d'artefacts afin de permettre leur utilisation lors de l'inférence.

## ⚙️ Architecture du projet

Le dépôt est organisé autour de plusieurs composants :

```text
PFA-MAIN/
│
├── src/                  # Code source
├── scripts/              # Scripts de traitement
├── notebooks/            # Expérimentations et analyses
├── data/
│   └── processed/        # Données prétraitées
├── artifacts/            # Artefacts des modèles
│
├── mlp_classification.h5
├── xgb_regression.json
├── requirements.txt
├── dockerfile
└── README.md
```

## 🛠️ Technologies utilisées

| Domaine | Technologies |
|---|---|
| Langage | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow |
| API Backend | FastAPI, Uvicorn |
| Interface | Gradio |
| Experiment Tracking | MLflow |
| Conteneurisation | Docker |

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/iisoml/PFA-MAIN.git
cd PFA-MAIN
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l'environnement

Windows :

```bash
venv\Scripts\activate
```

Linux / macOS :

```bash
source venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 📊 Perspectives

- Améliorer les performances et la robustesse des modèles.
- Approfondir l'analyse des facteurs influençant les délais.
- Enrichir les fonctionnalités de l'interface utilisateur.
- Améliorer le déploiement et le suivi des modèles.

## 👥 Collaboration

Projet réalisé en équipe dans le cadre d'un projet de Data Science appliquée au domaine médical.

Dépôt GitHub : [iisoml/PFA-MAIN](https://github.com/iisoml/PFA-MAIN)
