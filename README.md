
<div align="center">
<h1>🎙️ Forensic Audio Authentication System</h1>
<h3>Deep Learning Pipeline for Splicing & Deepfake Detection</h3>

<p>
An enterprise-grade forensic tool designed to identify audio tampering and synthetic speech injection.
Combining <strong>Linear Frequency Cepstral Coefficients (LFCC)</strong> with a hybrid <strong>CNN-BiLSTM</strong> architecture for judicial-grade reliability.
</p>

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Fedora%20Linux-5199DE?style=for-the-badge&logo=fedora&logoColor=white" />
<img src="https://img.shields.io/badge/Interface-Gradio-FFBB00?style=for-the-badge&logo=gradio&logoColor=black" />
<img src="https://img.shields.io/badge/XAI-SHAP%20%7C%20Grad--CAM-000000?style=for-the-badge" />
</div>

<br>

# **System Architecture**

```mermaid
graph LR
    A[Raw Audio] --> B[LFCC Feature Extraction]
    B --> C[CNN Backbone: Spectral Patterns]
    C --> D[BiLSTM Layer: Temporal Context]
    D --> E[Attention Mechanism]
    E --> F[Softmax Output]
    F --> G{Verdict: Authentic vs Tampered}
````

<br>

## 📝 Table of Contents

1. [Project Overview](#overview)
2. [Technical Rationale](#rationale)
3. [Model Architecture](#architecture)
4. [Dataset Pipeline](#pipeline)
5. [Performance & Forensic Metrics](#metrics)
6. [Explainable AI (XAI)](#xai)
7. [Installation & Usage](#usage)
8. [The Team](#team)

---

<a name="overview"></a>

## 🔭 Project Overview

Ce système traite l'audio non pas comme une simple onde, mais comme une preuve numérique. Il est capable de détecter :

* **Splicing :** Détection des discontinuités de phase et des bruits de montage aux frontières de collage.
* **Deepfakes :** Identification des artefacts de synthèse vocale (vocoders) invisibles à l'oreille humaine.
* **Speed/Pitch Manipulation :** Analyse des distorsions harmoniques liées aux changements de vitesse.

<a name="rationale"></a>

## 🧠 Technical Rationale

* **LFCC vs MFCC :** Contrairement aux MFCC qui compressent les hautes fréquences (mimant l'oreille humaine), les **LFCC** conservent une résolution linéaire. C'est crucial car les artefacts de synthèse et de montage se cachent souvent dans les hautes fréquences que l'oreille humaine ignore.
* **Temporal Context :** L'utilisation d'un **BiLSTM** permet d'analyser l'audio dans les deux sens (passé → futur et futur → passé), facilitant la détection des points de montage qui ne deviennent suspects qu'avec le contexte suivant la coupure.

<a name="architecture"></a>

## 🏗️ Model Architecture

Le moteur d'analyse repose sur une architecture hybride optimisée :

* **CNN Encoder :** 3 couches de convolution traitant la matrice LFCC comme une image pour extraire les textures spectrales.
* **Bidirectional LSTM :** 2 couches pour capturer la dynamique séquentielle du signal.
* **Global Attention :** Un mécanisme d'attention pondéré qui "écoute" plus attentivement les segments où l'anomalie est la plus probable.

<a name="pipeline"></a>

## 🌪️ Dataset & Pipeline

Le pipeline est conçu pour être robuste face au déséquilibre des classes (Data Imbalance) :

1. **Source Ingestion :** Chargement de TIMIT (Réel) et ASVspoof/WaveFake (Synthétique).
2. **Augmentation :** Injection de bruit blanc, masquage temporel et fréquentiel via le script de preprocessing.
3. **Automated Tampering :** Génération de clips falsifiés par "Splicing" aléatoire via `tampering/generate_dataset.py`.

<a name="metrics"></a>

## 📊 Performance & Forensic Metrics

Pour une admissibilité judiciaire, le modèle doit minimiser les faux positifs (FAR).

| Metric                 | Target | Description                                                     |
| :--------------------- | :----- | :-------------------------------------------------------------- |
| **EER**                | < 5%   | Point d'équilibre entre fausse acceptation et faux rejet.       |
| **AUC-ROC**            | > 0.95 | Capacité de discrimination globale du classifieur.              |
| **Precision @ 1% FAR** | > 80%  | Précision garantie avec seulement 1% de marge d'erreur tolérée. |

<a name="xai"></a>

## 🔍 Explainable AI (XAI)

Un verdict sans preuve n'est pas forensique. Le projet intègre :

* **Grad-CAM :** Visualisation thermique sur le spectrogramme pour montrer *où* le modèle a détecté l'anomalie.
* **SHAP Values :** Décomposition de l'impact de chaque coefficient LFCC sur la décision finale.

<a name="usage"></a>

## 💻 Installation & Usage

### Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Execution Workflow

```bash
# 1. Dataset Generation & Preprocessing
python tampering/generate_dataset.py --num_samples 2000
python features/preprocess.py

# 2. Training & Evaluation
python model/train.py
python model/evaluate.py

# 3. Serving (Gradio Web UI)
python -m app.app
```

<a name="team"></a>

## 👥 The Team

<div align="center">
<strong>Chahiri Abderrahmane</strong> • <strong>Yahya Hadir</strong> • <strong>Hafsa Benabou</strong> • <strong>Ilham Madihi</strong>
</div>

<br>

<div align="center">
Made with ❤️ for Audio Forensics • © 2026
</div>

