# CS180 Project 1 — Starter Kit

This folder contains a working baseline (single-scale + pyramid) and a tiny website template.

## 1) Installer l'environnement
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
```

## 2) Mettre les données
Téléchargez `data.zip` depuis la page du cours et décompressez-la dans `data/` à la racine du repo (ne commitez pas les `.tif`).
Placez les images `.jpg` / `.tif` ici: `data/`

## 3) Lancer l'alignement
```bash
python code/colorize.py data/cathedral.jpg outputs/cathedral.jpg
python code/colorize.py data/ outputs/ --metric ncc --levels auto
```

Les sorties arrivent dans `outputs/` + un `results.json` récapitulant les offsets.

## 4) Publier une page GitHub Pages
Le site d'exemple est dans `website/`. Copiez vos résultats (JPG) dans `website/assets/` et éditez `website/project1.html`.
Activez **GitHub Pages** (Settings → Pages → Deploy from branch) et sélectionnez la branche, dossier `/website` (ou `/docs`).

