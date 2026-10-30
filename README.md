# Projet 2 — Détecter des attaques réseau avec le Machine Learning

## Utilisation
```bash
pip install -r requirements.txt
python src/train.py      # entraîne, compare les modèles, enregistre models/model.joblib
python src/predict.py    # petit exemple de prédiction
```

## Résultats

**Nom / Prénom :** FAYE Jacqueline

Évaluation sur un jeu de validation (25 % des données, découpage stratifié, `random_state=42`, 750 flux dont 208 attaques).

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) |
|---|---|---|---|
| Régression logistique | 0,919 | 0,899 | 0,940 |
| Arbre de décision (profondeur 6) | 0,959 | 0,957 | 0,961 |
| Forêt aléatoire (300 arbres) | **0,973** | **0,971** | **0,976** |

**Modèle retenu et pourquoi :** la forêt aléatoire, qui a le meilleur F1 de la classe « attaque » (0,973)
ainsi que le meilleur rappel et la meilleure précision. Les relations entre variables ne sont pas
linéaires (par exemple, un `syn_rate` élevé combiné à beaucoup de connexions vers le même hôte), ce que
les arbres capturent mieux que la régression logistique. Le modèle final est ré-entraîné sur les 3000 flux
et enregistré dans `models/model.joblib` sous forme de `Pipeline` (imputation par la médiane, standardisation,
encodage one-hot de `protocol`, puis forêt). Tous les modèles utilisent `class_weight="balanced"` pour tenir
compte du déséquilibre (≈ 28 % d'attaques).

**Analyse des erreurs :** pour la forêt aléatoire, la matrice de confusion est `[[537, 5], [6, 202]]` :
**6 faux négatifs** (attaques non détectées sur 208, soit ≈ 2,9 %) et **5 faux positifs** (flux normaux
signalés à tort sur 542, soit ≈ 0,9 %). Les faux négatifs sont le pire cas en sécurité : une attaque
passée inaperçue (scan, force brute, DoS) peut aboutir à une intrusion ou une indisponibilité sans
qu'aucune alerte ne soit levée. Un faux positif coûte seulement une vérification manuelle. On pourrait
donc, en production, abaisser le seuil de décision pour privilégier le rappel, quitte à accepter un peu
plus de fausses alertes.
