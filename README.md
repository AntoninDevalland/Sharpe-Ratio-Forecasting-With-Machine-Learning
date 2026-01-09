# Prédiction du ratio de Sharpe par Machine Learning

Ce projet étudie la **prédiction du ratio de Sharpe futur** d’un portefeuille composé de plusieurs stratégies quantitatives, à l’aide de méthodes de machine learning. L’objectif est de prédire le **ratio de Sharpe annualisé sur les 5 prochains jours de trading**, pour une allocation donnée entre différentes stratégies.

---

## Présentation du projet

- Cible : ratio de Sharpe futur d’un portefeuille de 7 stratégies quantitatives  
- Horizon : 5 jours de trading (Sharpe annualisé)
- Données : poids des stratégies, trajectoires de prix sur 21 jours et valeurs de plusieurs instruments
- Approche : feature engineering + modèles supervisés avec une séparation stricte entraînement / validation / test afin d’éviter toute fuite d’information (data leakage)

Plusieurs modèles sont testés :
- Elastic Net
- Modèles à base d’arbres (Random Forest, ExtraTrees, XGBoost)
- Réseaux de neurones (MLP, ensemble de MLP)

---

## Structure du dépôt
- `main.ipynb`  
  Notebook principal exécutant l’ensemble du pipeline : chargement des données, feature engineering, entraînement des modèles et évaluation.

- `Sharpe_forecast.pdf`  
  Rapport détaillé présentant les données, la méthodologie, les modèles utilisés et les résultats.

- `src/`  
  Contient le code principal pour la construction des features et l’entraînement / l’évaluation des modèles.

- `data/`  
  Dossier des données d’entrée et des sorties générées par les modèles.


---
## Méthodologie (résumé)

- Feature engineering : rendements, structures de covariance, moments d’ordre supérieur, indice de concentration du portefeuille (HHI), etc. 
- Sélection des hyperparamètres sur le jeu de validation, puis ré-entraînement des modèles retenus avant l’évaluation finale.
- Comparaison de plusieurs modèles de machine learning : Elastic Net, Random Forest, ExtraTrees, XGBoost et réseaux de neurones (MLP) et ensemble sur le MLP.

## Résultats (vue d’ensemble)

- Les modèles parviennent à améliorer la prédiction de manière robuste par rapport à un benchmark naïf, malgré la difficulté intrinsèque de la prédiction d’un ratio de Sharpe futur à horizon 5 jours.
- L’analyse des hyperparamètres sélectionnés met en évidence que les modèles généralisant le mieux privilégient des architectures simples et une forte régularisation, ce qui indique que le signal est diffus, instable et réparti sur de nombreuses variables plutôt que porté par quelques facteurs dominants.
- Dans un contexte de signal faible et bruité, un ensemble de MLP permet de combiner la capacité des réseaux de neurones à capter des relations non linéaires avec l’effet stabilisant de l’agrégation.



