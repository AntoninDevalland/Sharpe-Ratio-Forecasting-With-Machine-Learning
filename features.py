from itertools import combinations

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import random

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV, ParameterSampler, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

from scipy import stats
from scipy.stats import kruskal

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from typing import Tuple,Optional

from types import SimpleNamespace

import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor

def extract_by_element_and_lag(stat: str, desc_df: pd.DataFrame ) -> pd.DataFrame:
    """
    Fonction qui extrait une statistique spécifique pour chaque élément (stratégie ou instrument) à chaque lag.

    Inputs :
    -----------
    stat : Statistique à extraire ('mean', 'std', etc.)
    desc_df : DataFrame contenant les statistiques descriptives

    Output :
    --------
    DataFrame avec colonnes 'element', 'lag' et la statistique demandée
    """
    # On crée un dictionnaire contenant le nom de la stratégie (I_lag_nbre) ou la variable macroéconomique (X_lag_nbre),
    # le lag converti en entier et la statistique cherchée pour cet élément à ce lag.
    results = [
        {
            'element': col.split('_lag_')[0],
            'lag': int(col.split('_lag_')[1]),
            stat: desc_df.loc[stat, col]
        }
        for col in desc_df.columns if '_lag_' in col
    ]

    # On convertit le dictionnaire en dataframe
    result_df = pd.DataFrame(results)
    return result_df.sort_values(['element', 'lag'])

def create_line_chart(data: pd.DataFrame, stat: str, elements_I: list = None, elements_X: list = None,
                     title: str = None, y_label: str = None) -> plt.Figure:
    """
    Fonction qui crée un graphique linéaire montrant l'évolution d'une statistique dans le temps pour différents éléments.

    Inputs :
    -----------
    data : DataFrame contenant les données à plot, au format [stratégie ou instrument    lag    statistique]
    stat : Nom de la colonne contenant la statistique à visualiser
    elements_I : Liste des stratégies à inclure (toutes les stratégies par défaut)
    elements_X : Liste des instruments à inclure (tous les instruments par défaut)
    title : Titre du graphique
    y_label : Label de l'axe des ordonnées
    """
    # Valeurs par défaut
    if elements_I is None:
        elements_I = sorted([e for e in data['element'].unique() if e.startswith('I_')])
    if elements_X is None:
        elements_X = sorted([e for e in data['element'].unique() if e.startswith('X_')])
    if title is None:
        title = f"Évolution de {stat} dans le temps"
    if y_label is None:
        y_label = stat.capitalize()

    fig, ax = plt.subplots(figsize= (14, 8))

    # On distingue les stratégies des instruments
    colors_I = plt.cm.Blues(np.linspace(0.4, 0.9, len(elements_I)))
    colors_X = plt.cm.Reds(np.linspace(0.4, 0.9, len(elements_X)))

    # Stratégies
    for i, elem in enumerate(elements_I):
        subset = data[data['element'] == elem]
        ax.plot(subset['lag'], subset[stat], 'o-',
                label=elem, color=colors_I[i], linewidth=2)

    # Instruments
    for i, elem in enumerate(elements_X):
        subset = data[data['element'] == elem]
        ax.plot(subset['lag'], subset[stat], 's-',
                label=elem, color=colors_X[i], linewidth=2)

    # Personnalisation du graphique
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Lag (0 = plus récent, 19 = plus ancien)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xticks(range(0, 20))
    ax.invert_xaxis()  # Inverser l'axe x pour que le temps aille de gauche à droite
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    return fig

def pearson_pvalue(r: float, n: int) -> float:
        """
        Calcule la p-value pour un coefficient de corrélation de Pearson donné
        Inputs :
        ---------
        r : coefficient de corrélation
        n : nombre de samples / observations
        """
        if abs(r) == 1.0:
            return 0.0
        t = r * np.sqrt((n-2) / (1-r**2))
        p = 2 * (1 - stats.t.cdf(abs(t), n-2))
        return p

def analyze_correlations(df: pd.DataFrame,
                         alpha: float=0.05) -> Tuple[pd.DataFrame, float]:
    """
    Analyse complète des corrélations d'un ensemble de données :
    - Matrice de corrélation
    - Distribution des coefficients de corrélation
    - Test de significativité des corrélations et distribution des p-values
    La corrélation de Pearson est utilisée et le nombre de données est suffisamment grand pour que
    les hypothèses de normalité soient supposées (théorème central limite)
    Inputs :
        df : DataFrame à analyser
        alpha : Seuil de significativité, par défaut 0.05
    Outputs :
        Tuple contenant la matrice de corrélation et le pourcentage de corrélations significatives
    """
    # 1. Setup
    ## Matrice de corrélation
    corr_matrix = df.corr(method='pearson')

    ## Création du masque pour le triangle inférieur de la matrice de corrélation
    mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    mask_lower = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)

    ## 2. Affichage de la matrice de corrélation
    plt.figure(figsize=(15,12))
    sns.heatmap(corr_matrix, mask=mask_upper, cmap="coolwarm",
                annot=False, linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Matrice de Corrélation (Triangle Inférieur)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # 3. Affichage de la distribution des coefficients de corrélation
    corr_values = corr_matrix.values[mask_lower]
    plt.figure(figsize=(10,6))
    sns.histplot(corr_values, bins=50, kde=True, color="royalblue")
    plt.xlabel("Coefficient de corrélation", fontsize=12)
    plt.ylabel("Fréquence", fontsize=12)
    plt.title("Distribution des coefficients de corrélation", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4. Test de significativité des corrélation et distribution des p-values
    n = len(df)

    ## Calcul des p-values pour chaque coefficient
    pvalues = np.array([pearson_pvalue(r, n) for r in corr_values])

    ## Affichage de la distribution des p-values
    plt.figure(figsize=(10,6))
    plt.hist(pvalues, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=alpha, color='red', linestyle='--',
                label=f'Seuil de significativité (p={alpha})')
    plt.title('Distribution des p-values des coefficients de corrélation', fontsize=14)
    plt.xlabel('p-value', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    ## Statistiques sur les p-values
    significant_count = np.sum(pvalues < alpha)
    total_count = len(pvalues)
    significant_percent = significant_count / total_count * 100

    print(f"Nombre de corrélations testées : {total_count}")
    print(f"Nombre de corrélations significatives (p < {alpha}) : {significant_count}")
    print(f"Pourcentage de corrélations significatives : {significant_percent:.2f}%")

    return corr_matrix, significant_percent

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction qui procède au calcul des rendements des stratégies et instruments financiers.

    Inputs :
    ----------
    df : DataFrame contenant les features initiaux

    Outputs :
    ----------
    DataFrame avec les poids, suivis des rendements
    """
    # 7 premières colonnes : poids
    rdt = df.iloc[:, :7].copy()

    # Calcul des rendements
    columns_value_strategies = df.columns[7:]
    for i in range(len(columns_value_strategies) - 1):
        current_prices = columns_value_strategies[i]
        next_prices = columns_value_strategies[i + 1]
        col_names = f"rendement_{df.columns[i+8]}"  # cela permet d'avoir
        # des colonnes nommées "rendement_I_lag_19" pour le rendement le plus ancien (le plus proche = lag_0)
        rdt[col_names] = np.log(df[next_prices] / df[current_prices])

    return rdt

def feature_engineering(df: pd.DataFrame, weights: Optional[pd.DataFrame]=None,
                        nb_strat: int=7, nb_indic: int=3, nb_days: int=20) -> pd.DataFrame:
    """
    Fonction qui calcule toutes les features :
    - Covariances entre stratégies, indicateurs, et entre les stratégies et les indicateurs
    - Volatilités des stratégies et indicateurs
    - Skewness et kurtosis
    - Ratio de Sharpe personnalisé (si weights est fourni)
    - HHI

    Inputs :
    -----------
        df: DataFrame contenant les rendements
        weights : DataFrame ou array contenant les poids (optionnel)
        nb_strat: Nombre de stratégies (7 par défaut)
        nb_indic: Nombre d'indicateurs (3 par défaut)
        nb_days: Nombre de jours de rendements (20 par défaut)

    Output :
    -----------
        DataFrame contenant toutes les features calculées pour chaque observation.
    """
    nb_rows = df.shape[0]

    # Création de paires et de noms de colonnes
    # On récupère les nb_strat * nb_days premières colonnes pour les rendements des stratégies,
    # et le reste pour les rendements des indicateurs
    strat_cols = df.columns[:nb_strat * nb_days].tolist()
    indic_cols = df.columns[nb_strat * nb_days:(nb_strat + nb_indic) * nb_days].tolist()

    # 1. Définition des colonnes pour les covariances
    strat_pairs = list(combinations(range(1, nb_strat+1), 2))
    strat_strat_cols = [f'cov_I_{i}_I_{j}' for i, j in strat_pairs]

    indicator_pairs = list(combinations(range(1, nb_indic+1), 2))
    indic_indic_columns = [f'cov_X_{i}_X_{j}' for i, j in indicator_pairs]

    strat_indic_pairs = [(i, j) for i in range(1, nb_strat+1) for j in range(1, nb_indic+1)]
    strat_indic_columns = [f'cov_I_{i}_X_{j}' for i, j in strat_indic_pairs]

    # 2. Définition des colonnes pour les volatilités
    vol_columns = [f'vol_I_{s}' for s in range(1, nb_strat+1)] + [f'vol_X_{i}' for i in range(1, nb_indic+1)]

    # 3. Définition des colonnes pour le skewness et kurtosis
    skew_columns = [f'skew_I_{s}' for s in range(1, nb_strat+1)] + [f'skew_X_{i}' for i in range(1, nb_indic+1)]
    kurt_columns = [f'kurt_I_{s}' for s in range(1, nb_strat+1)] + [f'kurt_X_{i}' for i in range(1, nb_indic+1)]

    # 4. Colonnes pour le ratio de Sharpe et HHI (si weights fourni)
    nb_weeks = nb_days // 5
    sharpe_columns = [f'sharpe_week_{w+1}' for w in range(nb_weeks)] if weights is not None else []
    hhi_column = ['HHI'] if weights is not None else []

    # Combinaison de toutes les colonnes pour le tableau final
    all_columns = strat_strat_cols + indic_indic_columns + strat_indic_columns + \
        vol_columns + skew_columns + kurt_columns + sharpe_columns + hhi_column
    results = pd.DataFrame(np.nan, index=df.index, columns=all_columns, dtype=float)


    # On récupère les indices des colonnes pour chaque stratégie et indicateur
    strat_indices = {}
    indic_indices = {}

    for s in range(1, nb_strat+1):
        strat_cols_s = strat_cols[(s-1)*nb_days:s*nb_days]
        strat_indices[s] = [df.columns.get_loc(col) for col in strat_cols_s]

    for i in range(1, nb_indic+1):
        indic_cols_i = indic_cols[(i-1)*nb_days:i*nb_days]
        indic_indices[i] = [df.columns.get_loc(col) for col in indic_cols_i]


    for row_pos in range(nb_rows):
        idx = df.index[row_pos]              # index initial (à conserver)
        row_values = df.iloc[row_pos].values


        # Extraction des données pour chaque stratégie et indicateur
        strat_data = {}
        indic_data = {}

        for s in range(1, nb_strat+1):
            strat_data[s] = row_values[strat_indices[s]]

        for i in range(1, nb_indic+1):
            indic_data[i] = row_values[indic_indices[i]]

        ########## Calcul des covariances ##########

        for col_idx, (s1, s2) in enumerate(strat_pairs):
            cov = np.cov(strat_data[s1], strat_data[s2])[0, 1]
            results.loc[idx, strat_strat_cols[col_idx]] = cov

        # Entre indicateurs
        for col_idx, (i1, i2) in enumerate(indicator_pairs):
            cov = np.cov(indic_data[i1], indic_data[i2])[0, 1]
            results.loc[idx, indic_indic_columns[col_idx]] = cov

        # Entre stratégies et indicateurs
        for col_idx, (s, i) in enumerate(strat_indic_pairs):
            cov = np.cov(strat_data[s], indic_data[i])[0, 1]
            results.loc[idx, strat_indic_columns[col_idx]] = cov

        ########## Calcul des volatilités ##########
        for s in range(1, nb_strat+1):
            vol = np.std(strat_data[s], ddof=1)
            results.loc[idx, f'vol_I_{s}'] = vol

        for i in range(1, nb_indic+1):
            vol = np.std(indic_data[i], ddof=1)
            results.loc[idx, f'vol_X_{i}'] = vol

        ########## Calcul des skewness et kurtosis ##########
        for s in range(1, nb_strat+1):
            results.loc[idx, f'skew_I_{s}'] = stats.skew(strat_data[s])
            results.loc[idx, f'kurt_I_{s}'] = stats.kurtosis(strat_data[s], fisher=True)

        for i in range(1, nb_indic+1):
            results.loc[idx, f'skew_X_{i}'] = stats.skew(indic_data[i])
            results.loc[idx, f'kurt_X_{i}'] = stats.kurtosis(indic_data[i], fisher=True)

        ########## Calcul du HHI ##########
        if weights is not None:
            # On récupère les poids pour la ligne
            if isinstance(weights, pd.DataFrame):
                w = weights.loc[idx].values
            else:
                w = weights[row_pos]

            hhi = np.sum(w ** 2)    
            results.loc[idx, 'HHI'] = hhi

        ########## Calcul des ratios de Sharpe ##########
        if weights is not None:
            # Récupérer les poids pour cette ligne
            if isinstance(weights, pd.DataFrame):
                w = weights.loc[idx].values
            else:
                w = weights[row_pos]

            # Calcul pour chaque semaine
            for week in range(nb_weeks):
                # Indices pour la semaine
                start_idx = week * 5
                end_idx = start_idx + 5

                # Calcul du numérateur
                weekly_returns = np.zeros(nb_strat)
                # On calcule la somme des rendements pour chaque stratégie
                for s in range(1, nb_strat+1):
                    weekly_returns[s-1] = np.sum(strat_data[s][start_idx:end_idx])
                # Somme pondérée des rendements hebdomadaires + annualisation
                weighted_return = np.sum(w * weekly_returns) * (252 / 5)

                # Calcul du dénominateur
                # Matrice de covariance
                # Remarque : le calcul "manuel" est nécessaire car on considère les rendements d'un seul échantillon
                cov_matrix = np.zeros((nb_strat, nb_strat))

                for i in range(1, nb_strat+1):
                    for j in range(1, nb_strat+1):
                        # On récupère tous les rendements pour les stratégies i et j
                        returns_i = strat_data[i]
                        returns_j = strat_data[j]

                        # Moyennes des rendements
                        mean_i = np.mean(returns_i)
                        mean_j = np.mean(returns_j)

                        # Covariance
                        dev_i = returns_i - mean_i
                        dev_j = returns_j - mean_j
                        cov_ij = np.sum(dev_i * dev_j) / (len(returns_i) - 1)

                        cov_matrix[i-1, j-1] = cov_ij

                # Calcul de la variance du portefeuille pondéré
                portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))

                # Volatilité annualisée
                portfolio_volatility = np.sqrt(252 * portfolio_variance)

                # Seuil minimum de 0.005
                denominator = max(portfolio_volatility, 0.005)

                # Ratio de Sharpe
                sharpe_ratio = weighted_return / denominator
                results.loc[idx, f'sharpe_week_{week+1}'] = sharpe_ratio

    return results

def process_features(features_rdt: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction qui permet de concaténer tous nos features :
        - ceux créés via la fonction ci-dessus,
        - les poids,
        - les rendements journaliers.

    Input :
        features_rdt : DataFrame contenant toutes les données

    Output :
        DataFrame contenant toutes les features
    """
    # Colonnes par type
    cols_strategies = [col for col in features_rdt.columns if "rendement_I" in col]
    cols_indicators = [col for col in features_rdt.columns if "X" in col]


    features_data = features_rdt[cols_strategies+cols_indicators]
    weight_columns = features_rdt.iloc[:,:7]

    # On calcule toutes les features
    new_features = feature_engineering(features_data, weight_columns)
    # On ajoute les rendements bruts aux features
    rendements_df = features_data
    # On ajoute les poids et les rendements
    new_features = pd.concat([weight_columns, rendements_df, new_features], axis=1)
    new_features = new_features.apply(pd.to_numeric, errors='coerce')
    # On vérifie les dimensions
    print(f"features_rdt: {features_rdt.shape}")
    print(f"new_features: {new_features.shape}")

    return new_features

