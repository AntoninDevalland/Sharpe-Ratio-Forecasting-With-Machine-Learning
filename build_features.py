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



