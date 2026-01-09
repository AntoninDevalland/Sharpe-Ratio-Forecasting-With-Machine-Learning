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

# Fonctions utilitaires simples
def y1d(Y):
    """Y peut être un DataFrame avec 'Target' ou une Series/ndarray -> tableau numpy 1D"""
    if isinstance(Y, pd.DataFrame):
        return Y["Target"].to_numpy().ravel()
    if isinstance(Y, pd.Series):
        return Y.to_numpy().ravel()
    return np.asarray(Y).ravel()

# Recherche aléatoire manuelle des hyperparamètres sur le jeu de validation (sans cross-validation)
def tune_on_val(
    base_pipe: Pipeline,
    param_dist: dict,
    X_train, Y_train,
    X_val, Y_val,
    n_iter: int = 40,
    seed: int = 42,
    clip=None,
    verbose: int = 1,
):
    # Mise en forme des cibles
    ytr = y1d(Y_train)
    yva = y1d(Y_val)

    # Initialisation des meilleurs résultats
    best_val = np.inf
    best_params = None
    best_pipe = None

    # Génération aléatoire des combinaisons d'hyperparamètres
    sampler = ParameterSampler(param_dist, n_iter=n_iter, random_state=seed)

    for i, params in enumerate(sampler, start=1):
        # Copie du pipeline et application des paramètres
        pipe = clone(base_pipe)
        pipe.set_params(**params)
        pipe.fit(X_train, ytr)

        # Prédictions sur le jeu de validation
        pred_val = pipe.predict(X_val).ravel()
        if clip is not None:
            pred_val = np.clip(pred_val, clip[0], clip[1])

        # Calcul de la métrique sur la validation
        val_score = mean_absolute_error(yva, pred_val)  # plus petit = meilleur

        if verbose:
            print(f"[{i:02d}/{n_iter}] VAL={val_score:.4f}  params={params}")

        # Mise à jour du meilleur modèle
        if val_score < best_val:
            best_val = val_score
            best_params = params
            best_pipe = pipe

    return best_pipe, best_params, best_val

# Ré-entraînement final sur Train + Validation, puis évaluation sur le jeu de test
def refit_trainval_and_test(best_params, base_pipe, X_train, Y_train, X_val, Y_val, X_test, Y_test, clip=None):
    # Mise en forme des cibles
    ytr = y1d(Y_train)
    yva = y1d(Y_val)
    yte = y1d(Y_test)

    # Fusion des jeux d'entraînement et de validation
    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = np.concatenate([ytr, yva], axis=0)

    # Entraînement du modèle final
    final_pipe = clone(base_pipe)
    final_pipe.set_params(**best_params)
    final_pipe.fit(X_trval, y_trval)

    # Prédictions sur le jeu de test
    pred_test = final_pipe.predict(X_test).ravel()
    if clip is not None:
        pred_test = np.clip(pred_test, clip[0], clip[1])

    # Évaluation finale sur le test
    test_score = mean_absolute_error(yte, pred_test)

    print("\n=== FINAL (réentraînement sur Train+Val) ===")
    print("Meilleurs paramètres :", best_params)
    print(f"Score TEST : {test_score:.4f}")
    print(f"corr(pred, y) : {np.corrcoef(pred_test, yte)[0,1]:.4f}")
    print(f"std(pred)={np.std(pred_test):.4f}  std(y)={np.std(yte):.4f}")

    return final_pipe, test_score, pred_test