from matplotlib.colors import LinearSegmentedColormap
from numpy import floating
from pandas import DataFrame, Series
from scipy.stats import t
from sklearn.utils import resample
# ML classes
from dataHandler import Config, MetadataHandler, OligosHandler, FeatureManager

import argparse

# basics
import time

from pathlib import Path
import re
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any
import joblib

# Plots
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.gridspec import GridSpec

#ML
from xgboost import XGBClassifier
import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_predict, cross_val_score, LeaveOneOut, StratifiedKFold, KFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest, VarianceThreshold, SelectFromModel, SelectPercentile
from sklearn import set_config
set_config(transform_output = "pandas")
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def search_best_model(estimator, param_grid, X_train, y_train, n_splits=5, n_iter=30, random_state=420, n_jobs = 1):
    # Create a custom scorer using C-index
    cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = BayesSearchCV(estimator, search_spaces=param_grid,
                                 cv=cv_inner,
                                 n_iter=n_iter,
                                 scoring='roc_auc',
                                 refit=True,
                                 random_state=random_state,
                                 n_jobs=n_jobs)

    search.fit(X_train, y_train)

    return search.best_estimator_

def calculate_mean_std_ci_tpr_auc(auc_list, tpr_list, mean_fpr, bootstrap=True):
        """
        Calculate mean FPR, mean TPR, and std TPR for ROC curves.
        """
        # Aggregate TPRs
        tprs = np.array(tpr_list)
        mean_tpr = tprs.mean(axis=0)
        mean_tpr[-1] = 1.0  # Ensure curve ends at (1, 1)
        std_tpr = tprs.std(axis=0)  # ddof=1

        se = std_tpr * 1.96 if bootstrap else std_tpr
        tprs_lower = np.maximum(mean_tpr - se, 0)
        tprs_upper = np.minimum(mean_tpr + se, 1)

        # Aggregate AUCs
        aucs = np.array(auc_list)
        auc_mean = aucs.mean()
        auc_std = aucs.std()

        if bootstrap:
            auc_ci_lower = np.percentile(aucs, 2.5)
            auc_ci_upper = np.percentile(aucs, 97.5)
            roc_metrics = {
                'boot_mean_fpr': mean_fpr,
                'boot_mean_tpr': mean_tpr,
                'boot_std_tpr': std_tpr,
                'boot_tprs_upper': tprs_upper,
                'boot_tprs_lower': tprs_lower,
                'boot_auc_mean': auc_mean,
                'boot_auc_std': auc_std,
                'boot_auc_ci_lower': auc_ci_lower,
                'boot_auc_ci_upper': auc_ci_upper
            }
            return roc_metrics
        else:
            t_value = t.ppf(0.975, len(aucs) - 1)
            se = t_value * (auc_std / np.sqrt(len(aucs)))
            auc_ci_lower = np.maximum(auc_mean - se, 0)  # 95% CI lower
            auc_ci_upper = np.minimum(auc_mean + se, 1)  # 95% CI upper
            roc_metrics = {
                'fpr': mean_fpr,
                'tpr': mean_tpr,
                'std_tpr': std_tpr,
                'tprs_upper': tprs_upper,
                'tprs_lower': tprs_lower,
                'auc': auc_mean,
                'auc_std': auc_std,
                'auc_ci_lower': auc_ci_lower,
                'auc_ci_upper': auc_ci_upper}

            return roc_metrics

def bootstrap_auc(mean_fpr=None, estimator=None, X=None, y_true=None, y_pred=None, n_bootstraps = 200, random_state=420):
        tpr_bootstraps = []
        auc_bootstraps = []

        if mean_fpr is None:
            mean_fpr = np.linspace(0, 1, 100)  # Define default mean_fpr here

        for i in range(n_bootstraps):
            if estimator is not None and X is not None and y_true is not None:
                X_resampled, y_resampled = resample(X, y_true, stratify=y_true, random_state=random_state + i)
                y_pred_resampled = estimator.predict_proba(X_resampled)[:, 1]
            elif y_pred is not None and y_true is not None:
                y_resampled, y_pred_resampled = resample(y_true, y_pred, stratify=y_true,
                                                         random_state=random_state + i)
            else:
                raise ValueError("Missing arguments. Estimator 'estimator', features 'X' and their true target 'y_true' must be provided for bootstrapping auc for test data."
                                 "For loocv, true target 'y_true' and predictions 'y_pred' must be provided.")

            interp_tpr, auc_value = compute_interp_tpr_auc(y_resampled, y_pred_resampled, mean_fpr)
            tpr_bootstraps.append(interp_tpr)
            auc_bootstraps.append(auc_value)

        roc_metrics = calculate_mean_std_ci_tpr_auc(auc_bootstraps, tpr_bootstraps, mean_fpr, bootstrap = True)

        return roc_metrics

def compute_interp_tpr_auc(y_true, y_pred_proba, mean_fpr):
        # Compute FPR and TPR
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

        # Interpolate TPR to the common mean FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # Ensure the curve starts at (0, 0)

        # Compute AUC
        auc_value = auc(fpr, tpr)

        return interp_tpr, auc_value

def make_pipeline(peptide_cols, demog_cols, estimator):
    """
    Build a pipeline which:
      - on peptide_cols: does variance threshold + SelectPercentile(mutual_info)
      - on demog_cols : just passes them through untouched
      - then fits whatever `estimator` you give it
    """
    transformers = []
    if peptide_cols:
        transformers.append((
            "peptides",
            Pipeline([
                ("variance_removal", VarianceThreshold(threshold=0.0)),
                ("feature_selection", SelectPercentile(mutual_info_classif, percentile=20)),
            ]),
            peptide_cols
        ))
    if demog_cols:
        transformers.append(
            ("demographics", "passthrough", demog_cols)
        )

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",   # drop anything not in peptide_cols or demog_cols
        verbose_feature_names_out=False   # <— disable automatic "peptides__…" prefixes
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator",     estimator)
    ])
    return pipe


def nested_cv_single(train_idx, valid_idx, X_train, y_train, pipeline = None,
                     param_grid = None, n_splits = 5, n_iter = 30,
                     random_state = 420, n_jobs = -1):

    set_config(transform_output = "pandas")
    mean_fpr = np.linspace(0, 1, 100)  # Define default mean_fpr here

    X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    if pipeline is None:
        params = {'objective': 'binary:logistic', 'eval_metric': 'auc',
                  'random_state': random_state, 'n_jobs': -1}

        ALL_DEMOG = {"Sex", "Age"}
        peptide_cols = [c for c in X_train.columns if c not in ALL_DEMOG]
        demog_cols = [c for c in X_train.columns if c in ALL_DEMOG]

        # Build the pipeline for this fold.
        pipeline = make_pipeline(peptide_cols=peptide_cols, demog_cols=demog_cols, estimator=XGBClassifier(**params))

    # Perform hyperparameter tuning if param_grid is provided.
    if param_grid is not None:
        # Use your search_best_model function or similar with BayesSearchCV
        best_estimator = search_best_model(pipeline, param_grid, X_train_fold, y_train_fold,
                                           n_splits=n_splits, n_iter=n_iter, random_state=random_state, n_jobs=n_jobs)
    else:
        best_estimator = pipeline
        best_estimator.fit(X_train_fold, y_train_fold)

    # Predict risk scores on the validation fold
    scores_fold = best_estimator.predict_proba(X_valid_fold)[:, 1]
    interp_tpr, auc_value = compute_interp_tpr_auc(y_valid_fold, scores_fold, mean_fpr)

    # Transform validation data for SHAP computation
    if len(best_estimator) > 1:
        try:
            X_valid_fold = best_estimator[:-1].transform(X_valid_fold)
        except Exception as e:
            logger.error(f"Error transforming validation data in fold: {e}")
            return None

    # Compute SHAP values using the regressor (last step)
    explainer = shap.TreeExplainer(best_estimator['classifier'])
    shap_values_fold = explainer.shap_values(X_valid_fold)
    shap_values_fold_df = pd.DataFrame(shap_values_fold, index=X_valid_fold.index, columns=X_valid_fold.columns)

    return valid_idx, scores_fold, shap_values_fold_df, best_estimator, interp_tpr, auc_value


def nested_cv(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        pipeline: Optional[Pipeline] = None,
        param_grid: Optional[Dict] = None,
        n_splits: int = 10,
        n_splits_inner: int = 5,
        n_iter: int = 30,
        random_state: int = 420,
        n_jobs: int = 1,
        n_jobs_inner: int = -1
) -> tuple[list[Any], DataFrame, Series, list[Any], dict[str, floating[Any] | Any] | dict[str, Any]]:
    """
    Perform nested cross-validation to tune hyperparameters and feature selection,
    and aggregate SHAP values and risk scores for each outer fold.

    For each outer fold:
      - Split data into training and validation sets.
      - Run hyperparameter tuning (BayesSearchCV) on the inner folds (if param_grid is provided)
        to select the best estimator.
      - Use the best estimator to predict risk scores on the validation set.
      - Compute SHAP values on the validation set. Note: because the pipeline's
        feature selection step may select a different subset of features per fold,
        the returned SHAP values DataFrame may have different columns.
      - Store the risk scores and SHAP values (with sample indices).

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train : pd.Series
        Classes for training samples.
    pipeline : dict, optional
        Default pipeline for hyperparameter tuning. If None, default parameters are used.
    param_grid : dict, optional
        Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
    n_splits : int, default 10
        Number of outer CV splits.
    n_splits_inner : int, default 5
        Number of inner CV splits.
    n_iter : int, default 30
        Number of iterations for Bayesian optimization.
    random_state : int, default 420
        Seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs to run in the outer CV (default is all cores -1).
    n_jobs_inner : int, default 1
        Number of parallel jobs to run in the inner CV (default is 1).
    Returns
    -------
    model_list : List[Pipeline]
        List of best estimators (one per fold).
    shap_values : pd.DataFrame
        DataFrame concatenating SHAP values from all folds, indexed by sample.
    scores : pd.Series
        Series of predicted scores from all folds, indexed by sample.
    validation_indices : List[np.ndarray]
        validation indices for each outer fold.
    roc_metrics : List[Dict]
        List of dictionaries containing ROC metrics for each outer fold.
    """
    #outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Run the outer folds in parallel.
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(nested_cv_single)(
            train_idx, valid_idx, X_train, y_train,
            pipeline = pipeline, param_grid = param_grid,
            n_splits = n_splits_inner, n_iter = n_iter,
            random_state = random_state, n_jobs = n_jobs_inner) for train_idx, valid_idx in outer_cv.split(X_train, y_train))
    # Filter out any folds that returned None (e.g., due to transformation errors).
    fold_results = [result for result in fold_results if result is not None]

    # Initialize master containers.
    validation_indices = []
    model_list = []
    auc_list, tpr_list = [], []
    mean_fpr = np.linspace(0, 1, 100)  # Define default mean_fpr here
    scores = pd.Series(index=X_train.index, name='score')
    shap_values = pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns)

    # Aggregate results from each fold.
    i=0
    #    return valid_idx, scores_fold, shap_values_fold_df, best_estimator, interp_tpr, auc_value
    for valid_idx, fold_scores, fold_shap_df, model,  interp_tpr, auc_value in fold_results:
        model_list.append(model)
        # Assign scores for the validation fold.
        scores.iloc[valid_idx] = fold_scores
        # Update master SHAP values. Since folds are disjoint, direct assignment works.
        shap_values.loc[fold_shap_df.index, fold_shap_df.columns] = fold_shap_df

        y_valid_fold = y_train.iloc[valid_idx]

        validation_indices.append(valid_idx)
        auc_list.append(auc_value)
        tpr_list.append(interp_tpr)
        i=i+1

    roc_metrics = bootstrap_auc(mean_fpr=mean_fpr, y_true=y_train, y_pred=scores.values, n_bootstraps = 200, random_state=420)
    kfold_metrics = calculate_mean_std_ci_tpr_auc(auc_list, tpr_list, mean_fpr, bootstrap=False)
    roc_metrics.update(kfold_metrics)

    logger.info(f"Mean AUC across folds: {np.mean(auc_list):.4f}")

    return model_list, shap_values, scores, validation_indices, roc_metrics



###########
def align_features(X_train, X_ext):
    """Align features between training and external test sets."""
    common_features = list(set(X_train.columns).intersection(X_ext.columns))
    if common_features:
        X_train = X_train[common_features]
        X_ext = X_ext[common_features]
        return X_train, X_ext
    else:
        raise ValueError(f"Training and External set shared no common features")

def _compute_roc_metrics_test(estimator, X_test, y_test, predicted_probs_test):
    """Compute ROC metrics for the test set."""
    fpr, tpr, _ = roc_curve(y_test, predicted_probs_test)
    roc_metrics = bootstrap_auc(estimator=estimator, X=X_test, y_true=y_test)
    roc_metrics.update({"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr), "auc_std": None})

    return roc_metrics

def train_and_validate_model(X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             pipeline: Optional[Pipeline] = None,
                             param_grid: Optional[Dict] = None,
                             n_splits: int = 10,
                             n_iter: int = 30,
                             random_state: int = 420,
                             n_jobs: int = -1,
                             get_only_model: bool = False)-> None | Pipeline | tuple[Pipeline | Any, DataFrame, Series, Dict]:

    """
    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train : pd.Series
        Target variable for training.
    X_test : pd.DataFrame
        Feature matrix for testing.
    y_test : pd.Series
        Target variable for testing.
    pipeline : dict, optional
        Default pipeline for hyperparameter tuning. If None, default parameters are used.
    param_grid : dict, optional
        Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
    n_splits : int, default 10
        Number of outer CV splits.
    n_iter : int, default 30
        Number of iterations for Bayesian optimization.
    random_state : int, default 420
        Seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs to run in the outer CV (default is all cores -1).
    get_only_model : bool, default False
        return only the fitted model
    Returns
    -------
    best_estimator : Pipeline
        best estimator
    shap_values_df : pd.DataFrame
        DataFrame SHAP values, indexed by sample.
    scores : pd.Series
        Series of predicted probabilities per class, indexed by sample.
    roc_metrics_test : Dict
        Dict containing ROC metrics for testing set.
    """

    set_config(transform_output = "pandas")
    X_train, X_test = align_features(X_train, X_test)
    shap_values_df = pd.DataFrame(0.0, index=X_test.index, columns=X_test.columns)

    if pipeline is None:
        params = {'objective': 'binary:logistic', 'eval_metric': 'auc',
                  'random_state': random_state, 'n_jobs': -1}

        ALL_DEMOG = {"Sex", "Age"}
        peptide_cols = [c for c in X_train.columns if c not in ALL_DEMOG]
        demog_cols = [c for c in X_train.columns if c in ALL_DEMOG]

        # Build the pipeline
        pipeline = make_pipeline(peptide_cols=peptide_cols, demog_cols=demog_cols, estimator=XGBClassifier(**params))

        # Perform hyperparameter tuning if param_grid is provided.
    if param_grid is not None:
        # Use your search_best_model function or similar with BayesSearchCV
        best_estimator = search_best_model(pipeline, param_grid, X_train, y_train,
                                           n_splits=n_splits, n_iter=n_iter, random_state=random_state, n_jobs=n_jobs)
    else:
        best_estimator = pipeline
        best_estimator.fit(X_train, y_train)

    if get_only_model:
        return best_estimator

        # Predict risk scores on the validation fold
    scores = best_estimator.predict_proba(X_test)[:, 1]
    scores = pd.Series(scores, index=X_test.index, name='Score')


    # Transform validation data for SHAP computation
    if len(best_estimator) > 1:
        try:
            X_test = best_estimator[:-1].transform(X_test)
        except Exception as e:
            logger.error(f"Error transforming validation data: {e}")
            return None

    roc_metrics_test = _compute_roc_metrics_test(best_estimator['classifier'], X_test, y_test, scores)


    # Compute SHAP values using the regressor (last step)
    explainer = shap.TreeExplainer(best_estimator['classifier'])
    shap_values = explainer.shap_values(X_test)
    shap_values = pd.DataFrame(shap_values, index=X_test.index, columns=X_test.columns)
    shap_values_df.loc[shap_values.index, shap_values.columns] = shap_values

    logger.info(f"AUC in testing set: {roc_metrics_test.get('auc'):.4f}")

    # # Create predictions DataFrame
    # predictions_test = _create_binary_class_predictions_df("XGBClassifier",
    #                                                             roc_metrics_test.get("auc"),
    #                                                             roc_metrics_test.get("auc_std"))

    return best_estimator, shap_values_df, scores, roc_metrics_test

if __name__ == '__main__':
    # Parse the command-line argument for the random seed
    parser = argparse.ArgumentParser(description="Run nested CV with custom random seed.")
    parser.add_argument("seed", type=int, nargs="?", default=420,
                        help="Random seed (default: 420)")
    args = parser.parse_args()
    random_seed = args.seed

    start_time = time.time()
    #config_file = "/home/creyna/Vogl-lab_Projects_git/HCC/Metadata/config_control_hcc.yaml"
    config_file = "/gpfs/data/fs71974/creynablanco/MLpackage/config_control_hcc.yaml"
    config = Config(config_file)
    metadata_handler = MetadataHandler(config)
    #oligos_handler = OligosHandler(config)
    oligos_handler = OligosHandler(config) #, data_type=config.data_types[0])
    feature_manager = FeatureManager(config, metadata_handler, oligos_handler,
                                 subgroup = 'all',
                                 with_oligos=True,
                                 with_additional_features = True,
                                 filter_by_entropy=False,
                                 prevalence_threshold_min=0,
                                 prevalence_threshold_max=100)
    X_train, y_train = feature_manager.get_features_target()


    params = {'objective': 'binary:logistic', 'eval_metric': 'auc',
              'random_state': 420, 'n_jobs': -1}

    ALL_DEMOG = {"Sex", "Age"}
    peptide_cols = [c for c in X_train.columns if c not in ALL_DEMOG]
    demog_cols = [c for c in X_train.columns if c in ALL_DEMOG]
    pipeline = make_pipeline(peptide_cols=peptide_cols, demog_cols=demog_cols, estimator=XGBClassifier(**params))
    param_grid = {
        'preprocessor__peptides__feature_selection__percentile': Integer(2, 50),
        'estimator__learning_rate': Real(0.01, 0.3, prior='uniform'),
        'estimator__max_depth': Integer(3, 10),
        'estimator__subsample': Real(0.6, 1.0, prior='uniform'),
        'estimator__colsample_bytree': Real(0.6, 1.0, prior='uniform'),
        'estimator__reg_lambda': Real(0.1, 10.0, prior='log-uniform'),
        'estimator__n_estimators': Integer(50, 500)
    }


    model_list, train_shap_values, scores_train, validation_indices, roc_metrics_train = nested_cv(X_train,
                                                                                                   y_train,
                                                                                                   pipeline=pipeline,
                                                                                                   param_grid=param_grid,
                                                                                                   n_splits=10,
                                                                                                   n_splits_inner=5,
                                                                                                   n_iter=10,
                                                                                                   random_state=random_seed,
                                                                                                   n_jobs=1,
                                                                                                   n_jobs_inner=-1)

    end_time = time.time()
    logger.info(f"Script runtime: {end_time - start_time:.2f} seconds")

    # Save the results as a dictionary
    # Save the results as a dictionary
    results = {
        'model_list': model_list,
        'train_shap_values': train_shap_values,
        'scores_train': scores_train,
        'validation_indices_train': validation_indices,
        'roc_metrics_train': roc_metrics_train
    }

    joblib.dump(results, f'classif_nestedCV_bestmodel_withSex_Age_{random_seed}.joblib')
