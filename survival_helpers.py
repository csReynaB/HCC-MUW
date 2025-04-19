import logging
from typing import Any, Dict, Optional, Union, List, Tuple

from pathlib import Path

import numpy as np
import pandas as pd
import re

# Plots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ML survival models and utils
import xgboost as xgb
import shap

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw, concordance_index_censored

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn import set_config

from skopt import BayesSearchCV

from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts, rmst_plot, qq_plot
from lifelines.utils import median_survival_times

# Parallelize
from joblib import Parallel, delayed

# Set up logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.family'] = 'Arial'

def c_index_scorer(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Custom scorer to calculate Concordance Index (C-index) for single-column `y_true`.
    Negative values in `y_true` indicate censored data; positive values indicate events.

    Parameters:
    - y_true: 1D array of survival times (negative for censored, positive for events).
    - y_pred: Predicted risk scores (higher scores indicate higher risk).

    Returns:
    - C-index: Concordance Index for the predictions.
    """
    # Separate survival times and event indicators
    events = np.where(y_true < 0, 0, 1)  # 0 for censored, 1 for event
    times = np.abs(y_true)  #survival_time=np.abs(y_true) Absolute values for survival times

    y_surv = Surv.from_arrays(event=events, time=times)

    # Calculate and return the C-index
    return concordance_index_censored(y_surv['event'], y_surv['time'], y_pred)[0]


def c_index_scorer_ipcw(y_train: pd.Series, y_val: pd.Series, y_pred: np.ndarray, tau: int = None) -> float:
    """
    Custom scorer to calculate Concordance Index (C-index).

    Parameters:
    - y_train: 1D array of survival times from training set (negative for censored, positive for events).
    - y_val: 1D array of survival times from validation set (negative for censored, positive for events).
    - y_pred: Predicted risk scores (higher scores indicate higher risk).
    - t: Tau for C-index calculation. Default is None.

    Returns:
    - C-index: Concordance Index for the predictions.
    """

    # Separate survival times and event indicators
    train_events = np.where(y_train < 0, 0, 1)  # 0 for censored, 1 for deceased
    train_times = np.abs(y_train)  #survival_time=np.abs(y_true) Absolute values for survival times
    y_train_surv = Surv.from_arrays(event=train_events, time=train_times)

    val_events = np.where(y_val < 0, 0, 1)  # 0 for censored, 1 for deceased
    val_times = np.abs(y_val)  #survival_time=np.abs(y_true) Absolute values for survival times

    valid_indices = val_times <= train_times.max()
    val_events = val_events[valid_indices]
    val_times = val_times[valid_indices]

    y_val_surv = Surv.from_arrays(event=val_events, time=val_times)
    y_pred = y_pred[valid_indices]

    if tau is None:
        c_index = concordance_index_ipcw(y_train_surv, y_val_surv, y_pred)[0]
    else:
        c_index = concordance_index_ipcw(y_train_surv, y_val_surv, y_pred,
                                      tau=tau)[0]
    # Calculate and return the C-index
    return c_index



def univariate_cox_score_single(j: int, X: np.ndarray, y_surv) -> float:
    """
    Computes the concordance score for a single feature (column j) using a univariate Cox model.

    Parameters
    ----------
    j : int
        Index of the feature to evaluate.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_surv : structured array
        Survival data as a structured array (e.g., from Surv.from_arrays).

    Returns
    -------
    float
        The score (e.g., concordance index) for feature j. Returns 0.0 if an error occurs.
    """
    model = CoxPHSurvivalAnalysis()
    try:
        # Extract the j-th feature as a 2D array
        Xj = X[:, j:j+1]
        model.fit(Xj, y_surv)
        score = model.score(Xj, y_surv)
        return score
    except Exception as e:
        #logger.warning(f"Feature index {j} failed with error: {e}")
        return 0.0

def univariate_cox_score(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """
    Computes univariate Cox scores for each feature in X in parallel.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        1D array of survival times; negative values indicate censored observations.
    n_jobs : int, optional
        Number of parallel jobs to run (default is -1 to use all available cores).

    Returns
    -------
    np.ndarray
        Array of scores for each feature (shape: (n_features,)).
    """
    # Convert y into a structured survival array: event==1 indicates event occurred,
    # and negative y-values indicate censored observations.
    events = np.where(y < 0, 0, 1)
    times = np.abs(y)
    y_surv = Surv.from_arrays(event=events, time=times)

    n_features = X.shape[1]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(univariate_cox_score_single)(j, X, y_surv) for j in range(n_features)
    )
    return np.array(scores)



def search_best_survival_model(
    estimator: Any,
    param_grid: Dict,
    X_train,
    y_train,
    method: str = "bayesian",  # "random", "grid", or "bayesian"
    n_splits: int = 5,
    n_iter: int = 30,
    random_state: int = 420,
    n_jobs: int = -1,
    **kwargs
) -> Any:
    """
    Tune hyperparameters for a model using one of three methods:
      - RandomizedSearchCV ("random")
      - GridSearchCV ("grid")
      - BayesSearchCV ("bayesian", if scikit-optimize is installed)

    Parameters
    ----------
    estimator : Any
        A scikit-learn style survival estimator.
    param_grid : Dict
        - For 'grid', a dict of parameter lists, e.g. {'param': [1, 2, 3]}.
        - For 'random', a dict of parameter distributions or lists.
        - For 'bayesian', a dict of parameter search spaces (from skopt.space).
    X_train : array-like or DataFrame
        Training feature data.
    y_train : array-like or structured array
        Training survival target (time + event).
    method : str, default="random"
        Which search method to use: "random", "grid", or "bayesian".
    n_splits : int, default=5
        Number of folds for StratifiedKFoldSurv cross-validation.
    n_iter : int, default=30
        - For 'random', number of draws from param distributions.
        - For 'bayesian', number of parameter settings to sample.
        - Ignored for 'grid'.
    random_state : int, default=420
        Seed for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. 
    **kwargs :
        Additional keyword arguments passed to the underlying search class.

    Returns
    -------
    best_estimator_ : Any
        The best-fitted estimator from the search.
    """
    # Create a custom scorer based on the c-index
    custom_scorer = make_scorer(c_index_scorer, greater_is_better=True)

    # Define the CV splitter
    cv = StratifiedKFoldSurv(n_splits=n_splits, random_state=random_state)

    method = method.lower()
    if method == "random":
        # RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=custom_scorer,
            cv=cv,
            refit=True,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
    elif method == "grid":
        # GridSearchCV
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=custom_scorer,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            **kwargs
        )
    elif method == "bayesian":
        # BayesSearchCV from scikit-optimize
        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_grid,
            n_iter=n_iter,
            scoring=custom_scorer,
            cv=cv,
            refit=True,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
    else:
        raise ValueError("method must be 'random', 'grid', or 'bayesian'.")

    # Run the search
    search.fit(X_train, y_train)

    # Return the best model
    return search.best_estimator_

def split_fold(X: pd.DataFrame, y: pd.Series, train_idx, valid_idx) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the input DataFrame and Series into training and validation folds based on provided indices.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target variable.
    train_idx : array-like
        Indices for the training fold.
    valid_idx : array-like
        Indices for the validation fold.

    Returns
    -------
    X_train_fold : pd.DataFrame
        Training fold features.
    X_valid_fold : pd.DataFrame
        Validation fold features.
    y_train_fold : pd.Series
        Training fold target.
    y_valid_fold : pd.Series
        Validation fold target.
    """
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    return X_train_fold, X_valid_fold, y_train_fold, y_valid_fold

def calculate_cumulative_dynamic_auc(y_train, y_valid, scores, time_points):

    valid_indices = y_valid.abs() <= y_train.abs().max()
    y_valid = y_valid[valid_indices]

    y_train_surv = Surv.from_arrays(
        event=np.where(y_train < 0, 0, 1),
        time=np.abs(y_train)
    )
    y_valid_surv = Surv.from_arrays(
        event=np.where(y_valid < 0, 0, 1),
        time=np.abs(y_valid)
    )
    # Get the observed time range from the validation fold
    min_time_point  = y_valid.abs().min()
    max_time_point  = y_valid.abs().max() - 0.001
    # Filter desired time points to only those within the observed range
    time_points_highlight = time_points[(time_points >= min_time_point) & (time_points <= max_time_point)]

    # Compute time-dependent AUC for this fold:
    auc_values, mean_auc_values = cumulative_dynamic_auc(
        y_train_surv,
        y_valid_surv,
        scores[valid_indices],
        time_points_highlight  # generate or use time points appropriate for this fold
    )

    return pd.Series(auc_values, index=time_points_highlight), mean_auc_values

def nested_cv_single(train_idx, valid_idx, X_train, y_time_train, pipeline=None,
                     param_grid=None, n_splits=5, n_iter=30,
                     random_state=420, n_jobs=1):
    set_config(transform_output="pandas")

    X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = split_fold(X_train, y_time_train, train_idx, valid_idx)

    if pipeline is None:
        params = {'objective': 'survival:cox', 'eval_metric': 'cox-nloglik',
                  'tree_method': 'hist', 'random_state': random_state, 'n_jobs': -1}
        # Build the pipeline for this fold.
        pipeline = Pipeline([
            ('variance_removal', VarianceThreshold(threshold=0.0)),
            ('feature_selection',
             SelectFromModel(CoxnetWrapper(l1_ratio=0.6, alpha_min_ratio=0.0001, n_alphas=100), threshold=1e-5)),
            ('regressor', xgb.XGBRegressor(**params))
        ])

    # Perform hyperparameter tuning if param_grid is provided.
    if param_grid is not None:
        # Use your search_best_model function or similar with BayesSearchCV
        best_estimator = search_best_survival_model(pipeline, param_grid, X_train_fold, y_train_fold, method="bayesian",
                                           n_splits=n_splits, n_iter=n_iter, random_state=random_state, n_jobs=n_jobs)
    else:
        best_estimator = pipeline
        best_estimator.fit(X_train_fold, y_train_fold)

    # Predict risk scores on the validation fold
    risk_scores_fold = best_estimator.predict(X_valid_fold)

    # Transform validation data for SHAP computation
    if len(best_estimator) > 1:
        try:
            X_valid_fold = best_estimator[:-1].transform(X_valid_fold)
        except Exception as e:
            logger.error(f"Error transforming validation data in fold: {e}")
            return None

    # Compute SHAP values using the regressor (last step)
    explainer = shap.TreeExplainer(best_estimator['regressor'])
    shap_values_fold = explainer.shap_values(X_valid_fold)
    shap_values_fold_df = pd.DataFrame(shap_values_fold, index=X_valid_fold.index, columns=X_valid_fold.columns)
    # Reindex fold SHAP values to the master feature set, filling missing values with 0
    # fold_shap_df_aligned = shap_values_fold_df.reindex(columns=shap_values.columns, fill_value=0)
    # If a sample appears in only one fold, just assign. If overlapping folds exist, you could sum and later average.
    # shap_values.loc[X_valid_fold.index] += fold_shap_df_aligned

    # Compute performance score (e.g., c-index)
    fold_cindex = c_index_scorer(y_valid_fold, risk_scores_fold)
    # fold_cindex = c_index_scorer_ipcw(y_train_fold, y_valid_fold, risk_scores_fold)

    return valid_idx, risk_scores_fold, shap_values_fold_df, fold_cindex, best_estimator


def nested_cv(
        X_train: pd.DataFrame,
        y_time_train: pd.Series,
        pipeline: Optional[Pipeline] = None,
        param_grid: Optional[Dict] = None,
        n_splits: int = 10,
        n_splits_inner: int = 5,
        n_iter: int = 30,
        max_time_point=None,
        random_state: int = 420,
        n_jobs: int = -1,
        n_jobs_inner: int = 1
) -> Tuple[List[Pipeline], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[float], List[np.ndarray]]:
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
      - Compute time-dependent AUC scores for the validation set.
      - Store the risk scores and SHAP values (with sample indices).

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_time_train : pd.Series
        Survival times for training samples.
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
    max_time_point : int, default None
        Max time point for time-dependent AUC estimation
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
    risk_scores : pd.Series
        Series of risk scores from all folds, indexed by sample.
    time_dependent_auc : pd.DataFrame
        DataFrame with time-dependent AUC for each fold.
    time_dependent_auc_mean : pd.Series
        Series with mean AUC values for each fold
    c_index : List[float]
        List of performance scores (e.g., c-index) for each outer fold.
    validation_indices : List[array-like]
        indices of the validation set for each outer fold.
    """
    # outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFoldSurv(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Run the outer folds in parallel.
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(nested_cv_single)(
            train_idx, valid_idx, X_train, y_time_train,
            pipeline=pipeline, param_grid=param_grid,
            n_splits=n_splits_inner, n_iter=n_iter,
            random_state=random_state, n_jobs=n_jobs_inner) for train_idx, valid_idx in
        outer_cv.split(X_train, y_time_train))
    # Filter out any folds that returned None (e.g., due to transformation errors).
    fold_results = [result for result in fold_results if result is not None]

    # Initialize master containers.
    validation_indices = []
    c_index = []
    model_list = []
    risk_scores = pd.Series(index=X_train.index, name='Risk score')
    shap_values = pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns)
    max_time_point = y_time_train.abs().max() if max_time_point is None else max_time_point
    time_points =  np.arange(1, max_time_point, step=1)
    time_dependent_auc = pd.DataFrame(index=np.arange(n_splits), columns=time_points)
    time_dependent_auc_mean = pd.Series(index=np.arange(n_splits))
    # Aggregate results from each fold.
    i=0
    for valid_idx, fold_risk, fold_shap_df, fold_cindex, model in fold_results:
        model_list.append(model)
        # Assign risk scores for the validation fold.
        risk_scores.iloc[valid_idx] = fold_risk
        # Update master SHAP values. Since folds are disjoint, direct assignment works.
        shap_values.loc[fold_shap_df.index, fold_shap_df.columns] = fold_shap_df
        # Reindex fold SHAP values to the master feature set, filling missing values with 0
        # fold_shap_df_aligned = fold_shap_df.reindex(columns=shap_values.columns, fill_value=0)
        # If a sample appears in only one fold, just assign. If overlapping folds exist, you could sum and later average.
        # shap_values.loc[fold_shap_df.index] += fold_shap_df_aligned
        y_valid_fold = y_time_train.iloc[valid_idx]
        auc_values_fold, mean_auc_values_fold = calculate_cumulative_dynamic_auc(y_time_train.loc[y_time_train.index.difference(y_valid_fold.index)] , y_valid_fold,
                                                           fold_risk, time_points)
        time_dependent_auc.loc[i, auc_values_fold.index] = auc_values_fold.values
        time_dependent_auc_mean.loc[i] = mean_auc_values_fold
        del fold_shap_df
        c_index.append(fold_cindex)
        validation_indices.append(valid_idx)

    logger.info(f"Mean Concordance Index (C-index) across folds: {np.mean(c_index):.4f}")
    logger.info(f"Mean Time-Dependent AUC across folds: {time_dependent_auc.mean(skipna=True).mean():.4f}")

    return model_list, shap_values, risk_scores, time_dependent_auc, time_dependent_auc_mean, c_index, validation_indices

def nested_cv_allfolds(
        X_train: pd.DataFrame,
        y_time_train: pd.Series,
        y_event_train: pd.Series,
        pipeline: Optional[Pipeline] = None,
        param_grid: Optional[Dict] = None,
        n_splits: int = 10,
        n_splits_inner: int = 5,
        n_iter: int = 30,
        max_time_point=None,
        random_state: int = 420,
        n_jobs: int = -1
) -> Tuple[ List[Pipeline], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[float], List[np.ndarray]]:
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
      - Compute time-dependent AUC scores for the validation set.
      - Store the risk scores and SHAP values (with sample indices).

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_time_train : pd.Series
        Survival times for training samples.
    y_event_train : pd.Series
        Event status (1 if event occurred, 0 if censored) for training samples.
    pipeline : dict, optional
        Default pipeline for hyperparameter tuning. If None, default parameters are used.
    param_grid : dict, optional
        Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
    n_splits : int, default 10
        Number of outer CV splits.
    n_iter : int, default 30
        Number of iterations for Bayesian optimization.
    max_time_point : int, default None
        Max time point for time-dependent AUC estimation
    random_state : int, default 420
        Seed for reproducibility.
    n_jobs: int, default 1
        Number of jobs in parallel in the inner CV

    Returns
    -------
    model_list : List[Pipeline]
        List of best estimators (one per fold).
    shap_values : pd.DataFrame
        DataFrame concatenating SHAP values from all folds, indexed by sample.
    risk_scores : pd.Series
        Series of risk scores from all folds, indexed by sample.
    time_dependent_auc : pd.DataFrame
        DataFrame with time-dependent AUC for each fold.
    time_dependent_auc_mean : pd.Series
        Series with AUC mean
    c_index : List[float]
        List of performance scores (e.g., c-index) for each outer fold.
    validation_indices : List[np.ndarray]
        validation indices for each outer fold.
    """
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    c_index = []
    model_list = []
    validation_indices = []
    risk_scores = pd.Series(index=X_train.index, name='Risk score')
    shap_values = pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns)
    max_time_point = y_time_train.abs().max() if max_time_point is None else max_time_point
    time_points =  np.arange(1, max_time_point, step=1)
    time_dependent_auc = pd.DataFrame(index=np.arange(n_splits), columns=time_points)
    time_dependent_auc_mean = pd.Series(index=np.arange(n_splits))
    i=0
    for train_idx, valid_idx in outer_cv.split(X_train, y_event_train):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_time_train.iloc[train_idx], y_time_train.iloc[valid_idx]

        if pipeline is None:
            params = {'objective': 'survival:cox', 'eval_metric': 'cox-nloglik',
                      'tree_method': 'hist', 'random_state': random_state, 'n_jobs': -1}
            # Build the pipeline for this fold.
            pipeline = Pipeline([
                ('variance_removal', VarianceThreshold(threshold=0.0)),
                ('feature_selection',
                 SelectFromModel(CoxnetWrapper(l1_ratio=0.6, alpha_min_ratio=0.0001, n_alphas=100), threshold=1e-5)),
                ('regressor', xgb.XGBRegressor(**params))
            ])

            # Perform hyperparameter tuning if param_grid is provided.
        if param_grid is not None:
            # Use your search_best_model function or similar with BayesSearchCV
            best_estimator = search_best_survival_model(pipeline, param_grid, X_train_fold, y_train_fold, method="bayesian",
                                               n_splits=n_splits, n_iter=n_iter, random_state=random_state,
                                               n_jobs=n_jobs)
        else:
            best_estimator = pipeline
            best_estimator.fit(X_train_fold, y_train_fold)


        model_list.append(best_estimator)

        # Predict risk scores on the validation fold
        risk_scores_fold = best_estimator.predict(X_valid_fold)
        risk_scores.iloc[valid_idx] = risk_scores_fold

        # Transform validation data for SHAP computation
        if len(best_estimator) > 1:
            try:
                X_valid_fold = best_estimator[:-1].transform(X_valid_fold)
            except Exception as e:
                logger.error(f"Error transforming validation data in fold: {e}")
                continue

        # Compute SHAP values using the regressor (last step)
        explainer = shap.TreeExplainer(best_estimator['regressor'])
        shap_values_fold = explainer.shap_values(X_valid_fold)
        shap_values_fold_df = pd.DataFrame(shap_values_fold, index=X_valid_fold.index, columns=X_valid_fold.columns)
        shap_values.loc[shap_values_fold_df.index, shap_values_fold_df.columns] = shap_values_fold_df

        # Reindex fold SHAP values to the master feature set, filling missing values with 0
        #fold_shap_df_aligned = shap_values_fold_df.reindex(columns=shap_values.columns, fill_value=0)
        # If a sample appears in only one fold, just assign. If overlapping folds exist, you could sum and later average.
        #shap_values.loc[X_valid_fold.index] += fold_shap_df_aligned

        # Compute performance score (e.g., c-index) for this fold using risk scores
        #fold_cindex = c_index_scorer(y_valid_fold, risk_scores)
        fold_cindex = c_index_scorer_ipcw(y_train_fold, y_valid_fold, risk_scores_fold)
        c_index.append(fold_cindex)

        auc_values_fold, mean_auc_values_fold = calculate_cumulative_dynamic_auc(y_train_fold, y_valid_fold,
            risk_scores_fold, time_points)
        time_dependent_auc.loc[i, auc_values_fold.index] = auc_values_fold.values
        time_dependent_auc_mean.loc[i] = mean_auc_values_fold

        validation_indices.append(valid_idx)  # Track SampleNames corresponding to the validation set
        i=i+1
    logger.info(f"Mean Concordance Index (C-index) across validation folds: {np.mean(c_index):.4f}")
    logger.info(f"Mean Time-Dependent AUC across folds: {time_dependent_auc.mean(skipna=True).mean():.4f}")

    return model_list, shap_values, risk_scores, time_dependent_auc, time_dependent_auc_mean, c_index, validation_indices


def train_and_evaluate_model(X_train: pd.DataFrame, y_time_train: pd.Series,
                             X_test: pd.DataFrame, y_time_test: pd.Series,
                             pipeline: Optional[Pipeline] = None,
                             param_grid: Optional[Dict] = None,
                             n_splits: int = 10,
                             n_iter: int = 30,
                             max_time_point = None,
                             random_state: int = 420,
                             n_jobs: int = 1,
                             get_only_model: bool = False) -> (
        Tuple)[Optional[Pipeline], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[float], Optional[float]]:

    """
    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_time_train : pd.Series
        Survival times for training samples (negative:censored, positive:event).
    X_test : pd.DataFrame
        Feature matrix for testing.
    y_time_test : pd.Series
        Survival times for testing samples (negative:censored, positive:event).
    pipeline : dict, optional
        Default pipeline for hyperparameter tuning. If None, default parameters are used.
    param_grid : dict, optional
        Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
    n_splits : int, default 10
        Number of outer CV splits.
    n_iter : int, default 30
        Number of iterations for Bayesian optimization.
    max_time_point : int, default None
        Max time point for time-dependent AUC estimation
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
    shap_values : pd.DataFrame
        DataFrame SHAP values, indexed by sample.
    risk_scores : pd.Series
        Series of risk scores, indexed by sample.
    time_dependent_auc : pd.Series
        DataFrame with time-dependent AUC, indexed by timepoint.
    time_dependent_auc_mean : float
        Time-dependent AUC mean based on given timepoints
    c_index : float
        Performance score (e.g., c-index).
    """

    set_config(transform_output = "pandas")
    X_train, X_test = align_features(X_train, X_test)

    if pipeline is None:
        params = {'objective': 'survival:cox', 'eval_metric': 'cox-nloglik',
                  'tree_method': 'hist', 'random_state': random_state, 'n_jobs': -1}
        # Build the pipeline for this fold.
        pipeline = Pipeline([
            ('variance_removal', VarianceThreshold(threshold=0.0)),
            ('feature_selection',
             SelectFromModel(CoxnetWrapper(l1_ratio=0.6, alpha_min_ratio=0.0001, n_alphas=100), threshold=1e-5)),
            ('regressor', xgb.XGBRegressor(**params))
        ])

    # Perform hyperparameter tuning if param_grid is provided.
    if param_grid is not None:
        # Use your search_best_model function or similar with BayesSearchCV
        best_estimator = search_best_survival_model(pipeline, param_grid, X_train, y_time_train, method="bayesian",
                                                    n_splits=n_splits, n_iter=n_iter, random_state=random_state,
                                                    n_jobs=n_jobs)
    else:
        best_estimator = pipeline
        best_estimator.fit(X_train, y_time_train)

    if get_only_model:
        return best_estimator

    # Predict risk scores on the validation fold
    risk_scores = best_estimator.predict(X_test)
    risk_scores = pd.Series(risk_scores, index=X_test.index, name='Risk score')

    # Transform validation data for SHAP computation
    if len(best_estimator) > 1:
        try:
            X_test = best_estimator[:-1].transform(X_test)
        except Exception as e:
            logger.error(f"Error transforming testing data: {e}")
            return None, None, None, None, None, None

    # Compute SHAP values using the regressor (last step)
    explainer = shap.TreeExplainer(best_estimator['regressor'])
    shap_values = explainer.shap_values(X_test)
    shap_values_df = pd.DataFrame(shap_values, index=X_test.index, columns=X_test.columns)
    # Reindex fold SHAP values to the master feature set, filling missing values with 0
    #shap_df_aligned = shap_values_df.reindex(columns=shap_values.columns, fill_value=0)
    # If a sample appears in only iteration just assign. If overlapping exist, you could sum and later average.
    #shap_values.loc[X_test.index] += shap_df_aligned

    # Compute performance score (e.g., c-index)
    #c_index = c_index_scorer(y_test, risk_scores.values)
    c_index = c_index_scorer_ipcw(y_time_train, y_time_test, risk_scores.values)

    max_time_point = y_time_train.abs().max() if max_time_point is None else max_time_point
    time_points =  np.arange(1, max_time_point, step=1)
    time_dependent_auc, time_dependent_auc_mean = calculate_cumulative_dynamic_auc(y_time_train, y_time_test, risk_scores, time_points)

    return best_estimator, shap_values_df, risk_scores, time_dependent_auc, time_dependent_auc_mean, c_index


def calculate_samples_at_risk(y_time, time_points) -> pd.DataFrame:
    """
    Calculate the number of individuals at risk, as well as the number of censored and event samples,
    at each time point.

    Parameters:
        y_time (array-like): Array or list of time-to-event data. Negative values indicate censored samples.
        time_points (iterable): Iterable of time points at which to compute the counts.

    Returns:
        pd.DataFrame: A DataFrame with columns 'At Risk', 'Censored', and 'Events' indexed by the time points.
    """
    at_risk = []
    samples_censored = []
    samples_events = []

    # At time 0, all individuals are considered at risk
    total_samples = len(y_time)

    for t in time_points:
        # Adjust the time point slightly
        t_adjusted = t + 0.01

        # Count censored samples: those with |y_time_train| <= t_adjusted and y_time_train < 0
        count_censored = np.sum((np.abs(y_time) <= t_adjusted) & (y_time < 0))

        # Count event samples: those with |y_time_train| <= t_adjusted and y_time_train >= 0
        count_events = np.sum((np.abs(y_time) <= t_adjusted) & (y_time >= 0))

        # Calculate the number at risk by subtracting events and censored samples from the total at risk
        count_at_risk = total_samples - count_events - count_censored

        at_risk.append(count_at_risk)
        samples_censored.append(count_censored)
        samples_events.append(count_events)

    # Create a DataFrame to store these counts, indexed by the original time points
    samples_at_risk = pd.DataFrame({
        'At Risk': at_risk,
        'Censored': samples_censored,
        'Events': samples_events
    }, index=time_points)

    return samples_at_risk

def calculate_time_dependent_auc(y_train, y_val, risk_scores, max_time_point = None,
                                  time_points_highlight=None, num_points=50, buffer=0.001) -> Tuple[pd.Series, float, pd.DataFrame]:
    """
     Calculate time-dependent AUC over a set of follow-up time points for the testing set and
     compute the number of samples at risk, censored, and events at each time point.

     Parameters
     ----------
     y_train : pd.Series
         Survival time for training data. (postive if event occurred, negative if censored).
     y_val : pd.Series
         Survival time for validation data. (postive if event occurred, negative if censored).
     risk_scores : pd.Series
         Predicted risk scores for the testing set.
     max_time_point : int default None
         Max time point for time-dependent AUC estimation.
     time_points_highlight : array-like, optional
     time_points_highlight : array-like, optional
         Specific time points to highlight on the AUC curve.
     num_points : int, default 50
         Number of evenly spaced time points for AUC calculation.
     buffer : float, default 0.001
         Small buffer added/subtracted to avoid boundary issues in time calculations.

     Returns
     -------
     auc_values : pd.Series
         Series of AUC values computed at each time point.
     mean_auc : float
         Mean AUC value across the time points.
     samples_at_risk : pd.DataFrame
         DataFrame with columns 'At Risk', 'Censored', and 'Events' for each time point.
     """
    # Set up time points within the range of the train and test follow-up times
    valid_indices = y_val.abs() <= y_train.abs().max()
    y_val = y_val[valid_indices]
    risk_scores = risk_scores[valid_indices]

    y_train_surv = Surv.from_arrays(
        event=np.where(y_train < 0, 0, 1),
        time=np.abs(y_train)
    )
    y_valid_surv = Surv.from_arrays(
        event=np.where(y_val < 0, 0, 1),
        time=np.abs(y_val)
    )

    min_time_point = max(y_val.abs().min(), y_train.abs().min())
    if max_time_point is None:
        max_time_point = y_val.abs().max() - buffer  # Ensuring max_time is strictly less than training and val max

    time_points = np.linspace(min_time_point, max_time_point, num=num_points)
    if time_points_highlight is None:
        time_points_highlight = np.concatenate(([1], np.arange(3, max_time_point+1, step=3))) #np.concatenate(([1], np.arange(6, max_time_point-1, step=6)))
    time_points = np.unique(np.concatenate((time_points, time_points_highlight)))

    # Calculate time-dependent AUC
    auc_values, mean_auc = cumulative_dynamic_auc(y_train_surv, y_valid_surv, risk_scores, time_points)
    auc_values = pd.Series(auc_values, index=time_points)

    # Calculate the number of individuals at risk at each time point
    samples_at_risk = calculate_samples_at_risk(y_val, time_points)

    return auc_values, mean_auc, samples_at_risk

def calculate_antigen_scores_scaled(
        shap_values: pd.DataFrame,
        y_time: pd.DataFrame,
        y_event: pd.DataFrame,
        top_peptides: list,
        scaler: Optional[MinMaxScaler] = None,
        scaler_antigens: Optional[MinMaxScaler] = None,
        threshold: Optional[float] = None,
        val_quantile: float = 50,
        return_all: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float, MinMaxScaler, MinMaxScaler]]:
    """
    Calculate antigen scores based on SHAP values for an external dataset.

    The function selects a subset of peptides (top_peptides), scales their SHAP values,
    applies the sign of the original values, and sums the values across features to obtain an antigen score.
    It then scales the antigen scores to the 0-1 range, dichotomizes them using a threshold (default quantile),
    and merges the scores with provided survival time and event DataFrames.

    Parameters
    ----------
    shap_values : pd.DataFrame
        DataFrame of SHAP values (with samples as rows and peptide features as columns).
    y_time : pd.DataFrame
        DataFrame of survival times, indexed by sample.
    y_event : pd.DataFrame
        DataFrame of event statuses (e.g., 1=event, 0=censored), indexed by sample.
    top_peptides : list
        List of peptide column names (subset of shap_values.columns) to be used.
    scaler : MinMaxScaler, optional
        Pre-fitted scaler for SHAP values; if None, a new scaler will be fit.
    scaler_antigens : MinMaxScaler, optional
        Pre-fitted scaler for antigen scores; if None, a new scaler will be fit.
    threshold : float, optional
        Threshold to dichotomize antigen scores. If None, computed as the val_quantile quantile of scaled scores.
    val_quantile : float, default 40
        Quantile (in percent) to use for threshold calculation if threshold is None.
    return_all : bool, default False
        If True, the function returns a tuple containing the antigen scores DataFrame,
        the threshold, the scaler used for SHAP values, and the scaler used for antigen scores.

    Returns
    -------
    antigen_scores_df : pd.DataFrame
        DataFrame with columns: 'Antigen Score', 'Antigen Score (Scaled)', and
        'Antigen Score (Dichotomized)', merged with y_time and y_event.
    If return_all is True, also returns (threshold, scaler, scaler_antigens).

    Raises
    ------
    ValueError
        If top_peptides are not a subset of shap_values.columns.
    """
    # Ensure that shap_values is a DataFrame and top_peptides are valid columns.
    if not isinstance(shap_values, pd.DataFrame):
        raise ValueError("shap_values must be a pandas DataFrame.")
    if not set(top_peptides).issubset(shap_values.columns):
        raise ValueError("Some top_peptides are not present in shap_values.columns.")

    # Work on absolute SHAP values for scaling; then reintroduce sign later.
    abs_shap = abs(shap_values)

    # Scale the SHAP values for the top peptides.
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_shap = pd.DataFrame(scaler.fit_transform(abs_shap[top_peptides]),
                                   index=abs_shap.index, columns=abs_shap[top_peptides].columns)
        #scaled_shap = scaler.fit_transform(scaled_shap)

    else:
        scaled_shap = pd.DataFrame(scaler.transform(abs_shap[top_peptides]),
                                   index=abs_shap.index, columns=abs_shap[top_peptides].columns)
        #scaled_shap = pd.DataFrame(scaler.transform(scaled_shap), index=scaled_shap.index)
        scaled_shap = np.clip(scaled_shap, 0, 1)

    #scaled_shap.columns = abs_shap[top_peptides].columns
    signed_shap = scaled_shap * np.sign(shap_values[top_peptides])
    antigen_scores = signed_shap.sum(axis=1).to_frame(name='Antigen Score')

    if scaler_antigens is None:
        scaler_antigens = MinMaxScaler(feature_range=(0, 1))
        antigen_scores_scaled = pd.DataFrame(scaler_antigens.fit_transform(antigen_scores), index=antigen_scores.index)
    else:
        antigen_scores_scaled = pd.DataFrame(scaler_antigens.transform(antigen_scores), index=antigen_scores.index)
        antigen_scores_scaled = np.clip(antigen_scores_scaled, 0, 1)

    # Determine the threshold to dichotomize the scores.
    if threshold is None:
        threshold = antigen_scores_scaled.quantile(val_quantile / 100).iloc[0]

    #antigen_scores = antigen_scores.rename(columns={antigen_scores.columns[0]: 'Antigen Score'})
    antigen_scores_scaled = antigen_scores_scaled.rename(columns={antigen_scores_scaled.columns[0]: 'Antigen Score (Scaled)'})
    antigen_scores_dichotomized = (antigen_scores_scaled.iloc[:,0] >= threshold).astype(int).to_frame(name='Antigen Score (Dichotomized)')
    #antigen_scores_dichotomized = antigen_scores_dichotomized.rename(columns={antigen_scores_dichotomized.columns[0]: 'Antigen Score (Dichotomized)'})

    # Merge with survival data (y_time and y_event)
    merged_df = pd.merge(y_time, y_event, left_index=True, right_index=True)
    merged_df = merged_df.merge(antigen_scores, left_index=True, right_index=True)
    merged_df = merged_df.merge(antigen_scores_scaled, left_index=True, right_index=True)
    antigen_scores_df = merged_df.merge(antigen_scores_dichotomized, left_index=True, right_index=True)

    if return_all:
        return antigen_scores_df, threshold, scaler, scaler_antigens
    else:
        return antigen_scores_df


def perform_logrank_test(df: pd.DataFrame,
                         time_column: str,
                         event_column: str,
                         group_column: str) -> float:
    """
    Perform a log-rank test between two groups defined by the group_column (expected values: 0 and 1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the survival data.
    time_column : str
        Name of the column containing the survival times.
    event_column : str
        Name of the column indicating the event occurrence (1=event, 0=censored).
    group_column : str
        Name of the column indicating the group assignment (0 for one group, 1 for the other).

    Returns
    -------
    float
        The p-value from the log-rank test.

    Raises
    ------
    ValueError
        If the group_column does not contain exactly two groups (e.g. 0 and 1).
    """
    # Verify that the group_column contains exactly two unique groups
    groups = np.sort(df[group_column].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected two groups in column '{group_column}', but found: {groups}")

    # Split the DataFrame into the two groups
    group0 = df[df[group_column] == groups[0]]
    group1 = df[df[group_column] == groups[1]]

    # Perform the log-rank test
    results = logrank_test(group0[time_column],
                           group1[time_column],
                           event_observed_A=group0[event_column],
                           event_observed_B=group1[event_column])

    p_value = results.p_value
    logger.info(f"Log-Rank Test p-value: {p_value:.4f}")

    return p_value


def align_features(X_train, X_ext):
    """Align features between training and external test sets."""
    common_features = list(set(X_train.columns).intersection(X_ext.columns))
    if common_features:
        X_train = X_train[common_features]
        X_ext = X_ext[common_features]
        return X_train, X_ext
    else:
        raise ValueError(f"Training and External set shared no common features")



def clean_predictor_name(label):
    """Clean the predictor name by removing extraneous substrings."""
    label = label.replace('_ctg', '')
    label = label.replace('Category', '')
    label = label.replace('(Scaled)', '')
    label = re.sub(r'\(Dichotomized\) ctg', '', label)
    return label.strip()

def run_univariate_km_analysis(df, variables, save_path="./", suffix_file="default"):
    """
    Run univariate Kaplan–Meier analyses for each variable in 'variables' and
    generate Kaplan–Meier and RMST plots for variables with two groups. A summary
    DataFrame is returned with the cleaned variable name, group, group size, median OS,
    95% confidence interval for median OS, and log-rank p-value.

    Parameters
    ----------
    df : pd.DataFrame
        The clinical metadata including survival time ('OS months'), event indicator ('OS Status'),
        and the candidate predictor variables.
    variables : list of str
        List of variable names to analyze (e.g., clinical variables and peptide markers).
    save_path : str
        path to save figures
    suffix_file : str
        An identifier used in filenames for saving figures.

    Returns
    -------
    univariate_results: pd.DataFrame
    univariate_kmf : pd.DataFrame
        DataFrame summarizing the univariate analysis with columns:
        ['Variable', 'Group', 'Size', 'Median OS (Months)', '95% CI Lower', '95% CI Upper', 'P-value'].
    """
    # Colors for plotting
    cc = ['dodgerblue', 'orange']
    results = []  # list to store results per variable and group

    # Loop over each variable
    for var in variables:
        clean_var = clean_predictor_name(var)
        groups = sorted(df[var].unique())
        kmf_list = []
        flag = 0  # used to cycle colors when plotting (for two-group variables)

        # Create a new figure if variable has less than 3 groups (i.e. binary)
        if len(groups) < 3:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Loop over each group for this variable
        for group in groups:
            group_data = df[df[var] == group]

            # Fit Kaplan–Meier model for this group
            kmf = KaplanMeierFitter()
            kmf.fit(group_data['OS months'], event_observed=group_data['OS Status'], label=str(group))

            # Calculate median survival and 95% CI using a helper function.
            # (Assumes median_survival_times() returns a DataFrame with lower and upper CIs)
            median_survival = kmf.median_survival_time_
            ci_df = median_survival_times(kmf.confidence_interval_)  # user-defined function
            ci_lower, ci_upper = ci_df.iloc[0, 0], ci_df.iloc[0, 1]

            # Get the number of patients in the group
            group_size = len(group_data)
            kmf_list.append(kmf)

            # If fewer than 3 groups, perform a pairwise log-rank test against the others.
            if len(groups) < 3:
                other_data = df[df[var] != group]
                lr_result = logrank_test(group_data['OS months'], other_data['OS months'],
                                         event_observed_A=group_data['OS Status'],
                                         event_observed_B=other_data['OS Status'])
                p_value = lr_result.p_value
                kmf.plot_survival_function(ax=ax, ci_show=True, color=cc[flag], show_censors=True)
                flag += 1
            else:
                # For >2 groups, use a multivariate log-rank test.
                lr_result = multivariate_logrank_test(
                    df['OS months'],
                    df[var],
                    event_observed=df['OS Status']
                )
                p_value = lr_result.p_value

            # Store the results for this group
            results.append([clean_var, group, group_size, median_survival, ci_lower, ci_upper, p_value])

        # For binary variables, add annotation and additional plots once both groups are processed.
        if len(groups) < 3 and flag == 2:
            p_value_text = f'p = {p_value:.3f}'
            ax.text(0.7, 0.05, p_value_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
            # Add at-risk counts (assumes add_at_risk_counts() is defined)
            add_at_risk_counts(kmf_list[0], kmf_list[1], ax=ax)
            ax.set_title(clean_var, fontsize=12)
            ax.set_xlabel("Time (Months)", fontsize=10)
            ax.set_ylabel("Survival Probability", fontsize=10)
            ax.grid(True, linestyle='--', color="black", alpha=0.15)
            # Right after add_at_risk_counts(...)
            fig = ax.get_figure()
            ax2 = fig.axes[-1]  # The newly created twinned axis for the risk table
            # Put the primary axis on top of the secondary axis in the z-order
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.legend(fontsize=8)
            plt.tight_layout()
            fig_path = Path(save_path) / f'kaplan_meier_{clean_var}_{suffix_file}.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Create RMST plot (assumes rmst_plot() is defined)
            fig, ax = plt.subplots(figsize=(8, 4))
            rmst_plot(kmf_list[0], model2=kmf_list[1], text_position=(10, 0.9), t=30, ax=ax)
            lines = ax.get_lines()
            lines[0].set_color(cc[0])
            lines[1].set_color(cc[1])
            ax.set_title(clean_var, fontsize=12)
            ax.set_xlabel("Time (Months)", fontsize=10)
            ax.set_ylabel("Survival Probability", fontsize=10)
            ax.grid(True, linestyle='--', color = "black", alpha=0.15)
            # Right after add_at_risk_counts(...)
            fig = ax.get_figure()
            ax2 = fig.axes[-1]  # The newly created twinned axis for the risk table
            # Put the primary axis on top of the secondary axis in the z-order
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.legend(fontsize=8)
            ax.legend([lines[0], lines[1]], [kmf_list[0].label, kmf_list[1].label], fontsize=8)
            plt.tight_layout()
            fig_path = Path(save_path) / f'kaplan_meier_RMST_{clean_var}_{suffix_file}.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()

    # Create a summary DataFrame from the results
    univariate = pd.DataFrame(results, columns=[
        'Variable', 'Group', 'Size', 'Median OS (Months)', '95% CI Lower', '95% CI Upper', 'P-value'
    ])
    univariate_kmf = univariate.copy(True)
    # Format the results: remove duplicates and apply bold formatting for significant predictors
    univariate_kmf['Group'] = univariate_kmf['Group'].str.replace(r'stage|Antigen Score', '', regex=True)
    univariate_kmf['Variable'] = univariate_kmf.apply(
        lambda row: f"\\textbf{{{row['Variable']}}}" if row['P-value'] < 0.1 else row['Variable'], axis=1
    )
    univariate_kmf['P-value'] = univariate_kmf['P-value'].apply(
        lambda x: f"\\textbf{{{x:.4f}}}" if pd.notna(x) and x < 0.1 else f"{x:.4f}"
    )
    univariate_kmf.loc[univariate_kmf['Variable'].duplicated(keep='last'), 'P-value'] = ""
    univariate_kmf.loc[univariate_kmf['Variable'].duplicated(), 'Variable'] = ""
    univariate_kmf['Median OS (Months)'] = univariate_kmf['Median OS (Months)'].round(2)
    univariate_kmf['95% CI Lower'] = univariate_kmf['95% CI Lower'].round(2)
    univariate_kmf['95% CI Upper'] = univariate_kmf['95% CI Upper'].round(2)

    return univariate, univariate_kmf



def extract_category_name(category):
    """
    Extract a clean category name from a model index string.
    E.g., 'Sex_ctg[T.Male]' becomes 'Male'.
    """
    # Split by '[' then take part after and remove trailing ']'
    category = category.split('[')[-1].split(']')[0].replace('T.', '')
    category = category.replace('stage ', '')
    category = category.replace('Antigen Score', '')
    return category.strip()

def run_univariate_cox_analysis(df, predictors_cont=None, predictors_ctg=None):
    """
    Run univariate Cox PH analyses for continuous and categorical predictors.

    For continuous predictors (list of names), a Cox PH model is fit using a formula with the predictor.
    For categorical predictors (provided as a dictionary with reference values), the model is fit
    forcing the predictor to be categorical. For each predictor (and each category, when applicable),
    the Hazard Ratio, 95% Confidence Intervals, and p-value are extracted.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing survival data with at least the following columns:
            - "OS months": duration
            - "OS Status": event indicator (e.g., 0/1)
            - predictor variables.
    predictors_cont : list of str
        List of continuous predictor names (e.g., ["Age", "AFP", "Antigen Score (Scaled)"]).
    predictors_ctg : dict
        Dictionary of categorical predictors with keys as variable names and values as
        the reference category label (e.g., {"Sex_ctg": "Female", ...}).

    Returns
    -------
    univariate_results: pd.DataFrame
    univariate_cph : pd.DataFrame
        A DataFrame with columns:
            ["Variable", "Group", "Hazard Ratio", "95% CI Lower", "95% CI Upper", "P-value"]
        containing the summary statistics from each univariate analysis.
    """
    results = []  # list to store result rows

    # Check if both predictor lists are empty:
    if (not predictors_cont) and (not predictors_ctg):
        return 0

    # --- Continuous Predictors ---
    if predictors_cont:
        for predictor in predictors_cont:
            # Fit Cox PH model using only the continuous predictor.
            cph = CoxPHFitter()
            formula = f"`{predictor}`"  # backticks in case the name has spaces/special characters
            cph.fit(df, duration_col="OS months", event_col="OS Status", formula=formula)
            summary = cph.summary

            # Extract statistics for the predictor (there is only one row)
            hr = summary.loc[predictor, 'exp(coef)']
            ll = summary.loc[predictor, 'exp(coef) lower 95%']
            ul = summary.loc[predictor, 'exp(coef) upper 95%']
            p_value = summary.loc[predictor, 'p']

            clean_name = clean_predictor_name(predictor)
            results.append((clean_name, "", hr, ll, ul, p_value))
    if predictors_ctg:
        # --- Categorical Predictors ---
        for predictor, ref_cat in predictors_ctg.items():
            cph = CoxPHFitter()
            # Force the predictor to be treated as categorical using C() in the formula.
            formula = f"C(`{predictor}`)"
            cph.fit(df, duration_col="OS months", event_col="OS Status", formula=formula)
            summary = cph.summary

            # For each row in the summary (each category level), extract results.
            for category in summary.index:
                hr = summary.loc[category, 'exp(coef)']
                ll = summary.loc[category, 'exp(coef) lower 95%']
                ul = summary.loc[category, 'exp(coef) upper 95%']
                p_value = summary.loc[category, 'p']

                clean_name = clean_predictor_name(predictor)
                # Create a label for the category by extracting the category name and appending the reference.
                cat_name = extract_category_name(category)
                group_label = f"{cat_name} - Ref {ref_cat}"
                results.append((clean_name, group_label, hr, ll, ul, p_value))

    # --- Create Summary DataFrame ---
    univariate_results = pd.DataFrame(results, columns=["Variable", "Group", "Hazard Ratio", "95% CI Lower", "95% CI Upper", "P-value"])
    univariate_cph = univariate_results.copy(True)
    # Format results:
    # Bold variables and groups with significant p-values (<0.1)
    univariate_cph['Variable'] = univariate_cph.apply(
        lambda row: f"\\textbf{{{row['Variable']}}}" if row['P-value'] < 0.1 else row['Variable'],
        axis=1)
    univariate_cph['Group'] = univariate_cph.apply(
        lambda row: f"\\textbf{{{row['Group']}}}" if row['P-value'] < 0.1 and row['Group'] != "" else row['Group'],
        axis=1)
    univariate_cph['P-value'] = univariate_cph['P-value'].apply(
        lambda x: f"\\textbf{{{x:.4f}}}" if pd.notna(x) and x < 0.1 else f"{x:.4f}")
    # Remove duplicate variable names (if repeated, only show once)
    univariate_cph.loc[univariate_cph['Variable'].duplicated(), 'Variable'] = ""

    # Round the numeric columns to 2 decimals
    univariate_cph['Hazard Ratio'] = univariate_cph['Hazard Ratio'].round(2)
    univariate_cph['95% CI Lower'] = univariate_cph['95% CI Lower'].round(2)
    univariate_cph['95% CI Upper'] = univariate_cph['95% CI Upper'].round(2)

    # Optionally, print as a LaTeX table (requires tabulate)
    # print(tabulate(univariate_cph, headers='keys', tablefmt='latex', showindex=False))

    return univariate_results, univariate_cph

""" 

Plots 

"""

# Example usage (use real data for these variables):
# plot_time_dependent_auc(time_points, auc_values, mean_auc, time_points_highlight=time_points_highlight, auc_values_highlight=auc_values_highlight)

def plot_time_dependent_auc(
        auc_values: pd.Series,
        samples_at_risk_df: pd.DataFrame,
        mean_auc: Optional[float] = None,
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
        time_points_highlight: Optional[np.ndarray] = None,
        max_time_point: Optional[float] = None,
        time_measure: str = 'Months',
        color_auc: str = 'dodgerblue',
        color_mean: str = 'orange',
        color_ci: str = 'gray',
        labels: list = None,
        suffix_file: Optional[str] = None,
        figures_dir: str = './',
        save_fig: bool = False) -> plt.Figure:
    """
    Plots time-dependent AUC over follow-up time with a table displaying samples at risk, censored, and events.
    The table is placed below the AUC plot as a horizontal table aligned with the x-axis.

    Parameters
    ----------
    auc_values : pd.Series
        The AUC values corresponding to each time point.
    samples_at_risk_df : pd.DataFrame
        DataFrame containing 'At Risk', 'Censored', and 'Events' per time point. Its index should
        correspond to time_points.
    mean_auc : float, optional
        The mean AUC value to be displayed as a horizontal line.
    ci_lower : np.ndarray, optional
    ci_upper : np.ndarray, optional
    time_points_highlight : array-like, optional
         Specific time points to highlight on the AUC curve.
    max_time_point: float, optional
        Maximum time point to plot. If not provided, all time points will be plotted.
    time_measure : str, default 'Months'
        Label for the time unit.
    color_auc : str, default 'dodgerblue'
        Color used for the AUC line.
    color_mean : str, default 'orange'
        Color used for the AUC mean line.
    color_ci : str, default 'gray'
        Color used for the CI line.
    labels: list, optional
        Labels for the legend
    suffix_file : str, optional
        Suffix to add to the saved figure filename.
    figures_dir : str, default './'
        Directory where the figure will be saved.
    save_fig : bool, default False
        Whether to save the figure to file.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    # Create a figure with GridSpec to control height ratios
    fig = plt.figure(figsize=(6, 4), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[0.8, 0.15], figure=fig)
    ax_auc = fig.add_subplot(gs[0])

    # Plot AUC curve and mean AUC line
    time_points = auc_values.index

    if mean_auc is None:
        mean_auc = auc_values.mean()

    ax_auc.plot(time_points, auc_values.values, color=color_auc, linestyle='-', linewidth=2)#, label=labels[0])

    # Plot the confidence interval if provided
    if ci_lower is not None and ci_upper is not None:
        ax_auc.fill_between(time_points, ci_lower, ci_upper, color=color_ci, alpha=0.3)

    ax_auc.axhline(y=mean_auc, color=color_mean, linestyle='--', linewidth=1.5)#, label=labels[1])
    ax_auc.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label=f"Random")
    ax_auc.set_xlabel(f'Time ({time_measure})', fontsize=12)
    ax_auc.set_ylabel('Time-Dependent AUC', fontsize=12)
    ax_auc.set_ylim([0.0, 1.05])
    ax_auc.grid(True, color = 'gray', linestyle='--', linewidth=0.7, alpha=0.7)

    # Highlight specific time points if provided
    if time_points_highlight is not None:
        #time_points_highlight = auc_values_highlight.index
        auc_values_highlight = auc_values.loc[time_points_highlight]
        ax_auc.scatter(time_points_highlight, auc_values_highlight, color="black", s=30, zorder=3)
        ax_auc.set_xticks(time_points_highlight)
        ax_auc.set_xticklabels(time_points_highlight.astype(int), fontsize=10)
        # If max_time_point is provided, filter out any time points greater than max_time_point.
        if max_time_point is not None:
            tp_high = [tp for tp in time_points_highlight if tp <= max_time_point]
            if max_time_point not in tp_high:
                tp_high.append(max_time_point)
            tp_high = sorted(tp_high)
        else:
            max_time_point = max(time_points)
            tp_high = sorted(time_points_highlight)[:-1] #remove last to avoid overlap with added max_point
            tp_high.append(max_time_point)
    else:
        tp_high = time_points  # if no highlight is provided, use all time points


    # If the labels list is provided, use it; otherwise, the legend will use default labels.
    handles = ax_auc.get_lines()
    if labels is not None:
        # Get all lines (in the order they were added)
        ax_auc.legend(handles, labels, fontsize=10, loc='lower right')
    else:
        ax_auc.legend(handles, ['Time-Dependent AUC',  f'Mean AUC = {mean_auc:.3f}', "Random"], fontsize=10, loc='lower right')

    # Bottom panel: Table for samples at risk, censored, and events
    ax_table = fig.add_subplot(gs[1])
    ax_table.set_xticks(time_points)
    ax_table.set_xticklabels([])  # No tick labels on x-axis
    # Set y-tick positions and labels for the table (order: Events, Censored, At Risk)
    y_positions = np.array([0.2,0.5,0.8])
    ax_table.set_yticks(y_positions)
    ax_table.set_yticklabels(['Events', 'Censored', 'At Risk'], ha='right', fontsize=10)
    ax_table.grid(False)
    ax_table.set_facecolor('white')
    ax_table.set_frame_on(False)
    ax_table.tick_params(axis='both', which='both', length=0)

    if max_time_point is not None:
        ax_auc.set_xlim([0, max_time_point])  # Ensure x-axis covers max time point given
        ax_table.set_xlim([0, max_time_point])  # Ensure x-axis covers max time point given
    else:
        ax_auc.set_xlim([0, max(time_points)])  # Ensure x-axis covers all time points
        ax_table.set_xlim([0, max(time_points)])  # Ensure x-axis covers all time points

    # Iterate over each highlighted time point and place text for table values
    for time_point in tp_high:
        try:
            at_risk_val = samples_at_risk_df.loc[time_point, 'At Risk']
            censored_val = samples_at_risk_df.loc[time_point, 'Censored']
            events_val = samples_at_risk_df.loc[time_point, 'Events']
        except KeyError:
            # Skip time points not found in the DataFrame
            continue

        ax_table.text(time_point, 0.2, f'{events_val}', ha='center', va='center', fontsize=10, color='black')
        ax_table.text(time_point, 0.5, f'{censored_val}', ha='center', va='center', fontsize=10, color='black')
        ax_table.text(time_point, 0.8, f'{at_risk_val}', ha='center', va='center', fontsize=10, color='black')


    #plt.tight_layout()

    # Save the figure if requested
    if save_fig:
        if suffix_file is None:
            suffix_file = "default"
        save_path = Path(figures_dir) / f'time-dependent_auc_{suffix_file}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_kaplan_meier(df: pd.DataFrame,
                      time_column: str,
                      event_column: str,
                      group_column: str,
                      labels: Dict[Union[int, str], str],
                      ax: Optional[plt.Axes] = None,
                      title: str = '',
                      xlabel: str = '',
                      ylabel: str = '',
                      suffix_file = None,
                      save_fig: bool = False,
                      figures_dir: str = './',
                      ) -> plt.Axes:
    """
    Create a Kaplan-Meier survival plot with log-rank test p-value annotation and at-risk counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing survival data.
    time_column : str
        Name of the column containing survival times.
    event_column : str
        Name of the column indicating event occurrence (1 = event, 0 = censored).
    group_column : str
        Name of the column indicating the group (expected to have exactly two unique values).
    labels : dict
        Dictionary mapping group values (keys) to display labels (values).
        The keys should correspond to the values in group_column.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot. If None, a new figure and axes are created.
    title : str, optional
        Title for the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    suffix_file : str, optional default None
        Suffix used in the filename when saving.
    save_fig : bool, default False
        If True, saves the figure to file.
    figures_dir : str, default './'
        Directory in which to save the figure.


    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the Kaplan-Meier plot.

    Raises
    ------
    ValueError
        If the group_column does not contain exactly two unique groups.
    """
    # Ensure there are exactly two groups.
    unique_groups = np.sort(df[group_column].unique())
    if len(unique_groups) != 2:
        raise ValueError(f"Expected exactly two groups in '{group_column}', but found: {unique_groups}")

    # Define colors explicitly for each group (or you can extend this mapping if needed)
    group_colors = {unique_groups[0]: 'dodgerblue', unique_groups[1]: 'darkorange'}

    # Create figure and axis if not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    else:
        fig = ax.get_figure()

    kmf_list = []
    # Plot survival functions in order based on the sorted unique groups
    for group_value in unique_groups:
        mask = df[group_column] == group_value
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_column][mask], event_observed=df[event_column][mask], label=labels[group_value])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=group_colors[group_value], show_censors=True)

        kmf_list.append(kmf)

    # Perform log-rank test between the two groups
    group1 = df[df[group_column] == unique_groups[0]]
    group2 = df[df[group_column] == unique_groups[1]]
    if not group1.empty and not group2.empty:
        results = logrank_test(group1[time_column], group2[time_column],
                               event_observed_A=group1[event_column],
                               event_observed_B=group2[event_column])
        p_value = results.p_value
        p_value_text = f'p = {p_value:.3f}'
        ax.text(0.7, 0.05, p_value_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
    else:
        logger.warning("One of the groups is empty; log-rank test not performed.")

    ax.grid(True, color='gray', linestyle='--', alpha=0.85) #linewidth=0.8, )

    # Add at-risk counts to the plot (from lifelines)
    if len(kmf_list) == 1:
        add_at_risk_counts(kmf_list[0], ax=ax)
    elif len(kmf_list) == 2:
        add_at_risk_counts(kmf_list[0], kmf_list[1], ax=ax)
    else:
        logger.warning("At-risk counts could not be added because fewer than two groups were found.")

    # Right after add_at_risk_counts(...)
    fig = ax.get_figure()
    ax2 = fig.axes[-1]  # The newly created twinned axis for the risk table
    # Put the primary axis on top of the secondary axis in the z-order
    ax.set_zorder(ax2.get_zorder() + 1)

    # Customize axes
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()

    # Save figure if required
    if save_fig:
        if suffix_file is None:
            suffix_file = "default"
        save_path = Path(figures_dir) / f'kaplan_meier_{suffix_file}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return ax


def plot_parametric_model_qq(survival_time: pd.Series, survival_status: pd.Series,
                             prefix_file: str = None, save_path: str = "./", save_fig: bool = False,
                             fig_size=(8, 6)) -> plt.Figure:
    """
    Generates QQ plots for a set of parametric survival models using the clinical metadata,
    prints their AIC values, and saves the figure.

    Parameters:
        survival_time (pd.Series ): Series containing at least the columns
            "OS months" (duration).
        survival_status (pd.Series ): Series containing at least the columns
            "OS Status" (event indicator).
        prefix_file (int): An integer used in the filename for the saved figure.
        save_path (str): The directory to save the figure.
        save_fig (bool): A boolean indicating whether to save the figure.
        fig_size (tuple): A tuple containing the width and height of the figure.

    Returns:
        plt.Figure: The figure object.
    """
    from lifelines import WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter
    # Define the models to be used.
    models = [
        WeibullFitter(),
        LogNormalFitter(),
        LogLogisticFitter(),
        ExponentialFitter()
    ]

    # Create a 2x2 grid of subplots and flatten to a 1D array for easier iteration.
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axes = axes.flatten()

    # Loop over models and corresponding axes.
    for ax, model in zip(axes, models):
        model.fit(survival_time, survival_status)
        qq_plot(model, ax=ax)
        print(f"The AIC value for {model.__class__.__name__} is {model.AIC_}")

    # Construct the figure path and save the figure.
    if save_fig:
        if prefix_file is None:
            filename = f"QQplot_Parametric_Models.png"
        else:
            filename = f"QQplot_Parametric_Models_{prefix_file}.png"
        fig_path = Path(save_path) / filename
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    return fig


"""

Classes

"""
class CoxnetWrapper(CoxnetSurvivalAnalysis):
    """
    A wrapper for CoxnetSurvivalAnalysis that automatically converts raw survival data
    (with negative values indicating censoring) into a structured array and selects
    coefficients corresponding to the final (smallest) alpha.
    """
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Coxnet model to X and y. If y is not already structured, it is converted.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            1D array of survival times; negative values indicate censoring, positive values indicate event.

        Returns
        -------
        self : CoxnetWrapper
            The fitted model.
        """
        # Convert y to structured format if needed
        if not (hasattr(y, "dtype") and y.dtype.names is not None):
            events = np.where(y < 0, 0, 1)
            times = np.abs(y)
            y = Surv.from_arrays(event=events, time=times)
        # Call the parent class's fit method
        super().fit(X, y)
        #If self.coef_ has multiple columns (one per alpha), choose the coefficients for the last alpha
        #if self.coef_.ndim > 1:
        #logger.info(f"Coefficient shape before selection: {self.coef_.shape}")
        self.coef_ = self.coef_[:, -1]

        return self


class StratifiedKFoldSurv:
    def __init__(self, n_splits=10, shuffle=True, random_state=420):
        self.n_splits = n_splits
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        # Define labels based on the sign of y (negative for censored, positive for deceased)

        if isinstance(y, np.ndarray):  # y is a numpy array from Surv.from_arrays
            labels = y['event']  # Event status is the first column (event = 1, censored = 0)
        else:
            labels = np.where(y < 0, 0, 1)  # 0 for censored, 1 for deceased

        # Stratify based on these labels
        return self.skf.split(X, labels)

    def get_n_splits(self, X=None, y=None, groups=None):
        # We don't need to use the `groups` argument here, just return `n_splits`
        return self.n_splits


