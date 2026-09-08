# ======================
# Standard library
# ======================
import logging
from collections.abc import Iterable, Sequence
from typing import Any, Dict, Optional, TypeAlias

# ======================
# Third-party libraries
# ======================
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sksurv.exceptions import NoComparablePairException
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv

logger = logging.getLogger(__name__)


def convert_to_survival_format(y):
    """
    Convert survival data from signed format to scikit-survival format.

    Parameters:
    - y: 1D array where negative values indicate censored observations,
         positive values indicate events.

    Returns:
    - Structured survival array compatible with scikit-survival functions.
    """
    events = np.where(y < 0, 0, 1)  # 0 for censored, 1 for deceased
    times = np.abs(y)  # Absolute values for survival times
    return Surv.from_arrays(event=events, time=times)


#############################
#         Classes           #
#############################


class StratifiedKFoldSurv:
    def __init__(self, n_splits=5, shuffle=True, random_state=420):
        self.n_splits = n_splits
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, X, y, groups=None):
        # Define labels based on the sign of y (negative for censored, positive for deceased)

        if isinstance(y, np.ndarray):  # y is a numpy array from Surv.from_arrays
            labels = y[
                "event"
            ]  # Event status is the first column (event = 1, censored = 0)
        else:
            labels = np.where(y < 0, 0, 1)  # 0 for censored, 1 for deceased

        # Stratify based on these labels
        return self.skf.split(X, labels)

    def get_n_splits(self, X=None, y=None, groups=None):
        # We don't need to use the `groups` argument here, just return `n_splits`
        return self.n_splits


class CoxnetWrapper(CoxnetSurvivalAnalysis):
    """
    A wrapper for CoxnetSurvivalAnalysis that automatically converts raw survival data
    (with negative values indicating censoring) into a structured array and selects
    coefficients corresponding to the final (smallest) alpha.
    """

    coef_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CoxnetWrapper":
        """
        Fit the Coxnet model to X and y. If y is not already structured, it is converted.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            1D array of survival times; negative values indicate censoring.

        Returns
        -------
        self : CoxnetWrapper
            The fitted model.
        """
        # Convert y to structured format if needed
        if not (hasattr(y, "dtype") and y.dtype.names is not None):
            y = convert_to_survival_format(y)
        # Call the parent class's fit method
        super().fit(X, y)
        # If self.coef_ has multiple columns (one per alpha), choose the coefficients for the last alpha
        # if self.coef_.ndim > 1:
        # logger.info(f"Coefficient shape before selection: {self.coef_.shape}")

        # self.coef_ = self.coef_[:, -1]
        coefficients = np.asarray(self.coef_)

        if coefficients.ndim == 2:
            if coefficients.shape[1] == 0:
                raise ValueError("CoxnetSurvivalAnalysis returned no coefficient paths")

            self.coef_ = coefficients[:, -1].copy()

        elif coefficients.ndim == 1:
            self.coef_ = coefficients

        else:
            raise ValueError(
                f"Unexpected Coxnet coefficient shape: {coefficients.shape}"
            )

        return self


#############################
#         Functions         #
#############################


def filter_validation_data(y_train, y_valid, scores=None):
    """
    Filter validation data to only include samples within the training time range.

    Parameters:
    - y_train: Training survival times (negative for censored, positive for events)
    - y_valid: Validation survival times (negative for censored, positive for events)
    - scores: Optional prediction scores to filter along with y_valid

    Returns:
    - Filtered y_valid, and optionally filtered scores
    """
    valid_indices = y_valid.abs() <= y_train.abs().max()
    y_valid_filtered = y_valid[valid_indices]

    if scores is not None:
        scores_filtered = scores[valid_indices]
        return y_valid_filtered, scores_filtered

    return y_valid_filtered


def c_index_scorer(y_true, y_pred):
    """
    Custom scorer to calculate Concordance Index (C-index) for single-column `y_true`.
    Negative values in `y_true` indicate censored data; positive values indicate events.

    Parameters:
    - y_true: 1D array of survival times (negative for censored, positive for events).
    - y_pred: Predicted risk scores (higher scores indicate higher risk).

    Returns:
    - C-index: Concordance Index for the predictions.
    """
    # Convert survival data to structured format
    y_surv = convert_to_survival_format(y_true)

    # Calculate and return the C-index
    return concordance_index_censored(y_surv["event"], y_surv["time"], y_pred)[0]


def c_index_scorer_ipcw(
    y_train: pd.Series, y_val: pd.Series, y_pred: np.ndarray, tau: float | None = None
) -> float:
    """
    Custom scorer to calculate Concordance Index (C-index).

    Parameters:
    - y_train: 1D array of survival times from training set (negative for censored, positive for events).
    - y_val: 1D array of survival times from validation set (negative for censored, positive for events).
    - y_pred: Predicted risk scores (higher scores indicate higher risk).

    Returns:
    - C-index: Concordance Index for the predictions.
    """

    # Convert survival data to structured format
    y_val_filtered, y_pred_filtered = filter_validation_data(y_train, y_val, y_pred)
    if len(y_val_filtered) == 0:
        raise ValueError("No validation samples fall within the training time range")

    # Filter validation data to only include times within training range
    y_train_surv = convert_to_survival_format(y_train)
    y_val_surv_filtered = convert_to_survival_format(y_val_filtered)

    if tau is None:
        c_index = concordance_index_ipcw(
            y_train_surv, y_val_surv_filtered, y_pred_filtered
        )[0]
    else:
        c_index = concordance_index_ipcw(
            y_train_surv, y_val_surv_filtered, y_pred_filtered, tau=tau
        )[0]

    # Calculate and return the C-index
    return c_index


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
    **kwargs,
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
    cv = StratifiedKFoldSurv(n_splits=n_splits, random_state=random_state)

    method = method.lower()
    if method == "bayesian":
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
            **kwargs,
        )
    elif method == "random":
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
            **kwargs,
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
            **kwargs,
        )
    else:
        raise ValueError("method must be 'bayesian', 'random' or 'grid'.")

    # Run the search
    search.fit(X_train, y_train)

    # Return the best model
    return search.best_estimator_


##########################################################
###Compute scores or coefficients for feature selection###
##########################################################


def univariate_cox_score_single(
    j: int,
    X: np.ndarray,
    y_surv: np.ndarray,
) -> float:
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

    Xj = np.asarray(
        X[:, j : j + 1],
        dtype=float,
    )

    # Constant or non-finite features cannot produce a useful Cox model.
    if not np.isfinite(Xj).all():
        return 0.0

    if np.ptp(Xj[:, 0]) == 0:
        return 0.0

    model = CoxPHSurvivalAnalysis()

    try:
        model.fit(Xj, y_surv)
        return float(model.score(Xj, y_surv))

    except (
        ValueError,
        ArithmeticError,
        np.linalg.LinAlgError,
        NoComparablePairException,
    ) as error:
        logger.debug(
            "Univariate Cox model failed for feature %d: %s",
            j,
            error,
        )
        return 0.0


def univariate_cox_score(X: np.ndarray, y: np.ndarray, n_jobs: int = 1) -> np.ndarray:
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
    y_surv = convert_to_survival_format(y)

    n_features = X.shape[1]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(univariate_cox_score_single)(j, X, y_surv) for j in range(n_features)
    )
    return np.array(scores)


# ##############################
#      Build Pipeline          #
# ##############################


def is_binary_numeric_column(series: pd.Series) -> bool:
    """Return whether a numeric column contains only 0/1, ignoring missing data."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = pd.Series(series.dropna().unique())
    return not values.empty and bool(values.isin([0, 1]).all())


def split_peptide_extra_columns(
    X: pd.DataFrame,
    peptide_prefixes: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Separate peptide features from all additional clinical variables."""
    prefixes = tuple(peptide_prefixes or ("agilent_", "corona2_", "twist_"))
    peptide_cols = [column for column in X.columns if column.startswith(prefixes)]
    extra_cols = [column for column in X.columns if column not in peptide_cols]
    return peptide_cols, extra_cols


def split_extra_columns_for_imputation(
    X: pd.DataFrame,
    extra_cols: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split extras into binary numeric, continuous numeric, and non-numeric."""
    binary: list[str] = []
    continuous: list[str] = []
    non_numeric: list[str] = []
    for column in extra_cols:
        if not pd.api.types.is_numeric_dtype(X[column]):
            non_numeric.append(column)
        elif is_binary_numeric_column(X[column]):
            binary.append(column)
        else:
            continuous.append(column)
    return binary, continuous, non_numeric


def make_pipeline(
    X: pd.DataFrame,
    peptide_cols: list[str],
    extra_cols: list[str],
    estimator: Any,
    random_state: int,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
) -> Pipeline:
    """Build peptide selection and clinical-feature preprocessing."""
    transformers = []
    if peptide_cols:
        transformers.append(
            (
                "peptides",
                Pipeline(
                    [
                        ("variance_removal", VarianceThreshold(threshold=0.0)),
                        (
                            "feature_selection",
                            SelectFromModel(
                                CoxnetWrapper(
                                    l1_ratio=0.6, alpha_min_ratio=0.0001, n_alphas=100
                                ),
                                threshold=1e-5,
                            ),
                        ),
                    ]
                ),
                peptide_cols,
            )
        )
    if extra_cols:
        binary, continuous, non_numeric = split_extra_columns_for_imputation(
            X,
            extra_cols,
        )
        if non_numeric:
            raise ValueError(
                "These additional features are non-numeric. Encode them before "
                f"building the survival pipeline: {non_numeric}"
            )
        if binary:
            transformers.append(("binary_extra", "passthrough", binary))
        if continuous:
            transformer: Any = "passthrough"
            if impute_extra_numeric:
                transformer = SimpleImputer(strategy=extra_numeric_impute_strategy)
            transformers.append(("continuous_extra", transformer, continuous))

    if not transformers:
        raise ValueError("No peptide or additional features were available")

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessor.set_output(transform="pandas")
    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def build_pipeline(
    X_train: pd.DataFrame,
    random_state: int = 420,
    peptide_prefixes: Sequence[str] | None = None,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
) -> Pipeline:
    """
    Create a Cox-survival XGBoost pipeline for peptides and numeric extras.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data. Peptides are detected by prefix; all remaining columns
        are treated as additional numeric clinical variables.
    random_state : int
        Seed for reproducibility.
    peptide_prefixes : sequence of str, optional
        Prefixes identifying peptide columns. All other columns are clinical extras.
    impute_extra_numeric : bool, optional
        Whether to impute numeric continous clinical variables.
    extra_numeric_impute_strategy : str, optional
        Imputation strategy for numeric clinical variables.

    Returns
    -------
    pipeline : sklearn.Pipeline
        Preprocessing + model pipeline.
    """
    peptide_cols, extra_cols = split_peptide_extra_columns(
        X_train,
        peptide_prefixes,
    )
    logger.info("Peptide features: %d", len(peptide_cols))
    logger.info("Additional features: %s", extra_cols)

    # Define estimator
    estimator = xgb.XGBRegressor(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )

    # Build pipeline
    return make_pipeline(
        X=X_train,
        peptide_cols=peptide_cols,
        extra_cols=extra_cols,
        estimator=estimator,
        random_state=random_state,
        impute_extra_numeric=impute_extra_numeric,
        extra_numeric_impute_strategy=extra_numeric_impute_strategy,
    )


#####################
#     Fit model     #
#####################


def _build_and_fit_pipeline(
    pipeline: Pipeline | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, Any] | None,
    n_splits: int,
    n_iter: int,
    random_state: int,
    n_jobs: int,
    peptide_prefixes: Sequence[str] | None = None,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
) -> Pipeline:
    """
    Helper function to build pipeline and perform hyperparameter tuning.

    Parameters
    ----------
    pipeline : Pipeline or None
        Existing pipeline or None to build default
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training targets
    param_grid : dict or None
        Hyperparameter grid for tuning
    n_splits : int
        Number of CV splits for tuning
    n_iter : int
        Number of iterations for Bayesian optimization
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs
    peptide_prefixes : sequence of str, optional
        Prefixes identifying peptide columns. All other columns are clinical extras.
    impute_extra_numeric : bool, optional
        Whether to impute numeric continous clinical variables.
    extra_numeric_impute_strategy : str, optional
        Imputation strategy for numeric clinical variables.

    Returns
    -------
    best_estimator : Pipeline
        Fitted pipeline
    """
    if pipeline is None:
        pipeline_candidate = build_pipeline(
            X_train,
            random_state=random_state,
            peptide_prefixes=peptide_prefixes,
            impute_extra_numeric=impute_extra_numeric,
            extra_numeric_impute_strategy=extra_numeric_impute_strategy,
        )
    else:
        pipeline_candidate = clone(pipeline)

    # Besides runtime validation, this narrows the inferred type to Pipeline.
    if not isinstance(pipeline_candidate, Pipeline):
        raise TypeError(
            "build_pipeline/clone did not produce an sklearn Pipeline; "
            f"found {type(pipeline_candidate).__name__}"
        )

    working_pipeline = pipeline_candidate

    if param_grid is None:
        working_pipeline.fit(X_train, y_train)
        return working_pipeline

    search_grid = dict(param_grid)
    valid_params = set(working_pipeline.get_params().keys())
    unknown = set(search_grid) - valid_params

    inactive_peptide_parameters: set[str] = set()

    if unknown:
        preprocessor = working_pipeline.named_steps.get("preprocessor")

        if isinstance(preprocessor, ColumnTransformer):
            transformer_names = {
                transformer_name for transformer_name, _, _ in preprocessor.transformers
            }

            inactive_peptide_parameters = {
                parameter
                for parameter in unknown
                if parameter.startswith("preprocessor__peptides__")
                and "peptides" not in transformer_names
            }

    if inactive_peptide_parameters:
        logger.info(
            "Ignoring %d peptide search parameters because this run has no "
            "peptide features",
            len(inactive_peptide_parameters),
        )

        search_grid = {
            parameter: space
            for parameter, space in search_grid.items()
            if parameter not in inactive_peptide_parameters
        }

        unknown -= inactive_peptide_parameters

    if unknown:
        raise ValueError(
            "Parameter-grid entries do not match the survival pipeline: "
            f"{sorted(unknown)}"
        )

    if not search_grid:
        logger.info("No active search parameters; fitting the pipeline directly")
        working_pipeline.fit(X_train, y_train)
        return working_pipeline

    best_estimator = search_best_survival_model(
        working_pipeline,
        search_grid,
        X_train,
        y_train,
        method="bayesian",
        n_splits=n_splits,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    if not isinstance(best_estimator, Pipeline):
        raise TypeError(
            "Hyperparameter search did not return an sklearn Pipeline; "
            f"found {type(best_estimator).__name__}"
        )

    return best_estimator


def calculate_cumulative_dynamic_auc(
    y_train: pd.Series,
    y_valid: pd.Series,
    scores: np.ndarray,
    time_points: np.ndarray,
) -> tuple[pd.Series, float]:
    y_valid, scores = filter_validation_data(
        y_train,
        y_valid,
        scores,
    )

    if len(y_valid) == 0:
        raise ValueError("No validation samples fall within the training time range")

    y_train_surv = convert_to_survival_format(y_train)
    y_valid_surv = convert_to_survival_format(y_valid)

    valid_absolute = np.abs(y_valid.to_numpy(dtype=np.float64))
    train_absolute = np.abs(y_train.to_numpy(dtype=np.float64))

    min_time_point: float = valid_absolute.min().item()
    valid_max_time: float = valid_absolute.max().item()
    train_max_time: float = train_absolute.max().item()

    max_time_point = (
        min(
            valid_max_time,
            train_max_time,
        )
        - 0.001
    )

    time_points_highlight = time_points[
        (time_points >= min_time_point) & (time_points <= max_time_point)
    ]

    if len(time_points_highlight) == 0:
        raise ValueError(
            "No requested AUC time points fall inside the evaluable interval "
            f"[{min_time_point}, {max_time_point}]"
        )

    auc_values, mean_auc_value = cumulative_dynamic_auc(
        y_train_surv,
        y_valid_surv,
        scores,
        time_points_highlight,
    )

    auc_series = pd.Series(
        auc_values,
        index=time_points_highlight,
    )

    return auc_series, float(mean_auc_value)


######################
#   Nested models    #
######################


def nested_cv_single(
    train_idx,
    valid_idx,
    X_train,
    y_time_train,
    pipeline=None,
    param_grid=None,
    n_splits=5,
    n_iter=30,
    random_state=420,
    n_jobs=-1,
    peptide_prefixes=None,
    impute_extra_numeric=False,
    extra_numeric_impute_strategy="median",
):
    X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_train_fold, y_valid_fold = (
        y_time_train.iloc[train_idx],
        y_time_train.iloc[valid_idx],
    )

    best_estimator = _build_and_fit_pipeline(
        pipeline,
        X_train_fold,
        y_train_fold,
        param_grid,
        n_splits,
        n_iter,
        random_state,
        n_jobs,
        peptide_prefixes,
        impute_extra_numeric,
        extra_numeric_impute_strategy,
    )

    # Predict risk scores on the validation fold
    risk_scores_fold = best_estimator.predict(X_valid_fold)

    # Transform validation data for SHAP computation
    preprocessor = best_estimator.named_steps["preprocessor"]
    final_estimator = best_estimator.named_steps["estimator"]
    X_valid_transformed = preprocessor.transform(X_valid_fold)
    if not isinstance(X_valid_transformed, pd.DataFrame):
        X_valid_transformed = pd.DataFrame(
            X_valid_transformed,
            index=X_valid_fold.index,
            columns=preprocessor.get_feature_names_out(),
        )
    logger.info("Transformed validation fold: %s", X_valid_transformed.shape)

    # Compute SHAP values using the regressor (last step)
    explainer = shap.TreeExplainer(final_estimator)
    shap_values_fold = explainer.shap_values(X_valid_transformed)
    shap_values_fold_df = pd.DataFrame(
        shap_values_fold,
        index=X_valid_transformed.index,
        columns=X_valid_transformed.columns,
    )

    # Compute performance score (e.g., c-index)
    # fold_cindex = c_index_scorer(y_valid_fold, risk_scores_fold)
    cindex_fold = c_index_scorer_ipcw(y_train_fold, y_valid_fold, risk_scores_fold)

    return valid_idx, risk_scores_fold, shap_values_fold_df, best_estimator, cindex_fold


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
    n_jobs: int = 1,
    n_jobs_inner: int = -1,
    peptide_prefixes: Sequence[str] | None = None,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
):
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
    peptide_prefixes : sequence of str, optional
        Prefixes identifying peptide columns. All other columns are clinical extras.
    impute_extra_numeric : bool, optional
        Whether to impute numeric continous clinical variables.
    extra_numeric_impute_strategy : str, optional
        Imputation strategy for numeric clinical variables.

    Returns
    -------
    model_list : List[Pipeline]
        List of the best estimators (one per fold).
    shap_values : pd.DataFrame
        DataFrame concatenating SHAP values from all folds, indexed by sample.
    risk_scores : pd.Series
        Series of risk scores from all folds, indexed by sample.
    validation_indices : List[np.ndarray]
        validation indices for each outer fold.
    time_dependent_auc : pd.DataFrame
        DataFrame with time-dependent AUC for each fold.
    c_index : List[float]
        List of performance c-index scores for each outer fold.
    mean_auc : List[float]
        List of performance mean auc scores for each outer fold.
    """
    # outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFoldSurv(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    # Run the outer folds in parallel.
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(nested_cv_single)(
            train_idx,
            valid_idx,
            X_train,
            y_time_train,
            pipeline=pipeline,
            param_grid=param_grid,
            n_splits=n_splits_inner,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs_inner,
            peptide_prefixes=peptide_prefixes,
            impute_extra_numeric=impute_extra_numeric,
            extra_numeric_impute_strategy=extra_numeric_impute_strategy,
        )
        for train_idx, valid_idx in outer_cv.split(X_train, y_time_train)
    )

    # Initialize master containers.
    model_list = []
    validation_indices = []
    c_index = []

    risk_scores = pd.Series(index=X_train.index, name="Risk score", dtype=float)

    shap_values = pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns)

    max_time_point = (
        y_time_train.abs().max() if max_time_point is None else max_time_point
    )
    time_points = np.arange(1, max_time_point, step=1)
    time_dependent_auc = pd.DataFrame(
        index=np.arange(n_splits),
        columns=time_points,
        dtype=float,
    )
    mean_auc = []

    # Aggregate results from each fold.
    i = 0
    for fold_result in fold_results:
        valid_idx, fold_risk, fold_shap_df, model, fold_cindex = fold_result

        model_list.append(model)
        validation_indices.append(valid_idx)
        c_index.append(fold_cindex)

        # Assign risk scores for the validation fold.
        risk_scores.iloc[valid_idx] = fold_risk

        # Update master SHAP values. Since folds are disjoint, direct assignment works.
        shap_values.loc[fold_shap_df.index, fold_shap_df.columns] = fold_shap_df
        del fold_shap_df

        # Build y_valid (same order as fold_risk) and y_train (complement)
        y_valid_fold = y_time_train.iloc[valid_idx]
        y_train_fold = y_time_train.drop(y_valid_fold.index)

        auc_values_fold, mean_auc_fold = calculate_cumulative_dynamic_auc(
            y_train_fold,  # y_time_train.loc[y_time_train.index.difference(y_valid_fold.index)],
            y_valid_fold,
            fold_risk,
            time_points,
        )

        time_dependent_auc.loc[i, auc_values_fold.index] = auc_values_fold.values
        mean_auc.append(mean_auc_fold)
        i = i + 1

    logger.info(
        f"Mean Concordance Index (C-index) across folds: {np.mean(c_index):.4f}"
    )
    logger.info(f"Mean Time-Dependent AUC across folds: {np.mean(mean_auc):.4f}")

    return (
        model_list,
        shap_values,
        risk_scores,
        validation_indices,
        time_dependent_auc,
        c_index,
        mean_auc,
    )


#
# def nested_cv_allfolds(
#     X_train: pd.DataFrame,
#     y_time_train: pd.Series,
#     y_event_train: pd.Series,
#     pipeline: Optional[Pipeline] = None,
#     param_grid: Optional[Dict] = None,
#     n_splits: int = 10,
#     n_splits_inner: int = 5,
#     n_iter: int = 30,
#     max_time_point=None,
#     random_state: int = 420,
#     n_jobs: int = -1,
# ):
#     """
#     Perform nested cross-validation to tune hyperparameters and feature selection,
#     and aggregate SHAP values and risk scores for each outer fold.
#
#     For each outer fold:
#       - Split data into training and validation sets.
#       - Run hyperparameter tuning (BayesSearchCV) on the inner folds (if param_grid is provided)
#         to select the best estimator.
#       - Use the best estimator to predict risk scores on the validation set.
#       - Compute SHAP values on the validation set. Note: because the pipeline's
#         feature selection step may select a different subset of features per fold,
#         the returned SHAP values DataFrame may have different columns.
#       - Compute time-dependent AUC scores for the validation set.
#       - Store the risk scores and SHAP values (with sample indices).
#
#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Feature matrix for training.
#     y_time_train : pd.Series
#         Survival times for training samples.
#     y_event_train : pd.Series
#         Event status (1 if event occurred, 0 if censored) for training samples.
#     pipeline : dict, optional
#         Default pipeline for hyperparameter tuning. If None, default parameters are used.
#     param_grid : dict, optional
#         Hyperparameter grid to search over. If None, no hyperparameter tuning is performed.
#     n_splits : int, default 10
#         Number of outer CV splits.
#     n_splits_inner : int, default 5
#         Number of inner CV splits
#     n_iter : int, default 30
#         Number of iterations for Bayesian optimization.
#     max_time_point : int, default None
#         Max time point for time-dependent AUC estimation
#     random_state : int, default 420
#         Seed for reproducibility.
#     n_jobs: int, default 1
#         Number of jobs in parallel in the inner CV
#
#     Returns
#     -------
#     model_list : List[Pipeline]
#         List of the best estimators (one per fold).
#     shap_values : pd.DataFrame
#         DataFrame concatenating SHAP values from all folds, indexed by sample.
#     risk_scores : pd.Series
#         Series of risk scores from all folds, indexed by sample.
#     time_dependent_auc : pd.DataFrame
#         DataFrame with time-dependent AUC for each fold.
#     time_dependent_auc_mean : pd.Series
#         Series with AUC mean
#     c_index : List[float]
#         List of performance scores (e.g., c-index) for each outer fold.
#     validation_indices : List[np.ndarray]
#         validation indices for each outer fold.
#     """
#     outer_cv = StratifiedKFold(
#         n_splits=n_splits, shuffle=True, random_state=random_state
#     )
#
#     c_index = []
#     model_list = []
#     validation_indices = []
#     risk_scores = pd.Series(index=X_train.index, name="Risk score")
#     shap_values = pd.DataFrame(0.0, index=X_train.index, columns=X_train.columns)
#     max_time_point = (
#         y_time_train.abs().max() if max_time_point is None else max_time_point
#     )
#     time_points = np.arange(1, max_time_point, step=1)
#     time_dependent_auc = pd.DataFrame(index=np.arange(n_splits), columns=time_points)
#     mean_auc = pd.Series(index=np.arange(n_splits))
#     i = 0
#     for train_idx, valid_idx in outer_cv.split(X_train, y_event_train):
#         X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
#         y_train_fold, y_valid_fold = (
#             y_time_train.iloc[train_idx],
#             y_time_train.iloc[valid_idx],
#         )
#
#         if pipeline is None:
#             params = {
#                 "objective": "survival:cox",
#                 "eval_metric": "cox-nloglik",
#                 "tree_method": "hist",
#                 "random_state": random_state,
#                 "n_jobs": -1,
#             }
#             # Build the pipeline for this fold.
#             pipeline = Pipeline(
#                 [
#                     ("variance_removal", VarianceThreshold(threshold=0.0)),
#                     (
#                         "feature_selection",
#                         SelectFromModel(
#                             CoxnetWrapper(
#                                 l1_ratio=0.6, alpha_min_ratio=0.0001, n_alphas=100
#                             ),
#                             threshold=1e-5,
#                         ),
#                     ),
#                     ("regressor", xgb.XGBRegressor(**params)),
#                 ]
#             )
#
#             # Perform hyperparameter tuning if param_grid is provided.
#         if param_grid is not None:
#             # Use your search_best_model function or similar with BayesSearchCV
#             best_estimator = search_best_survival_model(
#                 pipeline,
#                 param_grid,
#                 X_train_fold,
#                 y_train_fold,
#                 method="bayesian",
#                 n_splits=n_splits,
#                 n_iter=n_iter,
#                 random_state=random_state,
#                 n_jobs=n_jobs,
#             )
#         else:
#             best_estimator = pipeline
#             best_estimator.fit(X_train_fold, y_train_fold)
#
#         model_list.append(best_estimator)
#
#         # Predict risk scores on the validation fold
#         risk_scores_fold = best_estimator.predict(X_valid_fold)
#         risk_scores.iloc[valid_idx] = risk_scores_fold
#
#         # Transform validation data for SHAP computation
#         if len(best_estimator) > 1:
#             try:
#                 X_valid_fold = best_estimator[:-1].transform(X_valid_fold)
#             except Exception as e:
#                 logger.error(f"Error transforming validation data in fold: {e}")
#                 continue
#
#         # Compute SHAP values using the regressor (last step)
#         explainer = shap.TreeExplainer(best_estimator["regressor"])
#         shap_values_fold = explainer.shap_values(X_valid_fold)
#         shap_values_fold_df = pd.DataFrame(
#             shap_values_fold, index=X_valid_fold.index, columns=X_valid_fold.columns
#         )
#         shap_values.loc[shap_values_fold_df.index, shap_values_fold_df.columns] = (
#             shap_values_fold_df
#         )
#
#         fold_cindex = c_index_scorer_ipcw(y_train_fold, y_valid_fold, risk_scores_fold)
#         c_index.append(fold_cindex)
#
#         auc_values_fold, mean_auc_values_fold = calculate_cumulative_dynamic_auc(
#             y_train_fold, y_valid_fold, risk_scores_fold, time_points
#         )
#         time_dependent_auc.loc[i, auc_values_fold.index] = auc_values_fold.values
#         mean_auc.loc[i] = mean_auc_values_fold
#
#         validation_indices.append(
#             valid_idx
#         )  # Track SampleNames corresponding to the validation set
#         i = i + 1
#     logger.info(
#         f"Mean Concordance Index (C-index) across validation folds: {np.mean(c_index):.4f}"
#     )
#     logger.info(
#         f"Mean Time-Dependent AUC across folds: {time_dependent_auc.mean(skipna=True).mean():.4f}"
#     )
#
#     return (
#         model_list,
#         shap_values,
#         risk_scores,
#         validation_indices,
#         time_dependent_auc,
#         c_index,
#         mean_auc,
#     )


################################
#  External Validation model   #
################################


def align_external_to_pipeline(
    X_external: pd.DataFrame,
    fitted_pipeline: Pipeline,
    peptide_prefixes: Sequence[str] = ("agilent_", "twist_", "corona2_"),
    fill_missing_peptides_with_zero: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Align external inputs to the raw features seen by a fitted pipeline."""
    if not isinstance(X_external, pd.DataFrame):
        raise TypeError("X_external must be a pandas DataFrame")
    if X_external.columns.has_duplicates:
        duplicated = X_external.columns[X_external.columns.duplicated()].tolist()[:10]
        raise ValueError(f"External data has duplicate feature names: {duplicated}")

    expected_raw: Any = getattr(
        fitted_pipeline,
        "feature_names_in_",
        None,
    )

    if expected_raw is None:
        preprocessor = fitted_pipeline.named_steps.get("preprocessor")

        if preprocessor is not None:
            expected_raw = getattr(
                preprocessor,
                "feature_names_in_",
                None,
            )

    if expected_raw is None:
        raise ValueError(
            "The fitted pipeline does not expose feature_names_in_. "
            "It must be fitted with a pandas DataFrame before "
            "external alignment."
        )

    if isinstance(expected_raw, (str, bytes)) or not isinstance(
        expected_raw,
        Iterable,
    ):
        raise TypeError(
            "feature_names_in_ must be an iterable of feature names; "
            f"found {type(expected_raw).__name__}"
        )

    expected = [str(column) for column in expected_raw]

    external_columns = set(X_external.columns)
    expected_set = set(expected)
    missing = [column for column in expected if column not in external_columns]
    extra = [column for column in X_external.columns if column not in expected_set]
    prefixes = tuple(peptide_prefixes)
    missing_peptides = [column for column in missing if column.startswith(prefixes)]
    missing_non_peptides = [
        column for column in missing if not column.startswith(prefixes)
    ]

    if missing_non_peptides:
        raise ValueError(
            "External data is missing required non-peptide features; these cannot "
            f"be filled safely: {missing_non_peptides[:20]}"
        )
    if missing_peptides and not fill_missing_peptides_with_zero:
        raise ValueError(
            "External data is missing peptide features and zero filling is disabled: "
            f"{missing_peptides[:20]}"
        )

    aligned = X_external.copy()
    if missing_peptides:
        missing_frame = pd.DataFrame(
            0,
            index=aligned.index,
            columns=missing_peptides,
        )
        aligned = pd.concat([aligned, missing_frame], axis=1)
    aligned = aligned.loc[:, expected].copy()

    report = {
        "missing_features": missing,
        "missing_peptides": missing_peptides,
        "missing_non_peptides": missing_non_peptides,
        "extra_features": extra,
    }
    logger.info(
        "External alignment: expected=%d, missing peptides=%d, extras=%d",
        len(expected),
        len(missing_peptides),
        len(extra),
    )
    return aligned, report


ValidationResult: TypeAlias = tuple[
    Pipeline,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    float,
    float,
]

ValidationResultWithReport: TypeAlias = tuple[
    Pipeline,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    float,
    float,
    dict[str, list[str]],
]


def train_and_validate_model(
    X_train: pd.DataFrame,
    y_time_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_time_test: pd.Series | None = None,
    pipeline: Pipeline | None = None,
    param_grid: dict[str, Any] | None = None,
    best_estimator: Pipeline | None = None,
    n_splits: int = 10,
    n_iter: int = 30,
    max_time_point: float | None = None,
    random_state: int = 420,
    n_jobs: int = -1,
    get_only_model: bool = False,
    peptide_prefixes: Sequence[str] = (
        "agilent_",
        "twist_",
        "corona2_",
    ),
    fill_missing_peptides_with_zero: bool = True,
    impute_extra_numeric: bool = False,
    extra_numeric_impute_strategy: str = "median",
    return_feature_report: bool = False,
) -> Pipeline | ValidationResult | ValidationResultWithReport:
    """Train a survival model or evaluate a fitted model on external data.

    Parameters
    ----------
    X_train
        Training feature matrix.
    y_time_train
        Signed training survival times. Positive values indicate events and
        negative values indicate censored observations.
    X_test
        Optional external/testing feature matrix.
    y_time_test
        Optional signed external/testing survival times.
    pipeline
        Unfitted survival-analysis pipeline. A default pipeline is created when
        this is None.
    param_grid
        Optional Bayesian hyperparameter search space.
    best_estimator
        Previously fitted pipeline. When supplied, model fitting is skipped.
    n_splits
        Number of cross-validation splits used during hyperparameter tuning.
    n_iter
        Number of Bayesian optimization iterations.
    max_time_point
        Maximum evaluation time for cumulative dynamic AUC. When None, the
        maximum absolute training survival time is used.
    random_state
        Random seed.
    n_jobs
        Number of parallel jobs used during hyperparameter tuning.
    get_only_model
        If True, return only the fitted pipeline without external evaluation.
    peptide_prefixes
        Prefixes identifying peptide features.
    fill_missing_peptides_with_zero
        If True, missing external peptide features are added with value zero.
    impute_extra_numeric
        Whether to impute continuous numeric clinical variables.
    extra_numeric_impute_strategy
        Imputation strategy for continuous numeric clinical variables.
    return_feature_report
        If True, append the external feature-alignment report to the returned
        evaluation tuple.

    Returns
    -------
    Pipeline
        Returned when ``get_only_model=True``.
    ValidationResult
        Six-element external-validation result when
        ``return_feature_report=False``.
    ValidationResultWithReport
        Seven-element result including the feature-alignment report when
        ``return_feature_report=True``.
    """

    # Either fit a new pipeline or use the previously fitted pipeline.
    if best_estimator is None:
        fitted_pipeline = _build_and_fit_pipeline(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_time_train,
            param_grid=param_grid,
            n_splits=n_splits,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            peptide_prefixes=peptide_prefixes,
            impute_extra_numeric=impute_extra_numeric,
            extra_numeric_impute_strategy=extra_numeric_impute_strategy,
        )
    else:
        fitted_pipeline = best_estimator

    if get_only_model:
        return fitted_pipeline

    # External evaluation requires both features and survival times.
    if X_test is None or y_time_test is None:
        raise ValueError(
            "X_test and y_time_test are required when " "get_only_model=False"
        )

    # Align raw external features to those expected by the fitted pipeline.
    X_test_aligned, feature_report = align_external_to_pipeline(
        X_external=X_test,
        fitted_pipeline=fitted_pipeline,
        peptide_prefixes=peptide_prefixes,
        fill_missing_peptides_with_zero=fill_missing_peptides_with_zero,
    )

    # Predict risk scores using the complete pipeline.
    risk_predictions = np.asarray(
        fitted_pipeline.predict(X_test_aligned),
        dtype=np.float64,
    )

    # Obtain and validate the fitted preprocessor.
    preprocessor = fitted_pipeline.named_steps.get("preprocessor")

    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            "The fitted pipeline must contain a ColumnTransformer "
            "named 'preprocessor'"
        )

    final_estimator = fitted_pipeline.named_steps.get("estimator")

    if final_estimator is None:
        raise KeyError("The fitted pipeline does not contain an 'estimator' step")

    # Transform external data to the selected feature space used by the
    # final estimator.
    transformed = preprocessor.transform(X_test_aligned)

    if isinstance(transformed, pd.DataFrame):
        X_test_transformed = transformed
    else:
        feature_names = preprocessor.get_feature_names_out()

        X_test_transformed = pd.DataFrame(
            np.asarray(transformed),
            index=X_test_aligned.index,
            columns=feature_names,
        )

    logger.info(
        "Transformed test data: %s",
        X_test_transformed.shape,
    )

    # Calculate SHAP values using the final estimator and transformed data.
    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(X_test_transformed)

    shap_values_df = pd.DataFrame(
        shap_values,
        index=X_test_transformed.index,
        columns=X_test_transformed.columns,
    )

    # Calculate the IPCW concordance index.
    c_index = float(
        c_index_scorer_ipcw(
            y_train=y_time_train,
            y_val=y_time_test,
            y_pred=risk_predictions,
        )
    )

    # Determine the maximum AUC evaluation time.
    if max_time_point is None:
        training_times = np.abs(y_time_train.to_numpy(dtype=np.float64))

        if training_times.size == 0:
            raise ValueError("Training survival times cannot be empty")

        if not np.isfinite(training_times).all():
            raise ValueError("Training survival times contain non-finite values")

        resolved_max_time_point: float = training_times.max().item()
    else:
        resolved_max_time_point = float(max_time_point)

    if resolved_max_time_point <= 1:
        raise ValueError("max_time_point must be greater than 1")

    time_points = np.arange(
        1.0,
        resolved_max_time_point,
        step=1.0,
    )

    time_dependent_auc, mean_auc_value = calculate_cumulative_dynamic_auc(
        y_train=y_time_train,
        y_valid=y_time_test,
        scores=risk_predictions,
        time_points=time_points,
    )

    mean_auc_value = float(mean_auc_value)

    risk_scores = pd.Series(
        risk_predictions,
        index=X_test_aligned.index,
        name="Risk score",
        dtype=float,
    )

    logger.info(
        "Mean Concordance Index (C-index) in testing set: %.4f",
        c_index,
    )
    logger.info(
        "Mean Time-Dependent AUC in testing set: %.4f",
        mean_auc_value,
    )

    result: ValidationResult = (
        fitted_pipeline,
        shap_values_df,
        risk_scores,
        time_dependent_auc,
        mean_auc_value,
        c_index,
    )

    if return_feature_report:
        result_with_report: ValidationResultWithReport = (
            fitted_pipeline,
            shap_values_df,
            risk_scores,
            time_dependent_auc,
            mean_auc_value,
            c_index,
            feature_report,
        )

        return result_with_report

    return result
