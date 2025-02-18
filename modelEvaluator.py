from pathlib import Path
import logging
from dataclasses import dataclass, field
from typing import Optional, Union
import copy

import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from scipy.stats import t
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
import shap

from dataHandler import MetadataHandler, OligosHandler, FeatureManager
from joblib import dump
from types import SimpleNamespace


@dataclass
class PredResults:
    predictions_train: Optional[pd.DataFrame] = None
    roc_metrics_train: Optional[dict[str, Union[np.ndarray, float]]] = None
    features_names_train: Optional[pd.Index] = None
    target_samples_train: Optional[np.ndarray] = None
    shap_values_train: Optional[np.ndarray] = None
    permutation_importance_train: Optional[pd.DataFrame] = None

    predictions_test: Optional[pd.DataFrame] = None
    roc_metrics_test: Optional[dict[str, Union[np.ndarray, float]]] = None
    features_names_test: Optional[np.ndarray] = None
    target_samples_test: Optional[np.ndarray] = None
    shap_values_test: Optional[np.ndarray] = None
    permutation_importance_test: Optional[pd.DataFrame] = None

    predictions_ext: Optional[pd.DataFrame] = None
    roc_metrics_ext: Optional[dict[str, Union[np.ndarray, float]]] = None
    features_names_ext: Optional[np.ndarray] = None
    target_samples_ext: Optional[np.ndarray] = None
    shap_values_ext: Optional[np.ndarray] = None
    permutation_importance_ext: Optional[pd.DataFrame] = None

@dataclass
class CombinedPredResults:
    """Container for all results across combinations."""
    predictions_train: Union[list[pd.DataFrame], pd.DataFrame]  = field(default_factory=list)
    roc_metrics_train: list[dict] = field(default_factory=list)
    features_names_train: list[pd.Index] = field(default_factory=list)
    target_samples_train: list[np.ndarray] = field(default_factory=list)
    shap_values_train: list[np.ndarray] = field(default_factory=list)
    permutation_importance_train: list[pd.DataFrame] = field(default_factory=list)

    predictions_test: Union[list[pd.DataFrame], pd.DataFrame]  = field(default_factory=list)
    roc_metrics_test: list[dict] = field(default_factory=list)
    features_names_test: list[pd.Index] = field(default_factory=list)
    target_samples_test: list[np.ndarray] = field(default_factory=list)
    shap_values_test: list[np.ndarray] = field(default_factory=list)
    permutation_importance_test: list[pd.DataFrame] = field(default_factory=list)

    predictions_ext: Union[list[pd.DataFrame], pd.DataFrame] = field(default_factory=list)
    roc_metrics_ext: list[dict] = field(default_factory=list)
    features_names_ext: list[pd.Index] = field(default_factory=list)
    target_samples_ext: list[np.ndarray] = field(default_factory=list)
    shap_values_ext: list[np.ndarray] = field(default_factory=list)
    permutation_importance_ext: list[pd.DataFrame] = field(default_factory=list)

    def append_results(self, results: PredResults, feature_importance: bool, split_train_test: bool,
                       external_set: bool):
        """Append results from a single combination."""

        # Append predictions if they exist
        for attr, storage in [
            ("predictions_train", self.predictions_train),
            ("predictions_test", self.predictions_test),
            ("predictions_ext", self.predictions_ext),
        ]:
            value = getattr(results, attr, None)
            if value is not None and not value.empty:
                storage.append(value)

        # Define attributes to check
        attributes_to_check = ["roc_metrics_train", "features_names_train", "target_samples_train"]

        if split_train_test:
            attributes_to_check.extend(["roc_metrics_test", "features_names_test", "target_samples_test"])
            if external_set:
                attributes_to_check.extend(["roc_metrics_ext", "features_names_ext", "target_samples_ext"])
            if feature_importance:
                attributes_to_check.extend(["shap_values_train", "shap_values_test",
                                            "permutation_importance_train", "permutation_importance_test"])
                if external_set:
                    attributes_to_check.extend(["shap_values_ext", "permutation_importance_ext"])
        elif external_set:
            attributes_to_check.extend(["roc_metrics_ext", "features_names_ext", "target_samples_ext"])
            if feature_importance:
                attributes_to_check.extend(["shap_values_train", "shap_values_ext",
                                            "permutation_importance_train", "permutation_importance_ext"])
        elif feature_importance:
            attributes_to_check.extend(["shap_values_train", "permutation_importance_train"])

        # Append attributes if they exist in `results`
        for attr in attributes_to_check:
            value = getattr(results, attr, None)
            if value is not None:
                getattr(self, attr).append(value)


    def finalize_results(self):
        """Convert empty lists to None and return the results as SimpleNamespace ."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, list) and not field_value:  # Check if it's an empty list
                setattr(self, field_name, None)
        return self

class PerformanceValidator:
    def __init__(self, config, feature_manager,
                 estimator = None, estimator_name = None):

        # Parameter initializations with validations
        self.config = config
        self.feature_manager = feature_manager
        self.estimator_name = estimator_name if isinstance(estimator_name, str) else ""
        self.estimator_fn = None
        if estimator is not None:
            self.set_estimator(estimator)
        self.num_features = None
        self.num_oligos = None
        self.best_iteration = None

        # Default values from config
        self.libraries_prefixes = self.config.libraries_prefixes
        self.cv_method = self.config.cv_method #cv_method if cv_method in ("loo", "kfold") else "loo"
        self.split_train_test = self.config.split_train_test #split_train_test if isinstance(split_train_test, bool) else True
        self.compute_feature_importance = self.config.compute_feature_importance #compute_fpr_tpr_importance if isinstance(compute_fpr_tpr_importance, bool) else False
        self.return_train = self.config.return_train #return_train if isinstance(return_train, bool) else True
        self.return_test = self.config.return_test #return_test if isinstance(return_test, bool) else False
        self.external_set = self.config.external_set  #external_set if isinstance(external_set, bool) else False
        self.tuning_parameters = self.config.tuning_parameters #tuning_parameters if isinstance(tuning_parameters, bool) else True
        self.train_size = self.config.train_size #train_size if isinstance(train_size, (int, float)) else 0.8
        self.k = self.config.k #k if isinstance(k, int) else 10
        self.tuning_n_iter = self.config.tuning_n_iter #tuning_n_iter if isinstance(tuning_n_iter, int) else 20
        self.tuning_k = self.config.tuning_k #tuning_k if isinstance(tuning_k, int) else 3

    def set_estimator_fn(self, estimator_fn):
        if callable(estimator_fn):
            self.estimator_fn = estimator_fn
        else:
            raise ValueError("estimator_fn must be a callable function to instantiate a new estimator.")

    def set_estimator(self, estimator):
        """
        Validate and set an estimator by wrapping it in a lambda function.
        """
        if hasattr(estimator, "fit") and hasattr(estimator, "predict_proba"):
            # Wrap the estimator in a lambda to allow reinitialization
            self.set_estimator_fn(lambda: estimator)
            if self.estimator_name == "":
                self.estimator_name = estimator.__class__.__name__
        else:
            raise ValueError("The provided estimator must have 'fit' and 'predict_proba' methods.")

    def set_libraries_prefixes(self, libraries_prefixes):
        if isinstance(libraries_prefixes, list):
            self.libraries_prefixes = libraries_prefixes
        else:
            raise ValueError("The libraries_prefixes must be a list of strings.")

    def set_cv_method(self, cv_method):
        if cv_method == "loo" or cv_method == "kfold":
            self.cv_method = cv_method
        else:
            raise ValueError("cv_method must be either 'loo' or 'kfold'.")

    def set_k(self, k):
        if isinstance(k, int):
            self.k = k
        else:
            raise ValueError("k must be an integer.")

    def set_split_train_test(self, split_train_test):
        if isinstance(split_train_test, bool):
            self.split_train_test = split_train_test
        else:
            raise ValueError("split_train_test must be a boolean.")

    def set_feature_importance(self, feature_importance):
        if isinstance(feature_importance, bool):
            self.compute_feature_importance = feature_importance
        else:
            raise ValueError("feature_importance must be a boolean.")

    def set_train_size(self, train_size):
        if isinstance(train_size, (int, float)) and 0 < train_size < 1:
            self.train_size = train_size
        else:
            raise ValueError("train_size must be a numeric value between 0 and 1.")

    def _search_best_model(self, estimator, X_train, y_train):
        cv_inner = StratifiedKFold(n_splits=self.tuning_k, shuffle=True, random_state=self.config.random_state)
        random_search = RandomizedSearchCV(estimator, self.config.param_grid[self.estimator_name],
                                           cv=cv_inner, n_iter=self.tuning_n_iter,
                                           scoring='roc_auc', refit=True,
                                           random_state=self.config.random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        return best_model

    def _fit_estimator(self, estimator, X_train, y_train, X_val=None, y_val=None):
        try:
            if X_val is not None and y_val is not None:
                estimator.set_params(early_stopping_rounds=50, n_estimators=1000)
                estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                self.best_iteration.append(estimator.best_iteration)
            else:
                estimator.fit(X_train, y_train)
        except ValueError:
            estimator.fit(X_train, y_train)
        return estimator

    def _get_cross_validator(self, features, target):
        """Select and initialize the cross-validation method."""
        if self.cv_method == "loo":
            return LeaveOneOut().split(features)
        elif self.cv_method == "kfold":
            return StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.config.random_state).split(features,
                                                                                                               target)
        else:
            raise ValueError("Unsupported cross-validation method. Use 'loo' or 'kfold'.")

    def _train_model(self, X_train, y_train, X_test, y_test):
        """Train the model with or without hyperparameter tuning."""
        estimator = self.estimator_fn()
        if self.tuning_parameters:
            return self._search_best_model(estimator, X_train, y_train)
        return self._fit_estimator(estimator, X_train, y_train, X_test, y_test)

    @staticmethod
    def compute_interp_tpr_auc(y_true, y_pred_proba, mean_fpr):
        # Compute FPR and TPR
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

        # Interpolate TPR to the common mean FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # Ensure the curve starts at (0, 0)

        # Compute AUC
        auc_value = auc(fpr, tpr)

        return interp_tpr, auc_value

    @staticmethod
    def _calculate_mean_std_ci_tpr_auc(auc_list, tpr_list, mean_fpr, bootstrap=True):
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

    def _bootstrap_auc(self, mean_fpr=None, estimator=None, X=None, y_true=None, y_pred=None, n_bootstraps = 500):
        tpr_bootstraps = []
        auc_bootstraps = []

        if mean_fpr is None:
            mean_fpr = np.linspace(0, 1, 100)  # Define default mean_fpr here

        for i in range(n_bootstraps):
            if estimator is not None and X is not None and y_true is not None:
                X_resampled, y_resampled = resample(X, y_true, stratify=y_true, random_state=self.config.random_state + i)
                y_pred_resampled = estimator.predict_proba(X_resampled)[:, 1]
            elif y_pred is not None and y_true is not None:
                y_resampled, y_pred_resampled = resample(y_true, y_pred, stratify=y_true,
                                                         random_state=self.config.random_state + i)
            else:
                raise ValueError("Missing arguments. Estimator 'estimator', features 'X' and their true target 'y_true' must be provided for bootstrapping auc for test data."
                                 "For loocv, true target 'y_true' and predictions 'y_pred' must be provided.")

            interp_tpr, auc_value = self.compute_interp_tpr_auc(y_resampled, y_pred_resampled, mean_fpr)
            tpr_bootstraps.append(interp_tpr)
            auc_bootstraps.append(auc_value)

        roc_metrics = self._calculate_mean_std_ci_tpr_auc(auc_bootstraps, tpr_bootstraps, mean_fpr, bootstrap = True)

        return roc_metrics

    def _compute_feature_importance(self, estimator, X_test, y_test, shap_values_list, permutation_importance_list):
        """Compute SHAP and permutation importance."""
        explainer = shap.TreeExplainer(estimator)
        shap_values_list.append(explainer.shap_values(X_test))

        if self.cv_method == "kfold":
            perm_importances = permutation_importance(estimator, X_test, y_test, n_repeats=10,
                                                      random_state=self.config.random_state, n_jobs=-1).importances #None
            permutation_importance_list.append(perm_importances)

    @staticmethod
    def _aggregate_predictions(predict_proba_list, validation_indices_list):
        """Aggregate predictions and validation indices across folds."""
        predicted_probs_train = np.concatenate(predict_proba_list)
        validation_indices = np.concatenate(validation_indices_list)
        return predicted_probs_train, validation_indices

    def _compute_roc_metrics(self, target, validation_indices, predicted_probs_train, mean_fpr, auc_list, tpr_list):
        """Compute ROC metrics and handle KFold/LOOCV differences."""
        roc_metrics = self._bootstrap_auc(mean_fpr=mean_fpr, y_true=target.loc[validation_indices],
                                          y_pred=predicted_probs_train)

        if self.cv_method == "kfold":
            kfold_metrics = self._calculate_mean_std_ci_tpr_auc(auc_list, tpr_list, mean_fpr, bootstrap=False)
            roc_metrics.update(kfold_metrics)
        else:  # LOOCV
            fpr, tpr, _ = roc_curve(target.loc[validation_indices], predicted_probs_train)
            roc_metrics.update({"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr), "auc_std": None})

        return roc_metrics

    def _create_binary_class_predictions_df(self, predicted_probs, indices, target, _auc, auc_std):
        predictions_df = pd.DataFrame({
            self.config.col_predict: predicted_probs,
            'with_oligos':self.feature_manager.with_oligos, 'with_additional_features':self.feature_manager.with_additional_features,
            'with_run_plates':self.feature_manager.with_run_plates, 'estimator':self.estimator_name,
            'data_type':self.feature_manager.oligos_handler.data_type,
            'subgroup':self.feature_manager.subgroup,
            'filter_by_correlation':self.feature_manager.filter_by_correlation,
            'filter_by_entropy':self.feature_manager.filter_by_entropy, 'prevalence_threshold_min':self.feature_manager.prevalence_threshold_min,
            'subgroup_name': self.config.subgroups_to_name.get(self.feature_manager.subgroup),
            'auc': _auc, 'std': auc_std,
            'num_features': self.num_features, 'num_oligos': self.num_oligos,
        }, index=pd.MultiIndex.from_arrays([target.loc[indices], indices],
                                           names=[self.config.col_target, self.config.col_sample_name]))

        predictions_df = predictions_df.set_index(
            ['with_oligos', 'with_additional_features', 'with_run_plates',
             'estimator', 'data_type', 'subgroup',
             'filter_by_correlation', 'filter_by_entropy', 'prevalence_threshold_min'
            ,'subgroup_name', 'auc', 'std', 'num_features', 'num_oligos'
             ],
            append=True).reorder_levels([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1])
        return predictions_df

    @staticmethod
    def _compute_permutation_importance_summary(permutation_importances, feature_names):
        """
        Compute a summary DataFrame for permutation importance, including mean and std.

        Parameters:
        - permutation_importance (list or Dict-like object from ski-learn features_importance): if a List of np.ndarray, must contain raw permutation scores for each fold,
                                              each of shape (n_features, n_repeats).
        - feature_names (pd.index or np.ndarray): Array of feature names corresponding to the columns of the input data.

        Returns:
        - pd.DataFrame: A DataFrame with raw permutation scores, mean importance, and std importance.
        """
        if isinstance(permutation_importances, list):
            # Compute mean and std importance across folds and repeats
            importances = np.concatenate(permutation_importances, axis=1)  # Shape: (n_features, n_folds * n_repeats)
            importances_mean = importances.mean(axis=1)  # Shape: (n_features,)
            importances_std = importances.std(axis=1)  # Shape: (n_features,)
        else:
            importances = permutation_importances.importances
            importances_mean = permutation_importances.importances_mean
            importances_std = permutation_importances.importances_std

        # Sort features by descending mean importance
        sorted_indices = importances_mean.argsort()
        importances_df = pd.DataFrame(
            importances[sorted_indices].T,  # Transpose to make features columns
            columns=feature_names[sorted_indices]  # Feature names as columns
        )

        # Add mean and std importance as rows
        importances_df.loc["mean_importance"] = importances_mean[sorted_indices]
        importances_df.loc["std_importance"] = importances_std[sorted_indices]

        return importances_df

    def _process_feature_importance_results(self, shap_values_list, permutation_importance_list, feature_names):
        """Process and return feature importance results."""
        if self.compute_feature_importance:
            shap_values = np.concatenate(shap_values_list, axis=0)
            if self.cv_method == "kfold":
                permutation_importances = self._compute_permutation_importance_summary(permutation_importance_list, feature_names) #None
            else:
                permutation_importances = None
            return shap_values, permutation_importances
        return None, None

    def binaryClassification_train_val_predictions(self, features, target):
        """Perform binary classification with cross-validation and return metrics and predictions."""

        # Initialize cross-validation
        cv_outer = self._get_cross_validator(features, target)

        # Initialize result containers
        mean_fpr = np.linspace(0, 1, 100)
        validation_indices_list, predict_proba_list = [], []
        auc_list, tpr_list = [], []
        shap_values_list, permutation_importance_list = [], []
        self.best_iteration = []

        # Perform cross-validation
        for train_index, test_index in cv_outer:
            X_train_fold, X_test_fold = features.iloc[train_index], features.iloc[test_index]
            y_train_fold, y_test_fold = target.iloc[train_index], target.iloc[test_index]

            # Train model
            estimator = self._train_model(X_train_fold, y_train_fold, X_test_fold, y_test_fold)

            # Collect predictions
            y_pred_proba_val = estimator.predict_proba(X_test_fold)[:, 1]
            predict_proba_list.append(y_pred_proba_val)
            validation_indices_list.append(X_test_fold.index)

            # Compute AUC and TPR for KFold
            if self.cv_method == "kfold":
                interp_tpr, auc_value = self.compute_interp_tpr_auc(y_test_fold, y_pred_proba_val, mean_fpr)
                tpr_list.append(interp_tpr)
                auc_list.append(auc_value)

            # Compute feature importance (SHAP and permutation importance)
            if self.compute_feature_importance:
                self._compute_feature_importance(estimator, X_test_fold, y_test_fold, shap_values_list,
                                                 permutation_importance_list)

        if not self.tuning_parameters:
            self.best_iteration = {'n_estimators':int(np.mean(self.best_iteration))} if self.best_iteration else None

        # Aggregate results
        predicted_probs_train, validation_indices = self._aggregate_predictions(predict_proba_list, validation_indices_list)
        roc_metrics = self._compute_roc_metrics(target, validation_indices, predicted_probs_train, mean_fpr, auc_list,
                                                tpr_list)

        # Create predictions DataFrame
        predictions_train = self._create_binary_class_predictions_df(predicted_probs_train, validation_indices, target,
                                                                     roc_metrics.get("auc"), roc_metrics.get("auc_std"))

        # Handle feature importance results
        shap_values_train, permutation_importance_train = self._process_feature_importance_results(shap_values_list,
                                                                                                    permutation_importance_list,
                                                                                                    features.columns)

        return predictions_train, roc_metrics, features.columns, validation_indices,  shap_values_train, permutation_importance_train

    def _initialize_estimator_test(self):
        """Initialize and reset the estimator for testing."""
        estimator = self.estimator_fn()
        try:
            estimator.set_params(early_stopping_rounds=None)
        except ValueError:
            pass
        return estimator

    def _fit_estimator_with_best_iteration(self, estimator, X_train, y_train):
        """Fit the estimator with the best iteration parameters if available."""
        if self.best_iteration:
            estimator.set_params(**self.best_iteration)
        estimator.fit(X_train, y_train, verbose=False)

    @staticmethod
    def _align_features(X_train, X_ext):
        """Align features between training and external test sets."""
        common_features = list(set(X_train.columns).intersection(X_ext.columns))
        if common_features:
            X_train = X_train[common_features]
            X_ext = X_ext[common_features]
            return X_train, X_ext
        else:
            raise ValueError(f"Training and External set shared no common features")

    def _compute_roc_metrics_test(self, estimator, X_test, y_test, predicted_probs_test):
        """Compute ROC metrics for the test set."""
        fpr, tpr, _ = roc_curve(y_test, predicted_probs_test)
        roc_metrics = self._bootstrap_auc(estimator=estimator, X=X_test, y_true=y_test)
        roc_metrics.update({"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr), "auc_std": None})

        return roc_metrics

    def _compute_feature_importance_test(self, estimator, X_test, y_test):
        """Compute SHAP values and permutation importance for the test set."""
        explainer = shap.TreeExplainer(estimator)
        shap_values_test = explainer.shap_values(X_test)

        permutation_importances = permutation_importance(
            estimator, X_test, y_test, n_repeats=10, random_state=self.config.random_state, n_jobs=-1
        ) #None
        permutation_importance_test = self._compute_permutation_importance_summary(permutation_importances,
                                                                                   X_test.columns) #None

        return shap_values_test, permutation_importance_test

    def binaryClassification_test_predictions(self, X_train, y_train, X_test, y_test):

        # Ensure features in training and external sets are aligned
        if self.external_set:
            X_train, X_test = self._align_features(X_train, X_test)

        estimator = self._initialize_estimator_test()
        # Tune the model or set the best iteration parameters
        if self.tuning_parameters:
            estimator = self._search_best_model(estimator, X_train, y_train)
        else:
            self._fit_estimator_with_best_iteration(estimator, X_train, y_train)

        # Make predictions and compute ROC metrics
        predicted_probs_test = estimator.predict_proba(X_test)[:, 1]
        roc_metrics_test = self._compute_roc_metrics_test(estimator, X_test, y_test, predicted_probs_test)

        # Create predictions DataFrame
        predictions_test = self._create_binary_class_predictions_df(predicted_probs_test, y_test.index, y_test,
                                                                    roc_metrics_test.get("auc"), roc_metrics_test.get("auc_std"))

        # Compute feature importance (if applicable)
        shap_values_test, permutation_importance_test = None, None
        if self.compute_feature_importance:
            shap_values_test, permutation_importance_test = self._compute_feature_importance_test(estimator, X_test,
                                                                                                  y_test)

        return predictions_test,  roc_metrics_test, X_test.columns, y_test.index, shap_values_test, permutation_importance_test

    @staticmethod
    def _collect_results(results, dataset_type, values):
        """
        Collect and assign predictions, metrics, feature names, targets, and SHAP values
        for a specific dataset type (train/test/ext).
        """
        (
            predictions,
            roc_metrics,
            features_names,
            target_samples,
            shap_values,
            permutation_importances
        ) = values

        setattr(results, f"predictions_{dataset_type}", predictions)
        setattr(results, f"roc_metrics_{dataset_type}", roc_metrics)
        setattr(results, f"features_names_{dataset_type}", features_names)
        setattr(results, f"target_samples_{dataset_type}", target_samples)
        setattr(results, f"shap_values_{dataset_type}", shap_values)
        setattr(results, f"permutation_importance_{dataset_type}", permutation_importances)

    def get_binaryClassification(self, X_ext=None, y_ext=None):
        try:
            # Initialize results and retrieve features/target
            results = PredResults()
            features, target = self.feature_manager.get_features_target()

            # Skip processing if features are empty
            if features.empty:
                return results

            # Align features for external set if provided
            if self.external_set and X_ext is not None and y_ext is not None:
                features, X_ext = self._align_features(features, X_ext)

            # Set feature metadata
            self.num_features = len(features.columns)
            self.num_oligos = sum(features.columns.str.contains('|'.join(self.libraries_prefixes)))

            # Perform train/test split if required
            if self.split_train_test:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, train_size=self.train_size,
                    random_state=self.config.random_state, shuffle=True, stratify=target
                )

                # Train results
                if self.return_train:
                    self._collect_results(
                        results, "train",
                        self.binaryClassification_train_val_predictions(X_train, y_train)
                    )

                # Test results
                if self.return_test:
                    self._collect_results(
                        results, "test",
                        self.binaryClassification_test_predictions(X_train, y_train, X_test, y_test)
                    )

                # External set results
                if self.external_set and X_ext is not None and y_ext is not None:
                    self._collect_results(
                        results, "ext",
                        self.binaryClassification_test_predictions(X_train, y_train, X_ext, y_ext)
                    )
            else:
                # Cross-validation or external set without train/test split
                if self.return_train:
                    self._collect_results(
                        results, "train",
                        self.binaryClassification_train_val_predictions(features, target)
                    )
                if self.external_set and X_ext is not None and y_ext is not None:
                    self._collect_results(
                        results, "ext",
                        self.binaryClassification_test_predictions(features, target, X_ext, y_ext)
                    )

            return results

        except Exception as error:
            logging.error(f"Error during cross-validation: {error}")
            return PredResults()


class ModelRunner:
    def __init__(self, config, feature_manager, performance_validator):
        self.config = config
        self.feature_manager = feature_manager
        self.performance_validator = performance_validator

    @staticmethod
    def _log_progress(current_progress, total_combinations):
        """Log the progress of the current model run."""
        percentage_done = (current_progress / total_combinations) * 100
        progress_bar = f"[{'=' * int(percentage_done // 2)}{' ' * (50 - int(percentage_done // 2))}]"
        logging.info(f"Progress: {progress_bar} {percentage_done:.2f}%")


    def _calculate_total_combinations(self):
        """Calculate the total number of combinations for progress tracking."""
        return sum(
            len(list(product(self.config.data_types, self.config.subgroups_to_include,
                             self.config.prevalence_thresholds_min, self.config.filter_by_correlation,
                             self.config.filter_by_entropy)))
            if with_oligos else 1
            for with_oligos, _, _ in product(
                self.config.with_oligos_options, self.config.with_additional_features_options,
                self.config.with_run_plates_options)
        )

    def _set_feature_manager_parameters(self, **kwargs):
        """Set parameters for the feature manager."""
        for param, value in kwargs.items():
            setter = getattr(self.feature_manager, f"set_{param}", None)
            if setter:
                setter(value)

    def process_single_combination_binaryClassification(self, data_type, subgroup, prevalence_threshold_min,
                                                        filter_by_correlation, filter_by_entropy,
                                                        with_oligos, with_additional_features, with_run_plates,
                                                        X_ext = None, y_ext = None):
        # Set feature manager parameters
        self._set_feature_manager_parameters(
            data_type=data_type, subgroup=subgroup,
            prevalence_threshold_min=prevalence_threshold_min,
            filter_by_correlation=filter_by_correlation,
            filter_by_entropy=filter_by_entropy,
            with_oligos=with_oligos, with_additional_features=with_additional_features,
            with_run_plates=with_run_plates
        )

        return self.performance_validator.get_binaryClassification(X_ext, y_ext)



    def _process_combinations_binaryClassification(self, with_oligos, with_additional_features, with_run_plates,
                                                   results_aggregator, total_combinations, current_progress,
                                                   X_ext=None, y_ext=None):
        """Process combinations for binary classification."""
        if with_oligos:
            logging.info("Running models with data that contain peptides...")
            param_grid = product(self.config.data_types, self.config.subgroups_to_include,
                                 self.config.prevalence_thresholds_min, self.config.filter_by_correlation,
                                 self.config.filter_by_entropy)
        else:
            logging.info("Running models with data that do not contain peptides...")
            param_grid = [("exist", "all", 0.0, False, False)]

        for data_type, subgroup, prevalence_threshold_min, filter_by_correlation, filter_by_entropy in param_grid:
            self.process_single_combination_binaryClassification(data_type, subgroup, prevalence_threshold_min,
                                                                 filter_by_correlation, filter_by_entropy,
                                                                 with_oligos, with_additional_features, with_run_plates,
                                                                 X_ext, y_ext)
            # Process combination and append results
            results = self.performance_validator.get_binaryClassification(X_ext, y_ext)
            current_progress += 1
            self._log_progress(current_progress, total_combinations)
            results_aggregator.append_results(
                results,
                feature_importance=self.performance_validator.compute_feature_importance,
                split_train_test=self.performance_validator.split_train_test,
                external_set=self.performance_validator.external_set
            )

        return current_progress

    def get_all_model_predictions_binaryClassification(self, X_ext: Optional[pd.DataFrame] = None,
                                                       y_ext: Optional[pd.Series] = None):
        """Run binary classification for all parameter combinations."""
        total_combinations = self._calculate_total_combinations()
        current_progress = 0
        results_aggregator = CombinedPredResults()

        for with_oligos, with_additional_features, with_run_plates in product(
                self.config.with_oligos_options, self.config.with_additional_features_options,
                self.config.with_run_plates_options):
            # Skip combinations where all flags are False
            if not (with_oligos or with_run_plates or with_additional_features):
                continue
            current_progress = self._process_combinations_binaryClassification(
                with_oligos, with_additional_features, with_run_plates,
                results_aggregator, total_combinations, current_progress,
                X_ext, y_ext,
            )

        return results_aggregator.finalize_results()

class PredictionManager:
    def __init__(self, config):
        self.config = config

    def save_predictions_summary(self, summary_df, estimator_name, dataset):
        """
        Filters the summary DataFrame based on the given parameters and saves the filtered result to a CSV file.

        Args:
            summary_df (pd.DataFrame): The summary DataFrame to filter and save.
            estimator_name (str): The name of the estimator for file naming.
            dataset (str): trainSet or testSet
        """
        for with_oligos, with_additional_features, with_run_plates in product(
                self.config.with_oligos_options, self.config.with_additional_features_options,
                self.config.with_run_plates_options):
            if not (with_oligos or with_additional_features or with_run_plates):
                continue

                # Define filter index based on parameters
            filter_index = (
                slice(None),  # Select all 'model_id' values
                with_oligos,
                with_additional_features,
                with_run_plates,
                estimator_name,
                slice(None),  # Select all 'data_type' values
                slice(None),  # Select all 'subgroup' values
                slice(None),  # Select all 'filter_by_correlation' values
                slice(None),  # Select all 'filter_by_entropy' values
                slice(None)  # Select all 'prevalence_threshold' values
            )

            # Filter and reorder summary DataFrame
            try:
                filtered_summary = summary_df.loc[filter_index].unstack(1).T.reorder_levels([1, 0]).sort_index(level=0,
                                                                                                               sort_remaining=False).T


                filename = f"predictions_{dataset}Set_summary_{estimator_name}_{self.config.label_group_tests}"
                if with_oligos:
                    filename += "_with_oligos"
                if with_additional_features:
                    filename += "_with_additional_features"
                if with_run_plates:
                    filename += "_with_run_plates"
                filename += ".csv"

                # Save filtered DataFrame to CSV
                csv_path = Path(self.config.predictions_dir) / filename
                filtered_summary.to_csv(csv_path)
                logging.info(f"Saved {dataset} subset summary predictions csv files here: {csv_path}")

            except Exception as e:
                # Log the error and continue
                logging.error(f"Could not subset and save summary predictions for {dataset}: {e}")

    def summarize_models(self, model_predictions, estimator_name, dataset):
        if dataset == "train":
            predictions = pd.concat(model_predictions.predictions_train)
        elif dataset == "test":
            predictions = pd.concat(model_predictions.predictions_test)
        elif dataset == "ext":
            predictions = pd.concat(model_predictions.predictions_ext)
        else:
            raise ValueError(
                f"dataset '{dataset}' must be 'train', 'test' or 'ext' string.")

        summary_df = predictions.reset_index(
            [self.config.col_target, self.config.col_sample_name,
             'subgroup_name', 'auc', 'std', 'num_features', 'num_oligos']).drop(columns=[self.config.col_predict, self.config.col_sample_name, self.config.col_target])

        # Add a unique 'model_id' for each unique combination of the current MultiIndex
        summary_df['model_id'] = summary_df.groupby(level=summary_df.index.names, sort=False).ngroup()
        summary_df = summary_df.set_index('model_id', append=True)
        summary_df = summary_df.drop_duplicates().reorder_levels(['model_id'] + summary_df.index.names[:-1])

        # Save summary_df in csv and map metric to the 'model_id' in summary_df
        self.save_predictions_summary(summary_df, estimator_name, dataset)

        # Determine dataset and dynamically select attributes
        roc_metrics = {i: roc_metric for i, roc_metric in enumerate(getattr(model_predictions, f'roc_metrics_{dataset}'))}
        summary_df['roc_metrics'] = summary_df.index.get_level_values('model_id').map(roc_metrics)

        csv_path = Path(self.config.predictions_dir) / f"{estimator_name}_{self.config.label_group_tests}_{dataset}Set_summary_withROC_metrics.pkl"
        summary_df.to_pickle(csv_path)
        logging.info(f"Saved {dataset} set summary pkl file here: {csv_path}")

        return summary_df

    def _process_model_predictions(self, estimator_name, model_predictions, summarized_predictions):
        """Save and summarize predictions for train, test, and external datasets."""
        datasets = ["train", "test", "ext"]
        for dataset in datasets:
            predictions_attr = f"predictions_{dataset}"
            predictions = getattr(model_predictions, predictions_attr, None)

            if predictions is not None:
                # Save raw predictions to CSV
                csv_path = Path(
                    self.config.predictions_dir) / f"{estimator_name}_{self.config.label_group_tests}_predictions_{dataset}Set.csv"
                pd.concat(predictions).to_csv(csv_path)
                logging.info(f"Saved {dataset} set predictions csv file here: {csv_path}")

                # Summarize predictions and update model_predictions
                summarized = self.summarize_models(model_predictions, estimator_name, dataset=dataset)
                setattr(model_predictions, predictions_attr, summarized)

        summarized_predictions[estimator_name] = model_predictions

    def _save_results(self, predictions, summarized_predictions):
        """Save predictions and summarized predictions as joblib files."""
        predictions_path = Path(
            self.config.predictions_dir) / f"{self.config.label_group_tests}_all_model_predictions.joblib"
        summarized_path = Path(
            self.config.predictions_dir) / f"{self.config.label_group_tests}_all_model_summarized_predictions.joblib"

        dump(predictions, predictions_path)
        logging.info(f"Saved all model predictions here: {predictions_path}")

        dump(summarized_predictions, summarized_path)
        logging.info(f"Saved summarized model predictions here: {summarized_path}")


    def get_model_predictions_binaryClassification(self, X_ext = None, y_ext = None, return_predictions = True):
        predictions = {}
        summarized_predictions = {}
        for estimator_name, estimator_info in self.config.estimators_info.items():
            logging.info(f"Starting predictions for estimator: {estimator_name}")

            # Initialize components
            estimator = estimator_info['estimator_class'](**estimator_info['estimator_kwargs'])
            metadata_handler = MetadataHandler(self.config)
            oligos_handler = OligosHandler(self.config)
            feature_manager = FeatureManager(self.config, metadata_handler, oligos_handler)
            performance_validator = PerformanceValidator(self.config, feature_manager, estimator=estimator)
            model_runner = ModelRunner(self.config, feature_manager, performance_validator)

            # Generate model predictions
            model_predictions = model_runner.get_all_model_predictions_binaryClassification(X_ext, y_ext)

            predictions[estimator_name] = copy.deepcopy(model_predictions)

            # Save and summarize predictions for each dataset type
            self._process_model_predictions(estimator_name, model_predictions, summarized_predictions)

            predictions[estimator_name] = SimpleNamespace(**predictions[estimator_name].__dict__)
            summarized_predictions[estimator_name] = SimpleNamespace(**summarized_predictions[estimator_name].__dict__)

            logging.info(f"Finished predictions for estimator: {estimator_name}")

        # Save results
        self._save_results(predictions, summarized_predictions)

        return predictions, summarized_predictions if return_predictions else 0
