"""Command-line entry point for PhIP-seq survival modelling."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from phipsurv.io.data_handler import Config
from phipsurv.survival.helpers import (
    build_pipeline,
    nested_cv,
    train_and_validate_model,
)
from phipsurv.survival.train_test_utils import (
    SurvivalRunSettings,
    process_survival_data,
    setup_feature_manager,
)

logger = logging.getLogger(__name__)
MODEL_LABEL = "XGB-survivalCox"


class _ArgParser(argparse.ArgumentParser):
    """Argument parser supporting one shell-like option per @args-file line."""

    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        line = arg_line.split("#", 1)[0].strip()
        return shlex.split(line) if line else []


def _json_mapping(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Invalid JSON mapping: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def parse_args_survival(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI overrides; unspecified options remain available from YAML."""
    parser = _ArgParser(
        description="Train and validate a PhIP-seq survival model.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--config", "-c", required=True, help="YAML config path")

    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument(
        "--run-nested-cv",
        "--run_nested_cv",
        dest="run_nested_cv",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-pretrained",
        "--use_pretrained",
        dest="use_pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--only-train-model",
        "--only_train_model",
        dest="only_train_model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument("--subgroup", "-sub", default=None)
    parser.add_argument(
        "--with-oligos",
        "--with_oligos",
        dest="with_oligos",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--with-additional-features",
        "--with_additional_features",
        dest="with_additional_features",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--prevalence-threshold-min",
        "--prevalence_threshold_min",
        "-min",
        dest="prevalence_threshold_min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--prevalence-threshold-max",
        "--prevalence_threshold_max",
        "-max",
        dest="prevalence_threshold_max",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--outer-cv-splits",
        "--outer_cv_split",
        "-ocv",
        dest="outer_cv_splits",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--inner-cv-splits",
        "--inner_cv_split",
        "-icv",
        dest="inner_cv_splits",
        type=int,
        default=None,
    )
    parser.add_argument("--n-iter", dest="n_iter", type=int, default=None)
    parser.add_argument(
        "--max-time-point",
        "--max_timepoint",
        "-maxT",
        dest="max_time_point",
        type=float,
        default=None,
    )
    parser.add_argument("--n-jobs-outer", type=int, default=None)
    parser.add_argument("--n-jobs-inner", type=int, default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--param-grid-name", default=None)

    parser.add_argument(
        "--impute-extra-numeric",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--extra-numeric-impute-strategy",
        choices=("mean", "median", "most_frequent", "constant"),
        default=None,
    )

    parser.add_argument("--train", "-t", type=_json_mapping, default=None)
    parser.add_argument(
        "--validate",
        "-v",
        nargs=2,
        action="append",
        default=None,
        metavar=("FILTER_JSON", "OUTPUT_NAME"),
        help='Override YAML validations, e.g. -v \'{"treatment":"ICI"}\' HCC-ICI',
    )

    parser.add_argument("--input-dir", "--input_dir", "-id", default=None)
    parser.add_argument(
        "--output-dir", "--out_dir", "-d", dest="output_dir", default=None
    )
    parser.add_argument("--input-name", "--input_name", "-i", default=None)
    parser.add_argument(
        "--output-name", "--out_name", "-o", dest="output_name", default=None
    )
    return parser.parse_args(argv)


def _save_result(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    logger.info("Saved %s", path)


def _load_pretrained(settings: SurvivalRunSettings) -> Pipeline:
    supplied = Path(settings.input_name)
    candidates = []
    if supplied.suffix == ".joblib":
        candidates.append(
            supplied if supplied.is_absolute() else settings.input_dir / supplied
        )
    candidates.extend(
        [
            settings.input_dir
            / f"training_{MODEL_LABEL}_{settings.input_name}_{settings.seed}.joblib",
            settings.input_dir
            / f"validation_{MODEL_LABEL}_{settings.input_name}_{settings.seed}.joblib",
        ]
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "Could not find a pretrained model. Tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    loaded = joblib.load(path)
    estimator = loaded.get("best_estimator") if isinstance(loaded, dict) else loaded
    if not isinstance(estimator, Pipeline):
        raise TypeError(f"No sklearn Pipeline found in {path}")
    logger.info("Loaded pretrained model from %s", path)
    return estimator


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args_survival(argv)
    config = Config(args.config)
    settings = SurvivalRunSettings.from_sources(config, args)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing training cohort")
    train_manager = setup_feature_manager(
        config,
        settings.train_filters,
        settings,
    )
    X_train, y_time_train, _ = process_survival_data(
        train_manager,
        time_column=settings.time_column,
    )

    needs_fitted_model = settings.only_train_model or bool(settings.validation_sets)
    if not settings.run_nested_cv and not needs_fitted_model:
        logger.info("Nothing to run: nested CV, training, and validation are disabled")
        return 0

    pipeline: Pipeline | None = None
    param_grid: dict[str, Any] | None = None
    if settings.run_nested_cv or not settings.use_pretrained:
        pipeline = build_pipeline(
            X_train,
            random_state=settings.seed,
            peptide_prefixes=config.peptide_prefixes,
            impute_extra_numeric=settings.impute_extra_numeric,
            extra_numeric_impute_strategy=settings.extra_numeric_impute_strategy,
        )
        param_grid = config.get_bayesian_param_grid(settings.param_grid_name)

    if settings.run_nested_cv:
        if pipeline is None:
            raise RuntimeError("Nested CV requires a pipeline")
        started = time.perf_counter()
        nested_results = nested_cv(
            X_train,
            y_time_train,
            pipeline=pipeline,
            param_grid=param_grid,
            n_splits=settings.outer_cv_splits,
            n_splits_inner=settings.inner_cv_splits,
            n_iter=settings.n_iter,
            max_time_point=settings.max_time_point,
            random_state=settings.seed,
            n_jobs=settings.n_jobs_outer,
            n_jobs_inner=settings.n_jobs_inner,
        )
        keys = (
            "model_list",
            "train_shap_values",
            "risk_scores_train",
            "validation_indices_train",
            "time_dependent_auc_train",
            "c_index_train",
            "mean_auc_train",
        )
        _save_result(
            dict(zip(keys, nested_results)),
            settings.output_dir
            / f"nested_{MODEL_LABEL}_{settings.output_name}_{settings.seed}.joblib",
        )
        logger.info(
            "Nested CV completed in %.2f seconds", time.perf_counter() - started
        )

    if not needs_fitted_model:
        return 0

    started = time.perf_counter()
    if settings.use_pretrained:
        best_estimator = _load_pretrained(settings)
    else:
        if pipeline is None:
            raise RuntimeError("Model training requires a pipeline")
        best_estimator = train_and_validate_model(
            X_train,
            y_time_train,
            pipeline=pipeline,
            param_grid=param_grid,
            n_splits=settings.outer_cv_splits,
            n_iter=settings.n_iter,
            random_state=settings.seed,
            n_jobs=settings.n_jobs_inner,
            get_only_model=True,
        )
        if not isinstance(best_estimator, Pipeline):
            raise RuntimeError("Model training did not return a fitted Pipeline")
    logger.info("Model ready in %.2f seconds", time.perf_counter() - started)

    if settings.only_train_model:
        _save_result(
            {"best_estimator": best_estimator},
            settings.output_dir
            / f"training_{MODEL_LABEL}_{settings.output_name}_{settings.seed}.joblib",
        )
        return 0

    for validation in settings.validation_sets:
        started = time.perf_counter()
        validation_manager = setup_feature_manager(
            config,
            validation.filters,
            settings,
            validation=True,
        )
        X_test, y_time_test, _ = process_survival_data(
            validation_manager,
            time_column=settings.time_column,
        )
        validation_result = train_and_validate_model(
            X_train,
            y_time_train,
            X_test=X_test,
            y_time_test=y_time_test,
            best_estimator=best_estimator,
            max_time_point=settings.max_time_point,
            random_state=settings.seed,
            get_only_model=False,
            peptide_prefixes=config.peptide_prefixes,
            return_feature_report=False,
        )
        if not isinstance(validation_result, tuple):
            raise RuntimeError(f"Validation {validation.name!r} returned no result")
        keys = (
            "best_estimator",
            "test_shap_values",
            "risk_scores_test",
            "time_dependent_auc_test",
            "time_dependent_auc_mean_test",
            "c_index_test",
            # ,"feature_report",
        )
        _save_result(
            dict(zip(keys, validation_result)),
            settings.output_dir
            / f"validation_{MODEL_LABEL}_{validation.name}_{settings.seed}.joblib",
        )
        logger.info(
            "Validation %s completed in %.2f seconds",
            validation.name,
            time.perf_counter() - started,
        )
    return 0



if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
