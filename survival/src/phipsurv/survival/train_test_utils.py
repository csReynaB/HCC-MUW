"""Utilities shared by the survival-training CLI and notebooks."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from phipsurv.io.data_handler import (
    FeatureManager,
    MetadataHandler,
    OligosHandler,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationSpec:
    """Metadata filter and output label for one validation cohort."""

    filters: dict[str, Any]
    name: str


@dataclass(frozen=True)
class SurvivalRunSettings:
    """Validated execution settings resolved from YAML and CLI overrides."""

    seed: int
    run_nested_cv: bool
    use_pretrained: bool
    only_train_model: bool
    subgroup: str
    with_oligos: bool
    with_additional_features: bool
    prevalence_threshold_min: float
    prevalence_threshold_max: float
    outer_cv_splits: int
    inner_cv_splits: int
    n_iter: int
    max_time_point: float | None
    n_jobs_outer: int
    n_jobs_inner: int
    time_column: str
    param_grid_name: str
    impute_extra_numeric: bool
    extra_numeric_impute_strategy: str
    input_dir: Path
    output_dir: Path
    input_name: str
    output_name: str
    train_filters: dict[str, Any] | None
    validation_sets: tuple[ValidationSpec, ...]

    @classmethod
    def from_sources(cls, config: Any, args: Any) -> "SurvivalRunSettings":
        """Resolve defaults < YAML ``survival`` section < explicit CLI values."""
        raw = getattr(config, "survival", None) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("The YAML 'survival' section must be a mapping")

        def choose(setting_name: str, default: Any) -> Any:
            cli_value = getattr(args, setting_name, None)
            return (
                cli_value if cli_value is not None else raw.get(setting_name, default)
            )

        config_path = Path(config.config_file).expanduser().resolve()
        # config_dir = config.config_file.parent
        def resolve_path(value: str | Path) -> Path:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            return path.resolve()

        cli_train = getattr(args, "train", None)
        train_filters = cli_train if cli_train is not None else raw.get("train_filters")
        if train_filters is not None and not isinstance(train_filters, Mapping):
            raise TypeError("train_filters must be a mapping or null")

        cli_validations: list[list[str]] | None = getattr(
            args,
            "validate",
            None,
        )

        if cli_validations is not None:
            parsed_validations: list[ValidationSpec] = []
            for filter_json, name in cli_validations:
                filters = json.loads(filter_json)
                if not isinstance(filters, Mapping):
                    raise TypeError("Each --validate filter must be a JSON object")
                parsed_validations.append(
                    ValidationSpec(filters=dict(filters), name=name)
                )
            validation_sets = tuple(parsed_validations)
        else:
            validation_sets = cls._parse_validation_sets(raw.get("validation_sets", []))

        settings = cls(
            seed=int(choose("seed", getattr(config, "random_state", 420))),
            run_nested_cv=cls._as_bool(choose("run_nested_cv", True), "run_nested_cv"),
            use_pretrained=cls._as_bool(
                choose("use_pretrained", False), "use_pretrained"
            ),
            only_train_model=cls._as_bool(
                choose("only_train_model", False), "only_train_model"
            ),
            subgroup=str(choose("subgroup", "all")),
            with_oligos=cls._as_bool(choose("with_oligos", True), "with_oligos"),
            with_additional_features=cls._as_bool(
                choose("with_additional_features", False),
                "with_additional_features",
            ),
            prevalence_threshold_min=float(choose("prevalence_threshold_min", 2.0)),
            prevalence_threshold_max=float(choose("prevalence_threshold_max", 98.0)),
            outer_cv_splits=int(choose("outer_cv_splits", 5)),
            inner_cv_splits=int(choose("inner_cv_splits", 5)),
            n_iter=int(choose("n_iter", 30)),
            max_time_point=cls._optional_float(choose("max_time_point", None)),
            n_jobs_outer=int(choose("n_jobs_outer", 1)),
            n_jobs_inner=int(choose("n_jobs_inner", -1)),
            time_column=str(choose("time_column", "OS months")),
            param_grid_name=str(choose("param_grid_name", "xgboost")),
            impute_extra_numeric=cls._as_bool(
                choose("impute_extra_numeric", False),
                "impute_extra_numeric",
            ),
            extra_numeric_impute_strategy=str(
                choose("extra_numeric_impute_strategy", "median")
            ),
            input_dir=resolve_path(choose("input_dir", ".")),
            output_dir=resolve_path(choose("output_dir", ".")),
            input_name=str(choose("input_name", "input_name")),
            output_name=str(choose("output_name", "out_name")),
            train_filters=(dict(train_filters) if train_filters is not None else None),
            validation_sets=validation_sets,
        )
        settings.validate()
        return settings

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _as_bool(value: Any, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if value in (0, 1):
            return bool(value)
        raise TypeError(f"{name} must be a YAML/CLI Boolean, not {value!r}")

    @staticmethod
    def _parse_validation_sets(raw: Any) -> tuple[ValidationSpec, ...]:
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise TypeError("survival.validation_sets must be a list")

        parsed: list[ValidationSpec] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise TypeError(f"validation_sets[{index}] must be a mapping")
            name = item.get("name")
            filters = item.get("filters")
            if not isinstance(name, str) or not name:
                raise ValueError(f"validation_sets[{index}].name must be non-empty")
            if not isinstance(filters, Mapping):
                raise TypeError(f"validation_sets[{index}].filters must be a mapping")
            parsed.append(ValidationSpec(filters=dict(filters), name=name))
        return tuple(parsed)

    def validate(self) -> None:
        if not (0 <= self.prevalence_threshold_min <= 100):
            raise ValueError("prevalence_threshold_min must be between 0 and 100")
        if not (0 <= self.prevalence_threshold_max <= 100):
            raise ValueError("prevalence_threshold_max must be between 0 and 100")
        if self.prevalence_threshold_min > self.prevalence_threshold_max:
            raise ValueError("Minimum prevalence cannot exceed maximum prevalence")
        if self.outer_cv_splits < 2 or self.inner_cv_splits < 2:
            raise ValueError("Both CV split counts must be at least 2")
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if self.max_time_point is not None and self.max_time_point <= 1:
            raise ValueError("max_time_point must be greater than 1 or null")
        if self.extra_numeric_impute_strategy not in {
            "mean",
            "median",
            "most_frequent",
            "constant",
        }:
            raise ValueError("Unsupported extra_numeric_impute_strategy")
        if self.n_jobs_outer != 1 and self.n_jobs_inner != 1:
            logger.warning(
                "Both outer and inner CV parallelism are enabled; this can "
                "oversubscribe CPUs and memory."
            )


def process_survival_data(
    feature_manager: FeatureManager,
    *,
    time_column: str = "OS months",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build X and signed survival time from the configured event target."""
    features, event = feature_manager.get_features_target()
    if time_column not in features.columns:
        raise KeyError(
            f"Survival time column {time_column!r} is missing. Add it to "
            "extra_features_to_include and enable with_additional_features."
        )

    time = pd.to_numeric(features[time_column], errors="coerce")
    if time.isna().any():
        samples = time.index[time.isna()].tolist()[:5]
        raise ValueError(f"Missing/non-numeric survival times for samples: {samples}")
    if time.le(0).any():
        samples = time.index[time.le(0)].tolist()[:5]
        raise ValueError(f"Survival times must be positive; invalid samples: {samples}")

    unique_events = set(event.dropna().unique())
    if event.isna().any():
        samples = event.index[event.isna()].tolist()[:5]
        raise ValueError(f"Missing event indicators for samples: {samples}")
    if not unique_events.issubset({0, 1}):
        raise ValueError(f"Event target must contain only 0/1; found {unique_events}")

    X = features.drop(columns=[time_column]).copy()
    signed_time = time.where(event.eq(1), -time).rename(time_column)
    logger.info("Prepared survival data: X=%s, events=%s", X.shape, event.sum())
    return X, signed_time, event.astype(int)


def setup_feature_manager(
    config: Any,
    filters_metadata: Mapping[str, Any] | None,
    settings: SurvivalRunSettings,
    *,
    validation: bool = False,
) -> FeatureManager:
    """Create independent handlers so train/validation metadata caches cannot leak."""
    analysis_config = copy.copy(config)
    # When filters_metadata is None, retain the top-level config.filters_metadata.
    if filters_metadata is not None:
        analysis_config.filters_metadata = dict(filters_metadata)

    metadata_handler = MetadataHandler(analysis_config)
    oligos_handler = OligosHandler(analysis_config)
    prevalence_min = 0.0 if validation else settings.prevalence_threshold_min
    prevalence_max = 100.0 if validation else settings.prevalence_threshold_max

    return FeatureManager(
        analysis_config,
        metadata_handler,
        oligos_handler,
        subgroup=settings.subgroup,
        with_oligos=settings.with_oligos,
        with_additional_features=settings.with_additional_features,
        prevalence_threshold_min=prevalence_min,
        prevalence_threshold_max=prevalence_max,
    )
