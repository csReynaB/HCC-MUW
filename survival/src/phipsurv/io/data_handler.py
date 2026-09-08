"""Configuration-driven loading of peptide data, and sample and library metadata.

The public workflow remains:
    config = Config("config.yaml")
    metadata = MetadataHandler(config)
    oligos = OligosHandler(config)
    features = FeatureManager(config, metadata, oligos, ...)
    X, y = features.get_features_target()
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SUPPORTED_TABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
SUPPORTED_LIBRARY_METADATA_EXTENSIONS = SUPPORTED_TABLE_EXTENSIONS | {
    ".pkl",
    ".pickle",
}


def _read_table(path: Path, *, index_col: str | int | None = None) -> pd.DataFrame:
    """Read a delimited text or Excel table based on its filename suffix."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, index_col=index_col, low_memory=False)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", index_col=index_col, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0, index_col=index_col)
    raise ValueError(
        f"Unsupported table format '{suffix}' for {path}. "
        f"Supported formats: {sorted(SUPPORTED_TABLE_EXTENSIONS)}"
    )


@dataclass(init=False)
class Config:
    """Validated application configuration loaded from one YAML file."""

    data_input: Path
    metadata_input: Path
    lib_metadata_input: Path | None = None
    group_tests: list[str] = field(default_factory=list)

    project: str | None = None
    col_sample_name: str = "SampleName"
    col_target: str = "group_test"
    col_predict: str = "class1_proba"
    extra_features_to_include: list[str] = field(default_factory=list)
    filters_metadata: dict[str, Any] | None = None
    combined_filters_metadata: list[dict[str, Any]] | None = None
    oligo_filters: dict[str, Any] | None = None
    oligo_filter_mode: str = "all"
    peptide_prefixes: list[str] = field(
        default_factory=lambda: ["agilent_", "corona2_", "twist_"]
    )
    transposed: bool = True
    fillna_value: float | None = None
    random_state: int = 420
    param_grid: dict[str, Any] = field(default_factory=dict)
    survival: dict[str, Any] = field(default_factory=dict)

    # Old keys mapped to the new names. They may be removed after configs migrate.
    _ALIASES = {
        "lib_meta_data": "lib_metadata_input",
        "libraries_prefixes": "peptide_prefixes",
    }
    _IGNORED_LEGACY_KEYS = {
        "meta_typefile",
        "data_types",
        "with_oligos_options",
        "with_additional_features_options",
        "with_run_plates_options",
        "filter_by_entropy",
        "filter_by_correlation",
        "subgroups_to_name",
        "subgroups_order",
        "subgroups_to_include",
        "subgroups_colors",
        "estimators_info",
        "cv_method",
        "split_train_test",
        "compute_feature_importance",
        "return_train",
        "return_test",
        "external_set",
        "tuning_parameters",
        "train_size",
        "k",
        "tuning_n_iter",
        "tuning_k",
        "fillna",
        "imputed",
        "impute_additional_features",
    }

    def __init__(self, config_file: str | Path, **overrides: Any) -> None:
        path = Path(config_file).expanduser().resolve()
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("config_file must have a .yaml or .yml extension")
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {path}")

        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("The YAML root must be a mapping")

        values = dict(raw)
        values.update({k: v for k, v in overrides.items() if v is not None})
        for old, new in self._ALIASES.items():
            if old in values and new not in values:
                values[new] = values[old]

        known = {f.name for f in fields(type(self))}
        unknown = set(values) - known - set(self._ALIASES) - self._IGNORED_LEGACY_KEYS
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")

        for f in fields(type(self)):
            if f.name in values:
                setattr(self, f.name, values[f.name])
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())

        self.config_file = path
        self._normalise_paths(path.parent)
        self._normalise_prefixes()
        self._validate()
        self._update_group_metadata()

    def _update_group_metadata(self) -> None:
        """Rebuild values derived from the configured target-group order."""
        self.group_label_encoding = {
            label: index for index, label in enumerate(self.group_tests)
        }
        self.group_code_to_label = {
            index: label for index, label in enumerate(self.group_tests)
        }
        self.label_group_tests = "-".join(map(str, self.group_tests))

    def set_group_tests(self, group_tests: Sequence[Any]) -> None:
        """Replace target groups and refresh their positional encodings."""
        if isinstance(group_tests, (str, bytes)) or not isinstance(
            group_tests, Sequence
        ):
            raise TypeError("group_tests must be a sequence of at least two labels")
        self.group_tests = list(group_tests)
        self._validate_group_tests()
        self._update_group_metadata()

    def _normalise_paths(self, config_dir: Path) -> None:
        """Resolve every input filename using the configuration directory.

        Absolute paths are preserved, ``~`` is expanded, and relative paths are
        anchored to the YAML file rather than to the process working directory.
        """

        def resolve(
            value: str | Path | None,
        ) -> Path | None:
            if value is None:
                return None

            path = Path(value).expanduser()

            if not path.is_absolute():
                path = config_dir / path

            return path.resolve()

        data_input = resolve(getattr(self, "data_input", None))

        metadata_input = resolve(getattr(self, "metadata_input", None))

        if data_input is None:
            raise ValueError("Missing mandatory configuration key: 'data_input'")

        if metadata_input is None:
            raise ValueError("Missing mandatory configuration key: 'metadata_input'")

        # The type checker now knows these are Path, not Path | None.
        self.data_input = data_input
        self.metadata_input = metadata_input

        # This attribute is allowed to be Path | None.
        self.lib_metadata_input = resolve(getattr(self, "lib_metadata_input", None))

    def _normalise_prefixes(self) -> None:
        prefixes = self.peptide_prefixes
        if isinstance(prefixes, (str, bytes)) or not isinstance(prefixes, Sequence):
            raise TypeError("peptide_prefixes must be a sequence of strings")

        normalised: list[str] = []
        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix.strip():
                raise ValueError("peptide_prefixes cannot contain empty values")
            clean = prefix.strip()
            clean = clean if clean.endswith("_") else f"{clean}_"
            if clean not in normalised:
                normalised.append(clean)
        if not normalised:
            raise ValueError("peptide_prefixes cannot be empty")
        self.peptide_prefixes = normalised

    def _validate_group_tests(self) -> None:
        if not isinstance(self.group_tests, (list, tuple)) or len(self.group_tests) < 2:
            raise ValueError("group_tests must contain at least two target labels")
        self.group_tests = list(self.group_tests)
        if any(pd.isna(label) for label in self.group_tests):
            raise ValueError("group_tests cannot contain missing values")
        if len(set(self.group_tests)) != len(self.group_tests):
            raise ValueError("group_tests cannot contain duplicate labels")

    def _validate(self) -> None:
        missing = [
            name
            for name in ("data_input", "metadata_input", "group_tests")
            if not getattr(self, name, None)
        ]
        if missing:
            raise ValueError(f"Missing mandatory configuration keys: {missing}")
        self._validate_group_tests()
        if isinstance(self.extra_features_to_include, (str, bytes)) or not isinstance(
            self.extra_features_to_include, Sequence
        ):
            raise TypeError("extra_features_to_include must be a sequence of names")
        if not all(
            isinstance(column, str) and column
            for column in self.extra_features_to_include
        ):
            raise ValueError("extra_features_to_include contains an invalid name")
        self.extra_features_to_include = list(self.extra_features_to_include)

        if self.filters_metadata is not None and not isinstance(
            self.filters_metadata, Mapping
        ):
            raise TypeError("filters_metadata must be a mapping or null")
        if self.filters_metadata is not None:
            self.filters_metadata = dict(self.filters_metadata)

        if self.combined_filters_metadata is not None:
            if isinstance(
                self.combined_filters_metadata, (str, bytes)
            ) or not isinstance(self.combined_filters_metadata, Sequence):
                raise TypeError("combined_filters_metadata must be a sequence or null")
            if not all(
                isinstance(condition, Mapping)
                for condition in self.combined_filters_metadata
            ):
                raise TypeError(
                    "Every combined_filters_metadata condition must be a mapping"
                )
            self.combined_filters_metadata = [
                dict(condition) for condition in self.combined_filters_metadata
            ]

        if self.oligo_filter_mode not in {"all", "any"}:
            raise ValueError("oligo_filter_mode must be either 'all' or 'any'")
        if self.oligo_filters is not None and not isinstance(
            self.oligo_filters, Mapping
        ):
            raise TypeError("oligo_filters must be a mapping or null")
        if self.oligo_filters is not None:
            self.oligo_filters = dict(self.oligo_filters)
        if not isinstance(self.param_grid, Mapping):
            raise TypeError("param_grid must be a mapping")
        if not isinstance(self.survival, Mapping):
            raise TypeError("survival must be a mapping")

        self.param_grid = dict(self.param_grid)
        self.survival = dict(self.survival)

        for name in ("data_input", "metadata_input"):
            path = getattr(self, name)
            if path.suffix.lower() not in SUPPORTED_TABLE_EXTENSIONS:
                raise ValueError(f"Unsupported {name} extension: {path.suffix}")
        if (
            self.lib_metadata_input is not None
            and self.lib_metadata_input.suffix.lower()
            not in SUPPORTED_LIBRARY_METADATA_EXTENSIONS
        ):
            raise ValueError(
                "Unsupported lib_metadata_input extension: "
                f"{self.lib_metadata_input.suffix}"
            )

    def get_bayesian_param_grid(self, model_type: str) -> dict[str, Any]:
        """Convert one model's YAML search-space specifications to skopt spaces.

        This method intentionally leaves ``self.param_grid`` unchanged so the
        same Config instance can provide grids for more than one estimator.
        """
        try:
            from skopt.space import Categorical, Integer, Real
        except ImportError as error:
            raise ImportError(
                "Bayesian grid conversion requires scikit-optimize. "
                "Install it with `pip install scikit-optimize`."
            ) from error

        if model_type not in self.param_grid:
            available = sorted(self.param_grid)
            raise KeyError(
                f"No parameter grid for '{model_type}'. Available grids: {available}"
            )

        converted: dict[str, Any] = {}
        for parameter, specification in self.param_grid[model_type].items():
            if not isinstance(specification, Mapping):
                raise TypeError(
                    f"Grid specification for '{parameter}' must be a mapping"
                )
            space_type = str(specification.get("type", "")).lower()
            try:
                if space_type == "integer":
                    converted[parameter] = Integer(
                        specification["low"], specification["high"]
                    )
                elif space_type == "real":
                    converted[parameter] = Real(
                        specification["low"],
                        specification["high"],
                        prior=specification.get("prior", "uniform"),
                    )
                elif space_type == "categorical":
                    converted[parameter] = Categorical(specification["categories"])
                else:
                    raise ValueError(
                        f"Unknown search-space type '{space_type}' for '{parameter}'"
                    )
            except KeyError as error:
                raise ValueError(
                    f"Incomplete grid specification for '{parameter}': "
                    f"missing {error.args[0]!r}"
                ) from error
        return converted

    def get_bayesian_param_grid_from_dict_items(
        self, model_type: str = "xgboost"
    ) -> dict[str, Any]:
        """Old compatibility wrapper for existing code; converts the grid in place."""
        converted = self.get_bayesian_param_grid(model_type)
        self.param_grid = converted
        return converted


class MetadataHandler:
    def __init__(self, config: Config):
        self.config = config
        self._metadata: pd.DataFrame | None = None

    @staticmethod
    def _encode_sex(value: Any) -> Any:
        """Encode common Sex/Gender representations while preserving unknowns."""
        if pd.isna(value):
            return value

        if isinstance(value, str):
            normalised = value.strip().lower()

            mapping = {
                "f": 0,
                "female": 0,
                "m": 1,
                "male": 1,
                "0": 0,
                "1": 1,
            }

            return mapping.get(normalised, value)

        if value in (0, 1):
            return int(value)

        return value

    def get_individuals_metadata_df(self, *, refresh: bool = False) -> pd.DataFrame:
        if self._metadata is not None and not refresh:
            return self._metadata.copy()

        df = _read_table(self.config.metadata_input)

        unnamed_columns = df.columns[df.columns.astype(str).str.match(r"^Unnamed")]

        if len(unnamed_columns) > 0:
            df = df.drop(columns=unnamed_columns)

        sample = self.config.col_sample_name

        if sample in df.columns:
            df = df.set_index(sample)
        elif df.index.name != sample:
            raise KeyError(f"Sample column '{sample}' not found in metadata")

        df.index = df.index.astype(str)
        df.index.name = sample
        if df.index.has_duplicates:
            duplicated = df.index[df.index.duplicated()].unique().tolist()[:5]
            raise ValueError(f"Metadata contains duplicate sample IDs: {duplicated}")

        if "Gender" in df.columns and "Sex" not in df.columns:
            df = df.rename(columns={"Gender": "Sex"})

        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].map(self._encode_sex)

        if self.config.filters_metadata:
            df = self._apply_and_filter(df, self.config.filters_metadata)

        if self.config.combined_filters_metadata:
            masks = [
                self._condition_mask(df, condition)
                for condition in self.config.combined_filters_metadata
            ]

            combined_mask = pd.concat(
                masks,
                axis=1,
            ).any(axis=1)

            df = df.loc[combined_mask, :].copy()

        target = self.config.col_target
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in metadata")
        df = self._select_target_groups(df)

        self._metadata = df.copy()

        return df

    def _select_target_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select configured classes from textual or already encoded targets.

        Text targets are matched directly to ``group_tests``. Numeric targets
        use positional codes, so group_tests[0] is class 0, group_tests[1] is
        class 1, and so forth.
        """
        target = self.config.col_target
        values = df[target]

        direct_mask = values.isin(self.config.group_tests)
        numeric = pd.to_numeric(values, errors="coerce")
        integer_like = numeric.notna() & np.isclose(numeric, numeric.round())
        valid_codes = set(range(len(self.config.group_tests)))
        encoded_mask = integer_like & numeric.isin(valid_codes)
        encoded_only_mask = encoded_mask & ~direct_mask

        selected_mask = direct_mask | encoded_only_mask
        selected = df.loc[selected_mask].copy()
        if encoded_only_mask.any():
            encoded_index = df.index[np.asarray(encoded_only_mask, dtype=bool)]
            selected.loc[encoded_index, target] = numeric.loc[encoded_index].astype(int)

        if direct_mask.any() and encoded_only_mask.any():
            representation = "labels and numeric codes"
        elif direct_mask.any():
            representation = "labels"
        else:
            representation = "numeric codes"

        if selected.empty:
            available = values.dropna().drop_duplicates().tolist()
            raise ValueError(
                "No metadata rows remain after target/filter selection. "
                f"Configured group_tests={self.config.group_tests!r}; "
                f"expected either those labels or numeric codes "
                f"{list(range(len(self.config.group_tests)))!r}; "
                f"available values={available!r}; target column={target!r}."
            )

        selected.attrs["target_representation"] = representation
        return selected

    @staticmethod
    def _condition_mask(df: pd.DataFrame, condition: Mapping[str, Any]) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for column, accepted in condition.items():
            if column not in df.columns:
                raise KeyError(f"Metadata filter column '{column}' not found")
            values = (
                list(accepted)
                if isinstance(accepted, Sequence)
                and not isinstance(accepted, (str, bytes))
                else [accepted]
            )
            mask &= df[column].isin(values)
        return mask

    def _apply_and_filter(
        self, df: pd.DataFrame, condition: Mapping[str, Any]
    ) -> pd.DataFrame:
        return df.loc[self._condition_mask(df, condition)]

    def get_additional_features_df(self) -> pd.DataFrame:
        """Return unmodified clinical features for pipeline-level preprocessing."""
        columns = self.config.extra_features_to_include
        metadata = self.get_individuals_metadata_df()
        missing = sorted(set(columns) - set(metadata.columns))
        if missing:
            raise KeyError(f"Additional metadata features not found: {missing}")
        return metadata.loc[:, columns].copy()


class OligosHandler:
    def __init__(self, config: Config):
        self.config = config
        self._oligos: pd.DataFrame | None = None

    def get_oligos_df(self, *, refresh: bool = False) -> pd.DataFrame:
        if self._oligos is not None and not refresh:
            return self._oligos.copy()
        df = _read_table(self.config.data_input, index_col=0)
        if self.config.transposed:
            df = df.T
        df.index = df.index.astype(str)
        df.index.name = self.config.col_sample_name
        df.columns = df.columns.astype(str)
        if df.index.has_duplicates:
            raise ValueError("Peptide data contains duplicate sample IDs")
        if df.columns.has_duplicates:
            duplicated = df.columns[df.columns.duplicated()].unique().tolist()[:5]
            raise ValueError(
                f"Peptide data contains duplicate peptide IDs: {duplicated}"
            )
        self._oligos = df.copy()
        return df

    def get_oligos_metadata_df(self) -> pd.DataFrame:
        path = self.config.lib_metadata_input
        if path is None:
            raise ValueError("lib_metadata_input is required for subgroup selection")
        if not path.is_file():
            raise FileNotFoundError(f"Library metadata does not exist: {path}")

        df: pd.DataFrame

        if path.suffix.lower() in {".pkl", ".pickle"}:
            loaded = pd.read_pickle(path)

            if not isinstance(loaded, pd.DataFrame):
                raise TypeError(
                    "Library metadata pickle must contain a pandas DataFrame; "
                    f"found {type(loaded).__name__}"
                )

            df = loaded
        else:
            df = _read_table(path, index_col=0)

        df.index = df.index.astype(str)

        if df.index.has_duplicates:
            duplicated = df.index[df.index.duplicated()].unique().tolist()[:5]
            raise ValueError(
                f"Library metadata contains duplicate peptide IDs: {duplicated}"
            )

        return df


class FeatureManager:
    def __init__(
        self,
        config: Config,
        metadata_handler: MetadataHandler,
        oligos_handler: OligosHandler,
        *,
        subgroup: str = "all",
        oligo_filters: Mapping[str, Any] | None = None,
        oligo_filter_mode: str | None = None,
        with_oligos: bool = True,
        with_additional_features: bool = False,
        prevalence_threshold_min: float = 0,
        prevalence_threshold_max: float = 100,
    ):
        self.config = config
        self.metadata_handler = metadata_handler
        self.oligos_handler = oligos_handler
        if not isinstance(subgroup, str):
            raise TypeError("subgroup must be a string")
        if subgroup != "all" and oligo_filters is not None:
            raise ValueError(
                "Use either subgroup or oligo_filters, not both. "
                "subgroup is the compatibility shortcut for a True/False flag column."
            )
        self.subgroup = subgroup

        configured_oligo_filters = config.oligo_filters
        self.oligo_filters: dict[str, Any] | None

        if oligo_filters is not None:
            self.oligo_filters = dict(oligo_filters)

        elif subgroup == "all" and configured_oligo_filters is not None:
            self.oligo_filters = dict(configured_oligo_filters)

        else:
            self.oligo_filters = None

        self.oligo_filter_mode = oligo_filter_mode or config.oligo_filter_mode
        if self.oligo_filter_mode not in {"all", "any"}:
            raise ValueError("oligo_filter_mode must be either 'all' or 'any'")
        self.with_oligos = with_oligos
        self.with_additional_features = with_additional_features
        self.prevalence_threshold_min = self._validate_percentage(
            prevalence_threshold_min, "prevalence_threshold_min"
        )
        self.prevalence_threshold_max = self._validate_percentage(
            prevalence_threshold_max, "prevalence_threshold_max"
        )
        if self.prevalence_threshold_min > self.prevalence_threshold_max:
            raise ValueError("Minimum prevalence cannot exceed maximum prevalence")
        if not (with_oligos or with_additional_features):
            raise ValueError("Select oligos, additional features, or both")

    @staticmethod
    def _validate_percentage(value: float, name: str) -> float:
        if not isinstance(value, (int, float, np.number)) or not 0 <= value <= 100:
            raise ValueError(f"{name} must be numeric and between 0 and 100")
        return float(value)

    @staticmethod
    def _as_filter_values(value: Any) -> list[Any]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        return [value]

    @staticmethod
    def _boolean_mask(
        series: pd.Series,
        expected: bool,
    ) -> pd.Series:
        """Match common Boolean representations."""
        boolean_mapping: dict[str, bool] = {
            "true": True,
            "1": True,
            "1.0": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "0.0": False,
            "no": False,
            "n": False,
        }

        normalised = series.astype("string").str.strip().str.lower()

        encoded = normalised.map(boolean_mapping)

        return encoded.eq(expected).fillna(False).astype(bool)

    @staticmethod
    def _boolean_filter_value(series: pd.Series, value: Any) -> bool | None:
        """Interpret Boolean-like text only when the metadata column is Boolean-like."""
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if not isinstance(value, str):
            return None

        mapping = {
            "true": True,
            "1": True,
            "1.0": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "0.0": False,
            "no": False,
            "n": False,
        }
        expected = mapping.get(value.strip().lower())
        if expected is None:
            return None

        observed = series.dropna().astype("string").str.strip().str.lower()
        if observed.empty or not observed.isin(mapping).all():
            return None
        return expected

    def _effective_oligo_filters(self) -> dict[str, Any]:
        if self.oligo_filters is not None:
            return self.oligo_filters
        if self.subgroup == "all":
            return {}
        # Backward compatibility: subgroup="is_PNP" means is_PNP == True.
        return {self.subgroup: True}

    def _select_subgroup(self, peptides: pd.DataFrame) -> pd.DataFrame:
        filters = self._effective_oligo_filters()
        if not filters:
            return peptides

        library = self.oligos_handler.get_oligos_metadata_df()
        masks: list[pd.Series] = []
        for column, accepted in filters.items():
            if column not in library.columns:
                raise KeyError(f"Oligo metadata column '{column}' not found")
            expected_boolean = self._boolean_filter_value(
                library[column],
                accepted,
            )
            if expected_boolean is not None:
                mask = self._boolean_mask(library[column], expected_boolean)
            else:
                values = self._as_filter_values(accepted)
                if not values:
                    raise ValueError(
                        f"Oligo filter for '{column}' has no accepted values"
                    )
                mask = library[column].isin(values)
            masks.append(mask.rename(column))

        mask_table = pd.concat(masks, axis=1)
        combined = (
            mask_table.all(axis=1)
            if self.oligo_filter_mode == "all"
            else mask_table.any(axis=1)
        )
        selected_ids = set(library.index[combined])
        # Keep the original data-column order rather than the metadata-row order.
        keep = [column for column in peptides.columns if column in selected_ids]
        if not keep:
            raise ValueError(
                f"No peptide features matched oligo_filters={filters!r} "
                f"with mode='{self.oligo_filter_mode}'"
            )
        return peptides.loc[:, keep]

    def _filter_prevalence(self, peptides: pd.DataFrame) -> pd.DataFrame:
        if (self.prevalence_threshold_min, self.prevalence_threshold_max) == (0, 100):
            return peptides
        # Assumes binary hit/exist data. For continuous input, prevalence means non-zero.
        prevalence = (peptides.notna() & peptides.ne(0)).mean(axis=0).mul(100)

        keep = prevalence.between(
            self.prevalence_threshold_min,
            self.prevalence_threshold_max,
            inclusive="both",
        )
        # keep = (prevalence.gt(self.prevalence_threshold_min) &
        #        prevalence.lt(self.prevalence_threshold_max)
        #)

        return peptides.loc[:, keep]

    def get_features_target(self) -> tuple[pd.DataFrame, pd.Series]:
        metadata = self.metadata_handler.get_individuals_metadata_df()
        common_samples = metadata.index
        parts: list[pd.DataFrame] = []

        if self.with_oligos:
            peptides = self.oligos_handler.get_oligos_df()
            common_samples = common_samples.intersection(peptides.index, sort=False)
            peptides = peptides.loc[common_samples]
            peptides = self._filter_prevalence(self._select_subgroup(peptides))
            parts.append(peptides)

        if self.with_additional_features:
            additional = self.metadata_handler.get_additional_features_df()
            common_samples = common_samples.intersection(additional.index, sort=False)
            parts.append(additional.loc[common_samples])

        if len(common_samples) == 0:
            raise ValueError("Data and metadata have no sample IDs in common")
        X = pd.concat([part.loc[common_samples] for part in parts], axis=1)

        if X.shape[1] == 0:
            raise ValueError("No features remain")

        if X.columns.duplicated().any():
            duplicates = X.columns[X.columns.duplicated()].unique().tolist()[:5]
            raise ValueError(
                f"Duplicate feature names after combining data: {duplicates}"
            )
        if self.config.fillna_value is not None:
            X = X.fillna(self.config.fillna_value)

        labels = metadata.loc[common_samples, self.config.col_target]
        y = self._encode_target(labels)
        return X, y

    def _encode_target(self, labels: pd.Series) -> pd.Series:
        """Return integer class codes from labels and/or positional codes."""
        mapped = labels.map(self.config.group_label_encoding)
        numeric = pd.to_numeric(labels, errors="coerce")
        integer_like = numeric.notna() & np.isclose(numeric, numeric.round())
        valid_codes = set(range(len(self.config.group_tests)))
        valid = integer_like & numeric.isin(valid_codes)

        encoded = mapped.astype("Float64")
        use_numeric = encoded.isna() & valid
        encoded.loc[use_numeric] = numeric.loc[use_numeric]
        if encoded.notna().all():
            return encoded.astype(int).rename(self.config.col_target)

        invalid = labels.loc[~valid & mapped.isna()].drop_duplicates().tolist()
        raise ValueError(
            f"Target values {invalid!r} are neither configured labels "
            f"{self.config.group_tests!r} nor valid positional codes "
            f"{list(range(len(self.config.group_tests)))!r}."
        )
