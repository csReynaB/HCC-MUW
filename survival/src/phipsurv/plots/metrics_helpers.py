# =========================
# Standard library
# =========================
import logging
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# =========================
# Plotting
# =========================
import matplotlib.pyplot as plt

# =========================
# Third-party: core
# =========================
import numpy as np
import pandas as pd
from joblib import load

# =========================
# Survival analysis
# =========================
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts, qq_plot, rmst_plot
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import median_survival_times
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# =========================
# ML / explainability
# =========================
from sklearn.preprocessing import MinMaxScaler
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from tqdm import tqdm

# ======================
# Local / project imports
# ======================
from phipsurv.survival.helpers import calculate_cumulative_dynamic_auc

# ======================
# Global configuration
# ======================
logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.titlecolor": "black",
    }
)


def format_pval(p: Any, alpha: float = 0.05) -> str:
    """Format a p-value while handling missing and non-numeric values safely."""
    try:
        p_value = float(p)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(p_value):
        return "NA"
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    # 1. Non-significant (p > alpha)
    if p_value > alpha:
        raw = f"{p_value:.2f}".rstrip("0").rstrip(".")
        return f"ns [{raw}]"

    # 2. Normal fixed-decimal formatting (0.001 ≤ p ≤ alpha)
    if p_value >= 0.001:
        raw = f"{p_value:.3f}".rstrip("0").rstrip(".")
        return raw

    # 3. Scientific notation (< 0.001)
    # Format like 1e-4 instead of 1.00e-04
    raw = f"{p_value:.2e}"

    # Clean up trailing zeros in mantissa
    raw = raw.replace(".00e", "e")
    raw = raw.replace(".0e", "e")

    # Also trim e- notation like in R:
    # "1.10e-04" → "1.1e-04"
    if "e" in raw:
        mantissa, exp = raw.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        raw = mantissa + "e" + exp

    return raw

def get_significance_stars(p_value: Any) -> str:
    """Return significance stars for a p-value."""
    try:
        numeric = float(p_value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    if numeric < 0.001:
        return "***"
    elif numeric < 0.01:
        return "**"
    elif numeric < 0.05:
        return "*"
    else:
        return "ns"

#####################
#     Metrics       #
#####################


def _numeric_series(
    values: Union[pd.Series, Sequence[float], np.ndarray],
    *,
    name: str,
) -> pd.Series:
    """Return a finite numeric Series without silently changing its index."""
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name} contains missing or non-numeric values")
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return numeric.astype(float)


def calculate_samples_at_risk(
    y_time: Union[pd.Series, Sequence[float], np.ndarray],
    time_points: Iterable[float],
) -> pd.DataFrame:
    """
    Calculate the number of individuals at risk, as well as the number of censored and event samples,
    at each time point.

    Parameters:
        y_time (array-like): Array or list of time-to-event data. Negative values indicate censored samples.
        time_points (iterable): Iterable of time points at which to compute the counts.

    Returns:
        pd.DataFrame: A DataFrame with columns 'At Risk', 'Censored', and 'Events' indexed by the time points.
    """
    y_series = _numeric_series(y_time, name="y_time")
    y_array = y_series.to_numpy(dtype=float)
    points = np.asarray(list(time_points), dtype=float)
    if points.ndim != 1 or points.size == 0:
        raise ValueError("time_points must be a non-empty one-dimensional sequence")
    if not np.isfinite(points).all():
        raise ValueError("time_points contains non-finite values")

    at_risk = []
    samples_censored = []
    samples_events = []

    # At time 0, all individuals are considered at risk
    total_samples = len(y_array)

    for t in points:
        # Adjust the time point slightly
        t_adjusted = t + 0.01

        # Count censored samples: those with |y_time_train| <= t_adjusted and y_time_train < 0
        count_censored = np.sum(
            (np.abs(y_array) <= t_adjusted) & (y_array < 0)
        )

        # Count event samples: those with |y_time_train| <= t_adjusted and y_time_train >= 0
        count_events = np.sum(
            (np.abs(y_array) <= t_adjusted) & (y_array >= 0)
        )

        # Calculate the number at risk by subtracting events and censored samples from the total at risk
        count_at_risk = total_samples - count_events - count_censored

        at_risk.append(count_at_risk)
        samples_censored.append(count_censored)
        samples_events.append(count_events)

    # Create a DataFrame to store these counts, indexed by the original time points
    samples_at_risk = pd.DataFrame(
        {"At risk": at_risk, "Censored": samples_censored, "Events": samples_events},
        index=points,
    )

    return samples_at_risk


def calculate_time_dependent_auc(
    y_train: pd.Series,
    y_val: pd.Series,
    risk_scores: Union[pd.Series, Sequence[float], np.ndarray],
    max_time_point: Optional[float] = None,
    time_points_highlight: Optional[Sequence[float]] = None,
    num_points: int = 50,
    buffer: float = 0.001,
) -> Tuple[pd.Series, float, pd.DataFrame]:
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
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if buffer <= 0:
        raise ValueError("buffer must be positive")

    train = _numeric_series(y_train, name="y_train")
    validation = _numeric_series(y_val, name="y_val")
    if isinstance(risk_scores, pd.Series):
        scores = pd.to_numeric(
            risk_scores.reindex(validation.index),
            errors="coerce",
        )
    else:
        if len(risk_scores) != len(validation):
            raise ValueError("risk_scores and y_val must have the same length")
        scores = pd.Series(
            np.asarray(risk_scores, dtype=float),
            index=validation.index,
        )
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("risk_scores contains missing or non-finite values")

    train_max = float(train.abs().max())
    valid_indices = validation.abs() <= train_max
    validation = validation.loc[valid_indices]
    scores = scores.loc[valid_indices]
    if validation.empty:
        raise ValueError(
            "No validation samples fall within the training follow-up range"
        )

    y_train_surv = Surv.from_arrays(
        event=train.ge(0).to_numpy(),
        time=train.abs().to_numpy(dtype=float),
    )
    y_valid_surv = Surv.from_arrays(
        event=validation.ge(0).to_numpy(),
        time=validation.abs().to_numpy(dtype=float),
    )

    min_time_point = max(
        float(validation.abs().min()),
        float(train.abs().min()),
    )
    evaluable_max = min(
        float(validation.abs().max()),
        train_max,
    ) - buffer
    requested_max = (
        evaluable_max
        if max_time_point is None
        else min(float(max_time_point), evaluable_max)
    )
    if requested_max <= min_time_point:
        raise ValueError(
            "No evaluable AUC interval remains after applying follow-up bounds"
        )

    time_points = np.linspace(
        min_time_point,
        requested_max,
        num=num_points,
    )
    if time_points_highlight is None:
        highlights = np.concatenate(
            ([1.0], np.arange(3.0, requested_max + 1.0, step=3.0))
        )
    else:
        highlights = np.asarray(time_points_highlight, dtype=float)
    highlights = highlights[
        (highlights >= min_time_point) & (highlights <= requested_max)
    ]
    time_points = np.unique(np.concatenate((time_points, highlights)))

    # Calculate time-dependent AUC
    auc_values, mean_auc = cumulative_dynamic_auc(
        y_train_surv,
        y_valid_surv,
        scores.to_numpy(dtype=float),
        time_points,
    )
    auc_values = pd.Series(auc_values, index=time_points)

    # Calculate the number of individuals at risk at each time point
    samples_at_risk = calculate_samples_at_risk(validation, time_points)

    return auc_values, float(mean_auc), samples_at_risk


def calculate_antigen_scores_scaled(
    shap_values: pd.DataFrame,
    y_time: Union[pd.Series, pd.DataFrame],
    y_event: Union[pd.Series, pd.DataFrame],
    top_peptides: Sequence[str],
    scaler: Optional[MinMaxScaler] = None,
    scaler_antigens: Optional[MinMaxScaler] = None,
    threshold: Optional[float] = None,
    val_quantile: float = 50,
    return_all: bool = False,
    scale_feature_shap: bool = False,
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
    scale_feature_shap : bool, default False
        If True, sum signed, per-feature min-max-scaled SHAP magnitudes.
        False preserves the historical behavior of summing raw signed SHAP values.

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
    selected_peptides = list(dict.fromkeys(top_peptides))
    if not selected_peptides:
        raise ValueError("top_peptides cannot be empty")
    missing_peptides = sorted(set(selected_peptides) - set(shap_values.columns))
    if missing_peptides:
        raise ValueError(
            "Some top_peptides are not present in shap_values.columns: "
            f"{missing_peptides[:10]}"
        )
    if not 0 <= val_quantile <= 100:
        raise ValueError("val_quantile must be between 0 and 100")

    # Work on absolute SHAP values for scaling; then reintroduce sign later.
    selected_shap = shap_values.loc[:, selected_peptides].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if selected_shap.isna().any().any():
        raise ValueError("Selected SHAP values contain missing or non-numeric data")
    abs_shap = selected_shap.abs()

    # Scale the SHAP values for the top peptides.
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_shap = pd.DataFrame(
            scaler.fit_transform(abs_shap),
            index=abs_shap.index,
            columns=abs_shap.columns,
        )
        # scaled_shap = scaler.fit_transform(scaled_shap)

    else:
        scaled_shap = pd.DataFrame(
            scaler.transform(abs_shap),
            index=abs_shap.index,
            columns=abs_shap.columns,
        )
        scaled_shap = scaled_shap.clip(lower=0, upper=1)

    score_components = (
        scaled_shap * np.sign(selected_shap)
        if scale_feature_shap
        else selected_shap
    )
    antigen_scores = score_components.sum(axis=1).to_frame(name="Antigen Score")

    if scaler_antigens is None:
        scaler_antigens = MinMaxScaler(feature_range=(0, 1))
        antigen_scores_scaled = pd.DataFrame(
            scaler_antigens.fit_transform(antigen_scores), index=antigen_scores.index
        )
    else:
        antigen_scores_scaled = pd.DataFrame(
            scaler_antigens.transform(antigen_scores), index=antigen_scores.index
        )
        antigen_scores_scaled = antigen_scores_scaled.clip(lower=0, upper=1)

    # Determine the threshold to dichotomize the scores.
    if threshold is None:
        threshold = float(
            antigen_scores_scaled.quantile(val_quantile / 100).iloc[0]
        )
    else:
        threshold = float(threshold)
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")

    # antigen_scores = antigen_scores.rename(columns={antigen_scores.columns[0]: 'Antigen Score'})
    antigen_scores_scaled = antigen_scores_scaled.rename(
        columns={antigen_scores_scaled.columns[0]: "Antigen Score (Scaled)"}
    )
    antigen_scores_dichotomized = (
        (antigen_scores_scaled.iloc[:, 0] >= threshold)
        .astype(int)
        .to_frame(name="Antigen Score (Dichotomized)")
    )
    # antigen_scores_dichotomized = antigen_scores_dichotomized.rename(columns={antigen_scores_dichotomized.columns[0]: 'Antigen Score (Dichotomized)'})

    def one_column_frame(
        values: Union[pd.Series, pd.DataFrame],
        default_name: str,
    ) -> pd.DataFrame:
        if isinstance(values, pd.Series):
            return values.rename(values.name or default_name).to_frame()
        if isinstance(values, pd.DataFrame) and values.shape[1] == 1:
            result = values.copy()
            if result.columns[0] is None:
                result.columns = [default_name]
            return result
        raise ValueError(f"{default_name} must contain exactly one column")

    antigen_scores_df = pd.concat(
        [
            one_column_frame(y_time, "Time"),
            one_column_frame(y_event, "Event"),
            antigen_scores,
            antigen_scores_scaled,
            antigen_scores_dichotomized,
        ],
        axis=1,
        join="inner",
    )
    if antigen_scores_df.empty:
        raise ValueError("Survival and SHAP inputs have no sample IDs in common")

    if return_all:
        return antigen_scores_df, threshold, scaler, scaler_antigens
    else:
        return antigen_scores_df


######################################
#        Univariate analysis         #
######################################


def perform_logrank_test(
    df: pd.DataFrame, time_column: str, event_column: str, group_column: str
) -> float:
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
    required = {time_column, event_column, group_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing survival columns: {missing}")
    analysis_df = df.loc[:, [time_column, event_column, group_column]].dropna()
    if analysis_df.empty:
        raise ValueError("No complete observations remain for the log-rank test")

    groups = np.sort(analysis_df[group_column].unique())
    if len(groups) != 2:
        raise ValueError(
            f"Expected two groups in column '{group_column}', but found: {groups}"
        )

    # Split the DataFrame into the two groups
    group0 = analysis_df[analysis_df[group_column] == groups[0]]
    group1 = analysis_df[analysis_df[group_column] == groups[1]]

    # Perform the log-rank test
    results = logrank_test(
        group0[time_column],
        group1[time_column],
        event_observed_A=group0[event_column],
        event_observed_B=group1[event_column],
    )

    p_value = float(results.p_value)
    logger.info("Log-rank test p-value: %.4f", p_value)

    return p_value


def clean_predictor_name(label):
    """Clean the predictor name by removing extraneous substrings."""
    label = label.replace("_ctg", "")
    label = label.replace("Category", "")
    label = label.replace("(Scaled)", "")
    label = re.sub(r"\(Dichotomized\) ctg", "", label)
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
    variables = list(variables)
    required = {"OS months", "OS Status", *variables}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing Kaplan-Meier analysis columns: {missing}")
    output_dir = Path(save_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cc = ["dodgerblue", "orange"]
    results = []  # list to store results per variable and group

    # Loop over each variable
    for var in variables:
        clean_var = clean_predictor_name(var)
        groups = sorted(df[var].dropna().unique())
        if len(groups) < 2:
            logger.warning(
                "Skipping %s because it has fewer than two observed groups",
                var,
            )
            continue
        kmf_list = []
        flag = 0  # used to cycle colors when plotting (for two-group variables)

        # Create a new figure if variable has less than 3 groups (i.e. binary)
        if len(groups) == 2:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Loop over each group for this variable
        for group in groups:
            group_data = df.loc[
                df[var] == group,
                ["OS months", "OS Status", var],
            ].dropna(subset=["OS months", "OS Status"])
            if group_data.empty:
                logger.warning("Skipping empty group %r for %s", group, var)
                continue

            # Fit Kaplan–Meier model for this group
            kmf = KaplanMeierFitter()
            kmf.fit(
                group_data["OS months"],
                event_observed=group_data["OS Status"],
                label=str(group),
            )

            # Calculate median survival and 95% CI using a helper function.
            # (Assumes median_survival_times() returns a DataFrame with lower and upper CIs)
            median_survival = kmf.median_survival_time_
            ci_df = median_survival_times(
                kmf.confidence_interval_
            )  # user-defined function
            ci_lower, ci_upper = ci_df.iloc[0, 0], ci_df.iloc[0, 1]

            # Get the number of patients in the group
            group_size = len(group_data)
            kmf_list.append(kmf)

            # If fewer than 3 groups, perform a pairwise log-rank test against the others.
            if len(groups) == 2:
                other_data = df.loc[
                    df[var] != group,
                    ["OS months", "OS Status"],
                ].dropna()
                lr_result = logrank_test(
                    group_data["OS months"],
                    other_data["OS months"],
                    event_observed_A=group_data["OS Status"],
                    event_observed_B=other_data["OS Status"],
                )
                p_value = lr_result.p_value
                kmf.plot_survival_function(
                    ax=ax, ci_show=True, color=cc[flag], show_censors=True
                )
                flag += 1
            else:
                # For >2 groups, use a multivariate log-rank test.
                lr_result = multivariate_logrank_test(
                    df["OS months"], df[var], event_observed=df["OS Status"]
                )
                p_value = lr_result.p_value

            # Store the results for this group
            results.append(
                [
                    clean_var,
                    group,
                    group_size,
                    median_survival,
                    ci_lower,
                    ci_upper,
                    p_value,
                ]
            )

        # For binary variables, add annotation and additional plots once both groups are processed.
        if len(groups) == 2 and flag == 2:
            p_value_text = f"p = {p_value:.3f}"
            ax.text(
                0.7,
                0.05,
                p_value_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                bbox=dict(facecolor="white", alpha=0.5),
            )
            # Add at-risk counts (assumes add_at_risk_counts() is defined)
            add_at_risk_counts(kmf_list[0], kmf_list[1], ax=ax)
            ax.set_title(clean_var, fontsize=12)
            ax.set_xlabel("Time (Months)", fontsize=10)
            ax.set_ylabel("Survival Probability", fontsize=10)
            ax.grid(True, linestyle="--", color="black", alpha=0.15)
            # Right after add_at_risk_counts(...)
            fig = ax.get_figure()
            if len(fig.axes) > 1:
                ax2 = fig.axes[-1]
                ax.set_zorder(ax2.get_zorder() + 1)
            ax.legend(fontsize=8)
            plt.tight_layout()
            fig_path = output_dir / f"kaplan_meier_{clean_var}_{suffix_file}.pdf"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Create RMST plot (assumes rmst_plot() is defined)
            fig, ax = plt.subplots(figsize=(8, 4))
            rmst_plot(
                kmf_list[0], model2=kmf_list[1], text_position=(10, 0.9), t=30, ax=ax
            )
            lines = ax.get_lines()
            if len(lines) >= 2:
                lines[0].set_color(cc[0])
                lines[1].set_color(cc[1])
            ax.set_title(clean_var, fontsize=12)
            ax.set_xlabel("Time (Months)", fontsize=10)
            ax.set_ylabel("Survival Probability", fontsize=10)
            ax.grid(True, linestyle="--", color="black", alpha=0.15)
            # Right after add_at_risk_counts(...)
            fig = ax.get_figure()
            if len(fig.axes) > 1:
                ax2 = fig.axes[-1]
                ax.set_zorder(ax2.get_zorder() + 1)
            ax.legend(fontsize=8)
            if len(lines) >= 2:
                ax.legend(
                    [lines[0], lines[1]],
                    [kmf_list[0].label, kmf_list[1].label],
                    fontsize=8,
                )
            plt.tight_layout()
            fig_path = (
                output_dir / f"kaplan_meier_RMST_{clean_var}_{suffix_file}.pdf"
            )
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    # Create a summary DataFrame from the results
    univariate = pd.DataFrame(
        results,
        columns=[
            "Variable",
            "Group",
            "Size",
            "Median OS (Months)",
            "95% CI Lower",
            "95% CI Upper",
            "P-value",
        ],
    )
    univariate_kmf = univariate.copy(True)
    # Format the results: remove duplicates and apply bold formatting for significant predictors
    univariate_kmf["Group"] = univariate_kmf["Group"].astype(str).str.replace(
        r"stage|Antigen Score", "", regex=True
    )
    univariate_kmf["Variable"] = univariate_kmf.apply(
        lambda row: (
            f"\\textbf{{{row['Variable']}}}"
            if row["P-value"] < 0.1
            else row["Variable"]
        ),
        axis=1,
    )
    univariate_kmf["P-value"] = univariate_kmf["P-value"].apply(
        lambda x: f"\\textbf{{{x:.4f}}}" if pd.notna(x) and x < 0.1 else f"{x:.4f}"
    )
    univariate_kmf.loc[
        univariate_kmf["Variable"].duplicated(keep="last"), "P-value"
    ] = ""
    univariate_kmf.loc[univariate_kmf["Variable"].duplicated(), "Variable"] = ""
    univariate_kmf["Median OS (Months)"] = univariate_kmf["Median OS (Months)"].round(2)
    univariate_kmf["95% CI Lower"] = univariate_kmf["95% CI Lower"].round(2)
    univariate_kmf["95% CI Upper"] = univariate_kmf["95% CI Upper"].round(2)

    return univariate, univariate_kmf


def extract_category_name(category):
    """
    Extract a clean category name from a model index string.
    E.g., 'Sex_ctg[T.Male]' becomes 'Male'.
    """
    # Split by '[' then take part after and remove trailing ']'
    category = category.split("[")[-1].split("]")[0].replace("T.", "")
    category = category.replace("stage ", "")
    category = category.replace("Antigen Score", "")
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
            cph.fit(
                df, duration_col="OS months", event_col="OS Status", formula=formula
            )
            summary = cph.summary

            # Extract statistics for the predictor (there is only one row)
            hr = summary.loc[predictor, "exp(coef)"]
            ll = summary.loc[predictor, "exp(coef) lower 95%"]
            ul = summary.loc[predictor, "exp(coef) upper 95%"]
            p_value = summary.loc[predictor, "p"]

            clean_name = clean_predictor_name(predictor)
            results.append((clean_name, "", hr, ll, ul, p_value))
    if predictors_ctg:
        # --- Categorical Predictors ---
        for predictor, ref_cat in predictors_ctg.items():
            cph = CoxPHFitter()
            # Force the predictor to be treated as categorical using C() in the formula.
            formula = f"C(`{predictor}`)"
            cph.fit(
                df, duration_col="OS months", event_col="OS Status", formula=formula
            )
            summary = cph.summary

            # For each row in the summary (each category level), extract results.
            for category in summary.index:
                hr = summary.loc[category, "exp(coef)"]
                ll = summary.loc[category, "exp(coef) lower 95%"]
                ul = summary.loc[category, "exp(coef) upper 95%"]
                p_value = summary.loc[category, "p"]

                clean_name = clean_predictor_name(predictor)
                # Create a label for the category by extracting the category name and appending the reference.
                cat_name = extract_category_name(category)
                group_label = f"{cat_name} - Ref {ref_cat}"
                results.append((clean_name, group_label, hr, ll, ul, p_value))

    # --- Create Summary DataFrame ---
    univariate_results = pd.DataFrame(
        results,
        columns=[
            "Variable",
            "Group",
            "Hazard Ratio",
            "95% CI Lower",
            "95% CI Upper",
            "P-value",
        ],
    )
    univariate_cph = univariate_results.copy(True)
    # Format results:
    # Bold variables and groups with significant p-values (<0.1)
    univariate_cph["Variable"] = univariate_cph.apply(
        lambda row: (
            f"\\textbf{{{row['Variable']}}}"
            if row["P-value"] < 0.1
            else row["Variable"]
        ),
        axis=1,
    )
    univariate_cph["Group"] = univariate_cph.apply(
        lambda row: (
            f"\\textbf{{{row['Group']}}}"
            if row["P-value"] < 0.1 and row["Group"] != ""
            else row["Group"]
        ),
        axis=1,
    )
    univariate_cph["P-value"] = univariate_cph["P-value"].apply(
        lambda x: f"\\textbf{{{x:.4f}}}" if pd.notna(x) and x < 0.1 else f"{x:.4f}"
    )
    # Remove duplicate variable names (if repeated, only show once)
    univariate_cph.loc[univariate_cph["Variable"].duplicated(), "Variable"] = ""

    # Round the numeric columns to 2 decimals
    univariate_cph["Hazard Ratio"] = univariate_cph["Hazard Ratio"].round(2)
    univariate_cph["95% CI Lower"] = univariate_cph["95% CI Lower"].round(2)
    univariate_cph["95% CI Upper"] = univariate_cph["95% CI Upper"].round(2)

    # Optionally, print as a LaTeX table (requires tabulate)
    # print(tabulate(univariate_cph, headers='keys', tablefmt='latex', showindex=False))

    return univariate_results, univariate_cph


####################################
#      Process ML results          #
####################################


def process_runs_for_stats(template, start, end, top_k):
    """
    First pass:
      - Read each joblib (if present), get shap_df (rows=samples, cols=peptides).
      - Compute mean|SHAP| per peptide (Series).
      - Rank peptides within the run (1 = most important).
      - Mark top_k membership (stability).
    Returns:
      runs_used, all_feature_names, freq_counter, rank_sum, rank_n, mean_abs_sum
    """
    freq_counter = Counter()
    rank_sum = defaultdict(float)
    rank_n = defaultdict(int)
    mean_abs_sum = defaultdict(float)
    all_feature_names: list[str] = []
    seen_features: set[str] = set()
    runs_used = 0
    if start > end:
        raise ValueError("start must be less than or equal to end")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    for i in tqdm(range(start, end + 1), desc="Pass 1: stats"):
        path = template.format(i)
        if not os.path.exists(path):
            continue

        try:
            obj = load(path)
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            TypeError,
            ValueError,
        ) as error:
            logger.warning("Could not load %s: %s", path, error)
            continue

        if "train_shap_values" not in obj:
            continue

        shap_df = obj["train_shap_values"]
        if not isinstance(shap_df, pd.DataFrame):
            continue
        shap_df = shap_df.copy()
        shap_df.columns = shap_df.columns.astype(str)

        for feature in shap_df.columns:
            if feature not in seen_features:
                seen_features.add(feature)
                all_feature_names.append(feature)

        # mean |SHAP| per peptide
        mean_abs = shap_df.abs().mean(axis=0)
        ranks = mean_abs.rank(ascending=False, method="average")

        # stability top-k
        k = min(top_k, len(mean_abs))
        top_idx = mean_abs.nlargest(k).index
        for feature in top_idx:
            freq_counter[feature] += 1

        for feature, value in mean_abs.items():
            mean_abs_sum[feature] += float(value)
            rank_sum[feature] += float(ranks.loc[feature])
            rank_n[feature] += 1

        runs_used += 1

    return runs_used, all_feature_names, freq_counter, rank_sum, rank_n, mean_abs_sum


def hybrid_select(
    all_feature_names,
    runs_used,
    freq_counter,
    rank_sum,
    rank_n,
    mean_abs_sum,
    min_freq,
    n_select,
):
    """
    Apply hybrid rule:
      keep features with stability >= min_freq,
      then sort by mean_rank asc (best), tie-break by mean_abs_shap desc.
      If fewer than n_select survivors, relax threshold.
    """
    if runs_used < 1:
        raise ValueError("runs_used must be at least 1")
    if not 0 <= min_freq <= 1:
        raise ValueError("min_freq must be between 0 and 1")
    if n_select < 1:
        raise ValueError("n_select must be at least 1")
    if all_feature_names is None:
        raise ValueError("all_feature_names cannot be None")

    rows = []
    for f in all_feature_names:
        appearances = freq_counter.get(f, 0)
        stability = appearances / runs_used
        mrank = (rank_sum[f] / rank_n[f]) if rank_n[f] > 0 else np.nan
        mabs = (mean_abs_sum[f] / rank_n[f]) if rank_n[f] > 0 else np.nan
        rows.append(
            {
                "feature": f,
                "appearances_topk": appearances,
                "stability_freq": stability,
                "mean_rank": mrank,
                "mean_abs_shap": mabs,
                "runs_counted": rank_n[f],
            }
        )
    df = pd.DataFrame(rows)

    survivors = df[df["stability_freq"] >= min_freq].copy()

    if len(survivors) < n_select:
        logger.info(
            "Only %d peptides passed frequency >= %.2f; relaxing the threshold",
            len(survivors),
            min_freq,
        )
        survivors = df.sort_values(
            ["stability_freq", "mean_abs_shap"], ascending=[False, False]
        ).head(max(n_select, len(survivors)))

    selected = (
        survivors.sort_values(["mean_rank", "mean_abs_shap"], ascending=[True, False])
        .head(n_select)
        .copy()
    )
    selected["selected_rank"] = range(1, len(selected) + 1)

    return df, selected


# ---------- (1) Compute per-run and aggregated antigen scores ----------
def compute_antigen_score_from_joblibs(
    file_path_template: str,
    start: int,
    end: int,
    top_peptides: Iterable[str],
    time_train: pd.Series,
    time_test: pd.Series,
    min_timepoint: int = 1,
    max_timepoint: int = 25,
    agg: str = "median",
    shap_col: str = "train_shap_values",
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Calculate antigen scores from SHAP values stored across multiple joblib runs.

    For each run:

    1. Load the joblib result.
    2. Select the requested peptide SHAP columns.
    3. Sum their SHAP values for each sample.
    4. Align scores and survival times using their common sample IDs.
    5. Calculate the time-dependent AUC.
    6. Aggregate sample scores across runs using the mean or median.

    Individual runs are allowed to contain only a subset of ``time_test``
    samples. The final score is calculated from all runs available for each
    sample.

    Parameters
    ----------
    file_path_template
        Joblib path template containing a positional placeholder for the run
        number, for example ``"/results/run_{}.joblib"``.
    start
        First run number, inclusive.
    end
        Last run number, inclusive.
    top_peptides
        Peptide feature names used to calculate the antigen score.
    time_train
        Training survival times. Positive values represent events and negative
        values represent censored observations.
    time_test
        Survival times corresponding to the SHAP samples.
    min_timepoint
        Minimum requested time point for dynamic AUC calculation.
    max_timepoint
        Maximum requested time point, exclusive.
    agg
        Aggregation across runs: ``"mean"`` or ``"median"``.
    shap_col
        Key containing the SHAP DataFrame in each joblib artifact.

    Returns
    -------
    antigen_score
        Aggregated antigen score indexed by sample ID.
    time_dependent_auc
        Per-run time-dependent AUC values.
    mean_auc
        Mean time-dependent AUC for each run.

    Raises
    ------
    ValueError
        If arguments or sample indices are invalid.
    RuntimeError
        If none of the requested runs produces a valid antigen score.
    """
    if start > end:
        raise ValueError("start must be less than or equal to end")

    if min_timepoint >= max_timepoint:
        raise ValueError("min_timepoint must be smaller than max_timepoint")

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")

    top_peptides = list(dict.fromkeys(map(str, top_peptides)))
    if not top_peptides:
        raise ValueError("top_peptides must contain at least one feature")

    if not isinstance(time_train, pd.Series):
        raise TypeError("time_train must be a pandas Series")

    if not isinstance(time_test, pd.Series):
        raise TypeError("time_test must be a pandas Series")

    # Work with consistently formatted sample IDs. This handles artifacts in
    # which sample IDs were stored as integers in one object and strings in
    # another.
    time_test_aligned = time_test.copy()
    time_test_aligned.index = time_test_aligned.index.map(str)

    if time_test_aligned.index.has_duplicates:
        duplicated = (
            time_test_aligned.index[
                time_test_aligned.index.duplicated()
            ]
            .unique()
            .tolist()[:10]
        )
        raise ValueError(
            "time_test contains duplicate sample IDs after converting them "
            f"to strings: {duplicated}"
        )

    runs = range(start, end + 1)
    time_points = np.arange(
        min_timepoint,
        max_timepoint,
        step=1,
    )

    per_run_scores: list[pd.Series] = []

    time_dependent_auc_full = pd.DataFrame(
        np.nan,
        index=runs,
        columns=time_points,
        dtype=float,
    )

    mean_auc_full = pd.Series(
        np.nan,
        index=runs,
        dtype=float,
        name="Mean AUC",
    )

    files_found = 0
    partial_runs = 0
    missing_shap_runs = 0
    missing_feature_runs = 0
    failed_auc_runs = 0

    for run_number in tqdm(
        runs,
        desc="Computing per-run antigen scores",
    ):
        path = file_path_template.format(run_number)

        if not os.path.isfile(path):
            continue

        files_found += 1

        try:
            result = load(path)
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            AttributeError,
            ModuleNotFoundError,
            TypeError,
            ValueError,
        ) as error:
            logger.warning(
                "Skipping run %d because the joblib file could not be loaded: %s",
                run_number,
                error,
            )
            continue

        if not hasattr(result, "get"):
            logger.warning(
                "Skipping run %d: joblib content is not a mapping",
                run_number,
            )
            continue

        shap_df = result.get(shap_col)

        if not isinstance(shap_df, pd.DataFrame):
            missing_shap_runs += 1
            logger.debug(
                "Skipping run %d: key %r is not a DataFrame",
                run_number,
                shap_col,
            )
            continue

        shap_df = shap_df.copy()
        shap_df.columns = shap_df.columns.map(str)
        shap_df.index = shap_df.index.map(str)

        if shap_df.index.has_duplicates:
            duplicated = (
                shap_df.index[shap_df.index.duplicated()]
                .unique()
                .tolist()[:10]
            )
            logger.warning(
                "Skipping run %d because its SHAP DataFrame contains duplicate "
                "sample IDs: %s",
                run_number,
                duplicated,
            )
            continue

        if shap_df.columns.has_duplicates:
            duplicated = (
                shap_df.columns[shap_df.columns.duplicated()]
                .unique()
                .tolist()[:10]
            )
            logger.warning(
                "Skipping run %d because its SHAP DataFrame contains duplicate "
                "feature names: %s",
                run_number,
                duplicated,
            )
            continue

        available_peptides = [
            peptide
            for peptide in top_peptides
            if peptide in shap_df.columns
        ]

        if not available_peptides:
            missing_feature_runs += 1
            logger.debug(
                "Skipping run %d: none of the selected peptides are present",
                run_number,
            )
            continue

        # min_count=1 prevents a sample whose selected SHAP values are all NaN
        # from being silently converted to an antigen score of zero.
        run_scores = shap_df[available_peptides].sum(
            axis=1,
            min_count=1,
        )

        run_scores = pd.to_numeric(
            run_scores,
            errors="coerce",
        )

        # Preserve the order of time_test while selecting the overlap.
        common_samples = time_test_aligned.index[
            time_test_aligned.index.isin(run_scores.index)
        ]

        if len(common_samples) == 0:
            logger.warning(
                "Skipping run %d: no common sample IDs between SHAP values "
                "and time_test",
                run_number,
            )
            continue

        if len(common_samples) < len(time_test_aligned):
            partial_runs += 1

        run_scores = run_scores.reindex(common_samples)
        run_time_test = time_test_aligned.reindex(common_samples)

        finite_scores = np.isfinite(
            run_scores.to_numpy(dtype=float)
        )

        if not finite_scores.all():
            valid_samples = run_scores.index[finite_scores]
            run_scores = run_scores.loc[valid_samples]
            run_time_test = run_time_test.loc[valid_samples]

        if run_scores.empty:
            logger.warning(
                "Skipping run %d: no finite antigen scores remain",
                run_number,
            )
            continue

        try:
            auc_values, mean_auc = calculate_cumulative_dynamic_auc(
                time_train,
                run_time_test,
                run_scores,
                time_points,
            )
        except ValueError as error:
            failed_auc_runs += 1
            logger.warning(
                "Skipping run %d because its AUC could not be calculated: %s",
                run_number,
                error,
            )
            continue

        time_dependent_auc_full.loc[
            run_number,
            auc_values.index,
        ] = auc_values.to_numpy()

        mean_auc_full.loc[run_number] = mean_auc

        run_scores.name = f"run_{run_number}"
        per_run_scores.append(run_scores)

    if not per_run_scores:
        raise RuntimeError(
            "No runs produced antigen scores. "
            f"Requested runs={end - start + 1}, "
            f"files found={files_found}, "
            f"missing SHAP DataFrames={missing_shap_runs}, "
            f"runs without selected peptides={missing_feature_runs}, "
            f"runs with failed AUC={failed_auc_runs}. "
            "Check file_path_template, shap_col, top_peptides, and sample IDs."
        )

    # Different runs may contain different subsets of samples. Concatenation
    # aligns them by sample ID and aggregation ignores unavailable runs.
    per_run_df = pd.concat(
        per_run_scores,
        axis=1,
    )

    if agg == "mean":
        antigen_score = per_run_df.mean(
            axis=1,
            skipna=True,
        )
    else:
        antigen_score = per_run_df.median(
            axis=1,
            skipna=True,
        )

    antigen_score = antigen_score.dropna()
    antigen_score.name = "Antigen Score"

    samples_without_scores = time_test_aligned.index.difference(
        antigen_score.index
    )

    if len(samples_without_scores) > 0:
        logger.warning(
            "%d of %d test samples were not covered by any valid run",
            len(samples_without_scores),
            len(time_test_aligned),
        )

    logger.info(
        "Antigen scores calculated from %d of %d requested runs "
        "(%d runs used partial sample coverage)",
        len(per_run_scores),
        end - start + 1,
        partial_runs,
    )

    return (
        antigen_score,
        time_dependent_auc_full,
        mean_auc_full,
    )


# ---------- (2) Build tidy DF, find cutpoint, save ----------
def build_antigen_score_df(
    antigen_score: pd.DataFrame,
    y_time: pd.Series,
    y_event: pd.Series,
    out_dir: str,
    save_prefix: str = "antigen_scores_train",
    threshold: Optional[float] = None,  # allow fixed threshold
    threshold_method: str = "maxstat",  # "maxstat" (R), "median", or "percentile"
    percentile: float = 0.5,  # used if threshold_method == "percentile"
    r_seed: int = 1,
    r_B: int = 9999,
) -> Tuple[pd.DataFrame, float, str, Optional[str]]:
    """
    Create a tidy dataframe with OS, Status, Antigen Score, and dichotomized label.
    Optionally computes R maxstat cutpoint; otherwise median/percentile.
    Saves:
      - RAW (no dichotomization)
      - with-cut (adds dichotomized column)
    Returns: (df_with_cut, threshold, raw_csv_path, with_cut_csv_path)
    """
    output_dir = Path(out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if r_B < 1:
        raise ValueError("r_B must be at least 1")
    required_score_columns = {"Antigen Score", "Antigen Score (Scaled)"}
    missing = sorted(required_score_columns - set(antigen_score.columns))
    if missing:
        raise KeyError(f"antigen_score is missing columns: {missing}")

    # 1) Merge
    df = pd.DataFrame(
        {
            "OS months": pd.to_numeric(y_time, errors="coerce"),
            "OS Status": pd.to_numeric(y_event, errors="coerce"),
            "Antigen Score": pd.to_numeric(
                antigen_score["Antigen Score"], errors="coerce"
            ),
            "Antigen Score (Scaled)": pd.to_numeric(
                antigen_score["Antigen Score (Scaled)"], errors="coerce"
            ),
        },
        index=antigen_score.index,
    ).dropna(
        subset=["OS months", "OS Status", "Antigen Score", "Antigen Score (Scaled)"]
    )
    if df.empty:
        raise ValueError("No complete samples remain in the antigen-score data")

    # 2) Save RAW for provenance
    raw_path = output_dir / f"{save_prefix}.csv"
    df.rename_axis("SampleName").reset_index().to_csv(raw_path, index=False)

    # 3) Find threshold
    if threshold is None:
        if threshold_method.lower() == "maxstat":
            # Try rpy2; if missing, fall back to running a tiny Rscript; otherwise fallback to median.
            threshold = _compute_maxstat_threshold_via_Rcsv(
                str(raw_path), r_seed=r_seed, r_B=r_B
            )
            if threshold is None or np.isnan(threshold):
                logger.warning("Maxstat failed; falling back to the median cut")
                threshold = float(df["Antigen Score (Scaled)"].median())
        elif threshold_method.lower() == "median":
            threshold = float(df["Antigen Score (Scaled)"].median())
        elif threshold_method.lower() == "percentile":
            if not (0.0 < percentile < 1.0):
                raise ValueError("percentile must be in (0,1)")
            threshold = float(
                df["Antigen Score (Scaled)"].quantile(percentile)
            )
        else:
            raise ValueError(
                "threshold_method must be 'maxstat', 'median', or 'percentile'"
            )
    else:
        # Use provided threshold
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError("Provided threshold is not finite.")

    # 4) Add dichotomized label (high=1 if >= threshold)
    df_out = df.copy()
    df_out["Antigen Score (Dichotomized)"] = (
        df_out["Antigen Score (Scaled)"] >= threshold
    ).astype(int)

    with_cut_path = output_dir / f"{save_prefix}_with_cut.csv"
    df_out.rename_axis("SampleName").to_csv(with_cut_path, index=True)

    return df_out, threshold, str(raw_path), str(with_cut_path)


# ---------- Helper: call R maxstat using the saved CSV ----------
def _compute_maxstat_threshold_via_Rcsv(
    csv_path: str, r_seed: int = 1, r_B: int = 9999
) -> Optional[float]:
    """
    Minimal dependency helper: tries to call Rscript with maxstat on a CSV.
    Requires R with packages 'maxstat' and 'survival' available in PATH.
    Returns threshold (float) or None if it fails.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("Rscript") is None:
        return None

    r_csv_path = str(Path(csv_path).expanduser().resolve()).replace("\\", "/")
    r_csv_path = r_csv_path.replace('"', '\\"')
    r_code = f"""
    suppressMessages(library(maxstat))
    suppressMessages(library(survival))
    df <- read.csv("{r_csv_path}", stringsAsFactors=FALSE)
    # Expect columns: SampleName, OS.months, OS.Status, Antigen.Score
    # (We wrote them that way in build_antigen_score_df)
    # But if SampleName wasn't written, adapt names accordingly:
    nm <- names(df)
    if (!"OS.months" %in% nm) {{
      stop("OS.months not found")
    }}
    if (!"OS.Status" %in% nm) {{
      stop("OS.Status not found")
    }}
    if (!"Antigen.Score..Scaled." %in% nm) {{
      stop("Antigen Score (Scaled) not found")
    }}
    surv_obj <- Surv(df$OS.months, df$OS.Status)
    set.seed({int(r_seed)})
    res <- try(maxstat.test(
      surv_obj ~ `Antigen.Score..Scaled.`,
      data    = df,
      smethod = "LogRank",
      pmethod = "condMC",
      B       = {int(r_B)}
    ), silent=TRUE)
    if (inherits(res, "try-error")) {{
      cat("NA\\n")
    }} else {{
      cat(as.numeric(res$estimate), "\\n")
    }}
    """

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(r_code)
        rfile = fh.name

    try:
        out = subprocess.check_output(["Rscript", rfile], universal_newlines=True)
        out = out.strip()
        return float(out) if out not in ("", "NA", "NaN") else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    finally:
        Path(rfile).unlink(missing_ok=True)


"""""" """""" ""
"""" PLOTS """
"""""" """""" ""
#################################
#           PLOTS               #
#################################


# Example usage (use real data for these variables):
# plot_time_dependent_auc(auc_matrix_values, samples_at_risk, mean_auc, ci_lo, ci_up, limit_timepoints, timepoints_highligts, max_timepoint)
def plot_time_dependent_auc(
    auc_values: pd.DataFrame,  # rows=runs, cols=timepoints
    samples_at_risk_df: pd.DataFrame,
    mean_auc: pd.Series,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    extension_time_points: Optional[np.ndarray] = None,
    time_points_highlight: Optional[np.ndarray] = None,
    max_time_point: Optional[float] = None,
    mean_linewidth: float = 2,
    run_lines_alpha: float = 0.08,
    run_lines_color: str = "dodgerblue",
    run_linewidth: float = 0.8,
    time_measure: str = "Months",
    color_auc: str = "dodgerblue",
    color_mean: str = "dodgerblue",
    color_ci: str = "dodgerblue",
    labels: Optional[Sequence[str]] = None,
    suffix_file: Optional[str] = None,
    figures_dir: str = "./",
    save_fig: bool = False,
) -> plt.Figure:
    """
    Plots time-dependent AUC over follow-up time with a table displaying samples at risk, censored, and events.
    The table is placed below the AUC plot as a horizontal table aligned with the x-axis.

    Parameters
    ----------
    auc_values : pd.DataFrame
        The AUC values corresponding to each time point.
    samples_at_risk_df : pd.DataFrame
        DataFrame containing 'At Risk', 'Censored', and 'Events' per time point. Its index should
        correspond to time_points.
    mean_auc : pd.Series
        The mean AUC value to be displayed as a horizontal line.
    ci_lower : np.ndarray, optional
    ci_upper : np.ndarray, optional
    extension_time_points : array-like, optional
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
    mean_linewidth : float
    run_linewidth : float
        Color used for the AUC line.
    run_lines_alpha : float
    run_lines_color : str, default 'gray'
        Color used for the CI lines.
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
    if not isinstance(auc_values, pd.DataFrame) or auc_values.empty:
        raise ValueError("auc_values must be a non-empty DataFrame")
    required_risk_columns = {"At risk", "Censored", "Events"}
    missing_risk_columns = sorted(
        required_risk_columns - set(samples_at_risk_df.columns)
    )
    if missing_risk_columns:
        raise KeyError(
            f"samples_at_risk_df is missing columns: {missing_risk_columns}"
        )

    auc = auc_values.copy()
    try:
        auc.columns = pd.to_numeric(auc.columns, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("auc_values columns must be numeric time points") from error
    auc = auc.sort_index(axis=1)
    auc = auc.apply(pd.to_numeric, errors="coerce")
    if auc.notna().sum().sum() == 0:
        raise ValueError("auc_values contains no finite numeric values")

    mean_auc_values = pd.to_numeric(
        pd.Series(mean_auc),
        errors="coerce",
    ).dropna()
    if mean_auc_values.empty:
        raise ValueError("mean_auc contains no numeric values")

    fig = plt.figure(figsize=(4.5, 3.5), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[0.8, 0.15], figure=fig)
    ax_auc = fig.add_subplot(gs[0])

    # Plot AUC curves and mean AUC lines with CI
    time_points = auc.columns.to_numpy(dtype=float)

    # Mean curve to feed as auc_values

    for i in range(auc.shape[0]):
        ax_auc.plot(
            time_points,
            auc.iloc[i].to_numpy(dtype=float),
            color=run_lines_color,
            alpha=run_lines_alpha,
            linewidth=run_linewidth,
        )
    mean_curve = auc.mean(axis=0).to_numpy(dtype=float)
    ax_auc.plot(
        time_points,
        mean_curve,
        color=color_auc,
        linestyle="-",
        linewidth=mean_linewidth,
    )  # , label=labels[0])

    mean_line = ax_auc.axhline(
        y=float(mean_auc_values.mean()),
        color=color_mean,
        ls="--",
        lw=1.5,
    )
    rand_line = ax_auc.axhline(y=0.5, color="gray", ls="--", lw=1.5)

    # Plot the confidence interval if provided
    if ci_lower is not None and ci_upper is not None:
        lower = np.asarray(ci_lower, dtype=float)
        upper = np.asarray(ci_upper, dtype=float)
        if lower.shape != time_points.shape or upper.shape != time_points.shape:
            raise ValueError(
                "ci_lower and ci_upper must match the number of AUC time points"
            )
        ax_auc.fill_between(
            time_points,
            lower,
            upper,
            color=color_ci,
            alpha=0.2,
        )
    elif ci_lower is not None or ci_upper is not None:
        raise ValueError("ci_lower and ci_upper must be provided together")

    ci = np.nanpercentile(mean_auc_values.to_numpy(dtype=float), [2.5, 97.5])
    ci_handle = Patch(
        facecolor=color_ci, edgecolor="none", alpha=0.7
    )  # stronger alpha only in legend
    legend_labels = [
        f"Antigen Score (Mean AUC = {mean_auc_values.mean():.2f})",
        f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]",
        "Random Classifier",
    ]
    if labels is not None:
        for index, label in enumerate(labels[:3]):
            legend_labels[index] = str(label)
    ax_auc.legend(
        [mean_line, ci_handle, rand_line],
        legend_labels,
        loc="lower right",
        fontsize=11,
    )

    ax_auc.set_xlabel(f"Time ({time_measure})", fontsize=13)
    ax_auc.set_ylabel("Time-Dependent AUC", fontsize=13)
    ax_auc.set_ylim(0.0, 1.05)
    ax_auc.grid(True, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    ax_auc.tick_params(axis="y", labelsize=11)
    # Highlight specific time points if provided
    if time_points_highlight is not None:
        highlights = np.asarray(time_points_highlight, dtype=float)
        highlights = highlights[np.isin(highlights, time_points)]
        auc_values_highlight = auc.mean(axis=0).reindex(highlights)
        ax_auc.scatter(
            highlights,
            auc_values_highlight.to_numpy(dtype=float),
            color="black",
            s=30,
            zorder=4,
        )
        if extension_time_points is not None:
            extension_points = np.asarray(extension_time_points, dtype=float)
            ax_auc.set_xticks(extension_points)
            ax_auc.set_xticklabels(extension_points.astype(int), fontsize=11)
        else:
            ax_auc.set_xticks(highlights)
            ax_auc.set_xticklabels(highlights.astype(int), fontsize=11)
        # If max_time_point is provided, filter out any time points greater than max_time_point.
        if max_time_point is not None:
            tp_high = [tp for tp in highlights if tp <= max_time_point]
            if max_time_point not in tp_high:
                tp_high.append(max_time_point)
            tp_high = sorted(tp_high)
        else:
            max_time_point = max(time_points)
            tp_high = sorted(highlights)[
                :-1
            ]  # remove last to avoid overlap with added max_point
            tp_high.append(max_time_point)
    else:
        tp_high = time_points  # if no highlight is provided, use all time points

    # Bottom panel: Table for samples at risk, censored, and events
    ax_table = fig.add_subplot(gs[1])
    ax_table.set_xticks(time_points)
    ax_table.set_xticklabels([])  # No tick labels on x-axis
    # Set y-tick positions and labels for the table (order: Events, Censored, At Risk)
    y_positions = np.array([0.2, 0.5, 0.8])
    ax_table.set_yticks(y_positions)
    ax_table.set_yticklabels(["", "", ""], ha="right", fontsize=11)
    ax_table.grid(False)
    ax_table.set_facecolor("white")
    ax_table.set_frame_on(False)
    ax_table.tick_params(axis="both", which="both", length=0)

    if max_time_point is not None:
        ax_auc.set_xlim(0, max_time_point)  # Ensure x-axis covers max time point given
        ax_table.set_xlim(
            0, max_time_point
        )  # Ensure x-axis covers max time point given
    else:
        upper = max(float(t) for t in time_points)
        ax_auc.set_xlim(0, upper)  # Ensure x-axis covers all time points
        ax_table.set_xlim(0, upper)  # Ensure x-axis covers all time points
        # ax_auc.set_xlim(0, max(time_points))  # Ensure x-axis covers all time points
        # ax_table.set_xlim(0, max(time_points))  # Ensure x-axis covers all time points

    # Iterate over each highlighted time point and place text for table values
    if extension_time_points is not None:
        tps = np.asarray(extension_time_points, dtype=float)
    else:
        tps = tp_high

    label_x = -2  # or any value that fits nicely
    ax_table.text(
        label_x, 0.8, "At risk", ha="right", va="center", fontsize=11, color="black"
    )
    ax_table.text(label_x, 0.5, "Censored", ha="right", va="center", fontsize=11)
    ax_table.text(label_x, 0.2, "Events", ha="right", va="center", fontsize=11)

    for time_point in tps:  # ¼tp_high:
        try:
            at_risk_val = samples_at_risk_df.loc[time_point, "At risk"]
            censored_val = samples_at_risk_df.loc[time_point, "Censored"]
            events_val = samples_at_risk_df.loc[time_point, "Events"]
        except KeyError:
            # Skip time points not found in the DataFrame
            continue

        ax_table.text(
            time_point,
            0.2,
            f"{events_val}",
            ha="right",
            va="center",
            fontsize=11,
            color="black",
        )
        ax_table.text(
            time_point,
            0.5,
            f"{censored_val}",
            ha="right",
            va="center",
            fontsize=11,
            color="black",
        )
        ax_table.text(
            time_point,
            0.8,
            f"{at_risk_val}",
            ha="right",
            va="center",
            fontsize=11,
            color="black",
        )

    # plt.tight_layout()

    # Save the figure if requested
    if save_fig:
        if suffix_file is None:
            suffix_file = "default"
        output_dir = Path(figures_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = output_dir / f"time-dependent_auc_{suffix_file}.pdf"
        svg_path = output_dir / f"time-dependent_auc_{suffix_file}.svg"

        fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
        fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    return fig


def plot_kaplan_meier(
    df: pd.DataFrame,
    time_column: str,
    event_column: str,
    group_column: str,
    labels: Dict[Union[int, str], str],
    color_lines: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    tick_space: int = 3,
    suffix_file: Optional[str] = None,
    loc_legend: str = "best",
    save_fig: bool = False,
    figures_dir: str = "./",
) -> Tuple[plt.Figure, plt.Axes]:
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
    color_lines : List of str
        Color for the KM lines
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot. If None, a new figure and axes are created.
    title : str, optional
        Title for the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    tick_space : int, optional, defaul 3
        Space between ticks on the x-axis.
    suffix_file : str, optional default None
        Suffix used in the filename when saving.
    loc_legend : str, default 'best'
        Location for the legend.
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
    required = {time_column, event_column, group_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing Kaplan-Meier columns: {missing}")
    if tick_space <= 0:
        raise ValueError("tick_space must be positive")

    analysis_df = df.loc[:, [time_column, event_column, group_column]].copy()
    analysis_df[time_column] = pd.to_numeric(
        analysis_df[time_column],
        errors="coerce",
    )
    analysis_df[event_column] = pd.to_numeric(
        analysis_df[event_column],
        errors="coerce",
    )
    analysis_df = analysis_df.dropna()
    if analysis_df.empty:
        raise ValueError("No complete observations remain for Kaplan-Meier plotting")

    unique_groups = sorted(
        pd.unique(analysis_df[group_column]),
        key=lambda value: str(value),
    )
    if len(unique_groups) != 2:
        raise ValueError(
            f"Expected exactly two groups in '{group_column}', but found: {unique_groups}"
        )

    # Define colors explicitly for each group (or you can extend this mapping if needed)
    # group_colors = {unique_groups[0]: "#1f77b4", unique_groups[1]: "#d62728"}
    if color_lines is None:
        color_lines = ["#628D56", "#9B5C97"]
    if len(color_lines) < 2:
        raise ValueError("color_lines must contain at least two colors")
    missing_labels = [group for group in unique_groups if group not in labels]
    if missing_labels:
        raise KeyError(f"labels has no entries for groups: {missing_labels}")
    group_colors = {unique_groups[0]: color_lines[0], unique_groups[1]: color_lines[1]}

    # Create figure and axis if not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    else:
        fig = ax.get_figure()

    kmf_list = []
    # Plot survival functions in order based on the sorted unique groups
    for group_value in unique_groups:
        mask = analysis_df[group_column] == group_value
        kmf = KaplanMeierFitter()
        kmf.fit(
            analysis_df.loc[mask, time_column],
            event_observed=analysis_df.loc[mask, event_column],
            label=labels[group_value],
        )
        kmf.plot_survival_function(
            ax=ax, ci_show=True, color=group_colors[group_value], show_censors=True
        )

        kmf_list.append(kmf)

    # Perform log-rank test between the two groups
    group1 = analysis_df[analysis_df[group_column] == unique_groups[0]]
    group2 = analysis_df[analysis_df[group_column] == unique_groups[1]]
    if not group1.empty and not group2.empty:
        results = logrank_test(
            group1[time_column],
            group2[time_column],
            event_observed_A=group1[event_column],
            event_observed_B=group2[event_column],
        )
        p_value = results.p_value
        p_value_text = f"p = {format_pval(p_value)}"
        ax.text(
            0.1,
            0.05,
            p_value_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    else:
        logger.warning("One of the groups is empty; log-rank test not performed.")

    ax.grid(True, color="gray", linestyle="--", alpha=0.85)  # linewidth=0.8, )

    max_time = math.ceil(float(analysis_df[time_column].max()))
    custom_ticks = np.arange(0, max_time + 1, tick_space).astype(
        int
    )  # e.g. every 3 months: 0, 3, 6, 9, ...

    # Apply limits and ticks to main axis BEFORE at-risk counts
    ax.set_xlim(0, max_time)  # force axis to start at 0 (at frame intersection)
    ax.set_xticks(custom_ticks)  # custom tick spacing
    ax.set_xticklabels(custom_ticks)  # ([int(t) for t in custom_ticks]

    # Add at-risk counts to the plot (from lifelines)
    if len(kmf_list) == 2:
        add_at_risk_counts(kmf_list[0], kmf_list[1], ax=ax, fontsize=12)
    else:
        logger.warning(
            "At-risk counts could not be added because fewer than two groups were found."
        )

    # Right after add_at_risk_counts(...)
    fig = ax.get_figure()
    if len(fig.axes) > 1:
        ax2 = fig.axes[-1]
        ax.set_zorder(ax2.get_zorder() + 1)

    # Customize axes
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="x", labelsize=12)  # x-axis tick labels
    ax.tick_params(axis="y", labelsize=12)  # y-axis tick labels
    ax.legend(fontsize=12, loc=loc_legend)

    fig.tight_layout()

    # Save figure if required
    if save_fig:
        if suffix_file is None:
            suffix_file = "default"
        output_dir = Path(figures_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"kaplan_meier_{suffix_file}.pdf"
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)

        save_path = output_dir / f"kaplan_meier_{suffix_file}.svg"
        fig.savefig(save_path, format="svg", bbox_inches="tight", facecolor="white")
        logger.info("Figure saved to %s", save_path)


    return fig, ax


def plot_parametric_model_qq(
    survival_time: pd.Series,
    survival_status: pd.Series,
    prefix_file: str = None,
    save_path: str = "./",
    save_fig: bool = False,
    fig_size=(8, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates QQ plots for a set of parametric survival models using the clinical metadata,
    prints their AIC values, and saves the figure.

    Parameters:
        survival_time (pd.Series ): Series containing at least the columns
            "OS months" (duration).
        survival_status (pd.Series ): Series containing at least the columns
            "OS Status" (event indicator).
        prefix_file (str): A label used in the filename for the saved figure.
        save_path (str): The directory to save the figure.
        save_fig (bool): A boolean indicating whether to save the figure.
        fig_size (tuple): A tuple containing the width and height of the figure.

    Returns:
        plt.Figure: The figure object.
    """
    from lifelines import (
        ExponentialFitter,
        LogLogisticFitter,
        LogNormalFitter,
        WeibullFitter,
    )

    # Define the models to be used.
    models = [
        WeibullFitter(),
        LogNormalFitter(),
        LogLogisticFitter(),
        ExponentialFitter(),
    ]

    # Create a 2x2 grid of subplots and flatten to a 1D array for easier iteration.
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axes = axes.flatten()

    # Loop over models and corresponding axes.
    for ax, model in zip(axes, models):
        model.fit(survival_time, survival_status)
        qq_plot(model, ax=ax)
        logger.info(
            "AIC for %s: %.4f",
            model.__class__.__name__,
            model.AIC_,
        )

    # Construct the figure path and save the figure.
    if save_fig:
        if prefix_file is None:
            filename = "QQplot_Parametric_Models.png"
        else:
            filename = f"QQplot_Parametric_Models_{prefix_file}.pdf"
        output_dir = Path(save_path).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / filename
        fig.savefig(fig_path, dpi=600, bbox_inches="tight")

    return fig, axes


def forestplot_survival(
    multivariate_df: pd.DataFrame,
    save_path: Union[str, Path] = "./",
    title: str = "Multivariate Cox Proportional Hazards Model",
    suffix_file: Optional[str] = None,
    save_fig: bool = True,
    figsize: tuple[float, float] = (6.5, 4),
) -> plt.Figure:
    """
    Create and optionally save a forest plot for a multivariable Cox model.

    The DataFrame must include:
      - Variable
      - P-value
      - Group
      - N
      - log_HR
      - log_95% CI Lower
      - log_95% CI Upper
      - Hazard Ratio
      - 95% CI Lower
      - 95% CI Upper

    Parameters
    ----------
    multivariate_df
        DataFrame containing the forest-plot data.
    save_path
        Directory in which to save the figure.
    title
        Figure title.
    suffix_file
        Optional filename suffix.
    save_fig
        Whether to save PDF and SVG versions.
    figsize
        Figure width and height.

    Returns
    -------
    matplotlib.figure.Figure
        Created figure.
    """
    required = {
        "Variable",
        "P-value",
        "Group",
        "N",
        "log_HR",
        "log_95% CI Lower",
        "log_95% CI Upper",
        "Hazard Ratio",
        "95% CI Lower",
        "95% CI Upper",
    }

    missing = sorted(required - set(multivariate_df.columns))
    if missing:
        raise KeyError(
            f"Forest-plot data is missing columns: {missing}"
        )

    if multivariate_df.empty:
        raise ValueError("multivariate_df cannot be empty")

    variables = multivariate_df["Variable"]
    p_values = multivariate_df["P-value"]
    groups = multivariate_df["Group"]
    sample_sizes = multivariate_df["N"]

    hr = multivariate_df["log_HR"]
    lower_ci = multivariate_df["log_95% CI Lower"]
    upper_ci = multivariate_df["log_95% CI Upper"]

    actual_hr = multivariate_df["Hazard Ratio"]
    actual_lc = multivariate_df["95% CI Lower"]
    actual_uc = multivariate_df["95% CI Upper"]

    y_positions = np.arange(len(variables))

    # Create label and forest-plot panels with a shared y-axis.
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 4]},
        sharey=True,
    )

    fig.subplots_adjust(wspace=1)

    # ------------------------------------------------------------------
    # Left subplot: variable and group labels
    # ------------------------------------------------------------------
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(
        variables.values,
        ha="left",
        va="center",
    )

    # Remove the short y-axis tick lines while retaining the text labels.
    ax1.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        length=0,
        labelleft=True,
        labelsize=13,
    )

    for label in ax1.get_yticklabels():
        label.set_fontweight("bold")

    combined_labels = [
        f"{group} (n={sample_size})"
        for group, sample_size in zip(groups, sample_sizes)
    ]

    for position, label in enumerate(combined_labels):
        ax1.text(
            1.7,
            position,
            label,
            transform=ax1.get_yaxis_transform(),
            fontsize=13,
            ha="left",
            va="center",
            clip_on=False,
        )

    # Hide all borders around the label panel.
    for spine in ax1.spines.values():
        spine.set_alpha(0.0)

    # Hide the entire x-axis of the label panel.
    ax1.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )

    # ------------------------------------------------------------------
    # Right subplot: hazard ratios and confidence intervals
    # ------------------------------------------------------------------
    x_err_lower = np.asarray(hr) - np.asarray(lower_ci)
    x_err_upper = np.asarray(upper_ci) - np.asarray(hr)

    ax2.errorbar(
        hr,
        y_positions,
        xerr=[
            x_err_lower,
            x_err_upper,
        ],
        fmt="s",
        color="black",
        ecolor="dodgerblue",
        elinewidth=2,
        capsize=5,
    )

    # HR = 1 reference line.
    ax2.axvline(
        1,
        color="black",
        linestyle="--",
    )

    # Add p-values to the right of the confidence intervals.
    x_text = np.nanmax(upper_ci) + 0.5

    for position, p_value in enumerate(p_values):
        numeric_p = pd.to_numeric(
            pd.Series([p_value]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(numeric_p):
            continue

        significance = get_significance_stars(numeric_p)
        p_value_text = (
            f"{format_pval(numeric_p)} {significance}"
        )

        ax2.text(
            x_text,
            position,
            p_value_text,
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold" if numeric_p < 0.05 else "normal",
        )

    # Add HR and CI text for non-reference categories.
    for position, (h, actual, lower, upper) in enumerate(
        zip(
            hr,
            actual_hr,
            actual_lc,
            actual_uc,
        )
    ):
        if pd.notna(actual) and not np.isclose(actual, 1.0):
            ax2.text(
                h,
                position + 0.5,
                f"{actual:.2f} ({lower:.2f}-{upper:.2f})",
                ha="center",
                va="center",
                fontsize=13,
            )

    ax2.set_xlim(
        np.nanmin(lower_ci) - 0.5,
        np.nanmax(upper_ci) + 0.5,
    )

    ax2.set_xlabel(
        "adjusted Hazard Ratio (95% CI)",
        fontsize=13,
    )
    ax2.set_title(
        title,
        fontsize=12,
    )

    ax2.tick_params(
        axis="x",
        labelsize=13,
    )

    # Remove the short y-axis tick lines from the forest panel.
    ax2.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        length=0,
        labelleft=False,
        labelright=False,
    )

    # Hide all borders around the forest panel.
    for spine in ax2.spines.values():
        spine.set_alpha(0.0)

    # Show vertical grid lines only.
    ax2.grid(
        False,
        axis="y",
    )
    ax2.grid(
        True,
        axis="x",
        linestyle="-",
        color="black",
        alpha=0.15,
    )

    fig.tight_layout(pad=1)

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    if save_fig:
        if suffix_file is None:
            suffix_file = "default"

        output_dir = Path(save_path).expanduser()
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_path = output_dir / f"forest_plot_{suffix_file}.pdf"
        svg_path = output_dir / f"forest_plot_{suffix_file}.svg"

        fig.savefig(
            pdf_path,
            dpi=600,
            bbox_inches="tight",
        )
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            facecolor="white",
        )

    return fig