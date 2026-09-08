import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.axes as maxes

# Plots
import matplotlib.pyplot as plt
import numpy as np

# Data Analysis
import pandas as pd
import seaborn as sns

# ML
import shap
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.titlecolor": "black",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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


""" SHAP helpers """


def generate_feature_importance_table(
    values: Union[pd.DataFrame, np.ndarray],
    features: pd.DataFrame,
    target: pd.Series,
    oligos_metadata: pd.DataFrame,
    group_tests: Optional[Sequence[str]] = None,
    filename_label: Optional[str] = None,
    with_oligos: bool = True,
    with_additional_features: bool = False,
    with_run_plates: bool = False,
    figures_dir: str = "./",
) -> pd.DataFrame:
    """Generates feature importance table from SHAP values with metadata and group statistics.

    Args:
        values: SHAP values array
        features: Feature values DataFrame with SampleName index
        target: Target values Series with same index as features
        oligos_metadata: Oligos metadata DataFrame indexed by peptide ID
        group_tests: List of two group names for computing prevalence statistics
        filename_label: Name of the ouptut file name
        with_oligos: Include oligos metadata
        with_additional_features: Include additional feature columns
        with_run_plates: Include run plate information
        figures_dir: Output directory for saved CSV

    Returns:
        DataFrame with feature importance stats and metadata
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")
    if features.columns.has_duplicates:
        duplicates = features.columns[features.columns.duplicated()].tolist()[:10]
        raise ValueError(f"features contains duplicate columns: {duplicates}")
    if target.index.has_duplicates:
        raise ValueError("target contains duplicate sample IDs")

    common_index = features.index.intersection(target.index, sort=False)
    if common_index.empty:
        raise ValueError("features and target have no sample IDs in common")
    if len(common_index) != len(features.index) or len(common_index) != len(
        target.index
    ):
        logger.warning(
            "Feature-importance input was aligned to %d shared samples",
            len(common_index),
        )
    features_aligned = features.loc[common_index].copy()
    target_aligned = target.loc[common_index].copy()
    feature_names = pd.Index(features_aligned.columns.astype(str))
    features_aligned.columns = feature_names

    if isinstance(values, pd.DataFrame):
        shap_values = values.loc[common_index].copy()
        shap_values.columns = shap_values.columns.astype(str)
        missing = feature_names.difference(shap_values.columns)
        if not missing.empty:
            raise ValueError(
                "SHAP values are missing feature columns: " f"{missing.tolist()[:10]}"
            )
        shap_values = shap_values.loc[:, feature_names]
        values_array = shap_values.to_numpy(dtype=float)
    else:
        values_array = np.asarray(values)
        if values_array.ndim != 2:
            raise ValueError("values must be a two-dimensional SHAP array")
        if values_array.shape != features_aligned.shape:
            raise ValueError(
                "values and features must have the same shape; "
                f"got {values_array.shape} and {features_aligned.shape}"
            )

    importance_df = pd.DataFrame(
        {"SHAP value": np.abs(values_array).mean(axis=0)},
        index=feature_names,
    ).sort_values("SHAP value", ascending=False)

    # Add metadata if available
    if with_oligos:
        if not isinstance(oligos_metadata, pd.DataFrame):
            raise TypeError("oligos_metadata must be a DataFrame when with_oligos=True")
        metadata = oligos_metadata.copy()
        metadata.index = metadata.index.astype(str)
        importance_df = importance_df.merge(
            metadata, left_index=True, right_index=True, how="left"
        ).rename(
            columns={
                "species": "Species",
                "genus": "Genus",
                "family": "Family",
                "order": "Order",
            }
        )

    # Add group statistics if requested
    if group_tests is not None:
        if len(group_tests) != 2:
            raise ValueError("group_tests must contain exactly two group labels")
        group_labels = [str(label) for label in group_tests]

        observed = target_aligned.dropna()
        if observed.isin(group_tests).all():
            grouped_target = observed.astype(str)
        else:
            numeric_target = pd.to_numeric(observed, errors="coerce")
            valid_codes = set(range(len(group_labels)))
            if not (
                numeric_target.notna().all() and numeric_target.isin(valid_codes).all()
            ):
                raise ValueError(
                    "target must contain configured group labels or positional "
                    f"codes {sorted(valid_codes)}"
                )
            label_map = dict(enumerate(group_labels))
            grouped_target = numeric_target.astype(int).map(label_map)

        numeric_features = features_aligned.apply(pd.to_numeric, errors="coerce")
        grouped = numeric_features.loc[grouped_target.index].groupby(
            grouped_target,
            sort=False,
        )
        group_means = grouped.mean().reindex(group_labels)

        binary_columns = [
            column
            for column in numeric_features.columns
            if set(numeric_features[column].dropna().unique()).issubset({0, 1})
        ]
        group_means.loc[:, binary_columns] *= 100
        group_stats = group_means.T

        importance_df = importance_df.join(group_stats, how="left")
        denominator = importance_df[group_labels[1]].replace(0, np.nan)
        importance_df["Ratio"] = importance_df[group_labels[0]] / denominator

    # Format output
    importance_df = importance_df.rename_axis("Peptide ID").reset_index()

    # Save to CSV
    filename_parts = [
        "Table_shap",
        "-".join(str(label) for label in group_tests) if group_tests else None,
        filename_label or "model",
        "with_oligos" if with_oligos else None,
        "with_additional_features" if with_additional_features else None,
        "with_run_plates" if with_run_plates else None,
    ]
    filename = "_".join(filter(None, filename_parts)) + ".csv"

    output_dir = Path(figures_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_dir / filename, index=False)

    return importance_df


def _get_top_features(
    feature_shap_values_df: pd.DataFrame,
    group_tests: Optional[Sequence[str]] = None,
    to_select_features: int = 30,
    max_length_text: int = 75,
) -> pd.DataFrame:
    """
    Generate a table of top feature importances from a SHAP values DataFrame.

    The function selects the top features based on a hard-coded column order,
    rounds specified columns (based on group_tests and 'Ratio (log10)'),
    and truncates long text in the 'Description' column.

    Parameters
    ----------
    feature_shap_values_df : pd.DataFrame
        DataFrame containing feature importance values and additional metadata.
        Important to have Ratio (log 10), Description and Group_test columns. Run generate_feature_importance_df() first.
    group_tests : list of str
        A list of two group labels for binary classification. These labels are
        used as column names for rounding values. Must be provided with exactly two elements.
    to_select_features : int, default 30
        The number of top features to select.
    max_length_text : int, default 90
        Maximum allowed length for text in the 'Description' column. Longer texts are truncated.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the top features with rounded group test values,
        a computed ratio (in log10), and a truncated description.

    Raises
    ------
    ValueError
        If group_tests is not provided or does not have exactly two elements,
        or if the required columns are not found in the DataFrame.
    """
    # Check that group_tests is provided correctly
    if group_tests is None or len(group_tests) != 2:
        raise ValueError(
            "group_tests is expected to be provided as a list of two labels for binary classification."
        )
    if to_select_features < 1:
        raise ValueError("to_select_features must be at least 1")
    if max_length_text < 1:
        raise ValueError("max_length_text must be at least 1")

    # Helper function to truncate text
    def truncate_description(text: Any, max_length: int) -> str:
        if pd.isna(text):
            return ""
        text_value = str(text)
        return (
            text_value
            if len(text_value) <= max_length
            else text_value[:max_length] + "..."
        )

    selected_columns = [
        "Peptide ID",
        "Description",
        group_tests[0],
        group_tests[1],
        "Ratio",
        "SHAP value",
    ]
    missing = [
        column
        for column in selected_columns
        if column not in feature_shap_values_df.columns
    ]
    if missing:
        raise ValueError(f"Feature-importance table is missing columns: {missing}")
    top_features = (
        feature_shap_values_df.loc[:, selected_columns].head(to_select_features).copy()
    )
    # Round columns specified by group_tests and the 'Ratio' column.
    round_columns = [group_tests[0], group_tests[1], "Ratio"]
    for col in round_columns:
        top_features[col] = pd.to_numeric(top_features[col], errors="coerce").round(2)

    # Truncate text in the 'Description' column if it exists.
    top_features["Description"] = top_features["Description"].map(
        lambda value: truncate_description(value, max_length_text)
    )
    return top_features


def _colorize(
    val: float,
    min_val: float,
    max_val: float,
    palette: str = "Reds",
) -> tuple[float, float, float, float]:
    """
    Returns a color (as an RGBA tuple) based on a normalized value using a given palette.

    Parameters
    ----------
    val : float
        The value to colorize.
    min_val : float
        The minimum value for normalization.
    max_val : float
        The maximum value for normalization.
    palette : str, default 'Reds'
        The palette name to use. If 'Reds', then 'RdYlGn' colormap is used; otherwise 'BuPu' is used.

    Returns
    -------
    color : tuple or np.ndarray
        The RGBA color corresponding to the normalized value.
    """
    # Avoid division by zero if min_val equals max_val
    # if max_val == min_val:
    #   norm_val = 0.5
    # else:
    try:
        numeric_value = float(val)
    except (TypeError, ValueError):
        numeric_value = min_val
    if not math.isfinite(numeric_value):
        numeric_value = min_val
    norm_val = (
        0.5
        if math.isclose(max_val, min_val)
        else (numeric_value - min_val) / (max_val - min_val)
    )
    norm_val = float(np.clip(norm_val, 0.0, 1.0))

    if palette == "Reds":
        color = colormaps["RdYlGn"](norm_val)
    else:
        color = colormaps["BuPu"](norm_val)
    return tuple(color)


def _render_main_table(
    df: pd.DataFrame,
    col_widths: Sequence[float],
    row_height: float = 0.625,
    font_size: int = 9,
    header_color: str = "lightgray",
    row_colors: Optional[Sequence[str]] = None,
    edge_color: str = "w",
    bbox: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Render a DataFrame as a matplotlib table with custom cell coloring based on feature values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to display.
    col_widths : list of float
        List of widths for each column.
    row_height : float, default 0.625
        Height of each row.
    font_size : int, default 11
        Base font size for the table.
    header_color : str, default 'lightgray'
        Background color for the header cells.
    row_colors : list of str, optional
        List of alternating row colors. Defaults to ['#f1f1f2', 'w'].
    edge_color : str, default 'w'
        Color of the cell edges.
    bbox : list of float, optional
        Bounding box for the table in figure coordinates.
    ax : matplotlib.axes.Axes, optional
        Axes on which to render the table. If None, a new figure and axes are created.
    **kwargs : dict
        Additional keyword arguments to pass to ax.table.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes containing the rendered table.
    """
    if bbox is None:
        bbox = [0, 0, 1, 1]
    if row_colors is None:
        row_colors = ["#f1f1f2", "w"]
    if len(col_widths) != len(df.columns):
        raise ValueError("col_widths must contain one value per DataFrame column")
    if not row_colors:
        raise ValueError("row_colors cannot be empty")
    if ax is None:
        _, ax = plt.subplots(figsize=(np.sum(col_widths), len(df) * row_height + 1))
        ax.axis("off")

    # Predefined scaling values for percentage columns
    min_val = 0
    max_val = 100
    # For the ratio, using a symmetric scale (can be adjusted as needed)
    # min_val_ratio = -2
    # max_val_ratio = 2

    # Create the matplotlib table.
    mpl_table = ax.table(
        cellText=df.values,
        bbox=bbox,
        colLabels=df.columns,
        cellLoc="center",
        colWidths=col_widths,
        **kwargs,
    )

    for column_name in ("Peptide ID", "Description"):
        if column_name not in df.columns:
            continue
        column_index = df.columns.get_loc(column_name)
        for row in range(len(df) + 1):
            cell = mpl_table[(row, column_index)]
            cell.set_text_props(ha="left", va="center")
            cell.PAD = 0.01

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for (i, j), cell in mpl_table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_text_props(fontsize=8.5, color="black")

        if i == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor(header_color)
        else:
            if j == 2:
                cell_val = df.iloc[i - 1, 2]
                cell.set_facecolor(
                    _colorize(cell_val, min_val, max_val, palette="Reds")
                )
                cell.set_text_props(color="black")
            elif j == 3:
                cell_val = df.iloc[i - 1, 3]
                cell.set_facecolor(_colorize(cell_val, min_val, max_val))
                cell.set_text_props(color="black")
            else:
                cell.set_facecolor(row_colors[i % len(row_colors)])
                cell.set_text_props(color="black")

    return ax


def _render_header_main_table(ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Render the header portion of a table on the given matplotlib axis.

    This function draws a header table using fixed column widths and then adds
    custom text labels above the header cells.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The axes on which to render the header. If None, the current axes (plt.gca()) is used.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the rendered header table.
    """
    if ax is None:
        ax = plt.gca()

    # Define header colors and column widths.
    col_colors = ["lightgray", "lightgray", "lightgray"]
    col_widths = [0.563, 0.292, 0.107]
    bbox = [0, 1, 1, 0.12]

    # Create header table using matplotlib's table function.
    header_table = ax.table(
        cellLoc="center", colWidths=col_widths, bbox=bbox, cellColours=[col_colors]
    )

    # Customize cells in the header table.
    for (i, _), cell in header_table.get_celld().items():
        cell.set_edgecolor("w")
        if i == 0:  # Header row
            cell.set_text_props(weight="bold", color="black")

    # Add additional header text labels above the table.
    ax.text(
        0.29,
        1.05,
        "Peptide details",
        ha="center",
        color="black",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        0.74,
        1.035,
        "Antibody responses\nappearing % in ...",
        ha="center",
        fontsize=10,
        color="black",
        weight="bold",
    )
    ax.text(
        0.945,
        1.035,
        "Feature\nimportance",
        ha="center",
        fontsize=10,
        color="black",
        weight="bold",
    )

    return ax


def plot_table_top_features(
    feature_shap_values_df: pd.DataFrame,
    group_tests: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    to_select_features: int = 10,
    fig_size: tuple[float, float] = (8.5, 5),
    figure_dir: Union[str, Path] = "./",
    filename_label: str = "plot",
    save_fig: bool = False,
) -> tuple[Figure, Axes]:
    """
    Render a table of top features (with SHAP values) on a matplotlib axis.

    This function selects the top features using get_top_features(), renders a header table,
    renders the main table with custom column widths and formatting, hides the axes, and optionally saves the figure.

    Parameters
    ----------
    feature_shap_values_df : pd.DataFrame
        DataFrame with feature importance and metadata. Run generate_feature_importance_table() first to get the df.
    group_tests : list of str
        List of two group labels. Used in get_top_features() to add prevalence and ratio columns.
    ax : matplotlib.axes.Axes, optional
        The axis on which to render the tables. If None, a new figure and axis are created.
    to_select_features : int, default 10
        Number of top features to select.
    fig_size : tuple, default (8.5, 5)
        Size of the figure.
    figure_dir : str, default './'
        Directory to save the figure.
    filename_label : str, optional
        Suffix for the filename. If not provided, it is constructed from prefix_label, and group_tests.
    save_fig : bool, default False
        If True, the figure is saved to figure_dir.

    Returns
    -------
    Tuple[Figure, Axes]: fig and matplotlib.axes.Axes
        The axis with the rendered table.

    Raises
    ------
    ValueError
        If group_tests is provided but does not contain exactly two elements.
    """
    # Get top features with optional group tests. (Assumes get_top_features is defined elsewhere.)
    top_features_df = _get_top_features(
        feature_shap_values_df,
        group_tests=group_tests,
        to_select_features=to_select_features,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    # Render the header table.
    _render_header_main_table(ax)

    # Ensure SHAP values are formatted as strings (rounded) for display.
    if "SHAP value" in top_features_df.columns:
        top_features_df["SHAP value"] = (
            top_features_df["SHAP value"].round(3).astype(str)
        )
    else:
        raise ValueError("Expected column 'SHAP value' not found in the DataFrame.")

    # Render the main data table.
    # You can adjust col_widths as needed.
    _render_main_table(
        top_features_df, col_widths=[0.35, 1.5, 0.32, 0.32, 0.32, 0.35], ax=ax
    )
    #                      col_widths=[0.35, 2.15, 0.32, 0.32, 0.32, 0.35], ax=ax)

    # Hide axes.
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    fig.tight_layout()

    if save_fig:
        output_dir = Path(figure_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        group_label = "-".join(str(label) for label in group_tests)
        pdf_path = (
            output_dir
            / f"SHAP_table_top{to_select_features}_{filename_label}_{group_label}.pdf"
        )
        fig.savefig(
            pdf_path,
            dpi=600,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
        )

        svg_path = (
            output_dir
            / f"SHAP_table_top{to_select_features}_{filename_label}_{group_label}.svg"
        )
        fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")

    return fig, ax


def plot_shap_values(
    values: np.ndarray,
    features: pd.DataFrame,
    ax: Optional[Axes] = None,
    cmap: Union[str, LinearSegmentedColormap] = "viridis",
    max_display: int = 30,
    group_tests: Optional[Sequence[str]] = None,
    pattern: Optional[str] = None,
    filename_label: str = "plot",
    plot_title: str = "",
    save_fig: bool = False,
    figures_dir: Union[str, Path] = "./",
    fontsize: Optional[Mapping[str, int]] = None,
    figure_size: Tuple[float, float] = (6, 5),
    x_label: str = "Prediction toward...",
    legend_title: str = "Feature\nvalue",
    legend_labels: Optional[Sequence[str]] = None,
    add_group_labels: bool = True,
    add_binary_legend: bool = True,
    **shap_kwargs,
) -> tuple[Figure, Axes]:
    """
    Create a SHAP summary (beeswarm) plot with top x-axis and group labels.
    """

    # -------------------------
    # Defaults
    # -------------------------
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    values_array = np.asarray(values)
    if values_array.ndim != 2:
        raise ValueError("values must be a two-dimensional SHAP array")
    if values_array.shape != features.shape:
        raise ValueError(
            "values and features must have the same shape; "
            f"got {values_array.shape} and {features.shape}"
        )
    if max_display < 1:
        raise ValueError("max_display must be at least 1")

    fs_default: dict[str, int] = {
        "title": 12,
        "xlabel": 13,
        "ylabel": 13,
        "xticks": 12,
        "yticks": 12,
        "legend": 10,
        "colorbar": 13,
    }
    if fontsize:
        fs_default.update(fontsize)
    fontsize = fs_default

    if group_tests is None:
        group_tests = ["group1", "group2"]
    elif len(group_tests) != 2:
        raise ValueError("group_tests must contain exactly two labels")

    # -------------------------
    # Figure / axis handling
    # -------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.figure
        plt.figure(fig.number)
        plt.sca(ax)
    axes_before = set(fig.axes)

    # -------------------------
    # SHAP summary plot
    # -------------------------
    shap.summary_plot(
        values_array,
        features=features,
        plot_type="dot",
        cmap=cmap,
        max_display=max_display,
        plot_size=[figure_size[0], figure_size[1]],
        show=False,
        **shap_kwargs,
    )

    # Re-bind to the axis SHAP actually used
    fig = plt.gcf()
    ax = plt.gca()
    colorbar_axes = [
        candidate
        for candidate in fig.axes
        if candidate is not ax and candidate not in axes_before
    ]
    ax.set_xlabel("")
    # =====================================================
    # X-AXIS (TOP) + GROUP LABELS
    # =====================================================
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        labelsize=fontsize["xticks"],
    )
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(False)

    # ax.set_xlabel( x_label, fontsize=fontsize["xlabel"], fontweight="bold", labelpad=18, color="black", )
    if add_group_labels:
        xmin, xmax = ax.get_xlim()
        x0 = 0.0 if (xmin < 0 < xmax) else (xmin + xmax) / 2.0
        x_left = (xmin + x0) / 2.0
        x_right = (x0 + xmax) / 2.0
        x_mid = 0.5 * (x_left + x_right)
        trans = ax.get_xaxis_transform()

        ax.text(
            x_left,
            1.08,
            group_tests[0],
            ha="center",
            va="bottom",
            fontsize=fontsize["xlabel"],
            color="black",
            transform=trans,
            clip_on=False,
        )
        ax.text(
            x_right,
            1.08,
            group_tests[1],
            ha="center",
            va="bottom",
            fontsize=fontsize["xlabel"],
            color="black",
            transform=trans,
            clip_on=False,
        )

        # --- Center xlabel between the two groups (data-driven) ---
        ax.text(
            x_mid,
            1.14,  # above group labels
            x_label,
            ha="center",
            va="bottom",
            fontsize=fontsize["xlabel"],
            fontweight="bold",
            color="black",
            transform=trans,
            clip_on=False,
        )

        # fig.subplots_adjust(top=0.82)

    # -------------------------
    # Y tick labels cleanup
    # -------------------------
    if pattern:
        yticklabels = [
            re.sub(pattern, "", label.get_text()) for label in ax.get_yticklabels()
        ]
    else:
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks(), labels=yticklabels)
    ax.tick_params(axis="y", labelsize=fontsize["yticks"])

    # -------------------------
    # Title
    # -------------------------
    if plot_title:
        ax.set_title(plot_title, fontsize=fontsize["title"], weight="bold", pad=10)

    # -------------------------
    # Grid + aesthetics
    # -------------------------
    ax.tick_params(axis="y", pad=-10)
    ax.yaxis.set_label_coords(-0.3, 0.5)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color="lightgrey")
    ax.grid(False, axis="x")
    ax.patch.set_alpha(0.0)

    # -------------------------
    # Binary legend (right side)
    # -------------------------
    if add_binary_legend:
        if legend_labels is None:
            legend_labels = ["0", "1"]
        elif len(legend_labels) != 2:
            raise ValueError("legend_labels must contain exactly two labels")

        for colorbar_ax in colorbar_axes:
            colorbar_ax.remove()

        colormap = colormaps.get_cmap(cmap)
        handles = [
            Patch(facecolor=colormap(0.0), label=legend_labels[0]),
            Patch(facecolor=colormap(1.0), label=legend_labels[1]),
        ]
        legend = ax.legend(
            handles=handles,
            title=legend_title,
            frameon=False,
            bbox_to_anchor=(0.97, 0.5),
            loc="center left",
            fontsize=fontsize["legend"],
            title_fontsize=fontsize["legend"],
        )
        for txt in legend.get_texts():
            txt.set_color("black")
        legend.get_title().set_color("black")
    else:
        if legend_labels is None:
            legend_labels = ["Low", "High"]
        elif len(legend_labels) != 2:
            raise ValueError("legend_labels must contain exactly two labels")
        available_colorbars = colorbar_axes or [
            candidate for candidate in fig.axes if candidate is not ax
        ]
        if not available_colorbars:
            raise RuntimeError("SHAP did not create a colorbar axis")
        cbar = available_colorbars[-1]
        colorbar_ticks = cbar.get_yticks()
        if len(colorbar_ticks) < 2:
            colorbar_ticks = np.array([0.0, 1.0])
        cbar.set_yticks(
            [colorbar_ticks[0], colorbar_ticks[-1]],
            labels=legend_labels,
        )
        cbar.tick_params(
            axis="y",
            colors="black",
            labelsize=fontsize["colorbar"],
        )
        cbar.set_ylabel(legend_title, fontsize=fontsize["legend"], color="black")

    # -------------------------
    # Save
    # -------------------------
    if save_fig:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        safe_groups = "-".join(group_tests) if group_tests else "groups"
        save_path = figures_dir / f"shap_{safe_groups}_{filename_label}"
        fig.savefig(f"{save_path}.pdf", format="pdf", dpi=600, bbox_inches="tight")
        fig.savefig(
            f"{save_path}.svg", format="svg", bbox_inches="tight", facecolor="white"
        )

    return fig, ax



def test_contingency(
    contingency_table: Union[pd.DataFrame, Sequence[Sequence[int]]],
    method: str = "fisher",
) -> tuple[float, str]:
    """
    Perform Fisher's Exact test or Chi-square test.

    Parameters
    ----------
    contingency_table : 2×2 numpy array or list of lists
    method : "fisher" or "chisquare"

    Returns
    -------
    odds_ratio : float or None
    p_value : float
    """

    table = np.asarray(contingency_table)
    if table.ndim != 2 or table.size == 0:
        raise ValueError("contingency_table must be a non-empty 2D table")
    if not np.issubdtype(table.dtype, np.number):
        raise TypeError("contingency_table must contain numeric counts")
    if np.any(table < 0):
        raise ValueError("contingency_table cannot contain negative counts")

    if method.lower() == "fisher":
        if table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires a 2x2 contingency table")
        odds_ratio, p_value = fisher_exact(table)
        annotation = (
            f"Fisher Exact p = {format_pval(p_value)}\nOdds Ratio = {odds_ratio:.2f}"
        )

        return p_value, annotation

    elif method.lower() in ["chi", "chisq", "chisquare", "chi-square"]:
        _, p_value, _, _ = chi2_contingency(table)
        annotation = f"X² p = {format_pval(p_value)}"

        return p_value, annotation

    else:
        raise ValueError("method must be 'fisher' or 'chisquare'")

def barplot_counts_fisher_test(
    df,
    cat_var1,
    cat_var2,
    method="fisher",
    legend_title=None,
    y_label="Count",
    x_label=None,
    title=None,
    mapping1=None,
    mapping2=None,
    figsize=(6, 4),
    palette="viridis",
    save_path=None,
    suffix_file=None,
    fontsize=None,
):
    """
    Create a grouped bar plot from a 2x2 contingency table of two categorical variables,
    run Fisher's exact test, and annotate the plot with the p-value and odds ratio.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    cat_var1 : str
        The name of the first categorical variable (e.g., 'DCR').
    cat_var2 : str
        The name of the second categorical variable (e.g., 'Antigen Score (Dichotomized)').
    method : str, default "fisher"
        Method to use for the test. Can be "fisher" or "chisquare".
    legend_title : str, optional
        Title for the legend corresponding to cat_var1 (default: cat_var1).
    y_label : str, optional
        Label for the y-axis (default is 'Count').
    title : str, optional
        Title for the plot.
    mapping1 : dict, optional
        A dictionary to map the values of cat_var1 to descriptive labels (e.g., {0.0: 'Non-Responder', 1.0: 'Responder'}).
    mapping2 : dict, optional
        A dictionary to map the values of cat_var2 to descriptive labels (e.g., {0: 'Low', 1: 'High'}).
    figsize : tuple, optional
        Figure size (default is (6, 4)).
    palette : str or list, optional
        Color palette for the plot (default is 'viridis').
    save_path : str or Path, optional
        If provided, the plot will be saved to this path.
    suffix_file : str, default None
        Suffix name for to save the plot. If None, default is fisherTest.png
    fontsize : dict, optional
        Dictionary of font sizes: {"title": int, "xlabel": int, "ylabel": int, xtick": int, ytick": int,  "legend": int, "annotation": int}.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    if fontsize is None:
        fontsize = {
            "title": 12,
            "xlabel": 10,
            "ylabel": 10,
            "xtick": 10,
            "ytick": 10,
            "legend": 10,
            "annotation": 10,
        }

    # Compute the contingency table (rows: cat_var1, columns: cat_var2)
    contingency_table = pd.crosstab(df[cat_var1], df[cat_var2])

    # Run Chi2 or Fisher's exact test (works for 2x2 tables)
    p_value, annotation = test_contingency(contingency_table, method)

    # Convert the contingency table into tidy (long) format
    df_plot = contingency_table.stack().reset_index()
    df_plot.columns = [cat_var1, cat_var2, "Count"]

    # Map numeric codes to descriptive labels if mapping dictionaries are provided
    if mapping1 is not None:
        df_plot[cat_var1] = df_plot[cat_var1].map(mapping1)
    if mapping2 is not None:
        df_plot[cat_var2] = df_plot[cat_var2].map(mapping2)

    # Use the first categorical variable as the legend title if not provided
    if legend_title is None:
        legend_title = cat_var1

    # Create the grouped bar plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df_plot, x=cat_var2, y="Count", hue=cat_var1, palette=palette, ax=ax
    )

    if title is not None:
        ax.set_title(title, fontsize=fontsize["title"])
    if x_label is None:
        ax.set_xlabel(cat_var2, fontsize=fontsize["xlabel"])
    else:
        ax.set_xlabel(x_label, fontsize=fontsize["xlabel"])
    ax.set_ylabel(y_label, fontsize=fontsize["ylabel"])
    ax.legend(loc="lower left", fontsize=fontsize["legend"])

    ax.tick_params(axis="x", labelsize=fontsize["xtick"])
    ax.tick_params(axis="y", labelsize=fontsize["ytick"])

    # Annotate the plot with Fisher's test results in the top center of the axis
    ax.text(
        0.5,
        0.1,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize["annotation"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
    )

    # Place legend outside upper left, no background, no border/margin
    plt.legend(
        loc="upper left",  # anchor to upper left
        bbox_to_anchor=(0.01, 1.22),  # shift outside the axes
        frameon=False,  # no frame
        borderaxespad=0,  # no extra margin
        fontsize=fontsize["legend"],
    )

    plt.tight_layout()

    if save_path is not None:
        if suffix_file is not None:
            plt.savefig(
                Path(save_path) / f"fisherTest_{suffix_file}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                Path(save_path)  / f"fisherTest_{suffix_file}.svg",
                format="svg",
                bbox_inches="tight",
                facecolor="white",
            )

        else:
            plt.savefig(
                Path(save_path) / "fisherTest.pdf", dpi=600, bbox_inches="tight"
            )


    return fig


def boxplot_compare_distribution_by_category(
    df,
    cont_var,
    cat_var,
    cat_mapping=None,
    figsize=(6, 4),
    palette="viridis",
    x_label=None,
    y_label=None,
    title=None,
    save_path=None,
    suffix_file=None,
):
    """
    Plot the distribution of a continuous variable by a categorical variable,
    perform a Mann–Whitney U test if there are exactly 2 groups or a
    Kruskal–Wallis H test if there are more than 2 groups, and annotate
    the plot with the p-value and test statistic.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    cont_var : str
        Name of the continuous variable (e.g., "Antigen Score (Scaled)").
    cat_var : str
        Name of the categorical variable (e.g., "DCR" or "ORR").
    cat_mapping : dict, optional
        Mapping for the categorical variable values to descriptive labels
        (e.g., {0: 'Non-Responder', 1: 'Responder'}). If provided, a new
        column "cat_label" is created for plotting.
    figsize : tuple, optional
        Figure size (default is (6,4)).
    palette : str or list, optional
        Color palette for the boxplot (default is "viridis").
    x_label : str, optional
        Label for the x-axis. Defaults to the (mapped) categorical variable.
    y_label : str, optional
        Label for the y-axis. Defaults to the continuous variable name.
    title : str, optional
        Plot title.
    save_path : str or Path, optional
        If provided, the plot will be saved to this path.
    suffix_file : str, optional
        Suffix to append to the saved filename.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    # If a mapping is provided, add a new column for descriptive labels.
    if cat_mapping is not None:
        df = df.copy()  # Avoid modifying the original DataFrame
        df["cat_label"] = df[cat_var].map(cat_mapping)
        plot_cat = "cat_label"
        groups = list(cat_mapping.keys())
    else:
        plot_cat = cat_var
        groups = df[
            cat_var
        ].unique()  # Get the unique groups from the original categorical variable.

    # Depending on the number of groups, run the appropriate test.
    if len(groups) == 2:
        test_name = "Mann–Whitney"
        group0 = df.loc[df[cat_var] == groups[0], cont_var]
        group1 = df.loc[df[cat_var] == groups[1], cont_var]
        stat, p_value = mannwhitneyu(group0, group1, alternative="two-sided")
        # stat_str = f"U = {stat}"
    elif len(groups) > 2:
        test_name = "Kruskal–Wallis"
        group_data = [df.loc[df[cat_var] == g, cont_var] for g in groups]
        stat, p_value = kruskal(*group_data)
        # stat_str = f"H = {stat:.2f}"
    else:
        raise ValueError("The categorical variable must have at least 2 unique groups.")

    print(f"{test_name} Test statistic: {stat}, p-value: {p_value:.3f}")

    # Set default axis labels if not provided.
    if x_label is None:
        x_label = plot_cat
    if y_label is None:
        y_label = cont_var

    # Create the plot.
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        x=plot_cat, y=cont_var, data=df, palette=palette, ax=ax, showfliers=False
    )
    sns.swarmplot(x=plot_cat, y=cont_var, data=df, color="0.2", ax=ax)

    # Annotate the plot with the p-value and test statistic.
    # max_val = df[cont_var].max()
    annotation = f"{test_name} p = {format_pval(p_value)}"
    ax.text(
        0.5,
        0.85,
        annotation,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
    )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()

    # Save the figure if a save path is provided.

    if save_path is not None:
        if suffix_file is not None:
            plt.savefig(
                Path(save_path) / f"{test_name}_Test_{suffix_file}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                Path(save_path) / f"{test_name}_Test_{suffix_file}.svg",
                format="svg",
                bbox_inches="tight",
                facecolor="white",
            )

        else:
            plt.savefig(
                Path(save_path) / f"{test_name}_Test.pdf", dpi=600, bbox_inches="tight"
            )

    return fig

