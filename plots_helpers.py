from pathlib import Path
from typing import Optional, List

# Data Analysis
import pandas as pd
import numpy as np
import re
from scipy.stats import mannwhitneyu, fisher_exact

# Plots
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.axes as maxes
import seaborn as sns

# ML
import shap

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.family'] = 'Arial'

""" SHAP helpers """

def generate_feature_importance_table(values: np.ndarray,
                                      features: pd.DataFrame,
                                      target: pd.Series,
                                      oligos_metadata: pd.DataFrame,
                                      group_tests: Optional[List[str]] = None,
                                      suffix_file: Optional[str] = None,
                                      with_oligos: bool = True,
                                      with_additional_features: bool = False,
                                      with_run_plates: bool = False,
                                      figures_dir: str = './') -> pd.DataFrame:
    """
    Create a feature importance table from SHAP values, optionally merging metadata
    and group prevalence statistics if group_tests is provided.

    Parameters
    ----------
    values : np.ndarray
        Array of SHAP values.
    features : pd.DataFrame
        DataFrame of  feature values used for group calculations with SampleName as index.
    target : pd.Series
        Series of target values with same indices as features (SampleName) and a column name.
    oligos_metadata : pd.DataFrame
        Metadata for the oligos (peptides), indexed by peptide ID.
    group_tests : Optional[List[str]], default None
        List of two strings representing group names. If provided, extra columns for prevalence and ratio are added.
    suffix_file : Optional[str], default None
        Name of the estimator (for filename).
    with_oligos : bool, default True
        Flag to include oligos metadata columns.
    with_additional_features : bool, default False
        Flag to include additional features.
    with_run_plates : bool, default False
        Flag to include run plate information.
    figures_dir : str, default './'
        Directory where the CSV file will be saved.

    Returns
    -------
    pd.DataFrame
        A DataFrame with feature importance (mean absolute SHAP values) merged with metadata,
        and if group_tests is provided, extra columns with group prevalence and ratio.
    """
    # Feature names
    features = features.join(target).set_index(target.name, append=True)
    colnames = features.columns

    # Create base table of feature importance sorted by mean absolute SHAP value.
    feature_shap_values_df = pd.DataFrame({'SHAP value': np.abs(values).mean(0)},
                                          index=colnames).sort_values(by='SHAP value', ascending=False)

    # Merge metadata (if available) and rename columns as desired.
    feature_shap_values_df = feature_shap_values_df.merge(
        oligos_metadata,
        left_index=True, right_index=True, how='left'
    ).rename(columns={'full name': 'Protein name', 'len_seq': 'Protein length',
                      'pos': 'Start position', 'Organism_complete_name': 'Organism'})

    # If group_tests is provided, compute additional prevalence and ratio columns.
    if group_tests is not None:
        # Filter columns based on keywords in colnames
        keywords = ['agilent', 'corona2', 'twist', 'Sex', 'Age', 'run_plate']
        pattern = '|'.join(keywords)
        filtered_index = colnames[colnames.str.contains(pattern, case=False, regex=True)]

        # Compute average (mean) prevalence per group at the second level of the index in 'features'
        percentiles = features[filtered_index].groupby(level=features.index.names[1]).mean()
        #percentiles = features.groupby(level=features.index.names[1]).mean()

        # Identify columns for which to convert to percentage
        get_perc_columns = percentiles.columns[
            percentiles.columns.str.contains('agilent|corona2|twist|sex|run_plate', case=False)]
        percentiles[get_perc_columns] = percentiles[get_perc_columns].mul(100)
        percentiles = percentiles.T.rename(columns={0: group_tests[0], 1: group_tests[1]})

        # Merge with our feature importance DataFrame.
        feature_shap_values_df = pd.merge(feature_shap_values_df,
                                          percentiles,
                                          left_index=True, right_index=True, how='left')

        # Compute the ratio (in log10) between the two groups.
        feature_shap_values_df['Ratio (log10)'] = np.log10(
            feature_shap_values_df[group_tests[1]] / feature_shap_values_df[group_tests[0]]
        )

    # Reset index to get 'Peptide ID' column.
    feature_shap_values_df.reset_index(inplace=True)
    feature_shap_values_df.rename(columns={'index': 'Peptide ID'}, inplace=True)

    # Construct the filename.
    filename = f"shap_values_{suffix_file if suffix_file else 'model'}"
    if group_tests is not None:
        filename += f"_{'-'.join(group_tests)}"
    if with_oligos:
        filename += "_with_oligos"
    if with_additional_features:
        filename += "_with_additional_features"
    if with_run_plates:
        filename += "_with_run_plates"
    filename += ".csv"

    # Save the resulting DataFrame to CSV.
    feature_shap_values_df.to_csv(Path(figures_dir, filename), index=False)

    return feature_shap_values_df


def _get_top_features(feature_shap_values_df: pd.DataFrame,
                     group_tests: List[str] = None,
                     to_select_features: int = 30,
                     max_length_text: int = 85) -> pd.DataFrame:
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
        raise ValueError("group_tests is expected to be provided as a list of two labels for binary classification.")

    # Helper function to truncate text
    def truncate_description(text: Optional[str], max_length: int) -> Optional[str]:
        if text is None:
            return None
        return text if len(text) <= max_length else text[:max_length] + '...'

    # Select top features based on a hard-coded column order.
    # Adjust these indices as necessary for your DataFrame.
    top_features = feature_shap_values_df.iloc[:to_select_features, [0, 2, 7, 8, 9, 1]] #hard coded, assuming we have right columns

    # Round columns specified by group_tests and the 'Ratio (log10)' column.
    round_columns = [group_tests[0], group_tests[1], 'Ratio (log10)']
    for col in round_columns:
        if col in top_features.columns:
            top_features[col] = top_features[col].round(2)
        else:
            raise ValueError(
                f"Expected column '{col}' not found in the DataFrame. Check group_tests or the column order.")

    # Truncate text in the 'Description' column if it exists.
    if 'Description' in top_features.columns:
        top_features['Description'] = top_features['Description'].apply(
            lambda x: truncate_description(x, max_length_text))
    else:
        raise ValueError("Expected column 'Description' not found in the DataFrame.")
    return top_features


def _colorize(val: float, min_val: float, max_val: float, palette: str = 'Reds'):
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
    #if max_val == min_val:
     #   norm_val = 0.5
    #else:
    norm_val = (val - min_val) / (max_val - min_val)

    if palette == 'Reds':
        color = colormaps['RdYlGn'](norm_val)
    else:
        color = colormaps['BuPu'](norm_val)
    return color


def _render_main_table(df: pd.DataFrame,
                      col_widths: List[float],
                      row_height: float = 0.625,
                      font_size: int = 11,
                      header_color: str = 'lightgray',
                      row_colors: Optional[List[str]] = None,
                      edge_color: str = 'w',
                      bbox: Optional[List[float]] = None,
                      ax: Optional[plt.Axes] = None,
                      **kwargs) -> plt.Axes:
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
        row_colors = ['#f1f1f2', 'w']
    if ax is None:
        fig, ax = plt.subplots(figsize=(np.sum(col_widths), len(df) * row_height + 1))
        ax.axis('off')

    # Predefined scaling values for percentage columns
    min_val = 0
    max_val = 100
    # For the ratio, using a symmetric scale (can be adjusted as needed)
    min_val_ratio = -2
    max_val_ratio = 2

    # Create the matplotlib table.
    mpl_table = ax.table(cellText=df.values, bbox=bbox, colLabels=df.columns, cellLoc='center',
                         colWidths=col_widths, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    # Iterate over table cells
    for (i, j), cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        # Set a slightly smaller font for cell text (adjust as needed)
        cell.set_text_props(fontsize=7.5)

        if i == 0:
            # Header formatting
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor(header_color)
        else:
            # For specific columns, apply a gradient color
            # Note: Adjust the column indices based on your DataFrame layout.
            if j == 2:  # Assuming column 2 corresponds to 'CFS'
                cell_val = df.iloc[i - 1, 2]
                cell.set_facecolor(_colorize(cell_val, min_val, max_val, palette='Reds'))
                cell.set_text_props(color='black')
            elif j == 3:  # Assuming column 3 corresponds to 'Ctrl'
                cell_val = df.iloc[i - 1, 3]
                cell.set_facecolor(_colorize(cell_val, min_val, max_val))
                cell.set_text_props(color='black')
            elif j == 4:  # Assuming column 4 corresponds to 'Ratio (log)'
                cell_val = df.iloc[i - 1, 4]
                cell.set_facecolor(_colorize(cell_val, min_val_ratio, max_val_ratio, palette='Blues'))
                cell.set_text_props(color='black')
            else:
                # Alternate row colors for remaining cells.
                cell.set_facecolor(row_colors[i % len(row_colors)])

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
    col_colors = ['lightgray', 'lightgray', 'lightgray']
    col_widths = [0.76, 0.292, 0.107]
    bbox = [0, 1, 1, 0.12]

    # Create header table using matplotlib's table function.
    header_table = plt.table(cellLoc='center',
                             colWidths=col_widths, bbox=bbox,
                             cellColours=[col_colors])

    # Customize cells in the header table.
    for (i, j), cell in header_table._cells.items():
        cell.set_edgecolor('w')
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='black')

    # Add additional header text labels above the table.
    ax.text(0.34, 1.05, 'Peptide details', ha='center', color='black', weight='bold')
    ax.text(0.79, 1.035, 'Antibody responses\nappearing % in ...', ha='center', fontsize=10, color='black', weight='bold')
    ax.text(0.955, 1.035, 'Feature\nimportance', ha='center', fontsize=10, color='black', weight='bold')

    return ax

def plot_table_top_features(feature_shap_values_df: pd.DataFrame,
                            group_tests: List[str] = None,
                            ax: Optional[plt.Axes] = None,
                            to_select_features: int = 10,
                            set_type: str = "set_type",
                            figure_dir: str = './',
                            suffix_file: Optional[str] = None,
                            save_fig: bool = False) -> plt.Axes:
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
    set_type : str, default ""
        A string identifier for the dataset type: Train, Test, or Validation. Used in the filename for saving the figure.
    figure_dir : str, default './'
        Directory to save the figure.
    suffix_file : str, optional
        Suffix for the filename. If not provided, it is constructed from set_type, estimator_name, and group_tests.
    save_fig : bool, default False
        If True, the figure is saved to figure_dir.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the rendered table.

    Raises
    ------
    ValueError
        If group_tests is provided but does not contain exactly two elements.
    """
    # Get top features with optional group tests. (Assumes get_top_features is defined elsewhere.)
    top_features_df = _get_top_features(feature_shap_values_df,
                                       group_tests=group_tests,
                                       to_select_features=to_select_features)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 5))

    # Render the header table.
    _render_header_main_table(ax)

    # Ensure SHAP values are formatted as strings (rounded) for display.
    if 'SHAP value' in top_features_df.columns:
        top_features_df['SHAP value'] = top_features_df['SHAP value'].round(2).astype(str)
    else:
        raise ValueError("Expected column 'SHAP value' not found in the DataFrame.")

    # Render the main data table.
    # You can adjust col_widths as needed.
    _render_main_table(top_features_df,
                      col_widths=[0.35, 2.15, 0.32, 0.32, 0.32, 0.35], ax=ax)

    # Hide axes.
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    plt.tight_layout()

    # Construct filename and save the figure if required.
    if save_fig:
        if suffix_file is None:
            suffix_file = f"{set_type}_{'-'.join(group_tests)}"
        save_path = Path(figure_dir) / f'SHAP_table_top{to_select_features}_{suffix_file}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


def plot_shap_values(values: np.ndarray,
                     features: pd.DataFrame,
                     ax: Optional[maxes.Axes] = None,
                     cmap: str = 'viridis',
                     max_display: int = 30,
                     label_groups: Optional[List[str]] = None,
                     pattern: str = r'(bloodb|BloodB)', #agilent|twist...
                     suffix_file: str = 'plot',
                     plot_title: str = "",
                     save_fig: bool = False,
                     figures_dir: str = './',
                     **shap_kwargs) -> maxes.Axes:
    """
    Create a custom SHAP summary plot and add custom x-axis annotations for binary classification.

    Parameters
    ----------
    values : np.ndarray
        Array of SHAP values.
    features : pd.DataFrame
        DataFrame of features corresponding to the SHAP values.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    cmap : str, default 'viridis'
        Colormap used in the summary plot.
    max_display : int, default 30
        Maximum number of features to display.
    label_groups : list of str, optional
        List of two labels to annotate the x-axis. If not provided, defaults to ['group1', 'group2'].
    pattern : str, default r'(bloodb|BloodB)'
        Regular expression pattern to remove from y-axis tick labels.
    suffix_file : str, default 'default'
        Suffix to add to the filename when saving the figure.
    plot_title : str, default ""
        Title of the plot.
    save_fig : bool, default False
        Whether to save the figure as a file.
    figures_dir : str, default './'
        Directory to save the figure.
    **shap_kwargs : dict
        Additional keyword arguments to pass to shap.summary_plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the SHAP plot.
    """
    # Default group labels if not provided
    if label_groups is None:
        label_groups = ['group1', 'group2']

    # Create an axis if not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Create the SHAP summary plot (but do not display it)
    shap.summary_plot(values, features=features, plot_type='dot', cmap=cmap,
                      max_display=max_display, plot_size=[5, 5], show=False, **shap_kwargs)

    # Customize colorbar: get the last axis (assumed to be the colorbar)
    fig = plt.gcf()
    if fig.axes:
        cbar = fig.axes[-1]
        cbar.set_ylabel('Feature value', fontsize=10)
        cbar.tick_params(labelsize=9)

    # Get current x-ticks and x-tick labels
    xticks = ax.get_xticks()
    # Set dynamic x-ticks and their labels
    ax.set_xticks(xticks)

    # Remove the default x-axis label "SHAP value"
    ax.set_xlabel("")

    # Add custom x-axis labels below the tick labels using ax.text with transform=ax.transAxes
    ax.text(0.2, -0.06, label_groups[0], ha='center', va='top', fontsize=10, transform=ax.transAxes)
    ax.text(0.75, -0.06, label_groups[1], ha='center', va='top', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, -0.14, 'Prediction toward...', ha='center', va='top', fontsize=11, fontweight='bold',
            transform=ax.transAxes)

    # Adjust tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    # Process y-tick labels to remove pattern text
    yticklabels = [re.sub(pattern, '', label.get_text()) for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.tick_params(axis='y', pad=-10)
    ax.yaxis.set_label_coords(-0.3, 0.5)

    # Set x-limits based on the minimum and maximum of SHAP values
    ax.set_xlim([values.min(), values.max()])

    # Add title if provided
    if plot_title:
        ax.set_title(plot_title, fontsize=12, weight='bold', pad=10)

    # Draw horizontal grid lines
    for ytick in ax.get_yticks():
        ax.axhline(y=ytick, color='lightgrey', linestyle='--', linewidth=0.8)
    ax.grid(False)
    ax.patch.set_alpha(0.0)

    # Save the figure if requested
    if save_fig:
        save_path = Path(figures_dir, f'shap_{suffix_file}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax

def get_significance_stars(p_value):
    """Return significance stars for a p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'
def forestplot_survival(multivariate_df, save_path ="./", title="Multivariate Cox Proportional Hazards Model", suffix_file = None,
                        save_fig = True, figsize=(6.5, 4)):
    """
    Create and save a forest plot from a DataFrame containing forest plot data.

    The DataFrame should include the following columns:
      - 'Variable'
      - 'P-value'
      - 'Group'
      - 'N'
      - 'log_HR'
      - 'log_95% CI Lower'
      - 'log_95% CI Upper'
      - 'Hazard Ratio'
      - '95% CI Lower'
      - '95% CI Upper'

    Parameters
    ----------
    multivariate_df : pd.DataFrame
        DataFrame containing forest plot data.
    save_path : str or Path
        Directory in which to save the figure.
    title : str, optional
        Title for the plot (default is "Multivariate Cox Proportional Hazards Model").
    suffix_file : str
        Filename (without extension) for the saved figure.
    save_fig : bool
        Boolean to save or not the figure.
    figsize : tuple, optional
        Figure size (default is (7, 5)).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    # Extract the necessary columns from the DataFrame
    variables   = multivariate_df['Variable']
    p_values    = multivariate_df['P-value']
    groups      = multivariate_df['Group']
    N           = multivariate_df['N']
    hr          = multivariate_df['log_HR']
    lower_ci    = multivariate_df['log_95% CI Lower']
    upper_ci    = multivariate_df['log_95% CI Upper']
    actual_hr   = multivariate_df['Hazard Ratio']
    actual_lc   = multivariate_df['95% CI Lower']
    actual_uc   = multivariate_df['95% CI Upper']

    # Create the figure with two subplots and share y-axis for alignment
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize, gridspec_kw={'width_ratios': [1, 4]}, sharey = True)  # Adjusted width ratios

    # --- Left subplot: Variable (group) labels ---
    ax1.set_yticks(np.arange(len(variables)))  # Set y-ticks to match number of variables
    ax1.set_yticklabels(variables.values, ha='left', va='center')
    ax1.tick_params(axis='y', labelsize=9)  # Set font size for y-axis labels
    for label in ax1.get_yticklabels():
        label.set_fontweight("bold")

    # Combine the Variable and Group labels for the first subplot
    #combined_labels = [f"{grp} (n={n})" for var, grp, n in zip(variables, groups, N)]
    combined_labels = [f"{grp} (n={n})" for grp, n in zip(groups, N)]
    for i, var in enumerate(combined_labels):
        ax1.text(1, i, f"{var}", fontsize=8, ha='left', va='center')
    # Iterate over all spines (the borders) and make them transparent
    for spine in ax1.spines.values():
        spine.set_alpha(0.0)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    # --- Right subplot: Forest plot ---
    # Compute the error as the difference between log HR and its lower/upper bounds.
    y_positions = np.arange(len(variables))
    x_err_lower = np.array(hr) - np.array(lower_ci)
    x_err_upper = np.array(upper_ci) - np.array(hr)
    ax2.errorbar(hr, y_positions, xerr=[x_err_lower, x_err_upper],
                fmt='s', color='black', ecolor='dodgerblue', elinewidth=2, capsize=5)
    # Add reference vertical line at log(HR=1)=0
    ax2.axvline(0, color='black', linestyle='--')
    # Annotate with text: Adding p-values and CI on top of each HR point
    x_text = np.nanmax(upper_ci) + 0.5
    for i, (h, l, u, p) in enumerate(zip(hr, lower_ci, upper_ci, p_values)):
        if pd.notna(p) and p != 'ns':
            significance = get_significance_stars(p)
            p_value_text = f'{p:.3f} {significance}'
            if p < 0.05:
                ax2.text(x_text, i, p_value_text, va='center', ha='left', fontsize=8, fontweight='bold')
            else:
                ax2.text(x_text, i, p_value_text, va='center', ha='left', fontsize=8)

    # Annotate with actual HR values (if HR != 1)
    for i, (h,actual, l, u) in enumerate(zip(hr, actual_hr, actual_lc, actual_uc)):
        if actual != 1.0:  # Only add HR value if it is not equal to 1
            ax2.text(h, i+0.5, f'{actual:.2f} ({l:.2f}-{u:.2f})', ha='center', va='center', fontsize=6)
    # Set x-axis limits
    ax2.set_xlim(np.nanmin(lower_ci) - 0.5, np.nanmax(upper_ci) + 0.5)

    # Set x-axis label and title
    ax2.set_xlabel('log(Hazard Ratio) 95% CI', fontsize=8)
    ax2.set_title(title, fontsize=10)
    ax2.tick_params(axis='x', labelsize=8)
    # Make the spines of the right subplot transparent
    for spine in ax2.spines.values():
        spine.set_alpha(0.0)
    # Set vertical grid lines only on the x-axis
    ax2.grid(False, axis='y')
    ax2.grid(True, axis='x', linestyle='-', color='black', alpha=0.15)  # Only vertical grid lines
    #ax2.set_facecolor('white')
    plt.tight_layout(pad=1)

    # Save the figure to the specified directory and filename
    if save_fig == True:
        if suffix_file is not None:
            out_path = Path(save_path) / f'forest_plot_{suffix_file}.png'
        else:
            out_path = Path(save_path) / f'forest_plot.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    return 0

def barplot_counts_fisher_test(df, cat_var1, cat_var2,
                               legend_title=None, y_label='Count', x_label=None, title=None,
                               mapping1=None, mapping2=None,
                               figsize=(6, 4), palette='viridis',
                               save_path=None, suffix_file = None):
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    # Compute the contingency table (rows: cat_var1, columns: cat_var2)
    contingency_table = pd.crosstab(df[cat_var1], df[cat_var2])

    # Run Fisher's exact test (works for 2x2 tables)
    odds_ratio, p_value = fisher_exact(contingency_table)

    # Convert the contingency table into tidy (long) format
    df_plot = contingency_table.stack().reset_index()
    df_plot.columns = [cat_var1, cat_var2, 'Count']

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
    sns.barplot(data=df_plot, x=cat_var2, y='Count', hue=cat_var1, palette=palette, ax=ax)

    if title is not None:
        ax.set_title(title, fontsize=12)
    if x_label is None:
        ax.set_xlabel(cat_var2, fontsize=10)
    else:
        ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.legend(title=legend_title, loc='lower left')

    # Annotate the plot with Fisher's test results in the top center of the axis
    annotation = f"Fisher Exact p = {p_value:.3f}\nOdds Ratio = {odds_ratio:.3f}"
    ax.text(0.5, 0.9, annotation,
            transform=ax.transAxes,
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    plt.tight_layout()

    if save_path is not None:
        if suffix_file is not None:
            plt.savefig(Path(save_path) / f'fisherTest_{suffix_file}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(Path(save_path) / f'fisherTest.png', dpi=300, bbox_inches='tight')

    return fig

from scipy.stats import kruskal

def boxplot_compare_distribution_by_category(df, cont_var, cat_var,
                                             cat_mapping=None,
                                             figsize=(6,4), palette='viridis',
                                             x_label=None, y_label=None, title=None,
                                             save_path=None, suffix_file=None):
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
        df['cat_label'] = df[cat_var].map(cat_mapping)
        plot_cat = 'cat_label'
        groups = list(cat_mapping.keys())
    else:
        plot_cat = cat_var
        groups = df[cat_var].unique()  # Get the unique groups from the original categorical variable.

    # Depending on the number of groups, run the appropriate test.
    if len(groups) == 2:
        test_name = "Mann–Whitney"
        group0 = df.loc[df[cat_var] == groups[0], cont_var]
        group1 = df.loc[df[cat_var] == groups[1], cont_var]
        stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
        stat_str = f"U = {stat}"
    elif len(groups) > 2:
        test_name = "Kruskal–Wallis"
        group_data = [df.loc[df[cat_var] == g, cont_var] for g in groups]
        stat, p_value = kruskal(*group_data)
        stat_str = f"H = {stat:.2f}"
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
    sns.boxplot(x=plot_cat, y=cont_var, data=df, palette=palette, ax=ax)
    sns.swarmplot(x=plot_cat, y=cont_var, data=df, color='0.2', ax=ax)

    # Annotate the plot with the p-value and test statistic.
    max_val = df[cont_var].max()
    annotation = f"{test_name} p = {p_value:.3f}\n{stat_str}"
    ax.text(0.5, max_val * 0.95, annotation,
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=12)

    plt.tight_layout()

    # Save the figure if a save path is provided.
    if save_path is not None:
        if suffix_file is not None:
            file_name = f'{test_name}_Test_{suffix_file}.png'
        else:
            file_name = f'{test_name}_Test.png'
        plt.savefig(Path(save_path) / file_name, dpi=300, bbox_inches='tight')

    return fig