# ML classes
from dataHandler import Config, MetadataHandler, OligosHandler, FeatureManager
import time
import argparse

# basics
from pathlib import Path
import re
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import joblib

# Plots
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.gridspec import GridSpec

#ML
import xgboost as xgb
import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_predict, cross_val_score, LeaveOneOut, StratifiedKFold, KFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest, VarianceThreshold, SelectFromModel, SelectPercentile
from sklearn import set_config
set_config(transform_output = "pandas")
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

#Survival
from lifelines.utils import concordance_index, median_survival_times,find_best_parametric_model
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import rmst_plot, qq_plot,add_at_risk_counts
from lifelines import LogNormalFitter, LogLogisticFitter, WeibullFitter, ExponentialFitter, WeibullAFTFitter, KaplanMeierFitter, CoxPHFitter

from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw, concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Aggregate Feature Importances Across All Folds
def aggregate_feature_importances(importance_list):
    # Aggregate feature importance over multiple dictionaries
    all_keys = set(k for dic in importance_list for k in dic)
    aggregated_importance = {k: [] for k in all_keys}
    for dic in importance_list:
        for k in all_keys:
            aggregated_importance[k].append(dic.get(k, 0))

    # Calculate the mean for each feature
    mean_importance = {k: np.mean(v) for k, v in aggregated_importance.items()}
    return mean_importance


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
    # Separate survival times and event indicators
    events = np.where(y_true < 0, 0, 1)  # 0 for censored, 1 for deceased
    times = np.abs(y_true)  #survival_time=np.abs(y_true) Absolute values for survival times
    #event_indicator = (y_true > 0).astype(int)  # 1 for events (positive values), 0 for censored (negative values)

    y_surv = Surv.from_arrays(event=events, time=times)

    # Calculate and return the C-index
    #return concordance_index(survival_time, -y_pred, event_indicator)
    #return concordance_index_ipcw(y_surv, y_surv, y_pred)[0]
    return concordance_index_censored(y_surv['event'], y_surv['time'], y_pred)[0]

def c_index_scorer_ipcw(y_train, y_val, y_pred):
    """
    Custom scorer to calculate Concordance Index (C-index).

    Parameters:
    - y_train: 1D array of survival times from training set (negative for censored, positive for events).
    - y_val: 1D array of survival times from validation set (negative for censored, positive for events).
    - y_pred: Predicted risk scores (higher scores indicate higher risk).

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

    # Calculate and return the C-index
    return concordance_index_ipcw(y_train_surv, y_val_surv, y_pred)[0]


class CustomStratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=420):
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


def search_best_model(estimator, param_grid, X_train, y_train, n_splits=5, n_iter=30, random_state=420, n_jobs = 1):
    # Create a custom scorer using C-index
    custom_scorer = make_scorer(c_index_scorer, greater_is_better=True)
    cv_inner = CustomStratifiedKFold(n_splits=n_splits, random_state=random_state)
    # search = RandomizedSearchCV(estimator, param_grid,
    #                                        cv=cv_inner, n_iter=n_iter,
    #                                        scoring=custom_scorer, refit=True,
    #                                        random_state=random_state, n_jobs=n_jobs)
    search = BayesSearchCV(estimator, search_spaces=param_grid,
                                 cv=cv_inner,
                                 n_iter=n_iter,
                                 scoring=custom_scorer,
                                 refit=True,
                                 random_state=random_state,
                                 n_jobs=n_jobs)

    search.fit(X_train, y_train)

    return search.best_estimator_

def generate_feature_importance_table(values, colnames, oligos_metadata, features, estimator_name,
                                      group_tests = None, figures_dir = './',
                                      with_oligos = True, with_additional_features = False,  with_run_plates=False):
    if group_tests is None:
        return  pd.DataFrame()

    feature_shap_values_df = pd.DataFrame({'SHAP value': np.abs(values).mean(0)}, index=colnames).sort_values(by='SHAP value', ascending=False)

    feature_shap_values_df = feature_shap_values_df.merge(oligos_metadata,
                                                          left_index=True, right_index=True, how='left').rename(columns={'full name' : 'Protein name', 'len_seq' : 'Protein length', 'pos' : 'Start position', 'Organism_complete_name':'Organism'})

    #filtered_index = feature_shap_values_df.index[~feature_shap_values_df.index.str.contains('bloodb', case=False)]
    keywords = ['agilent', 'corona2', 'twist', 'Sex', 'Age', 'run_plate']
    pattern = '|'.join(keywords)
    filtered_index = colnames[colnames.str.contains(pattern, case=False, regex=True)]

    percentiles = features[filtered_index].groupby(level=features.index.names[1]).mean()
    #print(percentiles)
    get_perc_columns = percentiles.columns[percentiles.columns.str.contains('agilent|corona2|twist|sex|run_plate', case=False)]
    percentiles[get_perc_columns] = percentiles[get_perc_columns].mul(100)
    percentiles = percentiles.T.rename(columns={0: f'{group_tests[0]}', 1: f'{group_tests[1]}'})

    feature_shap_values_df = pd.merge(feature_shap_values_df, percentiles, left_index=True, right_index=True, how='left')

    feature_shap_values_df['Ratio (log10)'] = np.log10(feature_shap_values_df[f'{group_tests[1]}'] / feature_shap_values_df[f'{group_tests[0]}'])

    feature_shap_values_df.reset_index(inplace = True)
    feature_shap_values_df.rename(columns = {'index': 'Peptide ID'}, inplace = True)
    #feature_shap_values_df['Peptide ID'] = feature_shap_values_df['Peptide ID'].str.replace(r'^(agilent)_\s*', '', regex=True)

    # Define filename based on flags and save CSV
    filename = f"shap_values_{estimator_name}_{'-'.join(group_tests)}" + (
        "_with_oligos" if with_oligos else "") + (
        "_with_additional_features" if with_additional_features else "") + (
        "_with_run_plates" if with_run_plates else "") + ".csv"

    feature_shap_values_df.to_csv(Path(figures_dir, filename), index=False)

    return feature_shap_values_df


def plot_shap_values(values, features, ax = None, color_map = 'viridis', max_display = 30,
                     label_groups = None, pattern = r'(bloodb|BloodB)',
                     suffix_file = 'default', plot_title = "",
                     save_fig = False, figures_dir = './' ):

    if label_groups is None:
        label_groups = ['group1', 'group2']
    if ax is None:
        fig, ax = plt.subplots()

    # SHAP summary plot without displaying
    shap.summary_plot(values, features = features, plot_type='dot', cmap=color_map,
                      max_display = max_display,  plot_size=[5,5], show=False)

    cbar = plt.gcf().axes[-1]  # Get the last axes, which corresponds to the color bar
    cbar.set_ylabel('Feature value', fontsize=10)  # Change the font size of the color bar label
    cbar.tick_params(labelsize=9)

    # Get current x-ticks and x-tick labels
    xticks = ax.get_xticks()
    #xticklabels = [f"{tick:.1f}" for tick in xticks]  # Format ticks to one decimal place
    # Set dynamic x-ticks and their labels
    ax.set_xticks(xticks)

    # Remove the default x-axis label "SHAP value"
    ax.set_xlabel('')  # Remove SHAP label

    # Manually add custom x-axis labels below the numeric ticks using transform=ax.transAxes for alignment
    ax.text(0.2, -0.06, label_groups[0], ha='center', va='top', fontsize=10, transform=ax.transAxes)
    ax.text(0.75, -0.06, label_groups[1], ha='center', va='top', fontsize=10,  transform=ax.transAxes)

    # Add a centered label below the x-axis using the same transformation
    ax.text(0.5, -0.14, 'Prediction toward...', ha='center', va='top', fontsize=11, fontweight='bold', transform=ax.transAxes)

    # Set y-axis labels to avoid overlap (increase label spacing)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    yticklabels = [re.sub(pattern, '', label.get_text()) for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels, fontsize=8)
    # Move y-tick labels closer to the plot by adjusting labelpad
    ax.tick_params(axis='y', pad=-10)
    ax.yaxis.set_label_coords(-0.3, 0.5)
    ax.set_xlim([values.min(), values.max()])

    # Add title to the plot
    if plot_title:
        ax.set_title(plot_title, fontsize=12, weight='bold', pad=10)

    # Draw horizontal grid lines
    for ytick in ax.get_yticks():
        ax.axhline(y=ytick, color='lightgrey', linestyle='--', linewidth=0.8)  # Adjust color and style as needed
    # Adjust layout and show the plot
    ax.grid(False)
    ax.patch.set_alpha(0.0)
    #ax.set_facecolor("white")

    if save_fig:
        plt.savefig(Path(figures_dir, f'shap_plot_{suffix_file}'), dpi=300, bbox_inches='tight')
    return ax


def get_top_features(feature_shap_values_df, group_tests = None, to_select_features = 10):

    def truncate_description(text, max_length=90):
        # Check if text is None, and return None if it is
        if text is None:
            return None
        # If text is not None, proceed to truncate
        if len(text) > max_length:
            return text[:max_length] + '...'  # Truncate and add ellipsis
        return text

    top_features = feature_shap_values_df.iloc[0:to_select_features,[0,2,7,8,9,1]]
    top_features[[group_tests[0], group_tests[1], 'Ratio (log10)']] = top_features[[group_tests[0], group_tests[1], 'Ratio (log10)']].round(2)
    top_features['Description'] = top_features['Description'].apply(lambda x: truncate_description(x))
    return top_features

# Function to create a color based on value (gradient) for percentages (same as in your code)
def colorize(val, min_val, max_val, palette = 'Reds'):
    norm_val = (val - min_val) / (max_val - min_val)  # Normalize value between 0 and 1
    if palette == 'Reds':
        color = colormaps['RdYlGn'](norm_val)  # Use the 'RdYlGn' color map for percentages
    else:
        color = colormaps['BuPu'](norm_val)  # Use the 'BuPu' color map
    return color

# Render the main table (same as your code)
def render_main_table(df, col_widths, row_height=0.625, font_size=11, header_color='lightgray',
                      row_colors=None, edge_color='w', bbox=None, ax=None, **kwargs):
    if bbox is None:
        bbox = [0, 0, 1, 1]
    if row_colors is None:
        row_colors = ['#f1f1f2', 'w']
    if ax is None:
        fig, ax = plt.subplots(figsize=(np.sum(col_widths), len(df) * row_height + 1))
        ax.axis('off')  # Turn off the axis

    # Get min/max values for scaling
    min_val = 0
    max_val = 100

    min_val_ratio = -2
    max_val_ratio = 2

    # Create table
    mpl_table = ax.table(cellText=df.values, bbox=bbox, colLabels=df.columns, cellLoc='center', colWidths=col_widths, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    # Iterate over each cell and apply color
    for (i, j), cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        cell.set_text_props(fontsize=7.5)  # Set a smaller font size for wrapped text

        if i == 0:  # Header
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor(header_color)
        else:

            # Apply color gradient to 'Disease', 'Controls', and 'Ratio (log)' columns
            if j == 2:  # 'Disease' column
                cell_val = df.iloc[i - 1, 2]
                color = colorize(cell_val, min_val, max_val, 'Reds')  # Get color based on value
                cell.set_facecolor(color)
                cell.set_text_props(color='black')  # Set text color to white

            elif j == 3:  # 'Control' column
                cell_val = df.iloc[i - 1, 3]
                color = colorize(cell_val, min_val, max_val)  # Get color based on value
                cell.set_facecolor(color)
                cell.set_text_props(color='black')  # Set text color to white

            elif j == 4:  # 'Ratio (log)' column
                cell_val = df.iloc[i - 1, 4]
                color = colorize(cell_val, min_val_ratio, max_val_ratio, 'Blues')  # Get color based on value
                cell.set_facecolor(color)
                cell.set_text_props(color='black')  # Set text color to white

            else:
                # Alternate row colors for text columns
                cell.set_facecolor(row_colors[i % len(row_colors)])

    return ax

def render_header_main_table(ax=None):
    col_colors = ['lightgray', 'lightgray', 'lightgray']  # Color for the headers

    header_table = plt.table(cellLoc='center',
                             colWidths=[0.76, 0.292, 0.107], bbox=[0, 1, 1, 0.12],
                             cellColours=[col_colors])

    for (i, j), cell in header_table._cells.items():
        cell.set_edgecolor('w')
        if i == 0:  # Header
            cell.set_text_props(weight='bold', color='black')

    ax.text(0.34, 1.05, 'Peptide details', ha='center', color='black', weight='bold')
    ax.text(0.79, 1.035, 'Antibody responses\nappearing % in ...', ha='center', fontsize=10, color='black', weight='bold')
    ax.text(0.955, 1.035, 'Feature\nimportance', ha='center', fontsize=10, color='black', weight='bold')

    return ax


def plot_table_top_features(feature_shap_values_df, group_tests = None, ax = None, to_select_features = 10, set_type ="",
                              figure_dir = './', estimator_name = '', suffix_file = None, save_fig = False ):
    top_features_df = get_top_features(feature_shap_values_df, group_tests = group_tests, to_select_features = to_select_features)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Render the header table first
    render_header_main_table(ax)

   # Ensure SHAP values are correctly formatted before rendering
    top_features_df['SHAP value'] = top_features_df['SHAP value'].round(2).astype(str)  # Ensure rounding is preserved

    # Render the main data table
    render_main_table(top_features_df, col_widths=[0.35, 2.15, 0.32, 0.32, 0.32, 0.35], ax=ax)

    # Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    plt.tight_layout()
    if save_fig:
        if suffix_file is None:
            suffix_file = set_type+"_"+estimator_name+"_" + "-".join(group_tests)
        plt.savefig(Path(figure_dir) / f'table_top{to_select_features}_SHAPvalues_{suffix_file}.png', dpi=300, bbox_inches='tight')
    return ax


def plot_time_dependent_auc_with_table(
    time_points, auc_values, mean_auc, samples_at_risk_df,
    time_points_highlight=None, auc_values_highlight=None,
    time_measure='Months', color_auc='dodgerblue', figures_dir='./', save_fig=False, suffix_file='trainingSet'):
    """
    Plots time-dependent AUC over follow-up time with a table displaying samples at risk, censored, and events
    positioned just below the AUC plot as a horizontal table aligned with the x-axis.

    Parameters:
    - time_points (array-like): The time points (e.g., months) to plot on the x-axis.
    - auc_values (array-like): The AUC values corresponding to each time point.
    - mean_auc (float): The mean AUC value to be displayed as a horizontal line.
    - samples_at_risk_df (pd.DataFrame): DataFrame containing 'At risk', 'Censored', and 'Events' per time point.
    - time_points_highlight (array-like): Specific time points to highlight on the AUC curve.
    - auc_values_highlight (array-like): AUC values at the specific points to highlight.

    Returns:
    - fig, ax_auc, ax_table: The figure and axes objects for further customization.
    """
    # Create a figure
    #fig, (ax_auc, ax_table) = plt.subplots(2, 1, figsize=(10, 8))  # Simple subplot with two rows
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[0.8, 0.15], figure=fig)  # Adjust height ratios for tighter space
    ax_auc = fig.add_subplot(gs[0])

    # Top panel: AUC plot
    ax_auc.plot(time_points, auc_values, color=color_auc, linestyle='-', marker='', linewidth=2, label='Time-Dependent AUC')
    ax_auc.axhline(y=mean_auc, color='orange', linestyle='--', linewidth=1.5, label=f'Mean AUC = {mean_auc:.3f}')
    ax_auc.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5)
    if time_points_highlight is not None and auc_values_highlight is not None:
        ax_auc.scatter(time_points_highlight, auc_values_highlight, color=color_auc, s=50, zorder=3)
    ax_auc.set_xlabel(f'Time ({time_measure})', fontsize=12)
    ax_auc.set_ylabel('Time-Dependent AUC', fontsize=12)
    ax_auc.set_ylim([0.0, 1.05])
    ax_auc.set_xlim([0, max(time_points)])  # Add some padding to the x-axis for aesthetics
    ax_auc.legend(fontsize=10, loc='lower right')
    ax_auc.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    # Bottom panel: Table displaying At Risk, Censored, and Events horizontally
    ax_table = fig.add_subplot(gs[1])
    ax_table.set_xticks(time_points)
    ax_table.set_xticklabels([])
    #ax_table.set_xticklabels(time_points, ha='center', fontsize=8)

    # Set the y-ticks with the row labels for the table
    y_positions = ['Events', 'Censored', 'At Risk']
    ax_table.set_yticks(np.array([0.2,0.5,0.8]))  # 3 categories: At Risk, Censored, Events
    ax_table.set_yticklabels(y_positions, ha='right', fontsize=10)
    ax_table.grid(False)  # Disable the grid
    ax_table.set_facecolor('white')

    max_time_point = max(time_points)  # Find the maximum time point
    if max_time_point not in time_points_highlight:
        time_points_highlight = list(time_points_highlight) + [max_time_point]  # Add it to the list if not already included

    for i, time_point in enumerate(time_points_highlight):
        at_risk_val = samples_at_risk_df.loc[time_point, 'At Risk']
        censored_val = samples_at_risk_df.loc[time_point, 'Censored']
        events_val = samples_at_risk_df.loc[time_point, 'Events']

        # Display the values as text for At Risk, Censored, and Events for each time point
        ax_table.text(time_point, 0.2, f'{events_val}', ha='center', va='center', fontsize=10, color='black')
        ax_table.text(time_point, 0.5, f'{censored_val}', ha='center', va='center', fontsize=10, color='black')
        ax_table.text(time_point, 0.8, f'{at_risk_val}', ha='center', va='center', fontsize=10, color='black')

    # Save figure if required
    if save_fig:
        plt.savefig(Path(figures_dir) / f'time-dependent_auc_with_table_{suffix_file}.png', dpi=300, bbox_inches='tight')

    return fig, ax_auc, ax_table


def calculate_time_dependent_auc(y_event_train, y_time_train,
                                 y_event_test, y_time_test, risk_scores,
                                  time_points_highlight=None, num_points=50, buffer=0.001):
    """
    Calculate and plot Time-Dependent AUC based on risk scores for the testing set.

    Args:
        y_event_train (array-like): Event status for training data (1 if event occurred, 0 if censored).
        y_time_train (array-like): Survival time for training data.
        y_event_test (array-like): Event status for testing data (1 if event occurred, 0 if censored).
        y_time_test (array-like): Survival time for testing data.
        risk_scores (array-like): Predicted risk scores for the testing set.
        time_points_highlight (array-like, optional): Specific time points to highlight in AUC plot.
        num_points (int, optional): Number of time points to generate for the AUC calculation.
        buffer (float, optional): Small buffer to avoid boundary issues in time calculations.

    Returns:
        - auc_values (array): AUC values at each time point.
        - mean_auc (float): Mean AUC value across time points.
        - samples_at_risk (pd.Series): Number of samples at risk at each time point.
    """
    # Set up time points within the range of the train and test follow-up times
    valid_indices = y_time_test.abs() <= y_time_train.abs().max()
    y_event_test = y_event_test[valid_indices]
    y_time_test = y_time_test[valid_indices]

    min_time = max(y_time_test.abs().min(), y_time_train.abs().min()) + buffer  # Non-negative minimum time
    max_time = y_time_test.abs().max() - buffer  # Ensuring max_time is strictly less than training max

    time_points = np.linspace(min_time, max_time, num=num_points)
    if time_points_highlight is None:
        time_points_highlight = np.concatenate(([1], np.arange(6, max_time, step=6)))
    time_points = np.unique(np.concatenate((time_points, time_points_highlight)))

    # Step 2: Prepare survival data in structured format for sksurv
    y_structured_train = np.array([(e, t) for e, t in zip(y_event_train, y_time_train.abs())],
                                  dtype=[('event', 'bool'), ('time', 'float')])
    y_structured_test = np.array([(e, t) for e, t in zip(y_event_test, y_time_test.abs())],
                                 dtype=[('event', 'bool'), ('time', 'float')])

    # Predict risk scores for the filtered external dataset
    risk_scores = risk_scores[valid_indices]

    # Calculate time-dependent AUC
    auc_values, mean_auc = cumulative_dynamic_auc(y_structured_train, y_structured_test, risk_scores, time_points)
    auc_values_highlight, _ = cumulative_dynamic_auc(y_structured_train, y_structured_test, risk_scores, time_points_highlight)

    # Calculate the number of individuals at risk at each time point
    samples_at_risk = []  # At time 0, all individuals are at risk
    samples_censored = []
    samples_events = []
    at_risk = len(risk_scores)
    for t in time_points:
        t=t+0.01
        # Censored: count samples that were censored at time t
        count_censored = np.sum((y_time_test.abs() <= t) & (y_event_test == 0))  # Only censored samples
        # Events: count samples that experienced the event at or before time t
        count_events = np.sum((y_time_test.abs() <= t) & (y_event_test == 1))  # Only event samples
        count_at_risk = at_risk - count_events  - count_censored  # Subtract events and censored

        samples_at_risk.append(count_at_risk)
        samples_censored.append(count_censored)
        samples_events.append(count_events)

    # Create a DataFrame to store these counts
    samples_at_risk = pd.DataFrame({
        'At Risk': samples_at_risk,
        'Censored': samples_censored,
        'Events': samples_events
    }, index=time_points)
    # for t in time_points:
    #     count_at_risk = np.sum((y_time_test.abs() >= t)  & (y_event_test == 1)) # try with y_event_test==1
    #     samples_at_risk.append(count_at_risk)
    # samples_at_risk = pd.Series(samples_at_risk, index=time_points, name='Samples at risk')

    return auc_values, mean_auc, auc_values_highlight, samples_at_risk, time_points, time_points_highlight

def calculate_antigen_scores_scaled(shap_values, y_time, y_event, top_peptides, scaler=None, scaler_antigens=None, threshold=None, val_quantile=40, return_all = False):
    """
    Function to calculate antigen scores based on SHAP values for an external dataset.

    Parameters:
    - shap_values: np.ndarray
        SHAP values for the external dataset.
    - y_time, y_event: pd.DataFrame
        OS time and status with sample indices.
    - top_peptides: list
        List of top peptides selected during the training.
    - scaler: MinMaxScaler (optional)
        Pre-fitted MinMaxScaler to scale antigen scores.
    - threshold: float (optional)
        Threshold value to dichotomize the antigen score.

    Returns:
    - antigen_scores: pd.DataFrame
        DataFrame containing Antigen Score, Scaled Antigen Score (0-1), and Dichotomized Antigen Score.
    - threshold: how to dichotomize scores (optional)
    - scaler: how to scale other set (optional)
    """

    tmp_shap_values = abs(shap_values)
    # Scale antigen score (use provided scaler or fit a new one)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        tmp_shap_values =  pd.DataFrame(scaler.fit_transform(tmp_shap_values[top_peptides]), index=tmp_shap_values.index)
        #tmp_shap_values_scaled = scaler.fit_transform(tmp_shap_values)
    else:
        tmp_shap_values = pd.DataFrame(scaler.transform(tmp_shap_values[top_peptides]), index=tmp_shap_values.index)
        #tmp_shap_values_scaled = pd.DataFrame(scaler.transform(tmp_shap_values), index=tmp_shap_values.index)
        tmp_shap_values = np.clip(tmp_shap_values, 0, 1)

    #tmp_shap_values.columns = shap_values.columns
    #tmp_shap_values = tmp_shap_values[top_peptides]
    tmp_shap_values.columns = shap_values[top_peptides].columns
    tmp_shap_values = tmp_shap_values * np.sign(shap_values[top_peptides])
    antigen_scores = tmp_shap_values.sum(axis=1).to_frame()

    if scaler_antigens is None:
        scaler_antigens = MinMaxScaler(feature_range=(0, 1))
        antigen_scores_scaled = pd.DataFrame(scaler_antigens.fit_transform(antigen_scores), index=antigen_scores.index)
    else:
        antigen_scores_scaled = pd.DataFrame(scaler_antigens.transform(antigen_scores), index=antigen_scores.index)
        antigen_scores_scaled = np.clip(antigen_scores_scaled, 0, 1)


    # Set a threshold to dichotomize the scores (use provided threshold or the median)
    if threshold is None:
        #threshold = antigen_scores.quantile(35/100)
        threshold = antigen_scores_scaled.quantile(val_quantile/100)

    #antigen_scores_dichotomized = (antigen_scores >= threshold).astype(int)
    antigen_scores_dichotomized = (antigen_scores_scaled >= threshold).astype(int)
    antigen_scores = antigen_scores.rename(columns={antigen_scores.columns[0]: 'Antigen Score'})
    antigen_scores_scaled = antigen_scores_scaled.rename(columns={antigen_scores_scaled.columns[0]: 'Antigen Score (Scaled)'})
    antigen_scores_dichotomized = antigen_scores_dichotomized.rename(columns={antigen_scores_dichotomized.columns[0]: 'Antigen Score (Dichotomized)'})

    antigen_scores_tmp = pd.merge(y_time, y_event, left_index=True, right_index=True)
    antigen_scores_tmp = pd.merge(antigen_scores_tmp, antigen_scores, left_index=True, right_index=True)
    antigen_scores_tmp = pd.merge(antigen_scores_tmp, antigen_scores_scaled, left_index=True, right_index=True)
    antigen_scores_df = pd.merge(antigen_scores_tmp, antigen_scores_dichotomized, left_index=True, right_index=True)

    if return_all:
        return antigen_scores_df, threshold, scaler, scaler_antigens
    else:
        return antigen_scores_df

# Function to perform log-rank test between high and low antigen groups
def perform_logrank_test(df, time_column, event_column, group_column):
    group1 = df[df[group_column] == 0]
    group2 = df[df[group_column] == 1]
    results = logrank_test(group1[time_column], group2[time_column],
                           event_observed_A=group1[event_column], event_observed_B=group2[event_column])
    print(f"Log-Rank Test p-value: {results.p_value:.4f}")
    return results.p_value

# Function to create Kaplan-Meier plot with log-rank p-value annotation
def plot_kaplan_meier(df, time_column, event_column, group_column, labels,
                      ax=None, title='', xlabel='', ylabel='',
                      save_fig=False, figures_dir='./', suffix_file='defaultSet'):

    # Define colors explicitly for each group
    group_colors = {0: 'dodgerblue', 1: 'darkorange'}

    # Create figure for plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Fit and plot for each group in a consistent order
    kmf_list = []
    for group_value in sorted(labels.keys()):  # Ensure consistent order
        mask = df[group_column] == group_value
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_column][mask], event_observed=df[event_column][mask], label=labels[group_value])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=group_colors[group_value])
        kmf_list.append(kmf)

    # Perform log-rank test between groups
    group1 = df[df[group_column] == df[group_column].unique()[0]]
    group2 = df[df[group_column] == df[group_column].unique()[1]]

    # Ensure both groups are non-empty to perform the log-rank test
    if not group1.empty and not group2.empty:
        results = logrank_test(group1[time_column], group2[time_column],
                               event_observed_A=group1[event_column], event_observed_B=group2[event_column])
        p_value = results.p_value

        # Annotate the plot with the p-value from the log-rank test
        p_value_text = f'p = {p_value:.3f}'
        ax.text(0.7, 0.05, p_value_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

    # Customize the plot
    from lifelines.plotting import add_at_risk_counts
    add_at_risk_counts(kmf_list[0], kmf_list[1], ax=ax)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)
    plt.tight_layout()

    # Save figure if specified
    if save_fig:
        fig_path = Path(figures_dir) / f'kaplan_meier_{suffix_file}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
    return ax

def align_features(X_train, X_ext):
    """Align features between training and external test sets."""
    common_features = list(set(X_train.columns).intersection(X_ext.columns))
    if common_features:
        X_train = X_train[common_features]
        X_ext = X_ext[common_features]
        return X_train, X_ext
    else:
        raise ValueError(f"Training and External set shared no common features")


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
    events = np.where(y < 0, 0, 1)
    times = np.abs(y)
    y_surv = Surv.from_arrays(event=events, time=times)

    n_features = X.shape[1]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(univariate_cox_score_single)(j, X, y_surv) for j in range(n_features)
    )
    return np.array(scores)

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
            1D array of survival times; negative values indicate censoring.

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
        # If self.coef_ has multiple columns (one per alpha), choose the coefficients for the last alpha
        #if self.coef_.ndim > 1:
        #logger.info(f"Coefficient shape before selection: {self.coef_.shape}")
        self.coef_ = self.coef_[:, -1]

        return self

def calculate_cumulative_dynamic_auc(y_train, y_valid, risk_scores, time_points):

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
    auc_values, _ = cumulative_dynamic_auc(
        y_train_surv,
        y_valid_surv,
        risk_scores[valid_indices],
        time_points_highlight  # generate or use time points appropriate for this fold
    )

    return pd.Series(auc_values, index=time_points_highlight)

def nested_cv_single(train_idx, valid_idx, X_train, y_time_train, pipeline = None,
                     param_grid = None, n_splits = 5, n_iter = 30,
                     random_state = 420, n_jobs = -1):

    set_config(transform_output = "pandas")

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
        best_estimator = search_best_model(pipeline, param_grid, X_train_fold, y_train_fold,
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
    #fold_shap_df_aligned = shap_values_fold_df.reindex(columns=shap_values.columns, fill_value=0)
    # If a sample appears in only one fold, just assign. If overlapping folds exist, you could sum and later average.
    #shap_values.loc[X_valid_fold.index] += fold_shap_df_aligned
    # Compute performance score (e.g., c-index)
    #fold_cindex = c_index_scorer(y_valid_fold, risk_scores_fold)
    cindex_fold = c_index_scorer_ipcw(y_train_fold, y_valid_fold, risk_scores_fold)

    return valid_idx, risk_scores_fold, shap_values_fold_df, cindex_fold, best_estimator


def nested_cv(
        X_train: pd.DataFrame,
        y_time_train: pd.Series,
        pipeline: Optional[Pipeline] = None,
        param_grid: Optional[Dict] = None,
        n_splits: int = 10,
        n_splits_inner: int = 5,
        n_iter: int = 30,
        max_time_point = None,
        random_state: int = 420,
        n_jobs: int = 1,
        n_jobs_inner: int = -1
) -> Tuple[List[Pipeline], pd.DataFrame, pd.Series, pd.DataFrame, List[float], List[np.ndarray]]:
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
    max_time_points : int, default None
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
    c_index : List[float]
        List of performance scores (e.g., c-index) for each outer fold.
    validation_indices : List[np.ndarray]
        validation indices for each outer fold.
    """
    #outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_cv = CustomStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Run the outer folds in parallel.
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(nested_cv_single)(
            train_idx, valid_idx, X_train, y_time_train,
            pipeline = pipeline, param_grid = param_grid,
            n_splits = n_splits_inner, n_iter = n_iter,
            random_state = random_state, n_jobs = n_jobs_inner) for train_idx, valid_idx in outer_cv.split(X_train, y_time_train))
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
    # Aggregate results from each fold.
    i=0
    for valid_idx, fold_risk, fold_shap_df, fold_cindex, model in fold_results:
        model_list.append(model)
        # Assign risk scores for the validation fold.
        risk_scores.iloc[valid_idx] = fold_risk
        # Update master SHAP values. Since folds are disjoint, direct assignment works.
        shap_values.loc[fold_shap_df.index, fold_shap_df.columns] = fold_shap_df
        # Reindex fold SHAP values to the master feature set, filling missing values with 0
        #fold_shap_df_aligned = fold_shap_df.reindex(columns=shap_values.columns, fill_value=0)
        # If a sample appears in only one fold, just assign. If overlapping folds exist, you could sum and later average.
        #shap_values.loc[fold_shap_df.index] += fold_shap_df_aligned
        y_valid_fold = y_time_train.iloc[valid_idx]
        auc_values_fold = calculate_cumulative_dynamic_auc(y_time_train.loc[y_time_train.index.difference(y_valid_fold.index)] , y_valid_fold,
                                                           fold_risk, time_points)
        time_dependent_auc.loc[i, auc_values_fold.index] = auc_values_fold.values

        del fold_shap_df
        c_index.append(fold_cindex)
        validation_indices.append(valid_idx)
        i=i+1
    logger.info(f"Mean Concordance Index (C-index) across folds: {np.mean(c_index):.4f}")
    logger.info(f"Mean Time-Dependent AUC across folds: {time_dependent_auc.mean(skipna=True).mean():.4f}")

    return model_list, shap_values, risk_scores, time_dependent_auc, c_index, validation_indices

def train_and_evaluate_model(X_train: pd.DataFrame, y_time_train: pd.Series,
                             X_test: pd.DataFrame, y_time_test: pd.Series,
                             pipeline: Optional[Pipeline] = None,
                             param_grid: Optional[Dict] = None,
                             n_splits: int = 10,
                             n_iter: int = 30,
                             max_time_point = None,
                             random_state: int = 420,
                             n_jobs: int = 1,
                             get_only_model: bool = False)-> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series, float]:

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
        best_estimator = search_best_model(pipeline, param_grid, X_train, y_time_train,
                                           n_splits=n_splits, n_iter=n_iter, random_state=random_state, n_jobs=n_jobs)
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
            return None

    # Compute SHAP values using the regressor (last step)
    explainer = shap.TreeExplainer(best_estimator['regressor'])
    shap_values = explainer.shap_values(X_test)
    shap_values_df = pd.DataFrame(shap_values, index=X_test.index, columns=X_test.columns)
    # Reindex fold SHAP values to the master feature set, filling missing values with 0
    #shap_df_aligned = shap_values_df.reindex(columns=shap_values.columns, fill_value=0)
    # If a sample appears in only iteration just assign. If overlapping exist, you could sum and later average.
    #shap_values.loc[X_test.index] += shap_df_aligned

    # Compute performance score (e.g., c-index)
    #cindex = c_index_scorer(y_test, risk_scores)
    cindex = c_index_scorer_ipcw(y_time_train, y_time_test, risk_scores)

    max_time_point = y_time_train.abs().max() if max_time_point is None else max_time_point
    time_points =  np.arange(1, max_time_point, step=1)
    time_dependent_auc = calculate_cumulative_dynamic_auc(y_time_train, y_time_test, risk_scores, time_points)

    return best_estimator, shap_values_df, risk_scores, time_dependent_auc, cindex


if __name__ == '__main__':
    # Parse the command-line argument for the random seed
    parser = argparse.ArgumentParser(description="Run nested CV with custom random seed.")
    parser.add_argument("seed", type=int, nargs="?", default=420,
                        help="Random seed (default: 420)")
    args = parser.parse_args()
    random_seed = args.seed

    start_time = time.time()

    config_file = "/home/creyna/Vogl-lab_Projects_git/HCC/Metadata/config_survival_trainTest.yaml"
    config = Config(config_file)
    metadata_handler = MetadataHandler(config)
    oligos_handler = OligosHandler(config)
    feature_manager = FeatureManager(config, metadata_handler, oligos_handler,
                                     subgroup='all',
                                     with_oligos=True,
                                     with_additional_features=True,
                                     filter_by_entropy=False,
                                     prevalence_threshold_min=0,
                                     prevalence_threshold_max=100)

    X_train, y_event_train = feature_manager.get_features_target()
    y_time_train = X_train["OS months"]
    X_train.drop(columns=["OS months"], inplace=True)
    y_time_train = y_time_train.where(y_event_train == 1, -y_time_train)  # np.where(y_event, y_time, -y_time)

    # XGBoost parameters for Cox model
    params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'tree_method': 'hist',
        'random_state': 420 #random_seed,
        #'booster': 'dart'  # Use DART booster
    }

    pipeline = Pipeline([
        ('variance_removal', VarianceThreshold(threshold=0.0)),
        # ('feature_selection', SelectKBest(score_func=univariate_cox_score, k=500)),
        # ('feature_selection', SelectPercentile(score_func=univariate_cox_score, percentile=20)),
        ('feature_selection',
         SelectFromModel(CoxnetWrapper(l1_ratio=0.6, alpha_min_ratio=0.0001, n_alphas=100), threshold=1e-5)),
        ('regressor', xgb.XGBRegressor(**params, n_jobs=1))
    ])

    param_grid = {
        #'feature_selection__k': Integer(500, 2000),
        #'feature_selection__percentile': Integer(5, 50),
        'feature_selection__estimator__l1_ratio': Real(0.2,0.8, prior='uniform'),
        #'feature_selection__estimator__l1_ratio': Categorical([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        'feature_selection__estimator__alpha_min_ratio': Real(1e-5, 1e-3, prior='log-uniform'),
        #'feature_selection__estimator__alpha_min_ratio': Categorical([0.001, 0.0001, 0.00001]),
        # 'feature_selection__estimator__n_alphas': Integer(50, 100),
        'feature_selection__estimator__n_alphas': Categorical([50, 100]),
        'regressor__learning_rate': Real(0.01, 0.3, prior='uniform'),
        #'regressor__learning_rate': Categorical([0.01, 0.05, 0.1, 0.2]),
        'regressor__max_depth': Integer(3, 10),
        'regressor__subsample': Real(0.6, 1.0, prior='uniform'),
        'regressor__colsample_bytree': Real(0.6, 1.0, prior='uniform'),
        #'regressor__reg_lambda': Real(0.1, 10.0, prior='log-uniform'),
        #'regressor__reg_lambda': Categorical([0.1, 1.0, 10.0]),
        'regressor__n_estimators': Integer(50, 500)
        #'regressor__n_estimators': Categorical([50, 100, 200, 500])
    }

    model_list, train_shap_values, risk_scores_train, time_dependent_auc_train, c_index_train, validation_indices = nested_cv(X_train,
                                                                                y_time_train,
                                                                                pipeline = pipeline,
                                                                                param_grid=param_grid,
                                                                                n_splits=10, n_splits_inner=5,
                                                                                n_iter=50,  max_time_point=25,
                                                                                random_state=random_seed,
                                                                                n_jobs = 1, n_jobs_inner = 5)

    end_time = time.time()
    logger.info(f"Script runtime: {end_time - start_time:.2f} seconds")

    # Save the results as a dictionary
    # Save the results as a dictionary
    results = {
        'model_list': model_list,
        'train_shap_values': train_shap_values,
        'risk_scores_train': risk_scores_train,
        'time_dependent_auc_train': time_dependent_auc_train,
        'c_index_train': c_index_train,
        'validation_indices_train': validation_indices
    }

    joblib.dump(results, f'repeat_nested_cv_results_bestmodel_auc_{random_seed}.joblib')