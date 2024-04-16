"""
Module for plotting fairness metrics and accuracy.

This module contains functions for visualizing fairness metrics and accuracy
for different groups based on a specified attribute.

Functions:
- plot_attribute_distribution: Plot the distribution of a single attribute.
- plot_grouped_attribute_distribution: Plot the distribution of an attribute grouped by another attribute.

- plot_fairness_metrics: Plot fairness metrics and accuracy for a given attribute.

Prepared specifically for the censys_analysis notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_attribute_distribution(df, column_name, figsize=(5, 3), annotaion_rotation=0):
    """
    Plot the distribution of a categorical attribute in a DataFrame.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - column_name (str): The name of the column whose distribution is to be plotted.
    - figsize (tuple, optional): The size of the figure (width, height). Default is (5, 3).
    - annotaion_rotation (int, optional): The rotation angle of annotations on the bars. Default is 0.

    Returns:
    - None

    This function plots a bar chart showing the distribution of the specified categorical attribute in the DataFrame.
    Each category in the attribute is represented by a bar, and the height of each bar indicates the count of occurrences.
    Additionally, the percentage of occurrence for each category is annotated on its respective bar.

    Example:
    >>> plot_attribute_distribution(df, 'gender', figsize=(6, 4), annotaion_rotation=45)
    """

    # Calculate the counts of each category in the specified column
    attribute_counts = df[column_name].value_counts()

    # Define colors based on column values
    colors = []
    if column_name == "gender":
        colors = ["royalblue", "salmon"]
    elif column_name == "is_white":
        colors = ["lightgrey", "brown"]
    else:
        colors = "royalblue"
    # Calculate the percentage of occurrence for each category
    attribute_percentage = (attribute_counts / attribute_counts.sum()) * 100

    # Plot the histogram
    plt.figure(figsize=figsize)  # Adjust the figure size as needed
    bars = plt.bar(attribute_counts.index, attribute_counts.values, color=colors)
    plt.xlabel(
        column_name.capitalize(), fontsize=14
    )  # Capitalize the column name for better readability
    plt.ylabel("Count", fontsize=14)
    plt.title(f"{column_name} distribution", fontsize=16)
    plt.xticks(
        rotation=70, fontsize=12
    )  # Rotate x-axis labels for better readability if needed
    plt.yticks(fontsize=12)

    # Set the maximum y-axis limit to 30,000
    plt.ylim(top=29000)

    # Remove box around the graph
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Annotate bars with percentages with larger font size
    for bar, count, percentage in zip(
        bars, attribute_counts.values, attribute_percentage.values
    ):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.show()


def plot_grouped_attribute_distribution(
    df, group_column, attribute_column, figsize=(5, 4), annotaion_rotation=0
):
    """
    Plot the distribution of a categorical attribute within groups defined by another categorical attribute.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - group_column (str): The name of the column defining the groups.
    - attribute_column (str): The name of the column whose distribution within groups is to be plotted.
    - figsize (tuple, optional): The size of the figure (width, height). Default is (5, 4).
    - annotaion_rotation (int, optional): The rotation angle of annotations on the bars. Default is 0.

    Returns:
    - None

    This function plots a grouped bar chart showing the distribution of the specified categorical attribute
    within groups defined by another categorical attribute in the DataFrame. Each group is represented by a set
    of bars, and each category within the attribute_column is represented by a bar within its respective group.
    The height of each bar indicates the percentage of occurrences, and annotations show the percentages.

    Example:
    >>> plot_grouped_attribute_distribution(df, 'gender', 'income', figsize=(6, 4), annotaion_rotation=45)
    """

    # Define colors based on column values
    colors = []
    if "gender" in [group_column, attribute_column]:
        colors = ["royalblue", "salmon"]
    elif "is_white" in [group_column, attribute_column]:
        colors = ["lightgrey", "brown"]
    else:
        colors = "royalblue"
    # Calculate the percentage of
    # Group the DataFrame by the gender column and attribute column, and calculate the counts for each group
    attribute_counts = (
        df.groupby([group_column, attribute_column]).size().unstack(fill_value=0)
    )

    # Calculate the total counts for each category in the gender column
    total_counts = attribute_counts.sum(axis=1)

    # Calculate the percentages for each category in the gender column and attribute column
    attribute_percentages = attribute_counts.div(total_counts, axis=0) * 100
    attribute_percentages_trans = attribute_percentages.T
    attribute_percentages_trans = attribute_percentages_trans[
        attribute_percentages_trans.columns[::-1]
    ]

    N_groups = len(attribute_percentages_trans.index)
    x = np.arange(N_groups)  # the label locations
    width = 0.35  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained", figsize=figsize)

    for attribute, measurement in attribute_percentages_trans.items():
        offset = width * multiplier
        bars = ax.bar(
            x + offset, measurement, width, label=attribute, color=colors[multiplier]
        )
        ax.bar_label(
            bars,
            fmt=lambda x: f"{x:.1f}%",
            padding=3,
            fontsize=14,
            rotation=annotaion_rotation,
        )
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Percentage")
    ax.set_title(
        "{} distribution by {}".format(attribute_column, group_column), fontsize=16
    )
    ax.set_xticks(
        x + width / N_groups,
        attribute_percentages_trans.index,
        rotation=70,
        fontsize=12,
    )
    ax.legend(loc="upper right")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.show()


def plot_fairness_metrics(
    accuracy: float,
    approach: str,
    metrics: list[list[float]],
    attribute_name: str,
    figsize=(8, 5),
    annotaion_rotation=0,
):
    """
    Plot fairness metrics and accuracy for a given attribute.

    Parameters:
    - accuracy (float): The accuracy of the model.
    - approach (str): The name of the approach or method used.
    - metrics (list of lists of float): The fairness metrics data, where each inner list represents
      the metrics for a specific group, and each element in the inner list represents a metric value.
    - attribute_name (str): The name of the attribute being analyzed.
    - figsize (tuple, optional): The size of the figure (width, height). Default is (8, 5).
    - annotaion_rotation (int, optional): The rotation angle of annotations on the bars. Default is 0.



    Returns:
    - None

    This function plots fairness metrics and accuracy for a given attribute. Fairness metrics are shown
    as grouped bar charts, where each group represents a different group defined by the attribute.
    Accuracy is shown as a separate bar chart.

    Example:
    >>> plot_fairness_metrics(0.85, 'Optimized', [[80, 85, 90], [75, 82, 88]], 'gender', figsize=(8, 5), annotaion_rotation=45)
    """
    x_index = ["positive rate", "true positive rate", "false negtive rate"]

    if attribute_name == "gender":
        labels = ["male", "female"]
        colors = ["royalblue", "salmon"]
    elif attribute_name == "is_white":
        labels = ["white", "not-white"]
        colors = ["lightgrey", "brown"]
    else:
        labels = []
        colors = []

    # Determine the number of groups and the width of each bar
    num_groups = len(metrics[0])
    x = np.arange(num_groups)
    width = 0.35

    # Create a list of indices for the x-axis
    indices = np.arange(num_groups)
    multiplier = 0

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        layout="constrained",
        figsize=figsize,
        gridspec_kw={"width_ratios": [7, 1]},
    )
    for arr in metrics:
        offset = width * multiplier
        bars = ax1.bar(
            indices + offset,
            arr,
            width,
            label=labels[multiplier],
            color=colors[multiplier],
        )
        ax1.bar_label(
            bars,
            fmt=lambda x: f"{x:.1f}%",
            padding=3,
            fontsize=14,
            rotation=annotaion_rotation,
        )
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel("Percentage")
    ax1.set_title(
        "Fairness Metrics by {}".format(attribute_name.capitalize()), fontsize=16
    )
    ax1.set_xticks(x + width / num_groups, x_index, rotation=70, fontsize=12)
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 100)

    # Plot the accuracy bar in the second subplot
    bars_accuracy = ax2.bar(0, accuracy, width, label="Accuracy", color="dimgrey")
    ax2.bar_label(
        bars_accuracy,
        fmt=lambda x: f"{x:.1f}%",
        padding=3,
        fontsize=14,
        rotation=annotaion_rotation,
    )

    # Add labels, title, and legend for the accuracy subplot
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 100)
    ax2.set_title("Accuracy")
    ax2.set_xticks([0])
    ax2.set_xticklabels([approach], fontsize=14, rotation=70)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.show()
