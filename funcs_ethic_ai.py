"""
Module containing functions for data preprocessing, model training, and evaluation.

Functions:
- clean_dataset(df_local: pd.DataFrame) -> pd.DataFrame: Clean the given DataFrame by removing duplicates, empty values, and rows with '?'.
- delete_rows_with_question_mark(df_local: pd.DataFrame) -> pd.DataFrame: Delete rows containing '?' in any cell of the DataFrame.
- get_race_category_list(race_list: list) -> list: Assign race categories based on the provided list of races.
- strip_strings(df_local: pd.DataFrame) -> pd.DataFrame: Strip leading and trailing whitespace from string values in the DataFrame.
- get_X_Y_from_df(df_local: pd.DataFrame) -> (csr_matrix, np.ndarray): Extract features (X) and target variable (Y) from the DataFrame.
- get_TPR_FPR_per_group(df_truth_pred_local: pd.DataFrame, feature_of_interest: str, bool_val: bool) -> (float, float, float): Calculate True Positive Rate (TPR), False Positive Rate (FPR), and Positive Rate (PR) per group.
- get_truth_pred_dataframe(df_local: pd.DataFrame, df_local_test: pd.DataFrame) -> pd.DataFrame: Process the data for machine learning, train a model, and get predictions.
- create_cat_n_num_col_list: Create lists of numerical and categorical column names from a pandas DataFrame.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import csr_matrix

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "gain",
]


def clean_dataset(df_local: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the given DataFrame by performing the following steps:
    1. Strip strings in the DataFrame.
    2. Remove duplicate rows.
    3. Remove rows with empty values.
    4. Delete rows containing '?' as a value in any column.
    5. Set column names to predefined COLUMN_NAMES.
    6. Insert a new column 'is_white' based on the 'race' column.

    Parameters:
        df_local (pd.DataFrame): The DataFrame to be cleaned.

    Returns:
        pd.DataFrame: A cleaned DataFrame.
    """
    df_local = strip_strings(df_local)
    initial_entries = len(df_local)
    df_local.drop_duplicates(inplace=True)
    deduplicated_entries = len(df_local)
    duplicated_entries = initial_entries - deduplicated_entries
    print(
        "Deleted duplicated rows: {}, {}%".format(
            duplicated_entries, round(100 * duplicated_entries / initial_entries, 0)
        )
    )
    df_local.dropna(inplace=True)
    dedup_not_na_entries = len(df_local)
    notna_entries = deduplicated_entries - dedup_not_na_entries
    print(
        "Deleted rows with empty values: {}, {}%".format(
            notna_entries,
            round(
                100 * notna_entries / initial_entries,
            ),
        )
    )
    df_cleaned = delete_rows_with_question_mark(df_local)
    cleaned_entries = len(df_cleaned)
    question_mark_entries = dedup_not_na_entries - cleaned_entries
    print(
        "Number of entries in a dataset after cleaning: {}, {}% from initial dataset were dismissed".format(
            cleaned_entries,
            round(100 * (initial_entries - cleaned_entries) / initial_entries, 0),
        )
    )
    df_cleaned.columns = COLUMN_NAMES
    race_list = list(df_cleaned["race"])
    df_cleaned.insert(
        loc=len(df_cleaned.columns),
        column="is_white",
        value=get_race_category_list(race_list),
    )

    return df_cleaned


def delete_rows_with_question_mark(df_local: pd.DataFrame) -> pd.DataFrame:
    """
    Delete rows containing '?' in any cell of the DataFrame.

    Parameters:
        df_local (pd.DataFrame): The DataFrame to be cleaned.

    Returns:
        pd.DataFrame: DataFrame with rows containing '?' removed.
    """

    df_local.copy(deep=True)
    # Create a boolean mask for rows containing "?"
    mask = df_local.apply(lambda row: any(["?" in str(cell) for cell in row]), axis=1)

    # Invert the mask to keep rows not containing "?"
    inverted_mask = ~mask

    # Filter the DataFrame to keep only rows not containing "?"
    df_local = df_local[inverted_mask]

    return df_local


def get_race_category_list(race_list: list) -> list:
    """
    Assign race categories based on the provided list of races.

    Parameters:
        race_list (list): A list containing race values.

    Returns:
        list: A list of race categories, where each race is categorized as either 'white' or 'non-white'.
    """

    # Define a list of values that are considered as "White"
    white_values = ["White"]

    race_category = []
    for race in race_list:
        if race.strip() in white_values:
            race_category.append("white")
        else:
            race_category.append("non-white")
    return race_category


def strip_strings(df_local: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading and trailing whitespace from string values in the DataFrame.

    Parameters:
        df_local (pd.DataFrame): The DataFrame to be processed.

    Returns:
        pd.DataFrame: DataFrame with leading and trailing whitespace stripped from string values.
    """
    # Iterate over all columns
    for col in df_local.columns:
        # Check if the column contains string values
        if df_local[col].dtype == "object":
            # Strip string values
            df_local[col] = df_local[col].str.strip()
    return df_local


def get_X_Y_from_df(df_local: pd.DataFrame) -> (csr_matrix, np.ndarray):
    """
    Extract features (X) and target variable (Y) from the DataFrame.

    Parameters:
        df_local (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        tuple: A tuple containing:
            - X (csr_matrix): Matrix of features.
            - Y (np.ndarray): Target variable.
    """

    # Get X  - vector of features
    X = df_local.drop(
        columns=["gain", "education", "fnlwgt", "race"]
    )  # column "education" is categorical renaming of "education-num"
    # get lists of categorical and numerical features for X
    numerical_columns, categorical_columns = create_cat_n_num_col_list(X)

    # Perform one-hot-encoding on categorical variables
    df_onehot = pd.get_dummies(X[categorical_columns], drop_first=True)
    y_raw = pd.get_dummies(df_local["gain"])
    if ">50K." in y_raw.columns:
        y_raw = y_raw.rename(columns={">50K.": ">50K", "<=50K.": "<=50K"})
    y = y_raw.loc[:, ">50K"]

    # Fit and transform the numerical columns using RobustScaler
    scaler = RobustScaler()
    scaled_numerical = scaler.fit_transform(df_local[numerical_columns])
    scaled_df = pd.DataFrame(scaled_numerical, columns=numerical_columns, index=X.index)

    df_ready_for_ML = pd.concat([scaled_df, df_onehot], axis=1)

    return (df_ready_for_ML, y)


def get_TPR_FPR_per_group(
    df_truth_pred_local: pd.DataFrame, feature_of_interest: str, bool_val: bool
) -> (float, float, float):
    """
    Calculate True Positive Rate (TPR), False Positive Rate (FPR), and Positive Rate (PR) per group.

    Parameters:
        df_truth_pred_local (pd.DataFrame): DataFrame containing truth and prediction values.
        feature_of_interest (str): The feature of interest to split the dataset.
        bool_val (bool): The value of the feature of interest to filter the dataset.

    Returns:
        tuple: A tuple containing:
            - TPR (float): True Positive Rate.
            - FPR (float): False Positive Rate.
            - PR (float): Positive Rate.
    """
    y_truth = df_truth_pred_local[df_truth_pred_local[feature_of_interest] == bool_val][
        ">50K"
    ]
    y_pred = df_truth_pred_local[df_truth_pred_local[feature_of_interest] == bool_val][
        ">50K_pred"
    ]

    TP = confusion_matrix(y_truth, y_pred)[1, 1]
    FP = confusion_matrix(y_truth, y_pred)[0, 1]
    FN = confusion_matrix(y_truth, y_pred)[1, 0]
    TN = confusion_matrix(y_truth, y_pred)[1, 1]

    PR = sum(y_pred) / len(y_pred)
    TPR = TP / (TP + FN)
    FNR = FN / (FN + TP)

    return (100 * PR, 100 * TPR, 100 * FNR)


def get_truth_pred_dataframe(
    df_local: pd.DataFrame, df_local_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Process the data for machine learning, train a model, and get predictions.

    Parameters:
        df_local (pd.DataFrame): The DataFrame containing the training dataset.
        df_local_test (pd.DataFrame): The DataFrame containing the test dataset.

    Returns:
        pd.DataFrame: DataFrame containing truth and prediction values.
    """
    # process the data for ML
    X_processed, y = get_X_Y_from_df(df_local)
    X_test_processed, y_test = get_X_Y_from_df(df_local_test)

    # define the model
    model = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=13,
    )

    # train the model
    model.fit(X_processed, np.ravel(y))

    # get predictions using the previously trained model
    y_test_predictions = model.predict(X_test_processed)

    # concatinate dataframes - X, truth, predictions
    df_truth_pred_local = pd.concat(
        [
            y_test,
            pd.Series(
                y_test_predictions, index=y_test.index, name=y_test.name + "_pred"
            ),
        ],
        axis=1,
    )

    return df_truth_pred_local


def create_cat_n_num_col_list(df_local: pd.DataFrame) -> (list, list):
    """
    Create lists of numerical and categorical column names from a pandas DataFrame.

    Parameters:
    - df_local (DataFrame): The pandas DataFrame containing the data.

    Returns:
    - (tuple): A tuple containing two lists:
        - numerical_columns (list): A list of column names with numerical data types.
        - categorical_columns (list): A list of column names with categorical data types.

    This function iterates over the columns of the input DataFrame and classifies them into
    numerical or categorical based on their data types. It returns two lists containing the
    names of numerical and categorical columns, respectively.

    Example:
    >>> numerical_cols, categorical_cols = create_cat_n_num_col_list(df)
    """
    numerical_columns = []
    categorical_columns = []
    for col in df_local.columns:
        if df_local[col].dtype == "object":
            categorical_columns.append(col)
        elif df_local[col].dtype == "int64":
            numerical_columns.append(col)
    return (numerical_columns, categorical_columns)
