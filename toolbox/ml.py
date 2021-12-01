import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd

"""Tools for general machine learning projects.
"""


def plot_learning_curves(
    estimator,
    X,
    y,
    figsize=(9, 4),
    train_sizes=np.array([0.1, 0.33, 0.55, 0.78, 1.0]),
    cv=None,
    scoring=None,
):
    """Plot the learning curves for a given sklearn estimator.

    Args
    ----------
    estimator : object type that implements the “fit” and “predict” method
        An object of that type which is cloned for each validation.
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and 
        `n_features` is the number of features.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
    train_sizes : array-like of shape (n_ticks,), \
        default=np.linspace(0.1, 1.0, 5)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
    scoring : str or callable, default=None
        A str or a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    figsize : tuple
        The figsize of the plot, default=(9, 4).
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_scores_mean, label="Training Score", color="black")
    plt.plot(train_sizes, test_scores_mean, label="Test Score", color="blue")
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.title("Learning curves", fontsize=18, y=1.03)
    plt.legend()


def reduce_memory_usage(df):
    """Reduce the memory usage of a DataFrame by downcasting all of its numeric columns.

    Args:
        df (pd.DataFrame): The DataFrame to reduce

    Returns:
        pd.DataFrame: The reduced DataFrame
    """
    reduced_df = pd.DataFrame()
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != "object":
            if str(col_type)[:3] == "int":
                reduced_df[col] = pd.to_numeric(df[col], downcast="integer")
            else:
                reduced_df[col] = pd.to_numeric(df[col], downcast="float")
        else:
            reduced_df[col] = df[col].astype("category")

    return reduced_df


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "integer": [0, 2, 5, 5],
            "float": [5.43, 5.0, 3, 50324.42],
            "cat": ["jean", "edouard", "rud", "joe"],
        }
    )
    for col in df.columns:
        print(df[col].dtype)

    reduced_df = reduce_memory_usage(df)

    for col in reduced_df.columns:
        print(reduced_df[col].dtype)
