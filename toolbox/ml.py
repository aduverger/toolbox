import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch

"""Tools for general machine learning projects.
"""


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    scoring=None,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    scoring : str, default=None
        A str (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def plot_for_scaling(df, feature, figsize=(7, 3)):
    """Plot the distribution and the boxplot of a given column from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame from which you want to plot \
                           the distribution and boxplot.
        feature (str): The name of the column
        figsize (tuple, optional): Defaults to (7, 3).
    """
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(1, 4)
    ax = fig.add_subplot(gs[0, 0:3])
    sns.histplot(x=df[feature].dropna(), kde=True, ax=ax)
    ax1 = fig.add_subplot(gs[0, 3:])
    sns.boxplot(y=df[feature].dropna(), ax=ax1)
    sns.despine()
    plt.show()


def investigate_corr(df):
    """Plot a heatmap of the collinearity between columns of a DataFrame.
    Also return a DataFrame with the sorted collinearity.
    """
    corr = df.corr().fillna(0)
    corr_df = corr.unstack().reset_index()  # Unstack correlation matrix
    corr_df.columns = ["feature_1", "feature_2", "correlation"]  # rename columns
    corr_df.sort_values(
        by="correlation", ascending=False, inplace=True
    )  # sort by correlation
    corr_df = corr_df[
        corr_df["feature_1"] != corr_df["feature_2"]
    ]  # Remove self correlation

    # Remove duplicates
    duplicates_indexes = []
    for index in range(corr_df.shape[0] - 1):
        if (
            corr_df.iloc[index, 0] == corr_df.iloc[index + 1, 1]
            and corr_df.iloc[index, 1] == corr_df.iloc[index + 1, 0]
        ):
            duplicates_indexes.append(corr_df.iloc[index, :].name)
    corr_df.drop(index=duplicates_indexes, inplace=True)

    # Draw the reordered correlation heatmap
    plot_corr_heatmap(df)

    return corr_df


def plot_corr_heatmap(df):
    """Plot a heatmap of a correlation matrix reordered by similarity between \
        the variables, so that the "groups" of variables that are strongly correlated \
        appear close in the heatmap.
    """
    corr = df.corr().fillna(0)
    # Generate features and distance matrix.
    D = corr.values
    # Compute and plot dendrogram.
    Y = sch.linkage(D, method="centroid")
    Z = sch.dendrogram(Y, orientation="right", no_plot=True)
    # Compute distance matrix.
    index = Z["leaves"]
    D = D[index, :]
    D = D[:, index]

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    size = max(10, len(corr.columns) / 2.0)
    plt.figure(figsize=(size, size))

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(
        D, mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
    )
    ax.set_xticklabels(corr.columns[index], rotation=90, ha="center")
    ax.set_yticklabels(corr.columns[index], rotation=0)


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
