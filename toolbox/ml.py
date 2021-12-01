import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""Tools for general machine learning projects.
"""


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
