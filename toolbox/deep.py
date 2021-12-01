import matplotlib.pyplot as plt

"""Tools for general deep learning projects.
"""


def plot_loss_metrics(history, metric="accuracy", title=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(history.history["loss"])
    axs[0].plot(history.history["val_loss"])
    axs[0].set_title("Model loss")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(["Train", "Val"], loc="best")

    axs[1].plot(history.history[metric])
    axs[1].plot(history.history[f"val_{metric}"])
    axs[1].set_title(f"Model {metric}")
    axs[1].set_ylabel(f"{metric}")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(["Train", "Val"], loc="best")
    if title:
        fig.suptitle(title)
    return fig, axs
