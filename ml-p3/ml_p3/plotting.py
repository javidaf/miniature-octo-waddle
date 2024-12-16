import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sns


def visualize_classifications(masks, classifications):

    for classification, label in classifications:
        if classification == 1:
            color = "red"
        elif classification == 2:
            color = "blue"
        else:
            color = "green"

        region = np.where(masks == label)
        region = np.array(region).T
        plt.scatter(region[:, 1], region[:, 0], s=1, c=color)
        # Add label text near the center of each region
        center_y = np.mean(region[:, 0])
        center_x = np.mean(region[:, 1])
        plt.text(center_x, center_y, str(label), fontsize=8, ha="center", va="center")


def plot_eth_data(Y, title):
    Y_flattened = Y.flatten()
    plt.plot(Y_flattened, label=title)
    plt.legend()


def plot_grid_search_heatmap(
    results, score_key, title=None, xlabel="Lookback", ylabel="Hidden Dimension"
):
    """
    Plots a heatmap for the specified score from grid search results.

    Parameters:
    - results (list of dict): Grid search results.
    - score_key (str): The key of the score to visualize (e.g., 'MSE').
    - title (str, optional): Title of the heatmap.
    - xlabel (str, optional): Label for the X-axis.
    - ylabel (str, optional): Label for the Y-axis.
    """
    df = pd.DataFrame(
        [
            {
                "lookback": res["params"]["lookback"],
                "hidden_dim": res["params"]["hidden_dim"],
                score_key: res["val_scores"][score_key],
            }
            for res in results
        ]
    )

    # Pivot the DataFrame for heatmap
    heatmap_data = df.pivot(index="hidden_dim", columns="lookback", values=score_key)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis")
    plt.title(title if title else f"Heatmap of {score_key}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
