import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

RANDOM_SEED=42


data = pd.read_csv("../../data/interim/01_non_skew_data.csv")
# data = pd.read_csv("../../data/raw/creditcard.csv")

data.head()
data.describe()

# ----------------------------------------------------------------
# visualize the nature of fraud and non-fraud transactions using T-SNE
# ----------------------------------------------------------------

# To keep the computation time low, let's feed t-SNE only a small subsample (undersampling the clean transactions).

# undersample the clean transactions to 2000
clean_undersampled = data[data["Class"] == 0].sample(2000, random_state=RANDOM_SEED)
fraud = data[data["Class"] == 1]

# concatenate with fraud transactions into a single dataframe
visualisation_initial = pd.concat([fraud, clean_undersampled])
column_names = list(visualisation_initial.drop("Class", axis=1).columns)

# isolate features from labels
X, y = (
    visualisation_initial.drop("Class", axis=1).values,
    visualisation_initial.Class.values,
)


# transform the data using t-SNE
from mpl_toolkits.mplot3d import Axes3D


def tsne_scatter(features, labels, dimensions=2, save_as="graph.png"):
    if dimensions not in (2, 3):
        raise ValueError(
            'tsne_scatter can only plot in 2d or 3d (What are you? An alien that can visualise >3d?). Make sure the "dimensions" argument is in (2, 3)'
        )

    # t-SNE dimensionality reduction
    features_embedded = TSNE(
        n_components=dimensions, random_state=RANDOM_SEED
    ).fit_transform(features)

    # initialising the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # counting dimensions
    if dimensions == 3:
        ax = fig.add_subplot(111, projection="3d")

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 1)]),
        marker="o",
        color="r",
        s=2,
        alpha=0.7,
        label="Fraud"
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 0)]),
        marker="o",
        color="g",
        s=2,
        alpha=0.3,
        label="Clean"
    )

    # storing it to be displayed later
    plt.legend(loc="best")
    plt.savefig(save_as)
    plt.show

tsne_scatter(X, y, dimensions=2, save_as="../../reports/figures/tsne_initial_2d.png")

# ? From the above graph we can observe that some clusters are apparent, but a minority of fraud transactions is very close to non_fraud transactions, thus are difficult to accurately classify from a model.
