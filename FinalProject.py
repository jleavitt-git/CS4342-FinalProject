import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("train.csv")
    data = data.astype({"id":"category","hasYard":"category","hasPool":"category","cityPartRange":"category","isNewBuilt":"category","hasStormProtector":"category","hasStorageRoom":"category"})
    data = data.drop(["id", "cityCode"], axis=1)

    # print(data.columns)

    pca = PCA(n_components=2)

    pcaData = data.to_numpy()[:, :-1]

    pca.fit(pcaData)

    print(pca.explained_variance_ratio_)

    pcaT = pca.transform(pcaData)

    plt.scatter(pcaData[:, 0], pcaData[:, 1])

    plt.show()


if __name__ == "__main__":
    main()