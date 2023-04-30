import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    data = pd.read_csv("train.csv")
    data = data.drop(["id"], axis=1)

    # print(data.columns)

    pca = PCA(n_components=2)

    pcaData = data.to_numpy()[:, :-1]

    pca.fit(pcaData)

    print(pca.explained_variance_ratio_)

    pcaT = pca.transform(pcaData)

    plt.scatter(pcaT[:, 0], pcaT[:, 1], c=data.booking_status, cmap='coolwarm', alpha=0.5, s=4)

    plt.show()

    # data.plot(kind ='box',subplots = True,sharex= False,sharey=False,figsize=(15,15))
    # plt.show()


if __name__ == "__main__":
    main()