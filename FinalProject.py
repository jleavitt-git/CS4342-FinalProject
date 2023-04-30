import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def main():
    data = pd.read_csv("train.csv")
    data = data.astype({"id":"category","hasYard":"category","hasPool":"category","cityPartRange":"category","isNewBuilt":"category","hasStormProtector":"category","hasStorageRoom":"category"})
    data = data.drop(["id", "cityCode"], axis=1)



if __name__ == "__main__":
    main()