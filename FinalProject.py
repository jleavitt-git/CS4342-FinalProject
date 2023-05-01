import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    data = pd.read_csv("train.csv")
    data = data.drop(["id"], axis=1)

    data.insert(len(data.columns)-1, 'winter', data['arrival_month'].apply(lambda x: 1 if (x == 1) or (x == 2) or (x == 12) else 0))
    data.insert(len(data.columns)-1, 'spring', data['arrival_month'].apply(lambda x: 1 if (x == 3) or (x == 4) or (x == 5) else 0))
    data.insert(len(data.columns)-1, 'summer', data['arrival_month'].apply(lambda x: 1 if (x == 6) or (x == 7) or (x == 8) else 0))
    data.insert(len(data.columns)-1, '2017', data['arrival_year'].apply(lambda x: 1 if (x == 2017) else 0))
    data = data.drop(['arrival_month'], axis = 1)
    data = data.drop(['arrival_year'], axis = 1)

    # print(data.columns)

    pca = PCA(n_components=2)

    pcaData = data.to_numpy()[:, :-1]

    pca.fit(pcaData)

    print(pca.explained_variance_ratio_)

    pcaT = pca.transform(pcaData)

    # visualize(pcaT, data.booking_status)

    X_train, X_test, y_train, y_test = train_test_split(pcaData, data.booking_status, test_size=0.2, shuffle=True)
    # print(f"{X_train.shape} : {X_test.shape} : {y_train.shape} : {y_test.shape}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    shallow_model(X_train, X_test, y_train, y_test)

    # data.plot(kind ='box',subplots = True,sharex= False,sharey=False,figsize=(15,15))
    # plt.show()


def shallow_model(X_train, X_test, y_train, y_test):

    clf = SVC(kernel='rbf')

    clf.fit(X_train, y_train)

    print(accuracy_score(y_train, clf.predict(X_train)))

    predictions = clf.predict(X_test)

    ac = accuracy_score(y_test, predictions)
    print(ac)


def visualize(pcaT, data):

    x = pcaT[:, 0]
    y = pcaT[:, 1]

    plt.scatter(x, y, c=data, cmap='coolwarm', alpha=0.5, s=4)

    # #calculate equation for trendline
    # z = np.polyfit(x, y, 1)
    # p = np.poly1d(z)

    # plt.plot(x, p(x))

    plt.show()


if __name__ == "__main__":
    main()