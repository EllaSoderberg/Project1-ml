import dataset_functions as df
import plot_dataset as plot

import clean_google_dataset as clean

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split


def main_google_play():
    address = 'google-play-store-apps\googleplaystore.csv'
    feature_names = ["Category", "Rating", "Reviews", "Size", "Price", "Content Rating", "Genres", "Last Updated"]
    target_name = "Installs"

    dataset = pd.read_csv(address).drop(10472)
    dataset.to_csv("test.csv", sep='\t')
    print(dataset.values[10838])
    feature_data = clean.prepare_features(dataset[feature_names], feature_names)
    target_data = dataset[target_name]

    #  Simple data split
    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data)

    #  Visualisation of data
    #plot.plot_dataset(pd.concat([feature_data, target_data], axis=1, join='inner'), target_name, 3)
    #plot.plot_dataset.plot_tree(prepared_dataset, df.decision_tree(X_train, y_train), ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    #              'gplayfile.png')

    #  Validation of a decision tree.
    #  Methods used: accuracy score, confusion matrix, precision score, recall score and k-fold

    clf = df.decision_tree(X_train, y_train)
    df.validation(y_test, clf.predict(X_test), "Decision tree")
    df.validation(y_train, clf.predict(X_train), "Decision tree resubstitution")

    #k_fold = df.k_fold_dtree(feature_data, target_data, 5)
    #print("K-fold validation:", np.mean(k_fold))

    #  Validation of a random decision tree forest.
    #  Methods used: accuracy score, confusion matrix, precision score and recall score
    forest_pred = df.random_forest(feature_data, target_data, X_test, 4, 5)
    df.validation(y_test, forest_pred, "Random forest")

if __name__ == '__main__':
    main_google_play()