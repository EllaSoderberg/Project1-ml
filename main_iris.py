import dataset_functions as df
import plot_dataset as plot

import numpy as np  # linear algebra

from sklearn.model_selection import train_test_split


def main_iris():
    address = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    target_name = "class"

    iris_dataset, iris_features, iris_target = df.read_file(address, feature_names, target_name)

    #  Simple data split
    X_train, X_test, y_train, y_test = train_test_split(iris_features, iris_target)

    #  Visualisation of data
    # plot.plot_dataset(iris_dataset, target_name, 2)
    # plot.plot_tree(df.decision_tree(X_train, y_train), feature_names, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    #          'treefile.png')

    #  Validation of a decision tree.
    #  Methods used: accuracy score, confusion matrix, precision score, recall score and k-fold
    clf = df.decision_tree(X_train, y_train)
    df.validation(y_test, clf.predict(X_test), "Decision tree")
    df.validation(y_train, clf.predict(X_train), "Decision tree resubstitution")
    k_fold = df.k_fold_dtree(iris_features, iris_target, 5)
    print("K-fold validation:", np.mean(k_fold))

    #  Validation of a random decision tree forest.
    #  Methods used: accuracy score, confusion matrix, precision score and recall score
    forest_pred = df.random_forest(iris_features, iris_target, X_test, 0.1, 4)
    df.validation(y_test, forest_pred, "Random forest")


if __name__ == '__main__':
    main_iris()
