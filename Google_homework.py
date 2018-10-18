import pydotplus
import collections

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import random
import time
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from pandas.plotting import scatter_matrix

from statistics import mode
from statistics import StatisticsError

from datetime import datetime

"""
TODO:
    * Make the plot dataset function more specific (names on axes, different colors depending on values) might be easier to do in excel??
    * Add the confusion matrix + recall validation to every k-fold and sum it up to calculate the final precision/recall/accuracy
    * Decide what data to experiment with in the google play dataset
    * Comment the code
    * Write the report
"""


def read_fileG(file):
    """
    Reads the file containing the dataset using pandas
    :param file: path to file
    :param features_list: a list of the names of the parameters to be used
    :param target: the name of the target
    :return: A pandas dataset, features and target
    """
    #names_list = features_list.copy()
    #names_list.append(target)
    data = pd.read_csv(file)
    return data

def read_fileI(file, features_list, target):
    """
    Reads the file containing the dataset using pandas
    :param file: path to file
    :param features_list: a list of the names of the parameters to be used
    :param target: the name of the target
    :return: A pandas dataset, features and target
    """
    names_list = features_list.copy()
    names_list.append(target)
    data = pd.read_csv(file, names=names_list)
    return data

def plot_dataset(dataset):
    """
    Shows a scatter matrix, a histogram and a plot.
    :param dataset: A dataset
    """
    colors = ['red', 'green', 'yellow']
    print(dataset.shape)
    print(dataset.head(20))
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('class').size())
    scatter_matrix(dataset)
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    dataset.hist()
    plt.show()


def plot_tree(clf, features, target_names, file_out):
    """
    Visualizes a desicion tree, saves it as a PNG
    :param clf: a desicion tree classifier
    :param features: the chosen features
    :param target_names: the different classes in the target list
    :param file_out: File name of the out file
    :return: Saves a PNG
    """

    tree_data = tree.export_graphviz(clf, feature_names=features,
                                     class_names=target_names, out_file=None, filled=True, rounded=True)

    graph = pydotplus.graph_from_dot_data(tree_data)

    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    colors = ('turquoise', 'orange')
    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png(file_out)


def split_data(features, target, test_size=0.3, random_state=None):
    """
    Splits a dataset into training and testing data
    :param features: the features of the dataset
    :param target: the target of the dataset
    :param test_size: How much of the dataset that should be for testing
    :param random_state: What random state should be used. If None, a different state is generated every time.
    :return: four variables containing datasets for training and testing
    """

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def decision_tree(X_train, y_train):
    """
    Creates a decision tree classifier
    :param X_train: features from the testing data
    :param y_train: targets from the testing data
    :return: a decision tree classifier
    """
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf


def clf_accuracy(clf, X_test, y_test):
    """
    Calculates the accuracy score
    :param clf: Classifier
    :param X_test: Features testing data
    :param y_test: Target testing data
    :return: the value of the accuracy
    """
    pred = clf.predict(X_test)
    acc = accuracy_score(pred, y_test)
    return acc


def validation(true, pred, model):
    """
    Prints validation scores based on a prediction
    :param true: The true answers to the testing data
    :param pred: The predicted answers to the testing data
    :param model: What model is used, to clarify when printing
    """
    accuracy = accuracy_score(true, pred)
    print("Accuracy for {}: {}".format(model, accuracy))
    matrix = confusion_matrix(true, pred)
    print("Confusion matrix for {}:\n {}".format(model, matrix))
    prec_score = metrics.precision_score(true, pred, average=None)
    print("Precision score for {}: {}".format(model, prec_score))
    r_score = metrics.recall_score(true, pred, average=None)
    print("Recall score for {}: {}".format(model, r_score))


def k_fold_dtree(features, target, n_splits):
    """
    Takes a dataset, divides it with the Kfold method and creates a decision tree on every set of data
    :param features: the features of the dataset
    :param target: the target of the dataset
    :param n_splits: number of splits
    :return: a list of the accuracies
    """
    acc_list = []
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, test_index in kf.split(features):
        itemindex = np.where(train_index == 10472)
        train_index = np.delete(train_index, itemindex)

        itemindex = np.where(test_index == 10472)
        test_index = np.delete(test_index, itemindex)

        X_train, X_test = features.loc[train_index], features.loc[test_index]
        y_train, y_test = target[train_index], target[test_index]

        '''print("<><><><><>")
        i = 0
        for a in y_train.values:
            print(train_index[i])
            print(a)
            print(type(a))
            i += 1
        print("<><><><><>")'''

        clf = decision_tree(X_train, y_train)
        acc = clf_accuracy(clf, X_test, y_test)
        acc_list.append(acc)
    return acc_list


def random_forest(features, target, X_test, set_size, no_of_trees):
    """
    Random forest model, creates trees based on random pieces of the dataset
    :param dataset: a dataset
    :param X_test: features testing values
    :param set_size: Size of the set used to create the tree
    :param no_of_trees: the number of trees the forest should contain
    :return: a list of predictions
    """
    tree_list = []
    all_pred = []
    predictions = []

    for n in range(no_of_trees):
        random_numbers = []
        for x in range(0, int(len(target)*set_size)):
            number = random.randint(0, len(target) - 1)
            random_numbers.append(number if number != 10472 else number + 1)
        #  Randomly select data
        features_sample = features.loc[random_numbers]
        target_sample = target[random_numbers]
        #  Make a tree with this data
        tree_list.append(decision_tree(features_sample, target_sample))
        #  Repeat for n trees

    for tree in tree_list:
        all_pred.append(tree.predict(X_test))

    for i in range(len(all_pred[0])):
        pred_list = []
        for pred in all_pred:
            pred_list.append(pred[i])
        try:
            predictions.append(mode(pred_list))
        except StatisticsError:
            predictions.append(pred_list[0])

    return predictions


multipliers = {'k': 10e2, 'M': 10e5}
millions_converter = lambda x: int(float(x[:-1])*multipliers[x[-1]]) #not sure abour 10e5, not 10e6
plus_remover = lambda x: int(float(x[:-1].replace(',', '')))
type_converter = lambda x: bool(0 if x == 'Free' or x == 0 else 1)
price_converter = lambda x: float(x if type(x) == int else x.replace('$', ''))
date_converter = lambda x: int(time.mktime(datetime.strptime(x, '%B %d, %Y').timetuple()))


def convert_columns_to_int(value):
    if type(value) == int:
        value = value
    elif "M" in value or "k" in value:
        value = millions_converter(value)
    elif "Varies with device" in value:
        value = 0
    elif "+" in value:
        value = plus_remover(value)
    else:
        value = int(value)
    return value

def convert_types_to_int(value, types):
    return types.index(value)

def prepare_features(dataset, feature_names):
    dataset[dataset.keys()[1]] = dataset[dataset.keys()[1]].replace(np.NaN, -1)
    data_list = dataset.values.tolist()
    #data_list.remove(data_list[10472])
    #["Category", "Rating", "Reviews", "Size", "Price", "Content Rating", "Genres", "Last Updated"]
    categories = list(set([row[0] for row in data_list]))
    content_ratings = list(set([row[5] for row in data_list]))
    genres = list(set([row[6] for row in data_list]))

    for row in data_list:
        row[0] = convert_types_to_int(row[0], categories)
        row[2] = convert_columns_to_int(row[2])
        row[3] = convert_columns_to_int(row[3])
        row[4] = price_converter(row[4])
        row[5] = convert_types_to_int(row[5], content_ratings)
        row[6] = convert_types_to_int(row[6], genres)
        row[7] = date_converter(row[7])

    return pd.DataFrame(data_list, columns=feature_names)


def main_iris():
    address = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    target_name = "class"

    dataset = read_fileI(address, feature_names, target_name)
    feature_data = dataset[feature_names]
    target_data = dataset[target_name]

    #TODO refactor names
    iris = dataset
    iris_features = feature_data
    iris_target = target_data

    #  Simple data split
    X_train, X_test, y_train, y_test = split_data(iris_features, iris_target)

    #  Visualisation of data
    plot_dataset(iris)
    plot_tree(decision_tree(X_train, y_train), feature_names, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
              'treefile.png')

    #  Validation of a decision tree.
    #  Methods used: accuracy score, confusion matrix, precision score, recall score and k-fold
    clf = decision_tree(X_train, y_train)
    validation(y_test, clf.predict(X_test), "Decision tree")
    validation(y_train, clf.predict(X_train), "Decision tree resubstitution")
    k_fold = k_fold_dtree(iris_features, iris_target, 5)
    print("K-fold validation:", np.mean(k_fold))

    #  Validation of a random decision tree forest.
    #  Methods used: accuracy score, confusion matrix, precision score and recall score
    forest_pred = random_forest(iris_features, iris_target, X_test, 0.1, 4)
    validation(y_test, forest_pred, "Random forest")


def main_google_play():
    address = 'google-play-store-apps\googleplaystore1.csv'
    feature_names = ["Category", "Rating", "Reviews", "Size", "Price", "Content Rating", "Genres", "Last Updated"]
    target_name = "Installs"

    dataset = read_fileG(address).drop(10472)
    dataset.to_csv("test.csv", sep='\t')
    print(dataset.values[10838])
    feature_data = prepare_features(dataset[feature_names], feature_names)
    target_data = dataset[target_name]#.drop(10472)

    #  Simple data split
    X_train, X_test, y_train, y_test = split_data(feature_data, target_data)

    #  Visualisation of data
    #plot_dataset(prepared_dataset)
    #plot_tree(prepared_dataset, decision_tree(X_train, y_train), ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    #              'gplayfile.png')

    #  Validation of a decision tree.
    #  Methods used: accuracy score, confusion matrix, precision score, recall score and k-fold

    clf = decision_tree(X_train, y_train)
    validation(y_test, clf.predict(X_test), "Decision tree")
    validation(y_train, clf.predict(X_train), "Decision tree resubstitution")

    k_fold = k_fold_dtree(feature_data, target_data, 5)
    print("K-fold validation:", np.mean(k_fold))

    #  Validation of a random decision tree forest.
    #  Methods used: accuracy score, confusion matrix, precision score and recall score
    forest_pred = random_forest(feature_data, target_data, X_test, 4, 5)
    validation(y_test, forest_pred, "Random forest")

if __name__ == '__main__':
    main_iris()
    #main_google_play()
