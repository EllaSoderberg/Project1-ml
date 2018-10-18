import pydotplus
import collections

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

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
    * Make read file return desired features and target
    * Make the plot dataset function more specific (names on axes, different colors depending on values) might be easier to do in excel??
    * Change the dataset parameter in most functions to features and target parameters, to make the code flexible
    * Add the confusion matrix + recall validation to every k-fold and sum it up to calculate the final precision/recall/accuracy
    * Decide what data to experiment with in the google play dataset
    * Comment the code
    * Write the report
"""

def read_file(file):
    """
    Reads the file containing the dataset using pandas
    :param file: path to file
    :param names: names of features
    :return: A pandas dataset
    """
    data = pd.read_csv(file)
    print(data.head())
    return data


def plot_dataset(dataset):
    """
    Shows a scatter matrix, a histogram and a plot.
    :param dataset: A dataset
    """
    colors = ['red', 'green', 'yellow']
    scatter_matrix(dataset)
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    dataset.hist()
    plt.show()


def plot_tree(dataset, classifier, target_names, file_out):
    """
    Visualizes a desicion tree, saves it as a PNG
    :param dataset: a dataset
    :param classifier: a classifier
    :param target_names: the names of the targets
    :param file_out: File name of the out file
    :return: Saves a PNG
    """

    colors = ('turquoise', 'orange')

    tree_data = tree.export_graphviz(classifier, feature_names=dataset.keys()[:-1],
                                     class_names=target_names, out_file=None, filled=True, rounded=True)

    graph = pydotplus.graph_from_dot_data(tree_data)

    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png(file_out)


def split_features_targets(dataset):
    features = dataset[dataset.keys()[:-1]]
    target = dataset[dataset.keys()[-1]]
    return features, target


def split_data(dataset, test_size=0.3, random_state=None):
    """
    Splits a dataset into training and testing data
    :param dataset: A dataset
    :param test_size: How much of the dataset that should be for testing
    :param random_state: What random state should be used. If None, a different state is generated every time.
    :return: four variables containing datasets for training and testing
    """
    features, target = split_features_targets(dataset)

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


def k_fold_dtree(dataset, n_splits):
    """
    Takes a dataset, divides it with the Kfold method and creates a decision tree on every set of data
    :param dataset: a dataset
    :param n_splits: number of splits
    :return: a list of the accuracies
    """
    acc_list = []
    features, targets = split_features_targets(dataset)
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, test_index in kf.split(features):
        X_train, X_test = features.loc[train_index], features.loc[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        clf = decision_tree(X_train, y_train)
        acc = clf_accuracy(clf, X_test, y_test)
        acc_list.append(acc)
    return acc_list


def random_forest(dataset, X_test, no_of_trees):
    """
    Random forest model, creates trees based on random pieces of the dataset
    :param dataset: a dataset
    :param X_test: features testing values
    :param no_of_trees: the number of trees the forest should contain
    :return: a list of predictions
    """
    tree_list = []
    all_pred = []
    predictions = []

    for n in range(no_of_trees):
        #  Randomly select data
        data_fraction = dataset.sample(frac=0.2)
        features, targets = split_features_targets(data_fraction)
        #  Make a tree with this data
        tree_list.append(decision_tree(features, targets))
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
date_converter = lambda x: datetime.strptime(x, '%B %d, %Y')


def convert_columns_to_int(value):
    if "M" in value or "k" in value:
        value = millions_converter(value)
    elif "Varies with device" in value:
        value = None
    elif "+" in value:
        value = plus_remover(value)
    else:
        value = int(value)
    return value


def convert_types_to_int(value, types):
    return types.index(value)


def prepare_dataset(dataset):
    data_list = dataset.values.tolist()

    data_list.remove(data_list[10472])
    categories = list(set([row[1] for row in data_list]))
    content_ratings = list(set([row[8] for row in data_list]))
    genres = list(set([row[9] for row in data_list]))

    for row in data_list:
        row[1] = convert_types_to_int(row[1], categories)
        row[3] = convert_columns_to_int(row[3])
        row[4] = convert_columns_to_int(row[4])
        row[5] = convert_columns_to_int(row[5])
        row[7] = price_converter(row[7])
        row[8] = convert_types_to_int(row[8], content_ratings)
        row[9] = convert_types_to_int(row[9], genres)
        row[10] = date_converter(row[10])

    #test = [row[12] for row in data_list]
    #print(set(test))

    prepared_data = [row[1:6] + row[7:11] for row in data_list]

    return prepared_data


if __name__ == '__main__':
    address = 'google-play-store-apps\googleplaystore.csv'  # ".../MLHW1/google-play-store-apps/googleplaystore.csv"
    dataset = read_file(address)
    prepared_dataset = prepare_dataset(dataset)
    print(prepared_dataset)

    #  Simple data split
    X_train, X_test, y_train, y_test = split_data(prepared_dataset)

    #  Visualisation of data
    plot_dataset(prepared_dataset)
    #plot_tree(prepared_dataset, decision_tree(X_train, y_train), ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    #              'gplayfile.png')'''

    #  Validation of a decision tree.
    #  Methods used: accuracy score, confusion matrix, precision score, recall score and k-fold
    clf = decision_tree(X_train, y_train)
    validation(y_test, clf.predict(X_test), "Decision tree")
    validation(y_train, clf.predict(X_train), "Decision tree resubstitution")
    k_fold = k_fold_dtree(dataset, 5)
    print("K-fold validation:", np.mean(k_fold))

    #  Validation of a random decision tree forest.
    #  Methods used: accuracy score, confusion matrix, precision score and recall score
    forest_pred = random_forest(dataset, X_test, 4)
    validation(y_test, forest_pred, "Random forest")
