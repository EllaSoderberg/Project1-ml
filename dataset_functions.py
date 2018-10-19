import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics as metrics
import random

from sklearn import tree
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from statistics import mode
from statistics import StatisticsError

"""
TODO:
    * Make the plot dataset function more specific (names on axes, different colors depending on values) might be easier to do in excel??
    * Fix the final precision/recall/accuracy on random forest model
    * Decide what data to experiment with in the google play dataset
    * Comment the code
    * Write the report
    * k-fold on the random tree
"""


def read_file(file, features_list, target):
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
    features = data[features_list]
    target = data[target]
    return data, features, target


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
    prec_score = metrics.precision_score(true, pred, average='macro')
    print("Precision score for {}: {}".format(model, prec_score))
    r_score = metrics.recall_score(true, pred, average='macro')
    print("Recall score for {}: {}".format(model, r_score))

'''
def k_fold_dtree(features, target, n_splits):
    """
    Takes a dataset, divides it with the Kfold method and creates a decision tree on every set of data
    :param features: the features of the dataset
    :param target: the target of the dataset
    :param n_splits: number of splits
    :return: a list of the accuracies
    """
    acc_list = []
    different_values = len(set(target))
    c_matrix = np.zeros((different_values, different_values))
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, test_index in kf.split(features):
        X_train, X_test = features.loc[train_index], features.loc[test_index]
        y_train, y_test = target[train_index], target[test_index]

        clf = decision_tree(X_train, y_train)
        pred = clf.predict(X_test)
        c_matrix += confusion_matrix(y_test, pred)
        acc = accuracy_score(pred, y_test)
        acc_list.append(acc)
    print(c_matrix / n_splits)
    return acc_list
'''


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
    different_values = len(set(target))
    c_matrix = np.zeros((different_values, different_values))

    for train_index, test_index in kf.split(features):
        itemindex = np.where(train_index == 10472)
        train_index = np.delete(train_index, itemindex)

        itemindex = np.where(test_index == 10472)
        test_index = np.delete(test_index, itemindex)

        X_train, X_test = features.loc[train_index], features.loc[test_index]
        y_train, y_test = target[train_index], target[test_index]

        """print("<><><><><>")
        i = 0
        for a in y_train.values:
            print(train_index[i])
            print(a)
            print(type(a))
            i += 1
        print("<><><><><>")"""

        clf = decision_tree(X_train, y_train)
        pred = clf.predict(X_test)
        c_matrix += confusion_matrix(y_test, pred)
        acc = accuracy_score(pred, y_test)
        acc_list.append(acc)
    print(c_matrix/n_splits)
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


def k_fold_random_forest(features, target, n_splits):
    """
    Takes a dataset, divides it with the Kfold method and creates a decision tree on every set of data
    :param features: the features of the dataset
    :param target: the target of the dataset
    :param n_splits: number of splits
    :return: a list of the accuracies
    """
    acc_list = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    matrix = []
    prec_score = []
    r_score = []

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

     #   print(X_train.isna().sum())
     #   print(y_train.isna().sum())

        rforest_pred = random_forest(X_train, y_train, X_test, 4, 5)
        acc = accuracy_score(rforest_pred, y_test)

       # clf = decision_tree(X_train, y_train)
       # acc = clf_accuracy(clf, X_test, y_test)
        acc_list.append(acc)

        matrix.append(confusion_matrix(y_test,rforest_pred))
        prec_score.append(metrics.precision_score(y_test, rforest_pred, average=None))
        r_score.append(metrics.recall_score(y_test, rforest_pred, average=None))

    #for i in matrix:
    #    matrix[0] = np.add(matrix[0],i)

    #print("Confusion matrix for kfold validation ", matrix[0])

   # print("Precision score for k-fold validation ", np.mean(prec_score))

    #print("Recall score for k-fold validation ", np.mean(r_score))
    return acc_list







