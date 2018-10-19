import pydotplus
import collections
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import tree


def plot_dataset(dataset, target, layout_size):
    """
    Shows a scatter matrix, a histogram and a plot.
    :param dataset: A dataset
    """
    print(dataset.shape)
    print(dataset.head(20))
    print(dataset.describe())

    # class distribution
    print(dataset.groupby(target).size())
    scatter_matrix(dataset)
    dataset.plot(kind='box', subplots=True, layout=(layout_size, layout_size), sharex=False, sharey=False)
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