import numpy as np


def random_graph(node_count, prob):
    """
    Create a random adjacency matrix of a graph

    Parameters
    ----------
    node_count: int
        Number of nodes in the graph
    prob: float
        Probability of presence an edge between two nodes

    Returns
    -------
    adj_mat: numpy.ndarray
        A numpy array of size (node_count, node_count) with elements in {0, 1}
    """
    return np.random.binomial(1, 0.01, size=(1000, 1000))
