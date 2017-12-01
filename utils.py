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
    return np.random.binomial(1, prob, size=(node_count, node_count)).astype(
        np.float64)


def random_choice(arr):
    """
    Based on the suggestion by Radim Rehurek in this tweet:
    https://twitter.com/RadimRehurek/status/928671225861296128

    Results of comparison between this implementation and
    numpy.random.choice:
    ```
    >>>> lst = range(100000)
    >>>> timeit lst[np.searchsorted(uniform.cumsum(), np.random.random())]
    213 µs ± 1.32 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    >>>> timeit np.random.choice(lst)
    8.2 ms ± 70.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    ```

    Parameters
    ----------
    arr: list
        A 1-D list or array from which to select an element randomly

    Returns
    -------
    One element selected randomly from arr, with uniform probability
    """
    assert type(arr) is list or type(arr) is np.ndarray, 'List not provided!'
    uniform = np.ones((len(arr), )) / len(arr)
    return arr[np.searchsorted(uniform.cumsum(), np.random.random())]


def adj_mat_to_list(adj_mat):
    """
    Converts an adjacency matrix to an adjacency list.

    Parameters
    ----------
    adj_mat: numpy.ndarray
        Square adjacency matrix of a graph

    Returns
    -------
    adj_list: list
        The adjacency list of the given matrix
    """
    assert adj_mat.ndim == 2, 'Adjacency matrix should be of rank 2.'
    assert adj_mat.shape[0] == adj_mat.shape[1], 'Adjacency matrix' \
        ' should be square.'
    assert np.all(adj_mat >= 0), 'All elements of the adjaceny matrix ' \
        'should be nonnegative.'
    adj_list = []
    for i in range(adj_mat.shape[0]):
        adj_list.append([])
        for j in range(adj_mat.shape[0]):
            if adj_mat[i, j] > 0:
                adj_list[-1].append(j)

    return adj_list


def adj_list_to_mat(adj_list):
    """
    Converts an adjacency list to an adjacency matrix.

    Parameters
    ----------
    adj_list: list
        Adjacency list of a graph

    Returns
    -------
    adj_mat: numpy.ndarray
        Square adjacency matrix of a graph
    """
    assert type(adj_list) == list, 'Adjacency list should be provided'

    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype=np.float64)
    for i in range(len(adj_list)):
        for j in adj_list[i]:
            adj_mat[i, j] = 1.0

    return adj_mat
