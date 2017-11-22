import numpy as np
import scipy as sc
from scipy import linalg


class PageRank():
    """
    This is an implementation of the PageRank algorithm for web page importance
    analysis in the web.

    Paramteres
    ----------
    teleporting_prob: float
        The teleporting probability, the probability that the random surfer
        will jump to an arbitrary web page on the web.
    """
    def __init__(self, teleporting_prob):
        if not (0 <= teleporting_prob <= 1):
            raise ValueError('Teleporting probabilityt should be in the '
                             'range [0, 1].')
        self.teleporting_prob = teleporting_prob

    def calculate(self, adj_mat):
        """
        Calculates the PageRank value for each webpage using the PageRank
        algorithm. Instead of the power method, this implementation uses the
        eigenvalue of the normalized adjacency matrix, with teleportation added
        to prevent deadlocks, to find the PageRank of each webpage.

        Parameters
        ----------
        adj_mat: np.ndarray
            The nonnegative square adjacency matrix of the web. Each element
            (i, j) of the matrix is 1 if there is a link from web page
            i to web page j, and 0 otherwise.

        Returns
        -------
        page_rank: np.ndarray
            A vector indicating the PageRank value for each webpage.
        """
        assert adj_mat.ndim == 2, 'Adjacency matrix should be of rank 2.'
        assert adj_mat.shape[0] == adj_mat.shape[1], 'Adjacency matrix' \
            ' should be square.'
        assert np.all(adj_mat >= 0), 'All elements of the adjaceny matrix ' \
            'should be nonnegative.'

        A = np.divide(
            adj_mat,
            adj_mat.sum(axis=0, keepdims=True) + np.zeros_like(adj_mat),
            where=(adj_mat > 0))

        M = (1 - self.teleporting_prob) * A + \
            self.teleporting_prob * np.ones_like(A)

        w, vr = np.linalg.eig(M)
        max_idx = np.argmax(np.real(w))
        page_rank = vr[:, max_idx] / np.sum(vr[:, max_idx])
        return np.real(page_rank)

    def iterative_calculate(self, adj_mat, iter_count):
        """
        Calculates the PageRank value for each webpage using the PageRank
        algorithm. This implementation uses the Power method.

        Parameters
        ----------
        adj_mat: np.ndarray
            The nonnegative square adjacency matrix of the web. Each element
            (i, j) of the matrix is 1 if there is a link from web page
            i to web page j, and 0 otherwise.
        iter_count: int
            Number of iterations of the Power method

        Returns
        -------
        page_rank: np.ndarray
            A vector indicating the PageRank value for each webpage.
        """
        assert adj_mat.ndim == 2, 'Adjacency matrix should be of rank 2.'
        assert adj_mat.shape[0] == adj_mat.shape[1], 'Adjacency matrix' \
            ' should be square.'
        assert np.all(adj_mat >= 0), 'All elements of the adjaceny matrix ' \
            'should be nonnegative.'

        A = np.divide(
            adj_mat,
            adj_mat.sum(axis=0, keepdims=True) + np.zeros_like(adj_mat),
            where=(adj_mat > 0))
        M = (1 - self.teleporting_prob) * A + \
            self.teleporting_prob * np.ones_like(A)

        z = np.ones((M.shape[0], 1))
        for _ in range(iter_count):
            z = np.matmul(M, z)

        page_rank = z / z.sum()
        return page_rank.reshape((-1, ))


class MCPageRank():
    """
    This is an implementation of the Monte Carlo PageRank algorithm for web
    page importance analysis in the web.

    Paramteres
    ----------
    teleporting_prob: float
        The teleporting probability, the probability that the random surfer
        will jump to an arbitrary web page on the web.
    """

    def __init__(self, teleporting_prob):
        if not (0 <= teleporting_prob <= 1):
            raise ValueError('Teleporting probability should be in the '
                             'range [0, 1].')
        self.teleporting_prob = teleporting_prob

    def calculate(self, adj_list, num_iter):
        """
        Calculates the PageRank value for each webpage using the PageRank
        algorithm. In the Monte Carlo variant of the PageRank algorithm, each
        node is visited at each iteration and a random walk is initiated from
        the given node to find a final node when the random walk stops. By
        counting the number of stops for each node, we can get an estimate
        of the PageRank value for each node.

        Parameters
        ----------
        adj_list: list
            Adjacency list of the web matrix. Each element of the list is a
            list of indices that have connections to the corresponding node.
        num_iter: int
            Number of iterations to run the Monte Carlo PageRank algorithm.

        Returns
        -------
        page_rank: np.ndarray
            A vector indicating the PageRank value for each webpage.
        """
        assert type(adj_list) == list, 'The adjacency list should be a list'

        n = len(adj_list)
        page_rank = np.zeros((n, ), dtype=np.float64)
        for _ in range(num_iter):
            for j in range(n):
                idx = j
                while np.random.random() > self.teleporting_prob:
                    if len(adj_list[idx]) == 0:
                        idx = np.random.randint(0, n)
                    else:
                        idx = np.random.choice(adj_list[idx])
                page_rank[idx] += 1

        page_rank /= (n * num_iter)
        return page_rank
