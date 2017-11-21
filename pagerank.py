import numpy as np
import scipy as sc
from scipy import linalg


class PageRank():
    """
    This is an implementation of the PageRank algorithm for web page importance
    analysis in the web.

    Paramteres
    ----------
    damping_factor: float
        The damping factor, the probability that the random surfer will jump
        to an arbitrary web page on the web.
    """
    def __init__(self, damping_factor):
        if not (0 <= damping_factor <= 1):
            raise ValueError('Damping factor should be in the range [0, 1].')
        self.damping_factor = damping_factor

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
        M = (1 - self.damping_factor) * A + \
            self.damping_factor * np.ones_like(A)

        _, vr = sc.linalg.eig(M)
        page_rank = vr[:, 0] / np.sum(vr[:, 0])
        return page_rank

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
        M = (1 - self.damping_factor) * A + \
            self.damping_factor * np.ones_like(A)

        z = np.ones((M.shape[0], 1))
        for _ in range(iter_count):
            z = np.matmul(M, z)

        page_rank = z / z.sum()
        return page_rank
