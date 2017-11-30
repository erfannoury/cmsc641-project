import numpy as np


class HITS():
    """
    Implementation of Hyperlink Induced Topic Search (HITS) using SVD and
    power iteration methods.
    """
    def __init__(self):
        pass

    def calculate(self, adj_mat):
        """
        HITS calculated using SVD method, instead of power iteration method,
        to compute authority scores and hub scores. SVD decomposes A = U s V'.
        First columns of U and V are first eigenvectors of AA' and A'A,
        hub and authority scores. Authority scores of nodes are roughly
        equivalent to PageRank score of nodes.

        Parameters
        ----------
        adj_mat: np.ndarray
            This is the nonnegative square adjacency matrix of the web. Each
            element (i, j) of the matrix is 1 if there is a link from web page
            i to web page j, and 0 otherwise.

        Returns
        -------
        au_scores: numpy.ndarray
            A vector indicating the authority score for each webpage.
        """
        assert adj_mat.ndim == 2, 'Adjacency matrix should be of rank 2.'
        assert adj_mat.shape[0] == adj_mat.shape[1], 'Adjacency matrix' \
            ' should be square.'
        assert np.all(adj_mat >= 0), 'All elements of the adjaceny matrix ' \
            'should be nonnegative.'

        U, _, _ = np.linalg.svd(adj_mat)
        au_scores = -U[:, 0]
        au_scores /= au_scores.sum()

        return au_scores

    def iterative_calculate(self, adj_mat, iter_count):
        """
        HITS calculated using iterative power method, to compute the dominant
        eigenvectors of authority matrix and hub matrix.

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
        au_scores: numpy.ndarray
            A vector indicating the authority score for each webpage.
        """
        assert adj_mat.ndim == 2, 'Adjacency matrix should be of rank 2.'
        assert adj_mat.shape[0] == adj_mat.shape[1], 'Adjacency matrix' \
            ' should be square.'
        assert np.all(adj_mat >= 0), 'All elements of the adjaceny matrix ' \
            'should be nonnegative.'

        au_mat = np.matmul(np.transpose(adj_mat), adj_mat)
        hub_mat = np.matmul(adj_mat, np.transpose(adj_mat))

        a = np.random.rand(au_mat.shape[0])
        h = np.random.rand(hub_mat.shape[0])

        for _ in range(iter_count):
            # calculate the matrix-by-vector product A*au
            a1 = np.dot(au_mat, a)
            h1 = np.dot(hub_mat, h)
            # normalize the vector
            au_scores = a1 / np.linalg.norm(a1)
            hub_scores = h1 / np.linalg.norm(h1)

        return au_scores / au_scores.sum()
