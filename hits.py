from datetime import datetime
import numpy as np
import utils


class HITS():
    """
    Implementation of Hyperlink Induced Topic Search (HITS) using SVD and
    power iteration methods.
    """
    def __init__(self):
        pass

    def calculate(self, adj_mat, verbose=False):
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
        verbose: bool
            Whether to return the algorithm runtime information.

        Returns
        -------
        au_scores: numpy.ndarray
            A vector indicating the authority score for each webpage.
        delta: datetime.timedelta
            If verbose is True, then the runtime of the algorithm will be
            returned, too
        """
        assert adj_mat.ndim == 2, 'Adjacency matrix should be of rank 2.'
        assert adj_mat.shape[0] == adj_mat.shape[1], 'Adjacency matrix' \
            ' should be square.'
        assert np.all(adj_mat >= 0), 'All elements of the adjaceny matrix ' \
            'should be nonnegative.'

        now = datetime.now()

        U, _, _ = np.linalg.svd(adj_mat)
        au_scores = (-U[:, 0]).astype(np.float64)
        au_scores /= au_scores.sum()

        delta = datetime.now() - now
        if verbose:
            return au_scores, delta
        else:
            return au_scores

    def iterative_calculate(self, adj_mat, num_iter, verbose=False):
        """
        HITS calculated using iterative power method, to compute the dominant
        eigenvectors of authority matrix and hub matrix.

         Parameters
        ----------
        adj_mat: np.ndarray
            The nonnegative square adjacency matrix of the web. Each element
            (i, j) of the matrix is 1 if there is a link from web page
            i to web page j, and 0 otherwise.
        num_iter: int
            Number of iterations of the Power method
        verbose: bool
            Whether to return the algorithm runtime information.

        Returns
        -------
        au_scores: numpy.ndarray
            A vector indicating the authority score for each webpage.
        delta: datetime.timedelta
            If verbose is True, then the runtime of the algorithm will be
            returned, too
        """
        assert adj_mat.ndim == 2, 'Adjacency matrix should be of rank 2.'
        assert adj_mat.shape[0] == adj_mat.shape[1], 'Adjacency matrix' \
            ' should be square.'
        assert np.all(adj_mat >= 0), 'All elements of the adjaceny matrix ' \
            'should be nonnegative.'

        now = datetime.now()

        au_mat = np.matmul(adj_mat.T, adj_mat)

        au_scores = np.random.rand(au_mat.shape[0]).astype(np.float64)

        for _ in range(num_iter):
            au_scores = np.dot(au_mat, au_scores)

            au_scores /= np.linalg.norm(au_scores)

        au_scores /= au_scores.sum()
        delta = datetime.now() - now

        if verbose:
            return au_scores, delta
        else:
            return au_scores


class MCHITS():
    """
    This is an implementation of the Monte Carlo HITS algorithm for web
    page importance analysis in the web.

    Paramters
    ----------
    teleporting_prob: float
        The teleporting probability, the probability that the random surfer
        will jump to an arbitrary web page on the web.
    """

    def __init__(self, stopping_prob):
        if not (0 <= stopping_prob <= 1):
            raise ValueError('Stopping probability should be in the '
                             'range [0, 1].')
        self.stopping_prob = stopping_prob

    def calculate(self, adj_mat, num_iter, verbose=False):
        """
        In the Monte Carlo variant of the HITS algorithm, each node is visited
        at each iteration and a random walk is initiated from source node.
        This is the MC-all variant of the MCHITS algorithm from [1].

        [1] Jin, Zhaoyan, Dianxi Shi, Quanyuan Wu, and Hua Fan. "MCHITS: Monte
            Carlo based method for hyperlink induced topic search on networks."
            Journal of Networks 8, no. 10 (2013): 2376-2383.

        Parameters
        ----------
        adj_mat: np.ndarray
            The nonnegative square adjacency matrix of the web. Each element
            (i, j) of the matrix is 1 if there is a link from web page
            i to web page j, and 0 otherwise.
        num_iter: int
            Number of iterations to run the Monte Carlo PageRank algorithm.
        verbose: bool
            Whether to return the algorithm runtime information.

        Returns
        -------
        au_scores: np.ndarray
            A vector indicating the authority score for each webpage.
        delta: datetime.timedelta
            If verbose is True, then the runtime of the algorithm will be
            returned, too
        """

        n = adj_mat.shape[0]
        adj_list_children = utils.adj_mat_to_list(adj_mat)
        adj_list_parents = utils.adj_mat_to_list(adj_mat.T)

        now = datetime.now()

        au_scores = np.zeros((n, ), dtype=np.float64)

        total_steps = 0
        for _ in range(num_iter):
            for j in range(n):
                idx = j
                while np.random.random() > self.stopping_prob:
                    total_steps += 1
                    children = adj_list_children[idx]
                    parents = adj_list_parents[idx]
                    if len(parents) + len(children) == 0:
                        break
                    neighbours = list(set(children + parents))
                    next_idx = random_choice(neighbours)
                    if next_idx in children:
                        au_scores[next_idx] += 1
                    idx = next_idx

        au_scores = au_scores / total_steps
        au_scores /= au_scores.sum()

        delta = datetime.now() - now

        if verbose:
            return au_scores, delta
        else:
            return au_scores
