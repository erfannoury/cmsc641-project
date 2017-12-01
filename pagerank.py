from datetime import datetime
import numpy as np
import utils


class PageRank():
    """
    This is an implementation of the PageRank algorithm for web page importance
    ranking in the web.

    Paramteres
    ----------
    teleporting_prob: float
        The teleporting probability, the probability that the random surfer
        will jump to an arbitrary web page on the web.
    """
    def __init__(self, teleporting_prob):
        if not (0 < teleporting_prob < 1):
            raise ValueError('Teleporting probabilityt should be in the '
                             'range (0, 1).')
        self.teleporting_prob = teleporting_prob

    def calculate(self, adj_mat, verbose=False):
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
        verbose: bool
            Whether to return the algorithm runtime information.

        Returns
        -------
        page_rank: np.ndarray
            A vector indicating the PageRank value for each webpage.
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

        A = np.divide(
            adj_mat,
            adj_mat.sum(axis=0, keepdims=True) + np.zeros_like(adj_mat),
            where=(adj_mat > 0))
        M = (1 - self.teleporting_prob) * A + \
            self.teleporting_prob * np.ones_like(A)

        w, vr = np.linalg.eig(M)
        max_idx = np.argmax(np.real(w))
        page_rank = np.real(vr[:, max_idx] / np.sum(vr[:, max_idx]))

        delta = datetime.now() - now

        if verbose:
            return page_rank, delta
        else:
            return page_rank

    def iterative_calculate(self, adj_mat, iter_count, verbose=False):
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
        verbose: bool
            Whether to return the algorithm runtime information.

        Returns
        -------
        page_rank: np.ndarray
            A vector indicating the PageRank value for each webpage.
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

        A = np.divide(
            adj_mat,
            adj_mat.sum(axis=0, keepdims=True) + np.zeros_like(adj_mat),
            where=(adj_mat > 0))
        M = (1 - self.teleporting_prob) * A + \
            self.teleporting_prob * np.ones_like(A)

        z = np.ones((M.shape[0], 1), dtype=np.float64) / M.shape[0]
        for _ in range(iter_count):
            z = np.matmul(M, z)

        page_rank = z / z.sum()
        page_rank = page_rank.reshape((-1, ))

        delta = datetime.now() - now

        if verbose:
            return page_rank, delta
        else:
            return page_rank


class MCPageRank():
    """
    This is an implementation of the Monte Carlo PageRank algorithm for web
    page importance analysis in the web. This implementation is based on the
    Algorithm 4 MC complete path stopping at dangling nodes of [1].

    [1] Avrachenkov, Konstantin, Nelly Litvak, Danil Nemirovsky, and Natalia
        Osipova. "Monte Carlo methods in PageRank computation: When one
        iteration is sufficient." SIAM Journal on Numerical Analysis 45, no. 2
        (2007): 890-904.

    Paramteres
    ----------
    teleporting_prob: float
        The teleporting probability, the probability that the random surfer
        will jump to an arbitrary web page on the web.
    """

    def __init__(self, teleporting_prob):
        if not (0 < teleporting_prob < 1):
            raise ValueError('Teleporting probability should be in the '
                             'range (0, 1).')
        self.teleporting_prob = teleporting_prob

    def calculate(self, adj_mat, num_iter, verbose=False):
        """
        Calculates the PageRank value for each webpage using the PageRank
        algorithm. In the Monte Carlo variant of the PageRank algorithm, each
        node is visited at each iteration and a random walk is initiated from
        the given node to find a final node when the random walk stops. By
        counting the number of stops for each node, we can get an estimate
        of the PageRank value for each node.

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
        page_rank: np.ndarray
            A vector indicating the PageRank value for each webpage.
        delta: datetime.timedelta
            If verbose is True, then the runtime of the algorithm will be
            returned, too
        """
        assert adj_mat.ndim == 2, 'Adjacency matrix should be of rank 2.'
        assert adj_mat.shape[0] == adj_mat.shape[1], 'Adjacency matrix' \
            ' should be square.'
        assert np.all(adj_mat >= 0), 'All elements of the adjaceny matrix ' \
            'should be nonnegative.'

        adj_list = utils.adj_mat_to_list(adj_mat)

        now = datetime.now()

        n = len(adj_list)
        page_rank = np.zeros((n, ), dtype=np.float64)
        for _ in range(num_iter):
            for j in range(n):
                idx = j
                while np.random.random() > self.teleporting_prob:
                    page_rank[idx] += 1
                    if len(adj_list[idx]) == 0:
                        break
                    else:
                        idx = utils.random_choice(adj_list[idx])

        page_rank /= page_rank.sum()

        delta = datetime.now() - now

        if verbose:
            return page_rank, delta
        else:
            return page_rank
