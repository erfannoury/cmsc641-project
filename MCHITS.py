from datetime import datetime
import numpy as np
import utils


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

        au_scores = np.zeros((n, ), dtype=np.float64)
        hub_scores = np.zeros((n, ), dtype=np.float64)

        now = datetime.now()
        total_steps = 0
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
                else:
                    hub_scores[next_idx] += 1
                idx = next_idx

        au_scores = au_scores / total_steps
        au_scores /= au_scores.sum()

        delta = datetime.now() - now

        if verbose:
            return au_scores, delta
        else:
            return au_scores
