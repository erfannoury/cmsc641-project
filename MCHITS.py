import numpy as np

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

    def calculate(self, adj_mat, num_iter):
        """
         In the Monte Carlo variant of the HITS algorithm, each
        node is visited at each iteration and a random walk is initiated from source node. At each step, check termination condition. 
        If termination condition not satified,continue random walk; pick neighbor from list (if current node i, neighbours include - all j's such that aij=1 is a child
        and all j such that aji=1 is a father. If picked neighbour is child to current node i, then increment authority ai for i, else increment hub hi.  
        Repeat step num_iter times and approximate ai and hi by dividing by total number of steps.

        Parameters
        ----------
        adj_mat: np.ndarray
            The nonnegative square adjacency matrix of the web. Each element
            (i, j) of the matrix is 1 if there is a link from web page
            i to web page j, and 0 otherwise.
        num_iter: int
            Number of iterations to run the Monte Carlo PageRank algorithm.

        Returns
        -------
        au_scores: np.ndarray
            A vector indicating the authority score for each webpage.
        """

        flatten = lambda tupleOfTuples: [element for tupl in tupleOfTuples for element in tupl]

        n = adj_mat.shape[0]

        au_scores = np.zeros((n,1), dtype=np.float64)
        hub_scores = np.zeros((n,1), dtype=np.float64)
        total_steps = np.zeros((n,1), dtype=np.float64)

        for _ in range(num_iter):

            for j in range(n):
                
                node = j
                children = flatten(np.nonzero(adj_mat[j,:]))
                father = flatten(np.nonzero(adj_mat[:,j]))

                while np.random.random() > self.stopping_prob:
                    #if (np.sum(children) + np.sum(father)) == 0:  #if there are no neighbours --> stop

                    neighbours = list(set(children+father))
                    # pick one neighbour at random
                    next_step = np.random.choice(neighbours)
                    total_steps[node]+=1
                    if next_step in children:
                        au_scores[node]+=1
                    else:
                        hub_scores[node]+=1

        au_scores = au_scores/total_steps.sum()
        return au_scores.reshape((1,n))

"""
p1 = MCHITS(0.5)
adj_mat = np.array([[0, 0, 1, 0,0,0,0],[0, 1, 1, 0,0,0,0],[1, 0, 1, 2,0,0,0],[0,0,1, 1,0,0,0],[0, 0, 0, 0,0,0,1],[0, 0, 0, 0,0,1,1],[0, 0, 0, 2,1,0,1]])
print(adj_mat)
sol = p1.calculate(adj_mat,10)
print(sol.sum())
"""
