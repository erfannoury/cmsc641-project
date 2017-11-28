import numpy as np
import scipy as sc
from scipy import linalg



class HITS():
    """
    Implementation of Hyperlink Induced Topic Search (HITS)- Using iterative method and SVD. 
    """

    def calculate(self,adj_mat):

        """HITS calculated using SVD method, instead of power iteration method,to compute authority scores and hub scores. SVD decomposes A = UsV'. 
        First vectors of U and V, are first eigen vectors of AA' and A'A, hub and authority scores. Authority scores of nodes are roughly equivalent
        to page rank of nodes.

        
        Parameters
        ------------------
        adj_mat : n-by-n matrix

        Returns
        ------------------
        au_scores,hub_scores: 1-by-2 array of 1-by-n np.ndarray
        """

        U, s, VT = np.linalg.svd(adj_mat, full_matrices=True)
        V = np.transpose(VT)
        au_scores = -U[:,0]
        hub_scores = -V[:,0]

        return au_scores


    def iterative_calculate(self, adj_mat):

        """HITS calculated using iterative power method, to compute the dominant eigen vector of authority matrix and hub matrix. 
        Power iteration Method - Algorithm to compute the dominant eigen vector(v) of given matrix(A). Stops after num_simulations iterations.
        
        Parameters
        ------------------
        adj_mat : n-by-n matrix

        Returns
        ------------------
        au_scores,hub_scores: 1-by-2 array of 1-by-n np.ndarray
        """
        num_simulations = 1000
        au_mat = np.matmul(np.transpose(adj_mat), adj_mat)
        hub_mat = np.matmul(adj_mat,np.transpose(adj_mat))

        a = np.random.rand(au_mat.shape[0])
        h = np.random.rand(hub_mat.shape[0])

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product A*au
            a1 = np.dot(au_mat, a)
            h1 = np.dot(hub_mat,h)
            # normalize the vector
            au_scores = a1 / np.linalg.norm(a1)
            hub_scores = h1/np.linalg.norm(h1)

            # normalize again
            au_scores = au_scores/np.linalg.norm(au_scores)
            hub_scores = hub_scores/linalg.norm(hub_scores)

        return au_scores

    

""" Sample
adj_mat = np.array([[0, 0, 1, 0,0,0,0],[0, 1, 1, 0,0,0,0],[1, 0, 1, 2,0,0,0],[0,0,1, 1,0,0,0],[0, 0, 0, 0,0,0,1],[0, 0, 0, 0,0,1,1],[0, 0, 0, 2,1,0,1]])
print(adj_mat)
p1 = HITS()
sol1 = p1.iterative_calculate(adj_mat)
print('Power Interation')
print(sol1)

sol2 = p1.calculate(adj_mat)
print('SVD')
print(sol2) 
"""
