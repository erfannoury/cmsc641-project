import numpy as np
import scipy as sc
from scipy import linalg


def powerIteration(A,num_simulations):
        """
        Algorithm to compute the dominant eigen vector(v) of given matrix(A). Stops after num_simulations iterations.

        Parameters
        ----------
        A: np.ndarray
            
        Returns
        -------
        v:  np.ndarray
            Dominant eigen vector of A

        """
        v = np.random.rand(A.shape[0])

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Av
            v1 = np.dot(adj_mat, v)

            # calculate the norm
            v1_norm = np.linalg.norm(v1)

            # re-normalize the vector
            v = v1 / v1_norm
            
        return v


class HITS():
    """
    Implementation of Hyperlink Induced Topic Search (HITS) in python. Using iterative method and SVD. 

    """
    
    def HITS_PowerIteration(self,adj_mat):

        """HITS calculated using iterative power method - to compute the dominant eigen vector of authority matrix and hub matrix.
        
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


        au_scores = powerIteration(au_mat,num_simulations)
        au_scores = au_scores/linalg.norm(au_scores)
        hub_scores = powerIteration(hub_mat,num_simulations)
        hub_scores = hub_scores/linalg.norm(hub_scores)

        return au_scores,hub_scores



    def HITS_SVD(self,adj_mat):

        """HITS calculated using SVD method - to compute authority scores and hub scores.
        
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

        return au_scores,hub_scores



    

p1 = HITS()
adj_mat = np.array([[0, 0, 1, 0,0,0,0],[0, 1, 1, 0,0,0,0],[1, 0, 1, 2,0,0,0],[0,0,1, 1,0,0,0],[0, 0, 0, 0,0,0,1],[0, 0, 0, 0,0,1,1],[0, 0, 0, 2,1,0,1]])
print(adj_mat)
sol1 = p1.HITS_PowerIteration(adj_mat)
print('Power Interation')
print(sol1[0])

sol2 = p1.HITS_SVD(adj_mat)
print('SVD')
print(sol2[0])
