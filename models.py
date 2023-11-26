import numpy as np
from algorithms import *
from sklearn.decomposition import non_negative_factorization, \
    LatentDirichletAllocation

class MatrixFactorization:
    """
    Base class for Matrix Factorization models
    Attributes:
        W: (np.ndarray) Decomposition Matrix 1
        H: (np.ndarray) Decomposition Matrix 2
    """

    def __init__(self, n, d, k):
        """
        Args:
             n : (int) number of data points
             d : (int) dimension of the data
             k : (int) the hidden dimension
        """
        self.W = np.abs(np.random.randn(n, k))
        self.H = np.abs(np.random.randn(k, d))
        self.n = n
        self.d = d
        self.k = k
        self.inference_method = None

    def get_data_size(self):
        """
        Returns:
            (int) returns the data size
        """

        return self.n

    def get_data_dim(self):
        """
        Returns:
            (int) returns the data dim
        """

        return self.d

    def get_embedding_dim(self):
        """
        Returns:
            (int) returns the embedding dim
        """

        return self.k

    def get_predicted_mat(self):
        """
        Returns:
            (np.ndarray) the predicted matrix
        """

        return np.dot(self.W, self.H)

    def get_word_embedding(self):
        """
        Returns:
             (np.ndarray)
        """

        return self.H

    def get_top_words(self, m=10):
        """
        return the top m words index
        Args:
            m: (int)
        """
        if m > self.H.shape[1]:
            m = self.H.shape[1]
        top_indices = np.argsort(self.H, axis=1)[:, -m:]

        return top_indices


    def fit(self, X, method="EM", n_iters=100, thresh=1e-3):
        """
        fits a matrix factorization model to the data

        Args:
            X: (np.ndarray) the initial data
            method: (str) the method used for inference
            n_iters: (int) number of iterations
            thresh: (float) the threshold for stoppage
        """

        if X.shape != (self.n, self.d):
            raise Exception("Shape of data does not match ({}, {})".format(
                self.n, self.d ))
        pass

class NmfSk(MatrixFactorization):
    """
    Base class for non-negative matrix factorization using sk-learn
    """

    def __init__(self, n, d, k):
        super().__init__(n,d ,k)

    def fit(self, X, method="sk", n_iters=100, thresh=1e-3):

        self.W, self.H, n_iter = non_negative_factorization(X,
                                                            n_components=self.k)

class LDA:
    """
    Base class for implementing Latent Dirichlet Allocation using
    sk learn
    """
    def __init__(self, k):
        """
         Args:
             k: (int) the number of topics
        """
        self.k = k
        self.lda = LatentDirichletAllocation(n_components=self.k)

    def fit(self, X):
        """
        fits LDA model

        Args:
            X: (np.ndarray) the document term matrix
        """

        self.lda.fit(X)

    def get_top_words(self, m=10):
        """
        return the top m words index
        Args:
            m: (int)
        """
        all_topic = self.lda.components_
        if m > self.k:
            m = self.lda.components_
        top_indices = np.argsort(all_topic, axis=1)[:, -m:]

        return top_indices


class PoissonMatrixFactorization(MatrixFactorization):
    """
    Base Class for Poisson Matrix Factorization
    """

    def __init__(self, n, d, k):
        super().__init__(n,d ,k)

    def fit(self, X, method="EM", n_iters=10, thresh=1e-3):

        if method == "EM":
            print("Setting EM as the inference method")
            em = EMPMF(self)
            em.run_EM(X)
