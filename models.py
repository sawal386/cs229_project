## This file contains models used in the project

from algorithms import *
from sklearn.decomposition import NMF, \
    LatentDirichletAllocation, TruncatedSVD
import torch
from gensim.models.ldamodel import LdaModel
from sklearn.model_selection import train_test_split

class MatrixFactorization:
    """
    Base class for Matrix Factorization models
    Attributes:
        W: (np.ndarray) Decomposition Matrix 1
        H: (np.ndarray) Decomposition Matrix 2
    """

    def __init__(self, n, d, k, index_word_map):
        """
        Args:
             n : (int) number of data points
             d : (int) dimension of the data
             k : (int) the hidden dimension
             index_word_map: (dict) map index to words
        """
        self.W = np.abs(np.random.randn(n, k))
        self.H = np.abs(np.random.randn(k, d))
        self.index_word_map = index_word_map
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

    def get_topic_dim(self):
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

    def get_topic_matrix(self):
        """
        Returns:
             (np.ndarray)
        """

        return self.H

    def get_document_matrix(self):
        """
        Returns:
             (np.ndarray)
        """

        return self.W

    def get_map(self):

        return self.index_word_map


    def fit(self, X, n_iters=100, thresh=1e-3):
        """
        fits a matrix factorization model to the data

        Args:
            X: (np.ndarray) the initial data
            n_iters: (int) number of iterations
            thresh: (float) the threshold for stoppage
        """

        if X.shape != (self.n, self.d):
            raise Exception("Shape of data does not match ({}, {})".format(
                self.n, self.d ))
        pass

    def get_top_words(self, top_n=10):
        """
        returns the top words for each topic
        Args:
            top_n: (int) the number of words to return
        Returns:
            (dict) (int) number of topics -> List[str] the top n words
        """

        top_words = {}
        for i in range(self.k):
            top_k = np.argsort(self.H)[-top_n:][::-1]
            top_words[i] = [self.index_word_map[k] for k in top_k]

        return top_words
class NmfSk(MatrixFactorization):
    """
    Base class for non-negative matrix factorization using sk-learn
    """

    def __init__(self, n, d, k, index_word_map):
        super().__init__(n,d ,k, index_word_map)
        self.mf = NMF(n_components=self.k, max_iter=200, beta_loss=2)

    def fit(self, X, n_iters=200, thresh=1e-2):

        self.W = self.mf.fit_transform(X)
        self.H = self.mf.components_

class PoissonMatrixFactorization(MatrixFactorization):
    """
    Class for Poisson Matrix Factorization
    """

    def __init__(self, n, d, k, index_word_map):
        super().__init__(n,d ,k, index_word_map)
        self.dev = "cpu"
        if torch.backends.mps.is_available():
            self.dev = "mps"
        elif torch.cuda.is_available():
            self.dev = "cuda"

        self.H = torch.from_numpy(np.float32(self.H)).to(self.dev)
        self.W = torch.from_numpy(np.float32(self.W)).to(self.dev)

    def fit(self, X,  n_iters=200, thresh=1e-2):
        if torch.backends.mps.is_available():
            X = torch.from_numpy(np.float32(X))
            #X = torch.from_numpy(np.float32(X)).to("mps")

        em = EMPMF(self)
        em.run_EM(X, max_iters=n_iters, epsilon=thresh)

    def get_topic_matrix(self):

        return self.H.to("cpu").numpy()

    def get_document_matrix(self):

        return self.W.to("cpu").numpy()

class SVD(MatrixFactorization):
    """
    Class for SVD based models
    """

    def __init__(self, n, d, k, index_word_map):
        super().__init__(n,d ,k, index_word_map)
        self.svd = TruncatedSVD(n_components=k)

    def fit(self, X, n_iters=10, thresh=1e-3):

        self.svd.fit(X)
        self.H = self.svd.components_
        self.svd.fit(X.T)
        self.W = self.svd.components_.T


class LDA:
    """Base Class for LDA"""

    def __init__(self, k, index_word_map):

        self.k = k
        self.H = None
        self.W = None
        self.index_word_map = index_word_map
        self.lda = None

    def get_best_hyper_params(self, corpus):
        """

        Args:
            corpus: [List[List]] the training corpus

        Returns:
            (float, float)
        """

        print("finding the best hyper params")
        alpha_values = [0.001, 0.01, 0.1, 1, 10]
        eta_values = [0.001, 0.01, 0.1, 1, 10]
        train, test = train_test_split(corpus, test_size=0.1)
        lda = LdaModel(corpus=train, id2word=self.index_word_map, num_topics=self.k)
        best_eta, best_alpha = lda.eta, lda.alpha
        best_perplexity = lda.log_perplexity(test)
        for alpha in alpha_values:
            for eta in eta_values:
                lda = LdaModel(corpus=train, id2word=self.index_word_map, num_topics=self.k,
                               alpha=alpha, eta=eta)
                perp = lda.log_perplexity(test)
                if perp < best_perplexity:
                    best_eta, best_alpha = eta, alpha
                    best_perplexity = perp

        return best_eta, best_alpha

    def fit(self, X):
        """
        Args:
            X: (np.ndarray) document term matrix
        """

        #alpha, eta = self.get_best_hyper_params(X)
        #self.lda = LdaModel(corpus=X, id2word=self.index_word_map, num_topics=self.k,
        #                    alpha=alpha, eta=eta)
        self.lda = LdaModel(corpus=X, id2word=self.index_word_map, num_topics=self.k,
                            iterations=100)
        self.H = self.lda.get_topics()
        temp_W = list(self.lda.get_document_topics(X, minimum_probability=0))
        self.W = np.zeros((len(X), self.k))

        for i in range(len(temp_W)):
            for j in range(self.k):
                self.W[i, j] = temp_W[i][j][1]

    def get_topic_matrix(self):
        """
        Returns:
             (np.ndarray)
        """

        return self.H

    def get_document_matrix(self):
        """
        Returns:
             (np.ndarray)
        """

        return self.W

    def get_map(self):

        return self.index_word_map

    def get_top_words(self, top_n=10):
        """
        returns the top words for each topic
        Args:
            top_n: (int) the number of words to return
        Returns:
            (dict) (int) number of topics -> List[str] the top n words
        """

        top_words = {}
        for i in range(self.k):
            top_k = np.argsort(self.H)[-top_n:][::-1]
            top_words[i] = [self.index_word_map[k] for k in top_k]

        return top_words
