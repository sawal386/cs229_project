### This file contains the the inference algorithms

import numpy as np
from visualization import simple_plot

class EMSolver:
    """Base class for Expectation Maximization
       Attributes:
           model: the model we are optimizing for
           probability: (np.ndarray) the probability distributions
           elbo_all: ([float]) the evidence lower bound
    """
    def __init__(self, model, probs=0):
        """
        Args:
             model: the model we are optimizing for
        """
        self.model = model
        self.probability = probs
        self.elbo_all = []

    def E_step(self, X):
        """
        runs the expectation step.
        Args:
            X: (np.ndarray) the input data
        """
        pass

    def M_step(self, X):
        """
        runs the maximization step
        Args:
            X: (np.ndarray) the input data
        """

        pass

    def compute_elbo(self, X):
        """
        computes the elbo and updates the elbo list
        Args:
            X: (np.ndarray) the input data
        Returns:
            (float) elbo
        """

        pass

    def run_EM(self, X, n_iters=1000, epsilon=1e-3):
        """
        runs the EM algorithm till convergence
        Args:
            X: X: (np.ndarray) the input data
            n_iters: (int) total number of iterations
            epsilon: (float) the tolerance
        """

        pass


class EMPMF(EMSolver):
    """
    EM Solver for Poisson Matrix Factorization
    """
    def __init__(self, pmf_model):
        """
         Args:
             pmf_model: (models.PoissonMatrixFactorization
        """
        n = pmf_model.get_data_size()
        d = pmf_model.get_data_dim()
        k = pmf_model.get_embedding_dim()

        super().__init__(pmf_model, np.zeros((n, d, k)))

    def compute_elbo(self, X):

        W, H = self.model.W, self.model.H
        ep = 0
        WXH = np.einsum("ik,kj ->kij", W, H) + ep
        WH = np.dot(W, H)
        Y = (X / WH) * WXH * np.log(WXH) - WXH

        return np.sum(Y)

    def E_step(self, X):
        """
        for PMF the latent variable follows multinomial distribution.
        """
        W, H = self.model.W, self.model.H
        WH = np.dot(W, H)
        WXH = np.einsum("ik,kj ->kij", W, H)

        self.probability = WXH / WH

    def M_step(self, X):
        W, H = self.model.W, self.model.H
        WH = np.dot(W, H)
        WXH = np.einsum('ij,ik,kj->ik', X / WH, W, H)
        H_sum = np.sum(H, axis=1)

        W1 = WXH / H_sum

        WH1 = np.dot(W1, H)
        WXH1 = np.einsum('ij,ik,kj->kj', X / WH1, W1, H)
        W_sum = np.sum(W1, axis=0).reshape(-1, 1)
        H1 = WXH1 / W_sum

        self.model.W = W1
        self.model.H = H1

    def run_EM(self, X, max_iters=100, epsilon=1e-3, save_elbo=True,
               folder="figures", name="elbo_em"):

        loss_1 = self.compute_elbo(X)
        loss_0 = 0
        i = 0
        self.elbo_all.append(loss_1)
        while np.abs(loss_0 - loss_1) > epsilon and i < max_iters :
            loss_0 = loss_1
            self.E_step(X)
            self.M_step(X)
            loss_1 = self.compute_elbo(X)
            self.elbo_all.append(loss_1)
            print("iter: {}, ELBO: {}".format(i, loss_1))
            i += 1
        x_dict = {"1": np.arange(0, len(self.elbo_all))}
        y_dict = {"1": np.array(self.elbo_all)}
        #simple_plot(x_dict, y_dict, "iter #", "ELBO",
        #            save_name="elbo_pmf", legend_on=False)
