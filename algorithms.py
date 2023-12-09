### This file contains the the inference algorithms

import numpy as np
from visualization import simple_plot
import torch

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
        k = pmf_model.get_topic_dim()

        super().__init__(pmf_model, np.zeros((n, d, k)))

    def compute_elbo(self, X):

        W, H = self.model.W, self.model.H
        ep = 1e-10
        WXH = torch.einsum("ik,kj ->kij", W, H) + ep
        WH = torch.matmul(W, H)
        Y = (X / WH) * WXH * torch.log(WXH) - WXH

        return torch.sum(Y)

    def E_step(self, X):
        """
        for PMF the latent variable follows multinomial distribution.
        """
        W, H = self.model.W, self.model.H
        WH = torch.matmul(W, H)
        WXH = torch.einsum("ik,kj ->kij", W, H)

        self.probability = WXH / WH

    def M_step(self, X):

        W, H = self.model.W, self.model.H
        WH = torch.matmul(W, H)
        WXH = torch.einsum('ij,ik,kj->ik', X / WH, W, H)
        H_sum = torch.sum(H, axis=1)
        W1 = WXH / H_sum
        WH1 = torch.matmul(W1, H)

        WXH1 = torch.einsum('ij,ik,kj->kj', X / WH1, W1, H)
        W_sum = torch.sum(W1, axis=0).reshape(-1, 1)
        H1 = WXH1 / W_sum

        self.model.W = W1
        self.model.H = H1

    def run_EM(self, X, max_iters=1000, epsilon=1e-2, save_elbo=True,
               folder="figures", name="elbo_em"):

        loss_1 = 0
        loss_0 = 100
        i = 0
        X = X + 1e-10
        max_iters = 50
        while np.abs(loss_0 - loss_1) > epsilon and i < max_iters :
            loss_0 = loss_1
            self.E_step(X)
            self.M_step(X)
            loss_1 = self.compute_elbo(X).item()
            self.elbo_all.append(loss_1)
            print("iter: {}, ELBO: {}".format(i, loss_1))
            i += 1
        #print(np.dot(self.model.W, self.model.H))
        x_dict = {"1": np.arange(0, len(self.elbo_all))}
        y_dict = {"1": np.array(self.elbo_all) * -1}
        simple_plot(x_dict, y_dict, "iter #", "ELBO",
                    save_name="elbo_pmf_{}".format(self.model.get_topic_dim()),
                    save_path="figures/elbo_pmf", legend_on=False, use_log=False)
