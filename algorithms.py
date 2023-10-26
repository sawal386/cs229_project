import numpy as np

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
        runs the expectation step
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

    def compute_elbo(self, X, WV_k):
        """
        computes the elbo of the matrix factorization
        Args:
            X : (np.ndarray) the data matrix. shape: (n, d)
            WV_k : (np.ndarray) the einsum matrix. shape: (n, d, k)
                                WV[i, j, k] = W[i, k] * V[k, j].
        """

        log_WV_k = np.log(WV_k)
        exp_matrix = X.reshape(X.shape[0], X.shape[1], 1) * self.probability
        loss_matrix = exp_matrix * log_WV_k - exp_matrix

        return np.sum(loss_matrix)

    def E_step(self, X):

        einsum_mat = np.einsum('ik,kj->ijk', self.model.W, self.model.V)
        einsum_mat_sum = np.sum(einsum_mat, axis=2, keepdims=True)
        self.probability = einsum_mat / einsum_mat_sum
        #print(einsum_mat.shape)

        elbo = self.compute_elbo(X, einsum_mat)
        self.elbo_all.append(elbo)

    def M_step(self, X):

        V_sum = np.sum(self.model.V, axis=1)
        X_expanded = X[:, np.newaxis, :]
        self.model.W = np.sum(X_expanded * self.probability, axis=0) / V_sum

        W_sum = np.sum(self.model.W, axis=0)
        X_expanded = X[np.new_axis, :, :]
        self.model.V = np.sum(X_expanded * self.probability, axis=1) / W_sum








