#this file contains functions that computes metrics used in model evaluation

import numpy as np

def matrix_accuracy(pred_mat, true_mat):
    """
    Computes the Frobenius norm of the difference matrix
    Args:
         pred_mat: (np.ndarray) the predicted matrix
         true_mat: (np.ndarray) the true matrix
    """

    diff_mat = pred_mat - true_mat

    return np.linalg.norm(diff_mat)

## TODO
def evaluate_topic_coherence():
    """

    """
    ##code for evaluating topic coherence.
    ##
    pass


## TODO
def compute_perplexity():
    """
    computes the model perplexity
    """





