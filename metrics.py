# this file contains functions that computes metrics used in model evaluation

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

def compute_coherence(p1, p2, p12, metric="nmi"):
    """
    computes coherence given the probabilities
    Args:
        p1: (float) probability of observing word 1 in a document
        p2: (float) probability of observing word 2 in a document
        p12: (float) probability of observing both word 1 and
        metric: (str)
    Returns:

    """
    ep = 0
    if metric == "umass":
        return np.log((p12+1) / p2 )
    elif metric == "nmi":
        return -np.log(p12/ (p1 * p2) + ep) / np.log(p12 + ep)
    elif metric == "pmi":
        return np.log(p12/ (p1 * p2) + ep)


def evaluate_topic_coherence(topic_mat, doc_term_mat, n_top=10, occur_data=None,
                             co_occur_data=None, metric="nmi"):
    """
    compute topic coherence
    Args:
        topic_mat: (np.ndarray) the topic matrix. shape: d x |V|
        doc_term_mat: (np.ndarray) the wo
        n_top: (int) the number of top_words to consider
        occur_data: (np.ndarray)
        co_occur_data: (np.ndarray)
        metric: (str)
    Returns:
        (float)
    """

    cohere = 0
    s = 0

    for k in range(topic_mat.shape[0]):
        top_indices = np.argsort(topic_mat[k])[ -n_top:][::-1]
        s = 0
        for i in range(n_top):
            for j in range(i+1, n_top):
                index_i, index_j = top_indices[i], top_indices[j]
                s += 1
                cohere += compute_coherence(occur_data[index_i], occur_data[index_j],
                                    co_occur_data[index_i, index_j], metric=metric)
    return cohere / (topic_mat.shape[0] * s)


def softmax(x):
    """
    computes the softmax probabilities
    Args:
        x: (np.ndarray) a two-dimensional array. shape: # data x # dim

    Returns:
        (np.ndarray)
    """
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)

    return e_x / np.sum(e_x, axis=1, keepdims=True)


def compute_word_prob(topic_mat_prob, word_index):

    p_word = topic_mat_prob[:, word_index]
    return np.sum(p_word) / topic_mat_prob.shape[0]


def compute_perplexity(topic_mat, doc_term_mat, doc_mat=None):
    """
    computes the perplexity of the documents
    """

    N = 0
    P = 0
    topic_prob = softmax(topic_mat)

    if doc_mat is not None:
        doc_prob = softmax(doc_mat)
        word_mat = np.dot(doc_prob, topic_prob)
    else:
        word_mat = topic_mat

    word_prob = np.mean(word_mat, axis=0)

    ep = 1e-10

    total_count = np.sum(doc_term_mat)
    indiv_word_count = np.sum(doc_term_mat, axis=0)
    #print(doc_term_mat.shape, word_prob.shape)
    total_p = np.sum(np.log(word_prob+ep) * indiv_word_count)
    perplexity = np.exp(-total_p / total_count)
    return perplexity

def jacard_index(list_1, list_2):
    """
    Computers the jacard index: Intersection over Union
    Args:
        list_1: (List[object])
        list_2: (List[object])

    Returns:
        (float)
    """

    set_1 = set(list_1)
    set_2 = set(list_2)

    intersection = set_1.intersection(set_2)
    union = set_1.union(set_2)

    return len(intersection) / len(union)

def cosine_similarity(A, B):
    """
    computes the cosine similarities
    Args:
        A: (np.ndarray) matrix of shape: n x d
        B: (np.ndarray) matrix of shape: n x d

    Returns:
        the average cosine similarity
    """

    norm_A = np.linalg.norm(A, axis=1, keep_dims=True)
    norm_B = np.linalg.norm(B, axis=1, keep_dims=True)

    normalized_A = A / norm_A
    normalized_B = B / norm_B

    similarity = np.mean(np.sum(normalized_A * normalized_B, axis=1))

    return similarity

def compute_diversity(word_list):
    """
    computes word diversity
    Args:
        word_list: List of words

    """

    all_words = []
    for key in word_list:
        for word in word_list[key]:
            all_words.append(word)
    set_words = set(all_words)

    return len(set_words) / len(all_words)