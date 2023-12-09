### This file contains functions for analyzing the results

from metrics import jacard_index, cosine_similarity
import numpy as np
import scipy.spatial.distance as dist
def get_top_words(topic_mat, index_word_map,
                  n_topics=None, top_n=10):
    """
    returns the top words for each topic
    Args:
        topic_mat: (np.ndarray)
        index_word_map: (dict)
        top_n: (int) the number of words to return
        n_topics: (int) the total number of topics under consideration
    Returns:
        (dict) (int) number of topics -> List[str] the top n words
    """

    top_words = {}
    K = topic_mat.shape[0] if n_topics is None else n_topics

    for i in range(K):
        top_k = np.argsort(topic_mat[i])[-top_n:][::-1]
        top_words[i] = [index_word_map[k] for k in top_k]

    return top_words

def compute_topic_embedding_matrix(embedding_dict, top_words_li, embed_dim):
    """
    gets the topic embedding matrix
    Args:
        embedding_dict: (dict) (str) word -> (np.ndarray)
        top_words_li: (List[List[str]]) top words for each topic
        embed_dim: (int) the dimension of the embedding
    Returns:
        np.ndarray
    """

    embedding_array = np.zeros((len(top_words_li), embed_dim))
    for k in top_words_li:
        topic_k_embed = 0
        denom = 0
        for w in top_words_li[k]:
            if w in embedding_dict:
                topic_k_embed += embedding_dict[w]
                denom += 1

        embedding_array[k] = topic_k_embed / denom

    return embedding_array

def compute_similarity(model1, model2, n_words, embedding_dict):
    """
    computes the average cosine similarity between the topics of two models
    Args:
        model1: (model)
        model2: (model)
        n_words: (int) number of words to use
        embedding_dict: (dict) (str) word -> (np.ndarray) the embedding
    Returns:
        (float)
    """

    H1 = model1.get_topic_matrix()
    index_map1 = model1.get_map()
    H2 = model2.get_topic_matrix()
    index_map2 = model2.get_map()

    top_words_1 = get_top_words(H1, index_map1, top_n=n_words)
    top_words_2 = get_top_words(H2, index_map2,  top_n=n_words)


    topic_embed_matrix1 = compute_topic_embedding_matrix(embedding_dict, top_words_1,100)
    topic_embed_matrix2 = compute_topic_embedding_matrix(embedding_dict, top_words_2, 100)

    cosine_similarity = 1 - dist.cdist(topic_embed_matrix1, topic_embed_matrix2, "cosine")

    return np.mean(cosine_similarity)

def topic_word_analysis(model, n_words, embedding_dict):
    """
    analyze the words within the topic
    Args:
        model: (model) the topic model
        n_words: (int) the number of words to consider
        embedding_dict: (dict) (str) word > (np.ndarray) embedding

    Returns:
        (np.ndarray) cosine similarity matrix
    """

    H = model.get_topic_matrix()
    index_map = model.get_map()

    top_words = get_top_words(H, index_map, top_n=n_words*2)
    corr_mat = np.zeros((n_words, n_words))
    for key in top_words:
        i = 0
        j = 0
        embed_mat = np.zeros((n_words, 100))
        while i < n_words:
            word_i = top_words[key][j]
            if word_i in embedding_dict:
                embed_mat[i] += embedding_dict[word_i]
                i += 1
            j += 1
        corr_mat_key = 1 - dist.cdist(embed_mat, embed_mat, "cosine")
        if key == 10000:
            return corr_mat_key
        corr_mat += corr_mat_key
    n = corr_mat.shape[0]
    corr_mat[range(n), range(n)] = 0

    return corr_mat / H.shape[0]
