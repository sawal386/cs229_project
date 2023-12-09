# This file is used to run topic models

from models import *
import numpy as np
import pickle as pkl
from metrics import matrix_accuracy
from visualization import visualize_embeddings
from analysis import clusterize_embeddings
from util import *

def run_nmf(X, d=50, min_dim=20, max_dim=50):
    """
    runs non-negative matrix factorization
    Args:
        X: (np.ndarray) the document-term matrix. shape: (#documents, #words)
        d: (int) the dimension for the latent factors
        min_dim: (np.ndarray) the minimum dimension of the embedding
        max_dim: (np.ndarray)
    """
    print("Running Non-Negative Matrix Factorization")
    model = NmfSk(X.shape[0], X.shape[1], d)
    model.fit(X)

    return model

def run_LDA(X, d=15):
    """
    runs Latent Dirichlet Allocation
    Args:
        X: (np.ndarray) the document-term matrix. shape: (#documents, #words)
        d: (int) the dimension for the latent factors; the number of topics
    """

    print("Running LDA")
    model = LDA(d)
    model.fit(X)

    return model

def analyze_model(model, index_word_map, folder_loc, name, col_key):
    """
    analyzes the model
    Args:
        model:
        folder_loc:
        name:
        col_key:
        index_word_map: (dict)
    """

    top_words_index = model.get_top_words()
    top_words_topic = get_word_array(top_words_index, index_word_map)
    export_dict_csv(top_words_topic, name, col_key, folder_loc)

if __name__ == "__main__":

    # full path name document-term matrix stores as pkl file
    data_mat_file_name = "sample_data/US_words_train_mat.pkl"
    # full path name dictionary that maps word to indices
    index_word_file_name = "sample_data/US_words_dictionary.pkl"
    with open(data_mat_file_name, "rb") as f:
        data = pkl.load(f)
    with open(index_word_file_name, "rb") as f:
        word_index_dict = pkl.load(f)
    print(word_index_dict)
    print("data:", data.shape)
    index_word_dict = {word_index_dict[w]: w for w in word_index_dict}


    nmf = run_nmf(data)
    embed_nmf = nmf.get_word_embedding()
    print(embed_nmf)
    #visualize_embeddings(embed_nmf.T, save_name="nmf_embedding",
     #                    title="NMF embedding")
    clusters = clusterize_embeddings(embed_nmf.T, index_word_dict)
    export_dict_csv(clusters, "nmf_embed_clusters", "Cluster")
    analyze_model(nmf, index_word_dict, "analysis", "nmf_top_words_eng", "Topic")

    lda = run_LDA(data)
    analyze_model(lda, index_word_dict, "analysis", "lda_top_words_eng", "Topic")










