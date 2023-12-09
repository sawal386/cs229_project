# This file contains functions to perform correlation analysis
# between results from social media posts and policy documents

from analysis import *
from util import *
import argparse
from metrics import *

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", required=True
                        )

    return parser.parse_args()


def find_model_cosine_similarity(model1_dict, model2_dict, n_topic_dict1, n_topic_dict2,
                                 embedding_dict, num_words=10):
    """
    finds the cosine similarity index for the two models
    """
    similarity = {}
    for key in model1_dict:
        print("key:", key)
        model1 = load_pickle(model1_dict[key])[n_topic_dict1[key]]
        model2 = load_pickle(model2_dict[key])[n_topic_dict2[key]]
        similarity[key] = compute_similarity(model1, model2, num_words, embedding_dict)
    return similarity


def compute_model_diversity(model_dict, size_info_dict, num_top, language, doc_type):
    """
    computes the topic model diversity
    Args:
        model_dict: (dict) (str) model name -> (str) location of pkl file containing the model
        size_info_dict: (dict) model_name -> (int) number of topics
        num_top: (int) number of top words to consider
        language: (str) english / chinese
        doc_type: (str) policy / opinion
    Returns:
        (float)
    """

    diversity = {}
    for key in model_dict:
        model = load_pickle(model_dict[key])[size_info_dict[key]]
        H = model.get_topic_matrix()
        index_map = model.get_map()
        top_words = get_top_words(H, index_map, top_n=num_top)
        diversity[key] = compute_diversity(top_words)

    print("For {} {} documents, diversity is: {}".format(language,
                                                         doc_type, diversity))




if __name__ == "__main__":

    args = parse_arguments()
    language = args.language
    if not (language == "english" or language == "chinese"):
        raise ValueError("Language must be english or chinese")
    policy_file_name = "output/train_k_{}_policy".format(language)
    opinion_file_name = "output/train_k_{}_opinion".format(language)

    policy_dict_topic_size, policy_dict_model = dict_representaion(policy_file_name)
    opinion_dict_topic_size, opinion_dict_model = dict_representaion(opinion_file_name)
    embedding_loc = "project_data/embeddings/embed/{}_word_embeddings.pkl".format(language)
    word_embedding_dict = load_pickle(embedding_loc)

    similarity = find_model_cosine_similarity(policy_dict_model, opinion_dict_model,
                                              policy_dict_topic_size,
                       opinion_dict_topic_size, word_embedding_dict)
    print("Similarity for {} documents: {}".format(language, similarity))
    compute_model_diversity(policy_dict_model, policy_dict_topic_size,
                            25, language, "policy")
    compute_model_diversity(opinion_dict_model, opinion_dict_topic_size,
                            25, language, "opinion")

