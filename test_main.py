# Used for running the model on the test dataset, and producing
# figures for the paper

import argparse
import os
import pickle as pkl
import torch

from sklearn.model_selection import train_test_split
from metrics import *
from visualization import *
from util import create_folder_name, save_pkl
from analysis import get_top_words
from tqdm import tqdm
def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model_loc", required=True,
                        help="location folder containing the models")
    parser.add_argument("-o", "--output_loc", help="path to the folder"
                                            "where the output is saved")
    parser.add_argument("-d", "--dict_loc", required=True,
                        help="location of the dictionary")
    parser.add_argument("-n", "--input_loc", required=True,
                        help="location of the pkl file containing the "
                             "document term matrix")
    parser.add_argument("-c", "--in_chinese", default=False, action="store_true",
                        help="the input document is chinese text or not")
    parser.add_argument("-p", "--is_policy", default=False, action="store_true",
                        help="the input document is a policy document or not")
    return parser.parse_args()


def calculate_model_perplexity(dict_results, dt_matrix, model_name):
    """

    Args:
        dict_results: (dict) (int) Number of topics ->  (np.ndarray) the topic matrix
        dt_matrix: (np.ndarray) the document-term matrix
    Returns:
        (nd.array) perplexity scores
    """

    value = np.ndarray(len(dict_results))
    i = 0
    for key in sorted(dict_results.keys()):
        #print(key)
        model = dict_results[key]
        H = model.get_topic_matrix()
        W = model.get_document_matrix()
        if torch.is_tensor(H):
            H = H.to("cpu").numpy()
            W = W.to("cpu").numpy()
        if model_name == "lda":

            value[i] = compute_perplexity(H, dt_matrix, None)
        else:
            value[i] = compute_perplexity(H, dt_matrix, W)
        i += 1


    return value

def calculate_model_coherence(dict_results, dt_matrix, occur_data, co_occur_data,
                              metric_type):
    """
    Args:
        dict_results: (dict) (int) Number of topics ->  (np.ndarray) the model
        dt_matrix: (np.ndarray) the document-term matrix
        occur_data: (np.ndarray)
        co_occur_data: (np.ndarray)
        metric_type: (str)
    Returns:
        (nd.array) perplexity scores
    """
    value = np.ndarray(len(dict_results))
    i = 0
    for key in tqdm(sorted(dict_results.keys())):
        print(key)
        H = dict_results[key].get_topic_matrix()
        if torch.is_tensor(H):
            H = H.to("cpu").numpy()
        value[i] = evaluate_topic_coherence(H, dt_matrix, occur_data=occur_data,
                                            co_occur_data=co_occur_data, metric=metric_type)
        i += 1

    return value

def calculate_model_divergence(dict_results):
    """
    calculates the divergence of topics in the models. We use the divergence definition
    provided in the embedding models paper.

    Args:
        dict_results: (dict) (str) model name -> (model)

    Returns:
        (float)
    """
    value = np.ndarray(len(dict_results))
    i = 0
    for key in sorted(dict_results.keys()):
        model = dict_results[key]
        H = model.get_topic_matrix()
        index_map = model.get_map()
        top_words = get_top_words(H, index_map, top_n=10)
        value[i] = compute_diversity(top_words)
        i += 1

    return value

if __name__ == "__main__":

    args = parse_arguments()
    with open(args.input_loc, "rb") as f:
        data = pkl.load(f)
    with open(args.dict_loc, "rb") as f:
        dictionary = pkl.load(f)

    rev_dict = {dictionary[i]: i for i in dictionary}
    train_data, valid_data = train_test_split(data, test_size=0.1, random_state=13)

    perplexity_models = {}
    coherence_nmi_models = {}
    coherence_pmi_models = {}
    coherence_umass_models = {}
    diversity_models = {}
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "cpu" #"mps"
    train_data_torch = torch.from_numpy(train_data)
    x_array = None

    one_mat = np.where(train_data > 0, 1.0, 0.0)
    one_freq_mat = np.sum(one_mat, axis=0)
    one_co_mat = np.dot(one_mat.T, one_mat)
    one_co_mat_p = one_co_mat / (2 * (np.sum(one_co_mat) - np.sum(np.diagonal(one_co_mat))))
    one_freq_mat_p = np.sum(one_mat, axis=0) / one_mat.shape[0]

    model_folder = create_folder_name(args.model_loc, args.in_chinese,
                                      args.is_policy)
    for files in os.listdir(model_folder):
        smaller_model = {}
        if "pkl" in files:
            full_path = Path(model_folder) / files
            with open(full_path, "rb") as f:
                model_results = pkl.load(f)

            name = files.split(".")[0]

            perplexity_models[name] = calculate_model_perplexity(model_results,
                                                                valid_data, name)
            coherence_nmi_models[name] = calculate_model_coherence(model_results, train_data,
                                                               one_freq_mat_p, one_co_mat_p,
                                                               "nmi")
            coherence_pmi_models[name] = calculate_model_coherence(model_results, train_data,
                                                                   one_freq_mat_p, one_co_mat_p,
                                                                   "pmi")
            coherence_umass_models[name] = calculate_model_coherence(model_results, train_data,
                                                                   one_freq_mat, one_co_mat,
                                                                   "umass")
            diversity_models[name] = calculate_model_divergence(model_results)
            if x_array is None:
                x_array = np.array([k for k in sorted(model_results.keys())])

    x_dict = {k: x_array for k in sorted(perplexity_models.keys())}
    add_name = create_folder_name("_", args.in_chinese, args.is_policy)

    output_folder = create_folder_name("output/test_output", args.in_chinese,
                                       args.is_policy)
    save_pkl(output_folder, "perplexity.pkl", perplexity_models)
    save_pkl(output_folder, "coherence_nmi.pkl", coherence_nmi_models)
    save_pkl(output_folder, "coherence_umass.pkl", coherence_umass_models)
    save_pkl(output_folder, "coherence_pmi.pkl", coherence_pmi_models)
    save_pkl(output_folder, "topic_size.pkl",  x_dict)
    save_pkl(output_folder, "diversity.pkl", diversity_models)


    simple_plot(x_dict, coherence_pmi_models,"# topics", "Coherence", save=True,
                save_path="figures", save_name="coherence_pmi{}".format(add_name),
                legend_on=True, use_log=False)
    simple_plot(x_dict, coherence_nmi_models, "# topics", "Coherence", save=True,
                save_path="figures", save_name="coherence_nmi{}".format(add_name),
                legend_on=True, use_log=False)
    simple_plot(x_dict, coherence_umass_models, "# topics", "Coherence", save=True,
                save_path="figures", save_name="coherence_umass{}".format(add_name),
                legend_on=True, use_log=False)

    simple_plot(x_dict, perplexity_models,"# topics", "Perplexity", save=True,
                save_path="figures", save_name="perplexity{}".format(add_name),
                legend_on=True, use_log=False)
    simple_plot(x_dict, diversity_models, "# topics", "Topic Diversity", save=True,
                save_path="figures", save_name="diversity{}".format(add_name),
                legend_on=True, use_log=False)
