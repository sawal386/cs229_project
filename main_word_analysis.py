import argparse
import os
import pickle as pkl

from metrics import *
from visualization import *
from util import create_folder_name, export_dict_csv


def get_top_words(topic_mat, index_word_map,
                  n_topics, top_n=15):
    """
    returns the top words for each topic
    Args:
        topic_mat: (np.ndarray)
        index_word_map: (dict)
        top_n: (int) the number of words to return
        n_topics: (int) the number of topics
    Returns:
        (dict) (int) number of topics -> List[str] the top n words
    """

    top_words = {}
    K = topic_mat.shape[0] if n_topics is None else n_topics

    for i in range(K):
        top_k = np.argsort(topic_mat[i])[-top_n:][::-1]
        top_words[i] = [index_word_map[k] for k in top_k]

    return top_words
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


def export_words(model, folder_loc, name, col_key, n_topics=None):
    """
    exports the top words
    Args:
        model:(dict) (int) num_topics > (model)
        folder_loc: (str) location where the output is exported
        name: (str) name of the output_doc
        col_key:(str)
        n_topics: (int)
    """

    for key in model:
        model_k = model[key]
        H = model_k.get_topic_matrix()
        index_map = model_k.get_map()

        top_words = get_top_words(H, index_map, n_topics)
        export_dict_csv(top_words, name, col_key, folder_loc)

if __name__ == "__main__":

    args = parse_arguments()
    model_folder = create_folder_name(args.model_loc, args.in_chinese,
                                      args.is_policy)
    output_folder = create_folder_name(args.output_loc, args.in_chinese,
                                       args.is_policy)
    for files in os.listdir(model_folder):
        if "pkl" in files:
            full_path = Path(model_folder) / files
            with open(full_path, "rb") as f:
                model_results = pkl.load(f)
                model_name = files.split(".")[0]
                file_name = "top_words_{}".format(model_name)
                export_words(model_results,output_folder , file_name, "Topic",
                             n_topics=25)
