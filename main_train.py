# Used in training the model. The topic matrices are saved.

import argparse
import pickle as pkl
import gensim.corpora as corpora

from models import *
from pathlib import Path
from tqdm import tqdm
from metrics import *
from sklearn.model_selection import train_test_split

from util import create_gensim_corpus, create_folder_name

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--input_loc", required=True,
                        help="location of the pkl file containing the "
                             "document term matrix")
    parser.add_argument("-m", "--model", help="name of the model to use",
                        choices=["svd", "lda", "pmf", "etm"], default="lda")
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.001,
                        type=float)
    parser.add_argument("-o", "--output_loc", help="path to the folder"
                                                          "where the output is saved")
    parser.add_argument("-k", "--n_topics", type=int, default=10, required=False)
    parser.add_argument("--tune_param", default=False,
                       action="store_true", help="whether we are finding the hyperpameter or not")
    parser.add_argument("-d", "--dict_loc", required=True,
                        help="location of the dictionary")
    parser.add_argument("-c", "--in_chinese", default=False, action="store_true",
                        help="the input document is chinese text or not")
    parser.add_argument("-p", "--is_policy", default=False, action="store_true",
                        help="the input document is a policy document or not")
    parser.add_argument("-i", "--additional_info", default=None,
                        help="additional text to be included in output folder")

    return parser.parse_args()


def run_single_experiment(X, model_name, num_topics, index2word):
    """
    runs the experiment for a sigle topic size

    Args:
        X: (np.ndarray) the document-term matrix
        model_name: (str) name of the model
        num_topics: (int) the number of topics
        index2word: (dict) (str) word -> (int) index

    Returns:
        (np.ndarray) the topic distribution matrix
    """
    try:
        n_doc, n_vocab = X.shape
    except AttributeError:
        n_doc, n_vocab = len(X), len(index2word)
    if model_name == "pmf":
        model = PoissonMatrixFactorization(n_doc, n_vocab, num_topics,
                                           index2word)

    elif model_name == "lda":
        model = LDA(num_topics, index2word)
        model.fit(X)
    elif model_name == "svd":
        model = SVD(n_doc, n_vocab, num_topics, index2word)
        model.fit(X)
    else:
        raise Exception("{} not valid. Valid models are: lda, svd, etm, pmf")

    return model

def run_multiple_experiments(X, model_name, num_topic_values, save_result=True,
                             output_loc="output", save_name=None, index2word=None):
    """
    runs the model multiple times for different hyperparameters
    Args:
        X: (np.ndarray) the document-term matrix
        model_name: (str) name of the model
        num_topic_values: (List[int]) the number of topics to consider
        output_loc: (str) location to the folder where the output is saved
        save_result: (bool) whether to save the result or not
        save_name: (str) name used in saving results
        index2word: (dict) (int) -> (str)

    Returns:
        (dict) (int) topic number ->
    """
    all_results = {}
    path = Path(output_loc)
    path.mkdir(parents=True, exist_ok=True)
    if save_name is None:
        save_name = model_name
    full_path = path / "{}.pkl".format(save_name)

    if model_name == "lda":
        corpus = create_gensim_corpus(X, index2word)
        id2word = corpora.Dictionary()
        id2word.merge_with(index2word)
        index2word = id2word
        input_corpus = [id2word.doc2bow(text) for text in corpus]
        X = input_corpus

    for k in tqdm(range(len(num_topic_values))):
        #print("Fitting Model for {} topics".format(num_topic_values[k]))
        K = num_topic_values[k]
        fitted_model = run_single_experiment(X, model_name, K, index2word)
        all_results[K] = fitted_model

    with open(full_path, "wb") as f:
        pkl.dump(all_results, f)

    print("Saved trained model to: {}".format(full_path))
    return all_results

if __name__ == "__main__":

    args = parse_arguments()
    with open(args.input_loc, "rb") as f:
        data = pkl.load(f)
    with open(args.dict_loc, "rb") as f:
        dictionary = pkl.load(f)


    rev_dict = {dictionary[i]: i for i in dictionary}
    train_data, valid_data = train_test_split(data, test_size=0.1, random_state=13)
    output_folder = create_folder_name(args.output_loc, args.in_chinese,
                                       args.is_policy)

    if args.additional_info is not None:
        output_folder = output_folder + "_{}".format(args.additional_info)

    name = args.model
    if args.tune_param:
        topics = np.arange(10, 160, 3)
        result = run_multiple_experiments(train_data, args.model, topics,
                                           output_loc=output_folder, save_name=name,
                                           index2word=rev_dict)
    else:
        name = name + "_{}".format(args.n_topics)
        result = run_multiple_experiments(train_data, args.model, [args.n_topics],
                                          save_result=True, output_loc=output_folder,
                                          save_name=name, index2word=rev_dict)