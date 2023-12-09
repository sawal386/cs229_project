### This file contains helper functions

import json
from pathlib import Path
import pandas as pd
import numpy as np
import pickle as pkl
import os

def export_json(dict_, output_name, output_folder="analysis"):
    """
    exports a dictionary to json file
    Args:
        dict_: (dict) the dictionary we want to export
        output_name: (str) the output file name
        output_folder: (str) the path where the json is saved is saved
    """

    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    if ".json" not in output_name:
        output_name = output_name + ".json"
    full_path = path / output_name

    with open(full_path, "w") as f:
        json.dump(dict_, f)

    print("File exported to: {}".format(full_path))


def export_dict_csv(dict_, output_name, col_head="", output_folder="analysis"):
    """
    exports a dictionary of lists to csv file
    Args:
        dict_: (dict) the dictionary we want to export
        col_head: (str) string to add to the keys that make up the column heading
        output_name: (str) the output file name
        output_folder: (str) the path where the csv file is saved
    """

    new_dict = {}
    for k in sorted(dict_.keys()):
        new_dict[col_head + " " + str(k)] = dict_[k]
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in new_dict.items()]))
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    if ".csv" not in output_name:
        output_name = output_name + ".csv"
    full_path = path / output_name
    df.to_csv(full_path)
    print("File exported to: {}".format(full_path))

def get_word_array(index_array, index_word_map, key="topic"):
    """
    get the words corresponding to the indices
    """
    dict_ = {}
    for i in range(index_array.shape[0]):
        k = i + 1
        dict_[k] = []
        for j in range(index_array.shape[1]):
            dict_[k].append(index_word_map[index_array[i, j]])

    return dict_

def convert_json_document_matrix(file_loc):
    """
    converts document matrix stored as json to numpy array
    Args:
        file_loc: (str) location of the json file
    """

    try:
        with open(file_loc, "r") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("File not Found. Please enter correct location")

    matrix = np.asarray(raw_data["contents"])

    return matrix

def create_gensim_corpus(data, rev_dict):
    """
    Returns: (List[List])
    """
    corpus = []
    print("Creating Corpus. This may take a while.")
    for i in range(data.shape[0]):
        doc = []
        for j in range(data.shape[1]):
            doc = doc + [rev_dict[j]] * int(data[i][j])
        corpus.append(doc)

    return corpus


def create_folder_name(base_name, is_chinese, is_policy):
    """
    create a name for a give folder
    Args:
        base_name: (str) the base name
        is_chinese: (bool) whether the data is Chinese text or not
        is_policy: (bool) whether the data is policy doc text or not
    Returns:
        (str)
    """
    folder = base_name
    if is_chinese:
        folder = folder + "_chinese"
    else:
        folder = folder + "_english"

    if is_policy:
        folder = folder + "_policy"
    else:
        folder = folder + "_opinion"

    return folder


def save_pkl(path, name, data):
    """
    saves the data as a pkl file
    Args:
        path: (str) the path
        name: (str) name of the output file
        data: (object)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if ".pkl" not in name:
        name = name + ".pkl"
    full_path = path / name
    with open(full_path, "wb") as f:
        pkl.dump(data, f)

    print("Saved trained model to: {}".format(full_path))

def load_pickle(full_path):
    """
    loads the pickle file
    Args:
        full_path: (str) the full path to the file

    Returns:
        (object)
    """

    with open(full_path, "rb") as f:
        data = pkl.load(f)

    return data

def dict_representaion(file_loc):
    """
    stores the file names as dictionary
    Args:
        file_loc: (str)

    Returns:
        (dict) (str) -> (str), (dict) (str) -> (int)
    """
    topic_dict = {}
    model_dict = {}
    for file in os.listdir(file_loc):
        name, n_topic = file[:-4].split("_")
        topic_dict[name] = int(n_topic)
        model_dict[name] = file_loc + "/" + file

    return topic_dict, model_dict