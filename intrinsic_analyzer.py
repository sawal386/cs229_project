## This file performs intrinsic analysis i.e. it looks at the embeddings of the words in the topics

from analysis import *
from util import *
import matplotlib.pyplot as plt
from visualization import draw_heatmap

def topic_analyzer(model_dict, size_info_dict, num_word, embedding_dict):
    """
    analyzes the topics of topic models
    model_dict: (dict) (str) model name -> (str) location of the model saved as pkl file
    size_info_dict: (dict) (str) model name -> (int) number of topics used in the model
    num_top: (int) number of words under consideration
    embedding_dict: (dict) (str) word <- (np.ndarray) embeddings
    """
    within_topic_score = {}
    for key in model_dict:

        model = load_pickle(model_dict[key])[size_info_dict[key]]
        mat = topic_word_analysis(model, num_word, embedding_dict)
        within_topic_score[key] = mat

    return within_topic_score


if __name__ =="__main__":

    eng_folder = "output/train_k_english_opinion"
    chn_folder = "output/train_k_chinese_opinion"

    eng_dict_topic_size, eng_dict_model = dict_representaion(eng_folder)
    chn_dict_topic_size, chn_dict_model = dict_representaion(chn_folder)
    embedding_loc_chn = "project_data/embeddings/embed/chinese_word_embeddings.pkl"
    embedding_loc_eng = "project_data/embeddings/embed/english_word_embeddings.pkl"

    embedding_chn = load_pickle(embedding_loc_chn)
    embedding_eng = load_pickle(embedding_loc_eng)

    eng_topic_analysis = topic_analyzer(eng_dict_model, eng_dict_topic_size,
                                        10, embedding_eng)
    chn_topic_analysis = topic_analyzer(chn_dict_model, chn_dict_topic_size,
                                        10, embedding_chn)


    data = {"eng_pmf":eng_topic_analysis["pmf"], "eng_lda":eng_topic_analysis["lda"],
            "chn_pmf":chn_topic_analysis["pmf"], "chn_lda":chn_topic_analysis["lda"]}
    label = {"eng_pmf": "English PMF", "eng_lda":"English LDA",
             "chn_pmf":"Chinese PMF", "chn_lda":"Chinese LDA"}
    fig = plt.figure(figsize=(16, 4))
    axes_all = {"eng_pmf":fig.add_subplot(1, 4, 1), "eng_lda":fig.add_subplot(1, 4, 2),
                "chn_pmf":fig.add_subplot(1, 4, 3), "chn_lda":fig.add_subplot(1, 4, 4)}

    draw_heatmap(data, label, axes_all,  legend_lab="Cosine Similarity")
    fig.savefig("heatmap.pdf", bbox_inches="tight")
    #plt.show()


