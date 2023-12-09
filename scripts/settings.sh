BASE_ENG_DIR="/Users/sawal/Desktop/cs229_data/us_data_new"

export MODEL_OUTPUT_PATH_ALL="output_train/train_all"
export MODEL_OUTPUT_PATH_K="output_25/train_k"
export REPORT_OUTPUT_PATH="output/report_figures"
export WORD_OUTPUT_PATH="output/report_words"

export INPUT_ENG_POLICY_DIR="${BASE_ENG_DIR}/policy_train_mat.pkl"
export DICT_ENG_POLICY_DIR="${BASE_ENG_DIR}/policy_dictionary.pkl"
export INPUT_CHN_POLICY_DIR="project_data/policy_from_NDRC/NDRC_policy_matrix.pkl"
export DICT_CHN_POLICY_DIR="project_data/policy_from_NDRC/NDRC_policy_vocab.pkl"

export INPUT_ENG_SOCIAL_DIR="${BASE_ENG_DIR}/tweets_train_mat.pkl"
export DICT_ENG_SOCIAL_DIR="${BASE_ENG_DIR}/tweets_dictionary.pkl"
export INPUT_CHN_SOCIAL_DIR="project_data/china_social_posts_weibo/social_weibo_matrix.pkl"
export DICT_CHN_SOCIAL_DIR="project_data/china_social_posts_weibo/social_weibo_vocab.pkl"

export INPUT_CHN_STATE_DIR="project_data/china_social_posts_weibo/social_weibo_matrix.pkl"
export DICT_CHN_STATE_DIR="project_data/china_social_posts_weibo/social_weibo_vocab.pkl"

export EPOCHS=50
export LR=0.001

export SVD_ENG_SOC_TOPICS=25
export LDA_ENG_SOC_TOPICS=25
export PMF_ENG_SOC_TOPICS=25

export SVD_CHN_SOC_TOPICS=25
export LDA_CHN_SOC_TOPICS=25
export PMF_CHN_SOC_TOPICS=25

export SVD_ENG_POL_TOPICS=25
export LDA_ENG_POL_TOPICS=25
export PMF_ENG_POL_TOPICS=25

export SVD_CHN_POL_TOPICS=25
export LDA_CHN_POL_TOPICS=25
export PMF_CHN_POL_TOPICS=25

export SVD_CHN_STATE_TOPICS=20
export LDA_CHN_STATE_TOPICS=20
export PMF_CHN_STATE_TOPICS=20
