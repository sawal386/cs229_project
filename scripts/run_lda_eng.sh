#!/bin/bash

source scripts/settings.sh
: '
echo "Running LDA on English social media docs for multiple topic sizes"

python main_train.py \
    -n ${INPUT_ENG_SOCIAL_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_ENG_SOCIAL_DIR} \
    --tune_param

echo "Running LDA on English policy docs for multiple topic sizes"

python main_train.py \
    -n ${INPUT_ENG_POLICY_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_ENG_POLICY_DIR} \
    --tune_param \
    --is_policy
'
echo "Running LDA on English social media docs for ${LDA_ENG_SOC_TOPICS} topics"

python main_train.py \
    -n ${INPUT_ENG_SOCIAL_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_K}\
    -k ${LDA_ENG_SOC_TOPICS} \
    -d ${DICT_ENG_SOCIAL_DIR}

echo "Running LDA on English policy docs for ${LDA_ENG_POL_TOPICS}  topics"

python main_train.py \
    -n ${INPUT_ENG_POLICY_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_K}\
    -k ${LDA_ENG_POL_TOPICS} \
    -d ${DICT_ENG_POLICY_DIR} \
    --is_policy


