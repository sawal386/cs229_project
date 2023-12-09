#!/bin/bash

source scripts/settings.sh

echo "Running PMF on English social media docs for multiple topic sizes"
: '
python main_train.py \
    -n ${INPUT_ENG_SOCIAL_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_ENG_SOCIAL_DIR} \
    --tune_param

echo "Running PMF on English policy docs for multiple topic sizes"

python main_train.py \
    -n ${INPUT_ENG_POLICY_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_ENG_POLICY_DIR} \
    --tune_param \
    --is_policy
'
echo "Running PMF on English social media docs for ${PMF_ENG_SOC_TOPICS} topics"

python main_train.py \
    -n ${INPUT_ENG_SOCIAL_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_K}\
    -k ${PMF_ENG_SOC_TOPICS} \
    -d ${DICT_ENG_SOCIAL_DIR}

echo "Running PMF on English policy docs for ${PMF_ENG_POL_TOPICS}  topics"

python main_train.py \
    -n ${INPUT_ENG_POLICY_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_K}\
    -k ${PMF_ENG_POL_TOPICS} \
    -d ${DICT_ENG_POLICY_DIR} \
    --is_policy



