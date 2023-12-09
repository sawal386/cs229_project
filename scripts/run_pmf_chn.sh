#!/bin/bash

source scripts/settings.sh

: '
echo "Running PMF on Chinese social media docs for multiple topic sizes"

python main_train.py \
    -n ${INPUT_CHN_SOCIAL_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_CHN_SOCIAL_DIR} \
    --tune_param \
    --in_chinese

echo "Running PMF on Chinese policy docs for multiple topic sizes"

python main_train.py \
    -n ${INPUT_CHN_POLICY_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_CHN_POLICY_DIR} \
    --tune_param \
    --in_chinese \
    --is_policy
'
echo "Running PMF on Chinese social media docs for ${PMF_CHN_SOC_TOPICS} topics"

python main_train.py \
    -n ${INPUT_CHN_SOCIAL_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_K}\
    -k ${PMF_CHN_SOC_TOPICS} \
    -d ${DICT_CHN_SOCIAL_DIR} \
    --in_chinese

echo "Running PMF on Chinese policy docs for ${PMF_CHN_POL_TOPICS}  topics"

python main_train.py \
    -n ${INPUT_CHN_POLICY_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_K}\
    -k ${PMF_CHN_POL_TOPICS} \
    -d ${DICT_CHN_POLICY_DIR} \
    --in_chinese \
    --is_policy
: '
echo "Running PMF on Chinese state media docs for ${PMF_CHN_STATE_TOPICS} topics"

python main_train.py \
    -n ${INPUT_CHN_STATE_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -k ${PMF_CHN_STATE_TOPICS} \
    -d ${DICT_CHN_STATE_DIR} \
    --tune_param \
    --in_chinese \
    --additional_info state

echo "Running PMF on Chinese state media docs for multiple topic sizes"
python main_train.py \
    -n ${INPUT_CHN_STATE_DIR} \
    -m "pmf" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_CHN_STATE_DIR} \
    --tune_param \
    --in_chinese \
    --additional_info state
'
