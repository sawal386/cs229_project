#!/bin/bash

source scripts/settings.sh
: '
echo "Running LDA on Chinese social media docs for multiple topic sizes"

python main_train.py \
    -n ${INPUT_CHN_SOCIAL_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_CHN_SOCIAL_DIR} \
    --tune_param \
    --in_chinese

echo "Running LDA on Chinese policy docs for multiple topic sizes"

python main_train.py \
    -n ${INPUT_CHN_POLICY_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_CHN_POLICY_DIR} \
    --tune_param \
    --in_chinese \
    --is_policy
'
echo "Running LDA on Chinese social media docs for ${LDA_CHN_SOC_TOPICS} topics"

python main_train.py \
    -n ${INPUT_CHN_SOCIAL_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_K}\
    -k ${LDA_CHN_SOC_TOPICS} \
    -d ${DICT_CHN_SOCIAL_DIR} \
    --in_chinese

echo "Running LDA on Chinese policy docs for ${LDA_CHN_POL_TOPICS}  topics"

python main_train.py \
    -n ${INPUT_CHN_POLICY_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_K}\
    -k ${LDA_CHN_POL_TOPICS} \
    -d ${DICT_CHN_POLICY_DIR} \
    --in_chinese \
    --is_policy
: '
echo "Running LDA on Chinese state media docs for ${LDA_CHN_STATE_TOPICS} topics"

python main_train.py \
    -n ${INPUT_CHN_STATE_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -k ${LDA_CHN_STATE_TOPICS} \
    -d ${DICT_CHN_STATE_DIR} \
    --tune_param \
    --in_chinese \
    --additional_info state

    
echo "Running LDA on Chinese state media docs for multiple topic sizes"
python main_train.py \
    -n ${INPUT_CHN_STATE_DIR} \
    -m "lda" \
    -o ${MODEL_OUTPUT_PATH_ALL}\
    -d ${DICT_CHN_STATE_DIR} \
    --tune_param \
    --in_chinese \
    --is_policy \
    --additional_info state
'
