#!/bin/bash

#Scripts for generating top words for topics.
source scripts/settings.sh

echo "Analyzing words for social media documents in English"
python main_word_analysis.py \
    -n ${INPUT_ENG_SOCIAL_DIR} \
    -o ${WORD_OUTPUT_PATH}\
    -m ${MODEL_OUTPUT_PATH_K} \
    -d ${DICT_ENG_SOCIAL_DIR}
    
echo "Analyzing words for policy documents in English"
python main_word_analysis.py \
    -n ${INPUT_ENG_POLICY_DIR} \
    -o ${WORD_OUTPUT_PATH}\
    -m ${MODEL_OUTPUT_PATH_K} \
    -d ${DICT_ENG_POLICY_DIR}\
    --is_policy
    
echo "Analyzing words figures for social media documents in Chinese"
python main_word_analysis.py \
    -n ${INPUT_CHN_SOCIAL_DIR} \
    -o ${WORD_OUTPUT_PATH}\
    -m ${MODEL_OUTPUT_PATH_K} \
    -d ${DICT_CHN_SOCIAL_DIR} \
    --in_chinese

echo "Analyzing words figures for policy documents in Chinese"
python main_word_analysis.py \
    -n ${INPUT_CHN_POLICY_DIR} \
    -o ${WORD_OUTPUT_PATH}\
    -m ${MODEL_OUTPUT_PATH_K} \
    -d ${DICT_CHN_POLICY_DIR} \
    --in_chinese \
    --is_policy
