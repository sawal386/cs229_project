#!/bin/bash

source scripts/settings.sh

echo "Producing figures for documents in English"
python test_main.py \
    -n ${INPUT_ENG_SOCIAL_DIR} \
    -o ${REPORT_OUTPUT_PATH}\
    -m ${MODEL_OUTPUT_PATH_ALL} \
    -d ${DICT_ENG_SOCIAL_DIR}


echo "Producing figures for social media documents in Chinese"
python test_main.py \
    -n ${INPUT_CHN_SOCIAL_DIR} \
    -o ${REPORT_OUTPUT_PATH}\
    -m ${MODEL_OUTPUT_PATH_ALL} \
    -d ${DICT_CHN_SOCIAL_DIR} \
    --in_chinese


echo "Producing figures for policy documents in Chinese"
python test_main.py \
    -n ${INPUT_CHN_POLICY_DIR} \
    -o ${REPORT_OUTPUT_PATH}\
    -m ${MODEL_OUTPUT_PATH_ALL} \
    -d ${DICT_CHN_POLICY_DIR} \
    --in_chinese \
    --is_policy

echo "Producing figures for policy documents in English"
python test_main.py \
    -n ${INPUT_ENG_POLICY_DIR} \
    -o ${REPORT_OUTPUT_PATH}\
    -m ${MODEL_OUTPUT_PATH_ALL} \
    -d ${DICT_ENG_POLICY_DIR} \
    --is_policy

