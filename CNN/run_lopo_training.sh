#!/bin/bash

SUBJECTS=("SN001" "SN003" "SN004" "SN007" "SN009" "SN010" "SN013" "SN025" "SN027")

for SUBJECT in "${SUBJECTS[@]}"
do
    echo "Starte LOPO Training für Test-Subject: $SUBJECT"

    LOPO_TEST_SUBJECT=$SUBJECT CUDA_VISIBLE_DEVICES=0 python train.py

    echo "Fertig mit $SUBJECT"
done
