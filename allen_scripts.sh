#!/usr/bin/env bash

rm -rf models/*
cd src/

echo ""
echo "==============================="
echo "======= BEGIN TRAINING ========"
echo "==============================="
echo ""

allennlp train ../configs/demo.jsonnet -s ../models/tmp
# --include-package data --include-package model

echo ""
echo "====================================="
echo "======= EVALUATE ON TEST SET ========"
echo "====================================="
echo ""

allennlp evaluate ../models/tmp ../data/test.tsv
# --include-package data --include-package model

echo ""
echo "============================================"
echo "======= PREDICT ON CUSTOM SENTENCES ========"
echo "============================================"
echo ""

allennlp predict ../models/tmp ../data/small_test.json --predictor sentence_classifier
# --include-package data --include-package model --include-package predictor
