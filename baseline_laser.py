#!/usr/bin/env python3

"""
Laser baseline for the dev and test sets of the tatoeba corpus (eng-fin)
"""

import sys
import json
import numpy as np
from laserembeddings import Laser
import torch
from evaluation import eval_tatoeba_retrieval
import sklearn.metrics

def retrieve_most_similar(vectors1, vectors2):
    """
    Code modified from: https://github.com/TurkuNLP/Deep_Learning_in_LangTech_course/blob/master/laser.ipynb
    Given two vectors, return the most similar for every item in the first vector
    """
    #assert len(vectors1)==len(vectors2)
    all_dist = sklearn.metrics.pairwise_distances(vectors1, vectors2)
    nearest = all_dist.argmin(axis=-1)
    return nearest

def load(pos_dict_path, src_path, trg_path):
    with open(pos_dict_path, "r") as f:
        pos_dict = json.load(f)
    with open(src_path, "r") as f:
        src_sentences = f.readlines()
    src_sentences = [line.strip() for line in src_sentences]
    with open(trg_path, "r") as f:
        trg_sentences = f.readlines()
    trg_sentences = [line.strip() for line in trg_sentences]
    return pos_dict, src_sentences, trg_sentences

def embed(laser, src, trg):
    src_embedded = laser.embed_sentences(src, "en")
    trg_embedded = laser.embed_sentences(trg, "fi")
    return src_embedded, trg_embedded

def main():
    laser = Laser()
    # dev
    pos, src, trg = load("data/positives/dedup_src_trg_dev-positives.json",
                                  "data/eng-fin/dev.src.dedup",
                                  "data/eng-fin/dev.trg.dedup")
    src_dev, trg_dev = embed(laser, src, trg)
    selected_indices = retrieve_most_similar(src_dev, trg_dev)
    acc = eval_tatoeba_retrieval(selected_indices, pos)
    print("Laser for dev set:", acc)

    # test
    pos, src, trg = load("data/positives/dedup_src_trg_test-positives.json",
                                  "data/eng-fin/test.src.dedup",
                                  "data/eng-fin/test.trg.dedup")
    src_test, trg_test = embed(laser, src, trg)
    selected_indices = retrieve_most_similar(src_test, trg_test)
    acc = eval_tatoeba_retrieval(selected_indices, pos)
    print("Laser for test set:", acc)

    return 0

if __name__=="__main__":
    exit(main())

