#!/usr/bin/env python3

"""
Random baseline for the dev and test sets of the tatoeba corpus (eng-fin)
"""

import sys
import json
import numpy as np

def random_baseline(pos_dict, src_len, trg_len):
    numerator = sum([len(v_lst) for v_lst in pos_dict.values()])
    denominator = src_len*trg_len
    return numerator/denominator

def load_and_eval(pos_dict_path, src_path, trg_path):
    with open(pos_dict_path, "r") as f:
        pos_dict = json.load(f)
    with open(src_path, "r") as f:
        src_sentences = f.readlines()
    #src_sentences = [line.strip() for line in src_sentences]
    with open(trg_path, "r") as f:
        trg_sentences = f.readlines()
    #trg_sentences = [line.strip() for line in trg_sentences]
    acc = random_baseline(pos_dict, len(src_sentences), len(trg_sentences))
    return acc

def main():
    dev_acc = load_and_eval("data/positives/dedup_src_trg_dev-positives.json",
                            "data/eng-fin/dev.src.dedup",
                            "data/eng-fin/dev.trg.dedup")
    print("Random baseline for dev set:", dev_acc)
    test_acc = load_and_eval("data/positives/dedup_src_trg_test-positives.json",
                             "data/eng-fin/test.src.dedup",
                             "data/eng-fin/test.trg.dedup")
    print("Random baseline for test set:", test_acc)
    wmt_acc = load_and_eval("data/wmt/dedup_src_trg_wmt-positives.json",
                            "data/wmt/wmt-en.txt.dedup",
                            "data/wmt/wmt-fi.txt.dedup")
    print("Random baseline for wmt 2015-2018 test set:", wmt_acc)
    return 0
    
if __name__=="__main__":
    exit(main())

