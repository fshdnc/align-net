#!/usr/bin/env python3

"""
The source and target sentences contain duplicates.
This script deduplicates the sentences and generates
the dictionary containing the positive instances
after deduplication.

Original positive dictionary
"old_index": [old_indices]
     |             |
    text         texts
     |             |
"new index"  "new indices"
"""

import json
import argparse

def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--source-file", help="File containing the source sentences, one in a line", required=True)
    argparser.add_argument("--target-file", help="File containing the target sentences, one in a line", required=True)
    argparser.add_argument("--positive-dictionary", help="Json file of the dictionary storing the list of positives", required=True)
    args = argparser.parse_args()
    return args

def read(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    return data

def create_mapping_dict(sentences):
    d = {}
    for i, text in enumerate(sentences):
        if text in d:
            d[text].append(i)
        else:
            d[text] = [i]
    print(len(d))
    return d

def main():
    args = parse_arguments()

    # read in sentences from both files
    sentences_src = read(args.source_file)
    sentences_trg = read(args.target_file)

    # create dictionaries for the sentence files
    #src_mapping_dict = create_mapping_dict(sentences_src) # text -> list of indices
    #trg_mapping_dict = create_mapping_dict(sentences_trg)
    src_old_id_to_text = {i: t for i, t in enumerate(sentences_src)}
    trg_old_id_to_text = {i: t for i, t in enumerate(sentences_trg)}

    dedup_src = list(set(sentences_src))
    dedup_trg = list(set(sentences_trg))

    with open(args.source_file+".dedup", "w") as f:
        f.write("\n".join(dedup_src))

    with open(args.target_file+".dedup", "w") as f:
        f.write("\n".join(dedup_trg))

    src_text_to_newid = {t: i for i, t in enumerate(dedup_src)}
    trg_text_to_newid = {t: i for i, t in enumerate(dedup_trg)}

    # now dedup the dictionary as well
    with open(args.positive_dictionary, "r") as f:
        pos_dict = json.load(f)

    pos_dict_trg_src = {} # trg index -> [lst of src indices]
    for i, idx_lst in pos_dict.items():
        newid = trg_text_to_newid[trg_old_id_to_text[int(i)]]
        new_idx_lst = list(set([src_text_to_newid[src_old_id_to_text[idx]] for idx in idx_lst]))
        if newid in pos_dict_trg_src:
            try:
                assert set(pos_dict_trg_src[newid]) == set(new_idx_lst) # if a sentence appears twice, its positive list should be identical
            except AssertionError:
                print("non-identical!")
                print(new_idx_lst)
                print(pos_dict_trg_src[newid])
        else:
            pos_dict_trg_src[newid] = new_idx_lst

    with open("dedup_trg_src_"+args.positive_dictionary, "w") as f:
        json.dump(pos_dict_trg_src, f)
        
    # print out some results to check
    count = 0
    for i, lst in pos_dict_trg_src.items():
        count += 1
        print("Query:", dedup_trg[i])
        for idx in lst:
            print("\t", dedup_src[idx])
        if count ==10:
            break

    # now for the other direction
    pos_dict_src_trg = {} # src index -> [lst of trg indices]
    for i, idx_lst in pos_dict.items():
        newid = src_text_to_newid[src_old_id_to_text[int(i)]]
        new_idx_lst = list(set([trg_text_to_newid[trg_old_id_to_text[idx]] for idx in idx_lst]))
        if newid in pos_dict_src_trg:
            assert set(pos_dict_src_trg[newid]) == set(new_idx_lst) # if a sentence appears twice, its positive list should be identical
        else:
            pos_dict_src_trg[newid] = new_idx_lst

    with open("dedup_src_trg_"+args.positive_dictionary, "w") as f:
        json.dump(pos_dict_src_trg, f)
        
    return 0

if __name__=="__main__":
    exit(main())
