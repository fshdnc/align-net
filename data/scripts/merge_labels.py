#!/usr/bin/env python3

"""
Merge two positive dictionaries into one
(one src->trg the other trg->src)
"""

import json
import argparse

def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--source-dictionary", help="json file containing the dictionary generated using src->trg", required=True)
    argparser.add_argument("--target-dictionary", help="json file containing the dictionary generated using src->trg", required=True)
    argparser.add_argument("--merged-dictionary", help="Name of the json file to store the list of positives", required=True)
    args = argparser.parse_args()
    return args

def main():
    args = parse_arguments()

    with open(args.source_dictionary, "r") as f:
        src_dict = json.load(f)

    with open(args.target_dictionary, "r") as f:
        trg_dict = json.load(f)

    assert len(src_dict)==len(trg_dict)

    pos_dict = {k: list(set(v+trg_dict[k])) for k, v in src_dict.items()}

    with open(args.merged_dictionary, "w") as f:
        json.dump(pos_dict, f)

if __name__=="__main__":
    exit(main())
