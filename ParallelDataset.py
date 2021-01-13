#!/usr/bin/env python3

import torch
from typing import List, Dict , Generator
from torch.utils.data import Dataset, IterableDataset
import gzip
from itertools import cycle, islice

class ParallelDataset(Dataset):
    """
    """
    def __init__(self,
                 examples: List[Dict]):
        """
        Input: list of dict with keys "sentences": {"src", "trg"}, and "label"
        """
        self.examples = examples

    def __getitem__(self, item):
        label = self.examples[item]["label"]
        return self.examples[item]["sentences"], label

    def __len__(self):
        return len(self.examples)

    
class HugeParallelDataset(IterableDataset):
    """
    For the tatoeba training set
    Code taken from: https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """
    def __init__(self,
                 filepath: str
                 ):
        """
        Input: generator of dict with keys "sentences": {"src", "trg"}, and "label"
        """
        self.filepath = filepath
        
    def parse_file(self, filepath):
        with gzip.open(filepath, "rt") as f:
            for line in f:
                line = line.strip("\n").split("\t")
                try:
                    assert len(line)==3
                    if line[0]=="neg":
                        label = 0
                    elif line[0]=="pos":
                        label = 1
                    else:
                        raise AssertionError
                    yield {"src": line[1], "trg": line[2]}, label
                except AssertionError:
                    pass
            
    def get_stream(self, filepath):
        return cycle(self.parse_file(filepath))

    def __iter__(self):
        return self.get_stream(self.filepath)

        

def generate_candidate(candidate_matrix,
                       positive_dict,
                       src_sentences,
                       trg_sentences):
    """
    return pairs of sentences and their labels
    candidate_matrix: numpy matrix, [[candidate_indices], [candidate_indices], ...]
    positive_dict: {index: [candidates]}
    src_sentences: list of strings
    trg_sentences: list of strings
    """
    # import random; random.shuffle(candidate_matrix)
    for src_index, candidates in enumerate(candidate_matrix):
        src = src_sentences[src_index]
        for trg_index in candidates:
            trg_index = int(trg_index)
            trg = trg_sentences[trg_index]
            label = 1 if trg_index in positive_dict[str(src_index)] else 0
            yield {"sentences": {"src": src, "trg": trg}, "label": label}
        if float(src_index) not in candidates.tolist():
            trg = trg_sentences[src_index]
            yield {"sentences": {"src": src, "trg": trg}, "label": 1}
            

