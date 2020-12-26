#!/usr/bin/env python3

import sys
import json
import torch
import sklearn
import numpy as np
from math import ceil

def retrieve_most_similar(vectors1, vectors2):
    """
    Code modified from: https://github.com/TurkuNLP/Deep_Learning_in_LangTech_course/blob/master/laser.ipynb
    Given two vectors, return the most similar for every item in the first vector
    """
    assert len(vectors1)==len(vectors2)
    all_dist = sklearn.metrics.pairwise_distances(vectors1, vectors2)
    nearest = all_dist.argmin(axis=-1)
    return nearest

def eval_tatoeba_retrieval(nearest, positive_dict):
    """
    The tatoeba aligned corpus has many to many mapping
    i.e. multiple source sentences are parallel to multiple target sentences
    """
    correct = 0
    for i, selected_index in enumerate(nearest): # nearest: <class 'numpy.ndarray'>
        # selected_index <class 'numpy.int64'>
        #if i==0:
        #    print("selected index type", type(selected_index)) #<class 'numpy.int64'>
        #    print("positive dict list item type", type(positive_dict[str(i)][0])) #int
        if selected_index in positive_dict[str(i)]:
            correct += 1
        #    print("correct", selected_index, positive_dict[str(i)])
    print("Correct predictions", correct, "Total predictions", len(nearest))
    return correct/len(nearest)

def evaluate(model, pos_dict, src_sentences, trg_sentences, device):
    model.eval()
    model.to(device)
    
    embeddings = embed_by_batch(model, src_sentences, trg_sentences, batch_size=64)
    selected_indices = retrieve_most_similar(embeddings["src"], embeddings["trg"])
    acc1 = eval_tatoeba_retrieval(selected_indices, pos_dict)
    #print("Accuracy@1", acc1)
    return acc1

def embed_by_batch(model, src_sentences, trg_sentences, batch_size=128):
    # embedding sentences by batch
    embeddings = None
    for i in range(ceil(len(src_sentences)/batch_size)):
        embedded_segment = model({"src": src_sentences[i*batch_size:(i+1)*batch_size], "trg": trg_sentences[i*batch_size:(i+1)*batch_size]})
        if not isinstance(embeddings, dict): #torch.Tensor):
            embeddings = {k: v.cpu().detach().numpy() for k,v in embedded_segment.items()}
        else:
            #embeddings = {k: torch.cat((v, embedded_segment[k].to("cpu"))) for k, v in embeddings.items()}
            embeddings = {k: np.append(v, embedded_segment[k].cpu().detach().numpy(), axis=0) for k, v in embeddings.items()}
    assert len(embeddings["src"])==len(src_sentences)
    return embeddings

def main():
    """
    First argument, path to checkpoint for evaluation
    """
    from train import AlignLangNet

    ckpt_path = sys.argv[1]
    model = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.eval()

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # The dictionary of positives
    with open("dev-src-positives.json", "r") as f:
        pos_dict = json.load(f)

    # src sentences in text form
    with open("data/eng-fin/dev.src", "r") as f:
        src_sentences = f.readlines()
    src_sentences = [line.strip() for line in src_sentences]

    # trg sentences in text form
    with open("data/eng-fin/dev.trg", "r") as f:
        trg_sentences = f.readlines()
    trg_sentences = [line.strip() for line in trg_sentences]

    model.device = device
    model.to(model.device)

    # embed source and target sentences
    embeddings = embed_by_batch(model, src_sentences, trg_sentences)

    # find the nearest neighbor for all the source sentences
    selected_indices = retrieve_most_similar(embeddings["src"], embeddings["trg"])
    #with open("selected_indices.npy", "wb") as f:
    #    np.save(f, selected_indices)
    # calculate acc@1
    acc1 = eval_tatoeba_retrieval(selected_indices, pos_dict)
    print("Accuracy@1", acc1)


if __name__ == "__main__":
    exit(main())
