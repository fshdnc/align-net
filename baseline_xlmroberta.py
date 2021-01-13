#!/usr/bin/env python3

"""
XLM-RoBERTa baseline
"""
import sys
sys.path.insert(0,"/projappl/project_2002085/lihsin/transformers-electra/lib/python3.7/site-packages")
import os
import sys
import json
import numpy as np
import transformers
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
from evaluation import eval_tatoeba_retrieval
import sklearn.metrics
from math import ceil

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

def tokenize_texts(tokenizer, texts):
    # runs the BERT tokenizer, returns list of list of integers
    tokenized_ids = [tokenizer.encode(txt, add_special_tokens=True) for txt in texts]
    # turn lists of integers into torch tensors
    tokenized_ids_t = [torch.tensor(ids, dtype=torch.long) for ids in tokenized_ids]
    # zero-padding
    tokenized_single_batch = torch.nn.utils.rnn.pad_sequence(tokenized_ids_t, batch_first=True)
    return tokenized_single_batch

def embed(model, data, how_to_pool="CLS"):
    with torch.no_grad():
        mask = data.clone().float()
        mask[data>0] = 1.0
        emb = model(data.cuda(), attention_mask = mask.cuda())
        if how_to_pool == "AVG":
            pooled = emb[0]*(mask.unsqueeze(-1))
            pooled = pooled.sum(1)/mask.sum(-1).unsqueeze(-1)
        elif how_to_pool == "CLS":
            pooled = emb[0][:,0,:].squeeze()
        else:
            assert False, "how_to_pool should be CLS or AVG"
        print("Pooled shape:", pooled.shape)
    return pooled

def embed_by_batch(model, sentences, batch_size=64):
    # embedding sentences by batch
    embeddings = None
    for i in range(ceil(len(sentences)/batch_size)):
        embedded_segment = embed(model, sentences[i*batch_size:(i+1)*batch_size], how_to_pool="AVG")
        #print(len(embedded_segment))
        #print("embedded_segment type",type(embedded_segment))
        #print("embedded_segment", embedded_segment)
        if not isinstance(embeddings, np.ndarray):
            embeddings = embedded_segment.cpu().detach().numpy()
        else:
            embeddings = np.append(embeddings, embedded_segment.cpu().detach().numpy(), axis=0)
        #if not isinstance(embeddings, dict): #torch.Tensor):
        #    embeddings = {k: v.cpu().detach().numpy() for k,v in embedded_segment.items()}
        #else:
            #embeddings = {k: torch.cat((v, embedded_segment[k].to("cpu"))) for k, v in embeddings.items()}
        #    embeddings = {k: np.append(v, embedded_segment[k].cpu().detach().numpy(), axis=0) for k, v in embeddings.items()}
    assert len(embeddings)==len(sentences)
    return embeddings

def main():
    tokenizer = XLMRobertaTokenizer.from_pretrained("/scratch/project_2002820/lihsin/embeddings/xlm-roberta-base")
    model = XLMRobertaModel.from_pretrained("/scratch/project_2002820/lihsin/embeddings/xlm-roberta-base")
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    #inputs = tokenizer([], return_tensors="pt")
    #outputs = model(**inputs)
    #last_hidden_states = outputs.last_hidden_state

    # dev
    #pos, src, trg = load("data/positives/dedup_src_trg_dev-positives.json",
    #                              "data/eng-fin/dev.src.dedup",
    #                              "data/eng-fin/dev.trg.dedup")

    #pos, src, trg = load("data/positives/dedup_src_trg_test-positives.json",
    #                              "data/eng-fin/test.src.dedup",
    #                              "data/eng-fin/test.trg.dedup")

    pos, src, trg = load("data/wmt/dedup_src_trg_wmt-positives.json",
                            "data/wmt/wmt-en.txt.dedup",
                            "data/wmt/wmt-fi.txt.dedup")

    src_tok = tokenize_texts(tokenizer, src).to(device)
    trg_tok = tokenize_texts(tokenizer, trg).to(device)
    #src_dev, trg_dev = embed(laser, src, trg)
    src_emb = embed_by_batch(model, src_tok)
    trg_emb = embed_by_batch(model, trg_tok)
    selected_indices = retrieve_most_similar(src_emb, trg_emb)
    acc = eval_tatoeba_retrieval(selected_indices, pos)
    print("XLM-RoBERTa for dev set:", acc)

    # test
    
    #src_test, trg_test = embed(laser, src, trg)
    #selected_indices = retrieve_most_similar(src_test, trg_test)
    #acc = eval_tatoeba_retrieval(selected_indices, pos)
    #print("Laser for test set:", acc)

    #src_test, trg_test = embed(laser, src, trg)
    #selected_indices = retrieve_most_similar(src_test, trg_test)
    #acc = eval_tatoeba_retrieval(selected_indices, pos)
    #print("Laser for wmt:", acc)
    return 0

if __name__=="__main__":
    exit(main())

