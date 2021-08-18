#!/usr/bin/env python3

# experiment: encode the test file with bilingual BERT
# code modified from https://github.com/TurkuNLP/Deep_Learning_in_LangTech_course/blob/master/laser.ipynb

import numpy as np
import argparse
import transformers
import torch
import torch.nn
print("FLAG")
def tokenize_texts(texts):
    # runs the BERT tokenizer, returns list of list of integers
    tokenized_ids = [tokenizer.encode(txt, add_special_tokens=True) for txt in texts]
    # turn lists of integers into torch tensors
    tokenized_ids_t = [torch.tensor(ids, dtype=torch.long) for ids in tokenized_ids]
    # zero-padding
    tokenized_single_batch = torch.nn.utils.rnn.pad_sequence(tokenized_ids_t, batch_first=True)
    return tokenized_single_batch

def embed(data, how_to_pool="CLS"):
    # run BERT in torch
    with torch.no_grad(): # tell the model not to gather gradients for evaluation, saves memory
        mask = data.clone().float() # a mask telling which tokens are padding and which are real
        mask[data>0] = 1.0 # set to one for tokens that should not be masked
        emb = bert_model(data.cuda(), attention_mask = mask.cuda()) # applies the model and returns several things, documentation: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py#L648
        # print(emb[0].shape) # word x sequence x embedding

        if how_to_pool == "AVG":
            pooled = emb[0]*(mask.unsqueeze(-1)) # multiply everything by the mask
            pooled = pooled.sum(1)/mask.sum(-1).unsequeeze(-1) # sum and divide by non-zero elements in mask to get masked average
        elif how_to_pool == "CLS":
            pooled = emb[0][:,0,:].squeeze() # pick the first token as the embedding
        else:
            assert False, "how_to_pool should be CLS or AVG"
        print("Pooled shape:", pooled.shape)
    return pooled.cpu().numpy() # move data back to CPU and extract the numpy array

def get_nearest(texts1, texts2, vectors1, vectors2):
    all_dist = sklearn.metrics.pairwise_distances(vectors1, vectors2)
    nearest = all_dist.argmin(axis=-1) # lst of indices of the nearest sentences

# from evaluations import print_closest_sentence
    
if __name__=="__main__":
    # Embed a file of sentences into bert. One sentence per line.
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sentence-file", help="File containing sentences", required=True)
    argparser.add_argument("--bert-model-path", help="Path to the bert model", required=True)
    argparser.add_argument("--output-embedding-file", help="Output numpy file", required=True)
    args = argparser.parse_args()

    # Load BERT model
    bert_model = transformers.BertModel.from_pretrained(args.bert_model_path)
    bert_model = bert_model.cuda() # move the model to GPU
    bert_model.eval()
    print("BERT model {0} loaded".format(args.bert_model_path))

    # Load the Finnish BERT tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_model_path)

    # Load sentences
    with open(args.sentence_file) as f:
        data = f.readlines()
        data = [line.strip() for line in data][:10]
    print("Sentence file {0} read".format(args.sentence_file))
    
    data_tok = tokenize_texts(data).cuda() # tokenize and move to gpu
    data_emb_cls = embed(data_tok, "CLS")
    print("Sentences embedded, saving to output file {0}".format(args.output_embedding_file))

    # Save the embedding
    with open(args.output_embedding_file, "wb") as f:
        np.save(f, data_emb_cls)
    
    



