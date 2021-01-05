#!/usr/bin/env python3

import os
import sys
import time
import copy
import json
import numpy as np
import transformers
import torch
torch.manual_seed(1111)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from evaluation import load_and_evaluate
import ParallelDataset

# DEBUGGING
from tools import compare_models

# still need data preparation into the following format
# data["src"] (str), ["trg"] (str), ["label"] (0 or 1)
# batching by dataset


# the model here
class AlignLangNet(torch.nn.Module):
    """
    Neural network for aligning bilingual berts
    """
    def __init__(self, src_model_path, trg_model_path=None, device=None):
        if not trg_model_path:
            trg_model_path = src_model_path
        if not device:
            device = "cpu"

        super(AlignLangNet, self).__init__() # Initialize self._modules as OrderedDict
        self.bert_model_src = transformers.BertModel.from_pretrained(src_model_path) # eng
        self.bert_model_trg = transformers.BertModel.from_pretrained(trg_model_path) # fin

        # freeze the Finnish embeddings
        for param in self.bert_model_trg.parameters():
            param.requires_grad = False

        # move the models to GPU
        self.bert_model_src = self.bert_model_src.cuda()
        self.bert_model_trg = self.bert_model_trg.cuda()

        # we only need one tokenizer because we start off with the same model
        self.tokenizer = transformers.BertTokenizer.from_pretrained(src_model_path)

        self.device = device
        self.ckpt_name = "model_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pt"

    def save(self, path=None):
        if not path:
            path = self.ckpt_name
        torch.save(self, path)
        print("Newest best weights saved to", path) #, "(from gpu)")

    def tokenize_texts(self, texts):
        # runs the BERT tokenizer, returns list of list of integers
        tokenized_ids = [self.tokenizer.encode(txt, add_special_tokens=True) for txt in texts]
        # turn lists of integers into torch tensors
        tokenized_ids_t = [torch.tensor(ids, dtype=torch.long) for ids in tokenized_ids]
        # zero-padding
        tokenized_single_batch = torch.nn.utils.rnn.pad_sequence(tokenized_ids_t, batch_first=True)
        return tokenized_single_batch

    def embed(self, model, data, how_to_pool="CLS"):
        mask = data.clone().float() # a mask telling which tokens are padding and which are real
        mask[data>0] = 1.0 # set to one for tokens that should not be masked
        emb = model(data.to(self.device), attention_mask = mask.to(self.device)) # applies the model and returns several things, documentation: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py#L648
        # print(emb[0].shape) # word x sequence x embedding

        if how_to_pool == "AVG":
            pooled = emb[0]*(mask.unsqueeze(-1)) # multiply everything by the mask
            pooled = pooled.sum(1)/mask.sum(-1).unsqueeze(-1) # sum and divide by non-zero elements in mask to get masked average
        elif how_to_pool == "CLS":
            pooled = emb[0][:,0,:].squeeze() # pick the first token as the embedding
        else:
            assert False, "how_to_pool should be CLS or AVG"
            print("Pooled shape:", pooled.shape)
        return pooled

    def forward(self, data):
        """
        Data format x["src"]
        """
        # Use AVG: [from arxiv 2010.07761]
        # fsrc(X) and ftrg(Y) are the mean-pooled representations of the source sentence X and target sentence Y
        data_tok_src = self.tokenize_texts(data["src"]).to(self.device) # tokenize and move to device
        data_emb_src = self.embed(self.bert_model_src, data_tok_src, "AVG")

        # embed the trg sentences
        data_tok_trg = self.tokenize_texts(data["trg"]).to(self.device) # tokenize and move to device
        data_emb_trg = self.embed(self.bert_model_trg, data_tok_trg, "AVG")
        return {"src": data_emb_src, "trg": data_emb_trg}

def AlignLoss(data, gold, class_weight=None):
    """
    L(X,Y;theta_src) =
    |   f_src(X; theta_src)^T * f_tgt(Y)               |
    | ------------------------------------ - Par(X, Y) |
    | ||f_src(X; theta_src)|| ||f_tgt(Y)||             |

    Par(X, Y) is 1 if X, Y are parallel, 0 otherwise

    No sample_weights equivalent to {0:1, 1:1}
    """
    X = data["src"] # src
    Y = data["trg"] # trg

    X_T = torch.transpose(X, 0, 1) # shape: (bert_dim, batch_size)
    numerator = torch.matmul(Y, X_T) # shape: (batch_size, batch_size)
    # for a batch, we only need the diagonals X_n^T * Y_n
    numerator = torch.diagonal(numerator, 0) # shape: (batch_size)
    
    #X_norm = F.normalize(X, p=2, dim=-1)
    X_norm = torch.sqrt(torch.sum(torch.mul(X, X), 1)) # shape: (batch_size)
    Y_norm = torch.sqrt(torch.sum(torch.mul(Y, Y), 1)) # shape: (batch_size)
    denominator = torch.mul(X_norm, Y_norm) # shape: (batch_size)
    loss_per_sample = torch.abs(torch.div(numerator, denominator)-gold)
    if class_weight:
        #weight_tensor=gold
        #for g, w in class_weight.items():
        #    weight_tensor = torch.where(gold==g, torch.tensor(w, dtype=torch.float), weight_tensor.float())
        class_weight = torch.where(gold==0, class_weight[0], class_weight[1])
        loss_per_sample = torch.mul(loss_per_sample, class_weight)
    return torch.mean(loss_per_sample)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    global device, src_sentences, trg_sentences, pos_dict, val_src_sentences, val_trg_sentences, val_pos_dict
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    # for every epoch -> for train and eval -> loop over data
    for epoch in range(num_epochs):
        patience = 0
        patience_acc = 0

        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                class_weight = {0:torch.tensor(0.3).to(device).float(), 1: torch.tensor(1).to(device).float()}
            else:
                model.eval()
                class_weight={}

            running_loss = 0.0
            #eval_metric = 0
            batch_count = 0
            for inputs, labels in dataloaders[phase]:
                # tokenize the sentences and move to device
                labels = labels.to(device) #.float()

                optimizer.zero_grad()

                # forward, track history if only in the training phase
                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels, class_weight=class_weight)
                    print("EPOCH", str(epoch), "PHASE", phase, "BATCH_LOSS", loss.tolist(), file=sys.stderr)

                    # backward
                    if phase == "train":
                        batch_count += 1
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * labels.size(0)
                # eval_metrics += e.g. torch.sum(preds == labels.data)
                
                if phase == "train" and batch_count == 500: # check acc every 500 batches
                    batch_count = 0
                    epoch_acc = load_and_evaluate(model,
                                              "data/positives/dedup_src_trg_test-positives.json",
                                              "data/eng-fin/test.src.dedup",
                                              "data/eng-fin/test.trg.dedup",
                                              device)
                    print("Epoch\t{}\tPhase\tTRAINING\tAcc\t{:.4f}".format(epoch, epoch_acc))
                    if epoch_acc > patience_acc: # early stopping
                        patience_acc = epoch_acc
                        patience = 0
                    else:
                        patience += 1
                if patience > 10: # if acc does not improve in 10 checks x 50 batch/check x 128 sample/batch
                    print("Early stopping, best accuracy", patience_acc)
                    exit(0)
                    
            epoch_loss = running_loss / len(dataloaders[phase]) #dataset_sizes[phase]
            # in case of memory leakage, try epoch_loss += loss.detach().item()

            #if phase == "val" and epoch_acc > best_acc:
            if phase == "val":
                train_acc = load_and_evaluate(model,
                                              "data/positives/dedup_src_trg_dev-positives.json",
                                              "data/eng-fin/dev.src.dedup",
                                              "data/eng-fin/dev.trg.dedup",
                                              device)
                epoch_acc = load_and_evaluate(model,
                                              "data/positives/dedup_src_trg_test-positives.json",
                                              "data/eng-fin/test.src.dedup",
                                              "data/eng-fin/test.trg.dedup",
                                              device)
                print("Epoch\t{}\tPhase\t{}\tLoss\t{:.4f}\tTrain_acc\t{:.4f}\tVal_acc\t{:.4f}".format(epoch, phase, epoch_loss, train_acc, epoch_acc))
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    #best_model_wts = copy.deepcopy(model.state_dict())
                    #print("Best model weights copied")
                    model.save()
            else:
                print("Epoch\t{}\tPhase\t{}\tLoss\t{:.4f}".format(epoch, phase, epoch_loss))

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load the weights
    #model.load_state_dict(best_model_wts)
    
    return model

def main():
    global device, src_sentences, trg_sentences, pos_dict, val_src_sentences, val_trg_sentences, val_pos_dict
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model_path = "/scratch/project_2002820/lihsin/bert_checkpoints/biBERT80-transformers" #"/home/lhchan/embeddings/biBERT80-transformers"
    train_batch_size = 128

    # the selected candidates, in numpy array
    with open("dev-src-dev-trg-ivf1584.npy", "rb") as f:
        D, I = np.load(f)
    del D

    # The dictionary of positives
    with open("dev-positives.json", "r") as f:
        pos_dict = json.load(f)

    # src sentences in text form
    with open("data/eng-fin/dev.src", "r") as f:
        src_sentences = f.readlines()
    src_sentences = [line.strip() for line in src_sentences]

    # trg sentences in text form
    with open("data/eng-fin/dev.trg", "r") as f:
        trg_sentences = f.readlines()
    trg_sentences = [line.strip() for line in trg_sentences]

    # dataloaders
    train_dataset = ParallelDataset.ParallelDataset([*ParallelDataset.generate_candidate(I, pos_dict, src_sentences, trg_sentences)])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

    # load data for the valiation dataset
    with open("test-trg-test-src-flatL2.npy", "rb") as f:
        val_D, val_I = np.load(f)
    del val_D

    # The dictionary of positives
    with open("test-positives.json", "r") as f:
        val_pos_dict = json.load(f)

    # src sentences in text form
    with open("data/eng-fin/test.src", "r") as f:
        val_src_sentences = f.readlines()
    val_src_sentences = [line.strip() for line in val_src_sentences]

    # trg sentences in text form
    with open("data/eng-fin/test.trg", "r") as f:
        val_trg_sentences = f.readlines()
    val_trg_sentences = [line.strip() for line in val_trg_sentences]

    #val_dataset = ParallelDataset([*generate_candidate()])
    val_dataset = ParallelDataset.ParallelDataset([*ParallelDataset.generate_candidate(val_I, val_pos_dict, val_src_sentences, val_trg_sentences)])
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=train_batch_size)

    dataloaders = {"train": train_dataloader, "val": val_dataloader}
    
    model = AlignLangNet(model_path, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    #optimizer = torch.optim.SGD(model.parameters(), lr=2e-5, momentum=0.9)
    print("lr\t2e-5, adam")
    model.save("model_no_training")
    exit(0)

    model = train_model(model, dataloaders, AlignLoss, optimizer, num_epochs=10)
    assert not compare_models(model.bert_model_src.state_dict(), model.bert_model_trg.state_dict()), "the encoder weights should be different"

    # save the model (with best checkpoint loaded)
    #model_name = "model_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pt"
    #model.to("cpu")
    #model.save()
    #print("Final model (best checkpoint) saved to", model_name)

    return 0
    
if __name__=="__main__":
    exit(main())

