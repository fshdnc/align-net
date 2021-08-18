#!/usr/bin/env python3

import os
import sys
import time
import copy
import json
import numpy as np
import argparse
import transformers
import torch
torch.manual_seed(1111)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from evaluation import load_and_evaluate
import ParallelDataset
from itertools import islice
from train import AlignLangNet, AlignLoss, dataloader_generator
# DEBUGGING
from tools import compare_models


def train_model(model, dataloaders, criterion, optimizer, batch_no, num_epochs=10):
    global device, src_sentences, trg_sentences, pos_dict, val_src_sentences, val_trg_sentences, val_pos_dict, args
    since = time.time()

    best_acc = 0

    # for every epoch -> for train and eval -> loop over data
    for epoch in range(num_epochs):
        patience = 0
        patience_acc = 0

        print("Epoch {}/{}".format(epoch, num_epochs-1), flush=True)
        print("-"*10, flush=True)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                class_weight = {0:torch.tensor(0.22).to(device).float(), 1: torch.tensor(1).to(device).float()}
            else:
                model.eval()
                class_weight={}

            running_loss = 0.0
            #eval_metric = 0
            batch_count = 0
            for inputs, labels in dataloader_generator(dataloaders, phase, batch_no): #3866726): #123 735 236/batch_size
                if epoch == 0 and args.skip_batch > 0:
                    args.skip_batch = args.skip_batch - 1
                    continue
                # tokenize the sentences and move to device
                labels = labels.to(device) #.float()

                optimizer.zero_grad()

                # forward, track history if only in the training phase
                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels, class_weight=class_weight)
                    print("EPOCH", str(epoch), "PHASE", phase, "BATCH_LOSS", loss.tolist(), file=sys.stderr, flush=True)

                    # backward
                    if phase == "train":
                        batch_count += 1
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * labels.size(0)
                # eval_metrics += e.g. torch.sum(preds == labels.data)
                
                if phase == "train" and batch_count == 2500: # check acc every 500 batches
                    batch_count = 0
                    epoch_acc = load_and_evaluate(model,
                                              "data/positives/dedup_src_trg_test-positives.json",
                                              "data/eng-fin/test.src.dedup",
                                              "data/eng-fin/test.trg.dedup",
                                              device)
                    print("Epoch\t{}\tPhase\tTRAINING\tAcc\t{:.4f}".format(epoch, epoch_acc), flush=True)
                    # Puhti stdout doesn't show until the job ends
                    with open("DELME_training.txt","a") as f:
                        f.write("Epoch\t{}\tPhase\tTRAINING\tAcc\t{:.4f}\n".format(epoch, epoch_acc))
                    if epoch_acc > patience_acc: # early stopping
                        patience_acc = epoch_acc
                        patience = 0
                        model.save()
                    else:
                        patience += 1
                if patience > 10: # if acc does not improve in 10 checks x 50 batch/check x 128 sample/batch
                    print("Early stopping, best accuracy", patience_acc)
                    exit(0)
                    
            if phase == "train":
                epoch_loss = running_loss/(batch_no*labels.size(0))
            elif phase == "val":
                epoch_loss = running_loss / len(dataloaders[phase]) #dataset_sizes[phase]
            # in case of memory leakage, try epoch_loss += loss.detach().item()

            if phase == "val":
                epoch_acc = load_and_evaluate(model,
                                              "data/positives/dedup_src_trg_test-positives.json",
                                              "data/eng-fin/test.src.dedup",
                                              "data/eng-fin/test.trg.dedup",
                                              device)
                print("Epoch\t{}\tPhase\t{}\tLoss\t{:.4f}\tVal_acc\t{:.4f}".format(epoch, phase, epoch_loss, epoch_acc), flush=True)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    model.save()
            else:
                print("Epoch\t{}\tPhase\t{}\tLoss\t{:.4f}".format(epoch, phase, epoch_loss), flush=True)

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60), flush=True)
    print("Best val Acc: {:4f}".format(best_acc), flush=True)

    return model

def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-ckpt", help="Path to checkpoint to continue training", required=True)
    argparser.add_argument("--skip-batch", type=int, help="Number of batches to skip", required=True)
    argparser.add_argument("--train-batch-size", type=int, help="Batch size for training", required=True)
    argparser.add_argument("--learning-rate", type=float, help="Learning rate", required=True)
    args = argparser.parse_args()
    return args

def main():
    global device, src_sentences, trg_sentences, pos_dict, val_src_sentences, val_trg_sentences, val_pos_dict, args
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset = ParallelDataset.HugeParallelDataset("/scratch/project_2002820/lihsin/align-lang/data/laser-test-set/train_clean_shuffled.txt.gz")
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

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
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.train_batch_size)

    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    model = torch.load(args.model_ckpt, map_location=torch.device("cpu"))
    model.to(device)
    model.ckpt_name = "model_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pt"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model = train_model(model, dataloaders, AlignLoss, optimizer, 770000, num_epochs=1) #770000
    assert not compare_models(model.bert_model_src.state_dict(), model.bert_model_trg.state_dict()), "the encoder weights should be different"

    return 0
    
if __name__=="__main__":
    exit(main())

