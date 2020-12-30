#!/usr/bin/env python3

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader #, random_split
from torchvision import transforms
import pytorch_lightning as pl

class LitAlignLangNet(pl.LightningModule):
    """
    Lightning module for aligning bilingual bert
    """
    def __init__(self, src_model_path, trg_model_path=None):
        super().__init__()
        if not trg_model_path:
            trg_model_path = src_model_path

        self.bert_model_src = transformers.BertModel.from_pretrained(src_model_path) # eng
        self.bert_model_trg = transformers.BertModel.from_pretrained(trg_model_path) # fin

        # freeze the Finnish embeddings
        for param in self.bert_model_trg.parameters():
            param.requires_grad = False

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding =


        
