#!/usr/bin/env python3

import torch
from collections import OrderedDict

def compare_models(model1_state_dict: OrderedDict,
                   model2_state_dict: OrderedDict):
    """
    INPUTS: model.state_dict()
    """
    # any and all functions are defined on Byte Tensors.
    # torch.randn(10).byte().any()

    # compare two torch tensors torch.eq()
    print("Models have identical weights:", torch.Tensor([torch.eq(v, model2_state_dict[k]).byte().all() for k, v in model1_state_dict.items()]).byte().all())
