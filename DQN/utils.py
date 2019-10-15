# -*- coding: utf-8 -*-
"""
@created on: 10/14/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import torch
import numpy as np


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    if torch.cuda.is_available():
        x = torch.tensor(x, device=torch.device(0), dtype=torch.float32)
    else:
        x = torch.tensor(x, device=torch.device('cpu'), dtype=torch.float32)
    return x
