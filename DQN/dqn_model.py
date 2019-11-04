#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):
    """Initialize a deep Q-learning network
    
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    
    This is just a hint. You can build your own structure.
    """

    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(7 * 7 * 64, 500)
        self.fc2 = nn.Linear(500, 4)
        ###########################

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # print("input ", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print("after 1st conv ", x.shape)
        # x = F.max_pool2d(x, 2, 2)
        # print("after 1st poool ", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print("after 2nd conv ", x.shape)
        # x = F.max_pool2d(x, 2, 2)
        # print("after 2nd poool ", x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print("after 3nrd conv ", x.shape)
        # x = F.max_pool2d(x, 2, 2)
        # print("after 3rd poool ", x.shape)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        ###########################

    def set_weights(self, new_weights):
        """

        :param new_weights: dict of all layer weights
        :return:
        """
        self.conv1.weight = new_weights['conv1']
        self.conv2.weight = new_weights['conv2']
        self.conv3.weight = new_weights['conv3']
        self.fc1.weight = new_weights['fc1']
        self.fc2.weight = new_weights['fc2']
        self.bn1.weight = new_weights['bn1']
        self.bn2.weight = new_weights['bn2']
        self.bn3.weight = new_weights['bn3']