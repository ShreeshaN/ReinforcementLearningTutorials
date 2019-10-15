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
        self.conv1 = nn.Conv2d(4, 32, 4, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 1)
        self.conv3 = nn.Conv2d(64, 64, 4, 1)
        self.fc1 = nn.Linear(7 * 7 * 64, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 4)
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
        x = F.relu(self.conv1(x))
        # print("after 1st conv ", x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print("after 1st poool ", x.shape)
        x = F.relu(self.conv2(x))
        # print("after 2nd conv ", x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print("after 2nd poool ", x.shape)
        x = F.relu(self.conv3(x))
        # print("after 3nrd conv ", x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print("after 3rd poool ", x.shape)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        self.fc3.weight = new_weights['fc3']

    def get_weights(self):
        return {'conv1': self.conv1.weight,
                'conv2': self.conv2.weight,
                'conv3': self.conv3.weight,
                'fc1': self.fc1.weight,
                'fc2': self.fc2.weight,
                'fc3': self.fc3.weight
                }

    def save_model(self, path):
        torch.save(self.state_dict(), path)
