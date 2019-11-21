#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from functools import reduce
from torch import tensor



class DQN(nn.Module):
    """Initialize a deep Q-learning network
    
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    """

    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class BasicConvNet(nn.Module):


class ICM(nn.Module):

    def __init__(self, input_shape, num_actions):
        super(ICM, self).__init__()
        self.num_actions = num_actions
        self.input_shape = input_shape

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 8, 4),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.LeakyReLU(),
                nn.Linear(7 * 7 * 64, 512),
        )

        self.inverse_model = nn.Sequential(
                nn.Linear(512 * 2, 512),
                nn.LeakyReLU(),
                nn.Linear(512, num_actions)
        )

        self.forward_model = nn.Sequential(
                nn.Linear(512 + num_actions, 256),
                nn.LeakyReLU(),
                nn.Linear(256, self.encoded_next_state_shape)
        )

    def forward(self, state_batch, next_state_batch, onehot_action_batch):
        encoded_current_state_batch = self.feature_extractor(state_batch)
        encoded_next_state_batch = self.feature_extractor(next_state_batch)
        self.encoded_next_state_batch_shape = reduce(lambda x, y: x * y, encoded_next_state_batch.shape[1:])

        forward_model_inp = torch.cat((encoded_current_state_batch, onehot_action_batch), dim=1)
        predicted_next_state = self.forward_model(forward_model_inp)

        inverse_model_inp = torch.cat((encoded_current_state_batch, encoded_next_state_batch), dim=1)
        predicted_action = self.inverse_model(inverse_model_inp)

        return encoded_next_state_batch, predicted_next_state, predicted_action
