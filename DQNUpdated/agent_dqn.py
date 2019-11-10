#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# DOUBTS:
# 1. Replay buffer is updated every step ?


import random
import numpy as np
from collections import deque
import os
import sys

import torch
from torch import nn, tensor
import torch.nn.functional as F
import torch.optim as optim

from DQNUpdated.agent import Agent
from DQNUpdated.dqn_model import DQN
from torch.utils.tensorboard import SummaryWriter

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.run_name = args.run_name
        self.model_save_path = args.model_save_path
        self.model_save_interval = args.model_save_interval
        self.log_path = args.log_path
        self.tensorboard_summary_path = args.tensorboard_summary_path
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.model_test_path = args.model_test_path
        self.step = 0

        # Environment and network parameters
        self.env = env
        self.num_actions = env.action_space.n
        self.metrics_capture_window = args.metrics_capture_window
        self.replay_size = args.replay_size
        self.replay_memory = deque([], self.replay_size)
        self.total_num_steps = args.total_num_steps
        self.episodes = args.episodes
        self.gamma = args.gamma
        self.learning_rate = args.learning_rate
        self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.epsilon = self.initial_epsilon
        self.steps_to_explore = args.steps_to_explore
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.steps_to_explore
        self.network_update_interval = args.network_update_interval
        self.network_train_interval = args.network_train_interval
        self.last_n_rewards = deque([], self.metrics_capture_window)

        self.batch_size = args.batch_size
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.loss_function = F.smooth_l1_loss
        self.optimiser = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

        self.q_network.train()
        self.target_network.eval()

        # create necessary paths
        self.create_dirs()

        if args.test_dqn:
            print('loading trained model')
            self.q_network.load_state_dict(torch.load(self.model_test_path))

        self.log_file = open(self.model_save_path + '/' + self.run_name + '.log', 'w') if not args.test_dqn else None

        # Set target_network weight
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.writer = SummaryWriter(args.tensorboard_summary_path)

    def create_dirs(self):
        paths = [self.model_save_path, self.tensorboard_summary_path]
        [os.makedirs(path) for path in paths if not os.path.exists(path)]

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        observation = tensor(np.rollaxis(observation, 2)).to(self.device)
        if not test:
            if self.epsilon >= random.random() or self.step < self.replay_size:
                action = random.randrange(self.num_actions)
            else:
                action = torch.argmax(self.q_network(observation.unsqueeze(0).float()).detach()).item()
        else:
            if random.random() >= 0.005:
                action = random.randrange(self.num_actions)
            else:
                action = torch.argmax(self.q_network(observation.unsqueeze(0).float()).detach()).item()

        if self.epsilon > self.final_epsilon and self.step >= self.replay_size:
            self.epsilon -= self.epsilon_step

        return action

    ###########################

    # def push(self):
    #     """ You can add additional arguments as you need.
    #     Push new data to buffer and remove the old one if the buffer is full.
    #
    #     Hints:
    #     -----
    #         you can consider deque(maxlen = 10000) list
    #     """
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #
    #
    #     ###########################
    #
    # def replay_buffer(self):
    #     """ You can add additional arguments as you need.
    #     Select batch from buffer.
    #     """
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #
    #
    #     ###########################
    #     return

    def optimize_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        terminal_batch = np.array(terminal_batch) + 0

        q_values = self.q_network(tensor(state_batch).float())
        q_values = q_values.gather(1, tensor(action_batch).unsqueeze(1)).squeeze(1)

        target_values = self.target_network(tensor(next_state_batch).float())
        target_values, _ = target_values.max(1)

        target_values = (1 - tensor(terminal_batch)) * target_values.squeeze(0)
        target_values = tensor(reward_batch) + (self.gamma * target_values)

        loss = self.loss_function(q_values, target_values)
        self.optimiser.zero_grad()
        loss.backward()
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()
        return loss.cpu().detach().numpy()

    def log_summary(self, global_step, episode_loss, episode_reward):
        self.writer.add_scalar('Train/Episode Reward', sum(episode_reward), global_step)
        self.writer.add_scalar('Train/Average Loss', np.mean(episode_loss), global_step)
        self.writer.flush()

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        mode = 'random'
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            episode_reward = []
            episode_loss = []
            while not done:
                self.step += 1
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward.append(reward)
                self.replay_memory.append((np.rollaxis(state, 2), action, reward, np.rollaxis(next_state, 2), done))

                if self.step > self.replay_size:

                    mode = 'explore' if self.step <= self.replay_size + self.steps_to_explore else 'exploit'

                    # train network
                    if self.step % self.network_train_interval == 0:
                        loss = self.optimize_network()
                        episode_loss.append(loss)

                    # save network
                    if self.step % self.model_save_interval == 0:
                        save_path = self.model_save_path + '/' + self.run_name + '_' + str(self.step) + '.pt'
                        torch.save(self.q_network.state_dict(), save_path)
                        print('Successfully saved: ' + save_path)

                    # update target network
                    if self.step % self.network_update_interval == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())

                state = next_state
                if done:
                    # print(
                    #         f'Episode:{episode} | Steps:{self.step} | Reward:{sum(episode_reward)} | Loss: {np.mean(episode_loss)} | Mode: {mode}')
                    # print(
                    #         f'Episode:{episode} | Steps:{self.step} | Reward:{sum(episode_reward)} | Loss: {np.mean(episode_loss)} | Mode: {mode}',
                    #         file=self.log_file)

                    print('Episode:', episode, ' | Steps:', self.step, ' | Reward: ', sum(episode_reward), ' | Loss: ',
                          np.mean(episode_loss), ' | Mode: ', mode)
                    print('Episode:', episode, ' | Steps:', self.step, ' | Reward: ', sum(episode_reward), ' | Loss: ',
                          np.mean(episode_loss), ' | Mode: ', mode, file=self.log_file)
                    self.log_summary(episode, episode_loss, episode_reward)
                    episode_reward.clear()
                    episode_loss.clear()

        ###########################
