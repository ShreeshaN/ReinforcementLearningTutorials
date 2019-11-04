#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor

from DQN.agent import Agent
from DQN.dqn_model import DQN
from torch.autograd import Variable

from DQN.utils import tensor
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
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.num_steps = args.num_steps
        self.state_length = args.state_length
        self.gamma = args.gamma
        self.exploration_steps = args.exploration_steps
        self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.initial_replay_size = args.initial_replay_size
        self.num_replay_memory = args.num_replay_memory
        self.batch_size = args.batch_size
        self.target_update_interval = args.target_update_interval
        self.train_interval = args.train_interval
        self.learning_rate = args.learning_rate
        self.save_interval = args.save_interval
        self.no_op_steps = args.no_op_steps
        self.save_network_path = args.save_network_path
        self.save_summary_path = args.save_summary_path
        self.test_dqn_model_path = args.test_dqn_model_path
        self.exp_name = args.exp_name
        self.ddqn = args.ddqn
        self.dueling = args.dueling
        # self.test_path = args.test_path
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.log = args.logfile_path

        # environment setting
        self.env = env
        self.num_actions = env.action_space.n

        self.epsilon = self.initial_epsilon
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.exploration_steps
        self.t = 0

        # for summary & checkpoint
        self.total_reward = 0.0
        self.total_q_max = 0.0
        self.total_loss = 0.0
        self.duration = 0
        self.episode = 0
        self.last_100_reward = deque()

        # Input that is not used when fowarding for Q-value
        # or loss calculation on first output of model
        # self.dummy_input = np.zeros((1, self.num_actions))
        # self.dummy_batch = np.zeros((self.batch_size, self.num_actions))

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.q_network = DQN().to(self.device)
        # Create target network
        self.target_network = DQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

        self.q_network.train()
        self.target_network.eval()

        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)
        if not os.path.exists(self.save_summary_path):
            os.makedirs(self.save_summary_path)

        # load model for testing, train a new one otherwise
        if args.test_dqn:
            self.q_network.load_state_dict(torch.load(self.test_dqn_model_path))
        else:
            self.log = open(self.save_summary_path + self.exp_name + '.log', 'w')

        # Set target_network weight
        self.target_network.load_state_dict(self.q_network.state_dict())

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #

        self.writer = SummaryWriter(args.tensorboard_summary)

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

    def get_prediction(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state)
        if state.size() == (4, 84, 84):
            return self.q_network(state)
        elif state.size() == (84, 84, 4):
            # making state channel first
            state = state.permute(2, 0, 1)
            return self.q_network(state)
        else:
            raise Exception("Expecting state with (channel * X * Y) or (X * Y * channel). But recieved ", state.size())

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

        if not test:
            if self.epsilon >= random.random() or self.t < self.initial_replay_size:
                action = random.randrange(self.num_actions)
            else:
                action = torch.argmax(self.q_network(tensor(observation).unsqueeze(0).float()).detach()).item()

            # Anneal epsilon linearly over time
            if self.epsilon > self.final_epsilon and self.t >= self.initial_replay_size:
                self.epsilon -= self.epsilon_step
        else:
            if 0.005 >= random.random():
                action = random.randrange(self.num_actions)
            else:
                action = torch.argmax(self.q_network(tensor(observation).unsqueeze(0).float()).detach()).item()
        return action

    def push(self, state, action, reward, next_state, terminal):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.replay_memory.append((state, action, reward, next_state, terminal))
        ###########################

    def replay_buffer(self, state, action, reward, next_state, terminal):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Store transition in replay memory
        if len(self.replay_memory) > self.num_replay_memory:
            self.replay_memory.popleft()
        self.push(state, action, reward, next_state, terminal)

        ###########################

    def log_summary(self, global_step):
        self.writer.add_scalar('Train/Average Reward', np.mean(self.last_100_reward), global_step) if len(
                self.last_100_reward) > 0 else None
        self.writer.add_scalar('Train/Average Q', self.total_q_max / float(self.duration),
                               global_step) if self.duration > 0 else None
        self.writer.add_scalar('Train/Average Loss',
                               self.total_loss / (float(self.duration) / float(self.train_interval)),
                               global_step) if self.duration > 0 and self.train_interval > 0 else None

        self.writer.flush()

    def train_network(self):
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

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        # Q value from target network
        target_q_values_batch = self.target_network(tensor(next_state_batch).float())
        y_batch = tensor(reward_batch) + tensor(1 - terminal_batch) * self.gamma * torch.max(target_q_values_batch,
                                                                                             dim=-1).values
        # Loss
        output = self.q_network(tensor(state_batch).float())
        output = output[[x for x in range(self.batch_size)], [action_batch]].squeeze(0)
        loss = F.smooth_l1_loss(output, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.total_loss += loss

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        while self.t <= self.num_steps:
            terminal = False
            observation = self.env.reset()
            observation_channel_first = np.rollaxis(observation, 2)
            for _ in range(random.randint(1, self.no_op_steps)):
                observation, _, _, _ = self.env.step(0)  # Do nothing
            while not terminal:
                action = self.make_action(observation_channel_first, test=False)
                new_observation, reward, terminal, _ = self.env.step(action)
                # reward = max(-1.0, min(reward, 1.0))
                new_observation_channel_first = np.rollaxis(new_observation, 2)
                self.run(observation_channel_first, action, reward, terminal, new_observation_channel_first)
                observation_channel_first = new_observation_channel_first

        ###########################

    def run(self, state, action, reward, terminal, next_state):

        # Store transition in replay memory
        self.replay_buffer(state, action, reward, next_state, terminal)

        if self.t >= self.initial_replay_size:
            # Train network
            if self.t % self.train_interval == 0:
                self.train_network()

            # Update target network
            if self.t % self.target_update_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.save_network_path + '/' + self.exp_name + '_' + str(self.t) + '.pt'
                torch.save(self.q_network.state_dict(), save_path)
                # self.q_network.save_model(save_path)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += torch.max(self.q_network(tensor(state).unsqueeze(0).float()))
        self.duration += 1

        if terminal:
            # Observe the mean of rewards on last 100 episode
            self.last_100_reward.append(self.total_reward)
            if len(self.last_100_reward) > 100:
                self.last_100_reward.popleft()

            # Log message
            if self.t < self.initial_replay_size:
                mode = 'random'
            elif self.initial_replay_size <= self.t < self.initial_replay_size + self.exploration_steps:
                mode = 'explore'
            else:
                mode = 'exploit'
            self.episode = self.episode + 1
            print(
                    'EPISODE: {0:2d} | TIMESTEP: {1:2d} | DURATION: {2:2d} | EPSILON: {3:.5f} | REWARD: {4:.3f} | AVG_REWARD: {5:.3f} | AVG_MAX_Q: {6:.4f} | AVG_LOSS: {7:.5f} | MODE: {8}'.format(
                            self.episode, self.t, self.duration, self.epsilon, sum(self.last_100_reward),
                            np.mean(self.last_100_reward), self.total_q_max / float(self.duration),
                                                           self.total_loss / (float(self.duration) / float(
                                                               self.train_interval)), mode))
            print(
                    'EPISODE: {0:2d} | TIMESTEP: {1:2d} | DURATION: {2:2d} | EPSILON: {3:.5f} | REWARD: {4:.3f} | AVG_REWARD: {5:.3f} | AVG_MAX_Q: {6:.4f} | AVG_LOSS: {7:.5f} | MODE: {8}'.format(
                            self.episode, self.t, self.duration, self.epsilon, sum(self.last_100_reward),
                            np.mean(self.last_100_reward), self.total_q_max / float(self.duration),
                                                           self.total_loss / (float(self.duration) / float(
                                                               self.train_interval)), mode),
                    file=self.log)
            self.log_summary(global_step=self.episode)

            # Init for new game
            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1
