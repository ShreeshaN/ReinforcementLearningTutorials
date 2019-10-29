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
import gc
import torch.nn as nn
from DQN.utils import tensor
from torch.autograd import Variable
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
        self.total_episodes = args.total_episodes
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
        self.save_network_path = args.save_network_path
        self.log_path = args.log_path
        self.test_dqn_model_path = args.test_dqn_model_path
        self.run_name = args.run_name
        self.use_ddqn = args.use_ddqn
        self.dueling = args.use_dueling_network
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.log = args.logfile_path
        self.loss_fn = F.mse_loss if args.loss_fn == 'mse' else F.smooth_l1_loss if args.loss_fn == 'huber' else None
        self.env = env
        self.num_actions = env.action_space.n
        self.capture_window = args.capture_window
        self.epsilon = self.initial_epsilon
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.exploration_steps
        self.total_step_tracker = 0

        # Tensorboard log writer
        # Usage: tensorboard --logdir=/path_to_log_dir/
        self.writer = SummaryWriter(args.tensorboard_summary)

        self.total_loss_val = np.zeros(args.capture_window, np.float32)
        self.total_q_val = np.zeros(args.capture_window, np.float32)
        self.total_rewards = np.zeros(args.capture_window, np.float32)
        self.test_average_reward = []

        self.replay_memory = deque([], args.num_replay_memory)
        self.retrain = args.retrain

        # Create q network
        self.q_network = DQN().to(self.device)
        # Create target network
        self.target_network = DQN().to(self.device)
        # self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=args.learning_rate)

        self.q_network.train()
        self.target_network.eval()

        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # load model for testing, train a new one otherwise
        if args.test_dqn:
            self.q_network.load_state_dict(torch.load(self.test_dqn_model_path, map_location=self.device))
        else:
            # Log file for training
            self.log = open(self.log_path + self.run_name + '.log', 'w')

        if self.retrain:
            self.q_network.load_state_dict(torch.load(self.test_dqn_model_path, map_location=self.device))
            self.log = open(self.log_path + self.run_name + '.log', 'w')

        # Set target_network weight
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rewards = []
        self.q_values = []
        self.total_loss = []
        self.last_n_rewards = deque([], args.capture_window)
        self.last_n_qs = deque([], args.capture_window)

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
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
        observation = np.rollaxis(observation, 2)
        if test:
            if 0.005 >= random.random():
                return random.randrange(self.num_actions)
            else:
                return torch.argmax(self.q_network(tensor(observation).unsqueeze(0).float()).detach()).item()
        else:
            if self.epsilon >= random.random() or self.total_step_tracker < self.initial_replay_size:
                return random.randrange(self.num_actions)
            else:
                return torch.argmax(self.q_network(tensor(observation).unsqueeze(0).float()).detach()).item()

    def decay_epsilon(self):
        if self.epsilon > self.final_epsilon and self.total_step_tracker >= self.initial_replay_size:
            self.epsilon -= self.epsilon_step

    def push_to_replay_buffer(self, state, action, reward, next_state, terminal):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """

        # Store transition in replay memory
        state = np.rollaxis(state, 2)
        next_state = np.rollaxis(next_state, 2)
        self.replay_memory.append((state, action, reward, next_state, terminal))

    def optimize_network(self):
        current_states = []
        actions = []
        rewards = []
        future_states = []
        terminals = []

        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            current_states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            future_states.append(data[3])
            terminals.append(data[4])

        # converting arrays to tensors and assigning them to GPU if available
        current_states, actions, rewards, future_states, terminals = tensor(current_states).to(self.device), tensor(
                actions).to(self.device), tensor(
                rewards).to(self.device), tensor(future_states).to(self.device), terminals

        current_q_values = self.q_network(current_states).gather(1, actions.unsqueeze(1).long()).squeeze(1)
        future_q_values = self.target_network(future_states).detach()
        if self.use_ddqn:
            best_actions = torch.argmax(self.q_network(future_states), dim=-1)
            future_q_values = future_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        else:
            future_q_values = future_q_values.max(1)[0]

        target_q_values = rewards + (self.gamma * future_q_values * (1 - terminals))

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def test_network(self, total_episodes=100):
        self.test_average_reward = []
        print('Running test results ***')
        for i in range(total_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_num_states = 0

            # playing one game
            while (not done):
                action = self.make_action(state, test=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_num_states += 1
            self.test_average_reward.append(episode_reward)
            print('Running ', i, ' | Episode reward ', episode_reward, ' | Number of states ', episode_num_states,
                  ' | Moving average Reward ', np.mean(self.test_average_reward))
        print('Run %d episodes' % (total_episodes))
        print('Mean:', np.mean(self.test_average_reward))

    def log_summary(self, global_step, test=False):

        if not test:
            self.writer.add_scalar('Train/Episode Reward', sum(self.rewards), global_step) if len(
                    self.rewards) > 0 else None
            self.writer.add_scalar('Train/Average Reward', np.mean(self.last_n_rewards), global_step) if len(
                    self.total_rewards) > 0 else None
            self.writer.add_scalar('Train/Average Loss', np.mean(self.total_loss), global_step) if len(
                    self.total_loss) > 0 else None
            self.writer.add_scalar('Train/Average Q', np.mean(self.q_values), global_step) if len(
                self.q_values) > 0 else None
        else:
            self.writer.add_scalar('Test/Average Reward', np.mean(self.test_average_reward), global_step)
        self.writer.flush()

    def train(self):
        """
        Implement your training algorithm here
        """

        for i in range(self.total_episodes):
            terminal = False
            observation = self.env.reset()

            # # reset metrics
            # self.total_q_val[i % self.capture_window] = -1e9
            # self.total_rewards[i % self.capture_window] = 0

            while not terminal:

                action = self.make_action(observation, test=False)
                new_observation, reward, terminal, _ = self.env.step(action)

                # Decay epsilon over time
                self.decay_epsilon()

                # Store transition in replay memory
                self.push_to_replay_buffer(observation, action, reward, new_observation, terminal)

                # Do not train until the replay buffer is full
                if self.total_step_tracker >= self.initial_replay_size:
                    # Train network
                    if self.total_step_tracker % self.train_interval == 0:
                        loss = self.optimize_network()
                        # self.total_loss_val[i % self.capture_window] = loss
                        self.total_loss.append(loss)

                    # Update target network
                    if self.total_step_tracker % self.target_update_interval == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())
                        gc.collect()

                    # Save network
                    if self.total_step_tracker % self.save_interval == 0:
                        save_path = self.save_network_path + '/' + self.run_name + '_' + str(
                                self.total_step_tracker) + '.pt'
                        torch.save(self.q_network.state_dict(), save_path)
                        print('Successfully saved: ' + save_path)

                        # Testing model after every save to keep track of model performance at this checkpoint
                        # self.test_network()
                        self.log_summary(global_step=i, test=True)

                # Update rewards
                # self.total_rewards[i % self.capture_window] += reward
                self.rewards.append(reward)

                # Update Q value
                # self.total_q_val[i % self.capture_window] = max(
                #         torch.max(self.q_network(tensor(np.rollaxis(observation, 2)).unsqueeze(0).float())),
                #         self.total_q_val[i % self.capture_window])
                # self.q_network(tensor(np.rollaxis(observation, 2)).unsqueeze(0))
                self.q_values.append(
                        torch.max(self.q_network(tensor(np.rollaxis(observation, 2)).unsqueeze(0))).detach().numpy())

                if terminal:

                    self.last_n_rewards.append(sum(self.rewards))

                    if self.total_step_tracker < self.initial_replay_size:
                        mode = 'random'
                    elif self.initial_replay_size <= self.total_step_tracker < self.initial_replay_size + self.exploration_steps:
                        mode = 'explore'
                    else:
                        mode = 'exploit'

                    # Log metrics
                    # print(
                    #         f"Episode: {i} | Timestep: {self.total_step_tracker} | Epsilon: {self.epsilon:.3f} | Reward: {self.total_rewards[i % self.capture_window]} | Avg Reward: {np.mean(self.total_rewards):.3f} | AvgQ: {np.mean(self.total_q_val):.3f} | AvgLoss: {np.mean(self.total_loss_val):.3f} | Mode: {mode}")
                    # print(
                    #         f"Episode: {i} | Timestep: {self.total_step_tracker} | Epsilon: {self.epsilon:.3f} | Reward: {self.total_rewards[i % self.capture_window]} | Avg Reward: {np.mean(self.total_rewards):.3f} | AvgQ: {np.mean(self.total_q_val):.3f} | AvgLoss: {np.mean(self.total_loss_val):.3f} | Mode: {mode}",
                    #         file=self.log)
                    print(np.mean(self.q_values))
                    print(np.mean(self.last_n_rewards))
                    print(sum(self.rewards))
                    print(np.mean(self.total_loss))
                    print(
                            f"Episode: {i} | Timestep: {self.total_step_tracker} | Epsilon: {self.epsilon:.3f} | Reward: {sum(self.rewards)} | Avg Reward: {np.mean(self.last_n_rewards):.3f} | AvgQ: {np.mean(self.q_values):.3f} | AvgLoss: {np.mean(self.total_loss):.3f} | Mode: {mode}")
                    print(
                            f"Episode: {i} | Timestep: {self.total_step_tracker} | Epsilon: {self.epsilon:.3f} | Reward: {sum(self.rewards)} | Avg Reward: {np.mean(self.last_n_rewards):.3f} | AvgQ: {np.mean(self.q_values):.3f} | AvgLoss: {np.mean(self.total_loss):.3f} | Mode: {mode}",
                            file=self.log)
                    self.log_summary(global_step=i, test=False)
                    self.rewards = []
                    self.q_values = []
                    self.total_loss = []

                self.total_step_tracker += 1
                observation = new_observation
