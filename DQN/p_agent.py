#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from itertools import count
import gc
from DQN.agent import Agent
from DQN.p_model import DQN
import time
from torch.autograd import Variable
import json
import uuid

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class JsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            try:
                return obj.default()
            except Exception:
                return f'Object not serializable - {obj}'


class MetaData(object):
    """
    Medata for model monitor and restore purpose
    """

    def __init__(self, fp, args):
        self.transition = namedtuple('Data',
                                     (
                                         "episode", "step", "time", "time_elapsed", "ep_len", "buffer_len", "epsilon",
                                         "reward",
                                         "avg_reward", "max_q", "max_avg_q",
                                         "loss", "avg_loss", "mode"))
        self.fp = fp
        self.data = None
        self.args = args

    def update(self, *args):
        """
        Update metadata
        :param args: args
        """
        self.data = self.transition(*args)
        if self.data.episode % self.args.disp_freq == 0:
            # print(
            #         f"E: {self.data.episode} |  Step: {self.data.step} | T: {self.data.time:.2f} | ET: {self.data.time_elapsed:.2f}"
            #         f" | Len: {self.data.ep_len} | EPS: {self.data.epsilon:.5f} | R: {self.data.reward} | AR: {self.data.avg_reward:.3f}"
            #         f" | MAQ:{self.data.max_avg_q:.2f} | L: {self.data.loss:.2f} | AL: {self.data.avg_loss:.4f} | Mode: {self.data.mode}")

            print("E: ", self.data.episode, " |  Step: ", self.data.step, " | T: ", self.data.time, " | ET: ",
                  self.data.time_elapsed, "| Len: ", self.data.ep_len, " | EPS: ", self.data.epsilon, " | R: ",
                  self.data.reward, " | AR: ", self.data.avg_reward, " | MAQ:", self.data.max_avg_q, " | L: ",
                  self.data.loss, " | AL: ", self.data.avg_loss, " | Mode: ", self.data.mode)
        self.fp.write(self.data._asdict().values().__str__().replace('odict_values([', '').replace('])', '' + '\n'))

    def load(self, f):
        """
        Load Metadata
        :param f: File Pointer
        :return:
        """
        self.data = self.transition(*json.load(f).values())

    def dump(self, f):
        """
        JSONify metadata
        :param f: file pointer
        """
        json.dump(self.data._asdict(), f, cls=JsonEncoder, indent=2)


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

        # Declare variables
        self.exp_id = uuid.uuid4().__str__().replace('-', '_')
        self.args = args
        self.env = env
        self.eps_threshold = None
        self.nA = env.action_space.n
        self.action_list = np.arange(self.nA)
        self.reward_list = np.zeros(args.window, np.float32)
        self.max_q_list = np.zeros(args.window, np.float32)
        self.loss_list = np.zeros(args.window, np.float32)
        self.probability_list = np.zeros(env.action_space.n, np.float32)
        self.cur_eps = None
        self.t = 0
        self.ep_len = 0
        self.mode = None
        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))
        self.args.save_dir += f'/{self.exp_id}/'
        os.system(f"mkdir -p {self.args.save_dir}")
        self.meta = MetaData(fp=open(os.path.join(self.args.save_dir, 'result.csv'), 'w'), args=self.args)
        self.eps_delta = (self.args.eps - self.args.eps_min) / self.args.eps_decay_window
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")

        # Create Policy and Target Networks
        self.policy_net = DQN(env, args).to(self.device)
        self.target_net = DQN(env, args).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1.5e-4, eps=0.001)
        # Compute Huber loss
        self.loss = F.smooth_l1_loss

        # todo: Support for Multiprocessing. Bug in pytorch - https://github.com/pytorch/examples/issues/370
        # self.policy_net.share_memory()
        # self.target_net.share_memory()

        # Set defaults for networks
        self.policy_net.train()
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if args.test_dqn:
            # you can load your model here
            ###########################
            # YOUR IMPLEMENTATION HERE #
            print('loading trained model')
            self.load_model()

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
        with torch.no_grad():
            # Fill up probability list equal for all actions
            self.probability_list.fill(self.cur_eps / self.nA)
            # Fetch q from the model prediction
            q, argq = self.policy_net(Variable(self.channel_first(observation))).data.cpu().max(1)
            # Increase the probability for the selected best action
            self.probability_list[argq[0].item()] += 1 - self.cur_eps
            # Use random choice to decide between a random action / best action
            action = torch.tensor([np.random.choice(self.action_list, p=self.probability_list)])
            if not self.args.test_dqn:
                return action, q
        ###########################
        return action.item()

    def push(self, *args):
        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.memory) < self.args.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.args.capacity
        ###########################

    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        sample = random.sample(self.memory, batch_size)
        sample = map(lambda x: Variable(torch.cat(x, 0)), zip(*sample))
        ###########################
        return sample

    def optimize_model(self):
        """
        Function to perform optimization on DL Network
        :return: Loss
        """
        # Return if initial buffer is not filled.
        if len(self.memory) < self.args.mem_init_size:
            return 0

        self.mode = "Explore"
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = self.replay_buffer(self.args.batch_size)
        policy_max_q = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        target_max_q = self.target_net(batch_next_state).detach().max(1)[0].squeeze(0) * self.args.gamma * (
                1 - batch_done)

        # Compute Huber loss
        loss = self.loss(policy_max_q, batch_reward + target_max_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip rewards between -1 and 1
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss.cpu().detach().numpy()

    def channel_first(self, state):
        """
        The action returned from the environment is nhwc, hence convert to nchw
        :param state: state
        :return: nchw state
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.shape[1] == 4:
            return state
        return torch.reshape(state, [1, 84, 84, 4]).permute(0, 3, 1, 2)

    def save_model(self, i_episode):
        """
        Save Model based on condition
        :param i_episode: Episode Number
        """
        if i_episode % self.args.save_freq == 0:
            model_file = os.path.join(self.args.save_dir, f'model_e{i_episode}.th')
            meta_file = os.path.join(self.args.save_dir, f'model_e{i_episode}.meta')
            print("Saving model at ", model_file)
            with open(model_file, 'wb') as f:
                torch.save(self.policy_net, f)
            with open(meta_file, 'w') as f:
                self.meta.dump(f)

    def collect_garbage(self, i_episode):
        """
        Collect garbage based on condition
        :param i_episode: Episode Number
        """
        if i_episode % self.args.gc_freq == 0:
            print("Executing garbage collector . . .")
            gc.collect()

    def load_model(self):
        """
        Load Model
        :return:
        """
        print("Restoring model from ", self.args.load_dir)
        self.policy_net = torch.load(self.args.load_dir,
                                     map_location=self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if not self.args.test_dqn:
            self.meta.load(open(self.args.load_dir.replace('.th', '.meta')))
            self.t = self.meta.data.step
        else:
            self.cur_eps = 0.01
        print("Model successfully restored.")

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.t = 1
        self.mode = "Random"
        train_start = time.time()
        if not self.args.load_dir == '':
            self.load_model()
        for i_episode in range(1, self.args.max_episodes + 1):
            # Initialize the environment and state
            start_time = time.time()
            state = self.channel_first(self.env.reset())
            self.reward_list[i_episode % self.args.window] = 0
            self.loss_list[i_episode % self.args.window] = 0
            self.max_q_list[i_episode % self.args.window] = -1e9
            self.ep_len = 0
            done = False

            # Save Model
            self.save_model(i_episode)
            # Collect garbage
            self.collect_garbage(i_episode)

            # Run the game
            while not done:
                # Update the target network, copying all weights and biases in DQN
                if self.t % self.args.target_update == 0:
                    print("Updating target network . . .")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                # Select and perform an action
                self.cur_eps = max(self.args.eps_min, self.args.eps - self.eps_delta * self.t)
                if self.cur_eps == self.args.eps_min:
                    self.mode = 'Exploit'
                action, q = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = self.channel_first(next_state)
                reward = torch.tensor([reward], device=self.device)
                # Store the transition in memory
                self.push(state, torch.tensor([int(action)]), next_state, reward,
                          torch.tensor([done], dtype=torch.float32))

                self.reward_list[i_episode % self.args.window] += reward
                self.max_q_list[i_episode % self.args.window] = max(self.max_q_list[i_episode % self.args.window],
                                                                    q[0].item())

                # Increment step and Episode Length
                self.t += 1
                self.ep_len += 1

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if self.ep_len % self.args.learn_freq == 0:
                    loss = self.optimize_model()
                    self.loss_list[i_episode % self.args.window] += loss

            # Update meta
            self.meta.update(i_episode, self.t, time.time() - start_time, time.time() - train_start,
                             self.ep_len, len(self.memory), self.cur_eps,
                             self.reward_list[i_episode % self.args.window], np.mean(self.reward_list),
                             self.max_q_list[i_episode % self.args.window], np.mean(self.max_q_list),
                             self.loss_list[i_episode % self.args.window], np.mean(self.loss_list),
                             self.mode)

        ###########################
