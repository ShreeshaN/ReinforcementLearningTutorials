# -*- coding: utf-8 -*-
"""
@created on: 11/14/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

"""

### NOTICE ###
You DO NOT need to upload this file

"""

import argparse
import numpy as np
from DQN.environment import Environment
import time

seed = 11037


def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project3")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN', default=True)
    try:
        pass
        # from argument import add_arguments
        # parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    start_time = time.time()
    for i in range(total_episodes):
        state = env.reset()

        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        # playing one game
        frames = [state]
        while not done:
            env.env.render()
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            frames.append(state)

    print('Run %d episodes' % (total_episodes))
    print('Mean:', np.mean(rewards))
    print('rewards', rewards)
    print('running time', time.time() - start_time)


def run(args):
    env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
    from DQN.agent_dqn import Agent_DQN
    agent = Agent_DQN(env, args)
    test(agent, env, total_episodes=5)


if __name__ == '__main__':
    args = parse()
    run(args)
