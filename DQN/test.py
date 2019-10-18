"""

### NOTICE ###
DO NOT revise this file

"""

import argparse
import numpy as np
from DQN.environment import Environment

seed = 11037


def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN', default=True)
    try:
        from DQN.argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def norm_state(state):
    state_min = np.min(state)
    state_max = np.max(state)
    return (state - state_min) / (state_max - state_min)


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):

        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0
        episode_num_states = 0

        # playing one game
        while (not done):
            state = np.rollaxis(state, 2)
            # state = norm_state(state)
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_num_states += 1

        rewards.append(episode_reward)
        print('Running ', i, ' | Episode reward ', episode_reward, ' | Number of states ', episode_num_states,
              ' | Moving average Reward ', np.mean(rewards))
    print('Run %d episodes' % (total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)

        from DQN.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)

        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
