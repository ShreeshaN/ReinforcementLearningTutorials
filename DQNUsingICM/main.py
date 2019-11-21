import argparse
from DQNUsingICM.agent_dqn import Agent_DQN
from DQNUsingICM.environment import Environment
import numpy as np
from torch import tensor
import torch
import time

seed = 11037


def parse():
    parser = argparse.ArgumentParser(description="runner")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--run_name', type=str, default='icm_dqn', help='')
    parser.add_argument('--model_save_path', type=str, default='trained_models', help='')
    parser.add_argument('--model_save_interval', type=int, default=500, help='')
    parser.add_argument('--log_path', type=str, default='train_log.out', help='')
    parser.add_argument('--tensorboard_summary_path', type=str, default='tensorboard_summary', help='')
    parser.add_argument('--model_test_path', type=str, default='/Users/badgod/Downloads/dqn_model_23500.pt', help='')
    parser.add_argument('--metrics_capture_window', type=int, default=100, help='')
    parser.add_argument('--replay_size', type=int, default=10000, help='')
    parser.add_argument('--start_to_learn', type=int, default=5000, help='')
    parser.add_argument('--total_num_steps', type=int, default=5e7, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='')
    parser.add_argument('--gamma', type=float, default=0.99, help='')
    parser.add_argument('--initial_epsilon', type=float, default=1.0, help='')
    parser.add_argument('--final_epsilon', type=float, default=0.005, help='')
    parser.add_argument('--steps_to_explore', type=int, default=5000000, help='')
    parser.add_argument('--network_update_interval', type=int, default=5000, help='')
    parser.add_argument('--episodes', type=int, default=50000000, help='')
    parser.add_argument('--network_train_interval', type=int, default=10, help='')
    parser.add_argument('--ddqn', type=bool, default=False, help='')
    parser.add_argument('--use_icm', type=bool, default=True, help='')
    parser.add_argument('--beta', type=float, default=0.2, help='')
    parser.add_argument('--lambda_val', type=float, default=0.1, help='')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


# def test(agent, env, total_episodes=30):
#     rewards = []
#     env.seed(seed)
#     for i in range(total_episodes):
#         state = env.reset()
#         done = False
#         episode_reward = 0.0
#         count = 0
#
#         # playing one game
#         while not done:
#             count += 1
#             state = tensor(np.rollaxis(state, 2)).unsqueeze(0)
#             action = agent.make_action(state, test=True)
#             state, reward, done, info = env.step(action)
#             episode_reward += reward
#             if count > 5000:
#                 break
#         rewards.append(episode_reward)
#         print('Episode', i, '. . . Reward', episode_reward, '. . . Avg Reward', np.mean(rewards), '. . . States', count)
#     print('Run %d episodes' % (total_episodes))
#     print('Mean:', np.mean(rewards))
def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    start_time = time.time()
    for i in range(total_episodes):
        count = 0
        state = env.reset()

        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        # playing one game
        # frames = [state]
        while not done:
            count += 1
            # env.env.render()
            state = tensor(np.rollaxis(state, 2)).unsqueeze(0)
            action = agent.make_action(state, state_count=count, test=True, )
            state, reward, done, info = env.step(action)
            episode_reward += reward
            # frames.append(state)
        rewards.append(episode_reward)
        print('Episode', i, '. . . Reward', episode_reward, '. . . Avg Reward', np.mean(rewards), '. . . States',
              count)
    print('Run %d episodes' % (total_episodes))
    print('Mean:', np.mean(rewards))
    print('rewards', rewards)
    print('running time', time.time() - start_time)


if __name__ == '__main__':
    args = parse()
    run(args)
