import argparse
from DQNUpdated.agent_dqn import Agent_DQN
from DQNUpdated.environment import Environment
import numpy as np
import torch

seed = 11037


def parse():
    parser = argparse.ArgumentParser(description="runner")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN', default=True)
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--run_name', type=str, default='dqn_model', help='')
    parser.add_argument('--model_save_path', type=str, default='trained_models', help='')
    parser.add_argument('--model_save_interval', type=int, default=20000, help='')
    parser.add_argument('--log_path', type=str, default='train_log.out', help='')
    parser.add_argument('--tensorboard_summary_path', type=str, default='tensorboard_summary', help='')
    parser.add_argument('--model_test_path', type=str, default='dqn_model_5000.pt', help='')
    parser.add_argument('--metrics_capture_window', type=int, default=10000, help='')
    parser.add_argument('--replay_size', type=int, default=10000, help='')
    parser.add_argument('--total_num_steps', type=int, default=5e7, help='')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='')
    parser.add_argument('--gamma', type=float, default=0.99, help='')
    parser.add_argument('--initial_epsilon', type=float, default=1.0, help='')
    parser.add_argument('--final_epsilon', type=float, default=0.005, help='')
    parser.add_argument('--steps_to_explore', type=int, default=1000000, help='')
    parser.add_argument('--network_update_interval', type=int, default=5000, help='')
    parser.add_argument('--episodes', type=int, default=50000000, help='')
    parser.add_argument('--network_train_interval', type=int, default=4, help='')
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


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0

        # playing one game
        while not done:
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
    print('Run %d episodes' % (total_episodes))
    print('Mean:', np.mean(rewards))


if __name__ == '__main__':
    args = parse()
    run(args)
