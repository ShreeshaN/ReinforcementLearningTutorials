def add_arguments(parser):
    parser.add_argument('--run_name', type=str, default="dqn_network_for_breakout")
    parser.add_argument('--use_ddqn', type=bool, default=False)
    parser.add_argument('--use_dueling_network', type=bool, default=False)
    parser.add_argument('--logfile_path', type=str, default='dqn_log')
    parser.add_argument('--total_episodes', type=int, default=100000)
    parser.add_argument('--capture_window', type=int, default=100, help='Capture average metrics of last "n" episodes')
    parser.add_argument('--retrain', type=bool, default=False,
                        help='Make it true to start training network from the pretrained model mentioned in "test_dqn_model_path"')
    parser.add_argument('--train_metrics_file', type=str, help='Json file that saves metrics. Loaded while retraining',
                        default='metrics.json')
    parser.add_argument('--tensorboard_summary', type=str, default='tensorboard_summary')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--loss_fn', type=str, default='huber')
    parser.add_argument('--exploration_steps', type=int, default=1000000)
    parser.add_argument('--initial_epsilon', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=0.00015)
    parser.add_argument('--final_epsilon', type=float, default=0.1)
    parser.add_argument('--initial_replay_size', type=int, default=10000,
                        help='Number of steps to populate the replay memory before training starts')
    parser.add_argument('--log_path', type=str, default="dqn_log/", help='')
    parser.add_argument('--num_replay_memory', type=int, default=10000,
                        help='Number of replay memory the agent uses for training')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target_update_interval', type=int, default=5000,
                        help='The frequency with which the target network is updated')
    parser.add_argument('--train_interval', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=25000,
                        help='The frequency with which the network is saved')
    parser.add_argument('--save_network_path', type=str, default="dqn_models/")
    parser.add_argument('--test_dqn_model_path', type=str,
                        default="/Users/badgod/Documents/Documents - badgod's MacBook Pro/DOCS__/MS in DS/Semester 1/DS595 - RL/homeworks/project3_submission/project-3 submission/model_file/dqn_network_for_breakout.pt",
                        help='')
    return parser
