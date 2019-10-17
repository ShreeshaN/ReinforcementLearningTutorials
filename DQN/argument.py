def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--frame_width', type=int, default=84, help='Resized frame width')
    parser.add_argument('--frame_height', type=int, default=84, help='Resized frame height')
    parser.add_argument('--num_steps', type=int, default=10000000, help='Number of episodes the agent plays')
    parser.add_argument('--state_length', type=int, default=4,
                        help='Number of most recent frames to produce the input to the network')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--exploration_steps', type=int, default=1000000,
                        help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')
    parser.add_argument('--initial_epsilon', type=float, default=0.99, help='Initial value of epsilon in epsilon-greedy')
    parser.add_argument('--final_epsilon', type=float, default=0.05, help='Final value of epsilon in epsilon-greedy')
    parser.add_argument('--initial_replay_size', type=int, default=10000,
                        help='Number of steps to populate the replay memory before training starts')
    parser.add_argument('--num_replay_memory', type=int, default=10000,
                        help='Number of replay memory the agent uses for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size')
    parser.add_argument('--target_update_interval', type=int, default=5000,
                        help='The frequency with which the target network is updated')
    parser.add_argument('--train_interval', type=int, default=4,
                        help='The agent selects 4 actions between successive updates')
    parser.add_argument('--save_summary_path', type=str, default="dqn_summary/", help='')
    parser.add_argument('--learning_rate', type=float, default=0.00015, help='Learning rate used by RMSProp')
    parser.add_argument('--save_interval', type=int, default=100000,
                        help='The frequency with which the network is saved')
    parser.add_argument('--no_op_steps', type=int, default=2,
                        help='Maximum number of "do nothing" actions to be performed by the agent at the start of an episode')
    parser.add_argument('--save_network_path', type=str, default="saved_dqn_networks/", help='')
    # parser.add_argument('--test_dqn_model_path', type=str, default="saved_dqn_networks/breakout_dqn_4700000.pt", help='')
    parser.add_argument('--test_dqn_model_path', type=str, default="saved_dqn_networks/breakout_dqn_6000000.pt", help='')

    parser.add_argument('--exp_name', type=str, default="breakout_dqn", help='')
    parser.add_argument('--ddqn', type=bool, default=False, help='Set True to apply Double Q-learning')
    parser.add_argument('--dueling', type=bool, default=False, help='Set True to apply Duelinng Network')
    parser.add_argument('--test_path', type=str, default='saved_dqn_networks/breakout_dqn_4700000.pt',
                        help='Model used during testing')
    parser.add_argument('--logfile_path', type=str, default='dqn_log',
                        help='File used for logging')
    return parser