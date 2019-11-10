def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--run_name', type=str, default='dqn_model', help='')
    parser.add_argument('--model_save_path', type=str, default='trained_models', help='')
    parser.add_argument('--model_save_interval', type=int, default=20000, help='')
    parser.add_argument('--log_path', type=str, default='train_log.out', help='')
    parser.add_argument('--tensorboard_summary_path', type=str, default='tensorboard_summary', help='')
    parser.add_argument('--model_test_path', type=str, default='dqn_model_5000.pt', help='')
    parser.add_argument('--metrics_capture_window', type=int, default=100, help='')
    parser.add_argument('--replay_size', type=int, default=10000, help='')
    parser.add_argument('--total_num_steps', type=int, default=5e7, help='')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='')
    parser.add_argument('--gamma', type=float, default=0.99, help='')
    parser.add_argument('--initial_epsilon', type=float, default=1.0, help='')
    parser.add_argument('--final_epsilon', type=float, default=0.005, help='')
    parser.add_argument('--steps_to_explore', type=int, default=100000, help='')
    parser.add_argument('--network_update_interval', type=int, default=5000, help='')
    return parser
