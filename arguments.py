# Defines all simulation parameters

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate behaviour cloning for continuous control')
    parser.add_argument('--env_name', default='LunarLanderContinuous-v2', type=str, help='Environment to be cloned')
    parser.add_argument('--hidden_dims', default=512, type=int, help='Number of units in hidden layers')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for behavior cloning')
    parser.add_argument('--n_epochs', default=3000, type=int, help='Number of training epochs')
    parser.add_argument('--dir_demo', default='./demos_heuristic', type=str, help='Demonstrations directory')
    parser.add_argument('--dir_trained', default='./trained_agents', type=str, help='Trained agents bunker')
    parser.add_argument('--dir_results', default='./results', type=str, help='Demonstrations directory')
    parser.add_argument('--n_demo_traj', default=10, type=int, help='Number of trajectories in demonstration set ')
    parser.add_argument('--n_eval_traj', default=1000, type=int, help='Number of trajectories for evaluation')
    parser.add_argument('--screen_update_freq', default=10, type=int, help='How often training loss is shown')
    parser.add_argument('--min_score', default=200., type=float, help='Minimum score for solving the game')
    args = parser.parse_args()

    return args