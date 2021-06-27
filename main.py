import gym
import torch
import argparse
from policy import Policy
from agents import ExpertAgent, CloneAgent
from evaluate import evaluate


def main():
    # Define main inputs
    parser = argparse.ArgumentParser(description='Train and evaluate behaviour cloning for continuous control')
    parser.add_argument('--env_name', default='LunarLanderContinuous-v2', type=str, help='Environment to be cloned')
    parser.add_argument('--hidden_dims', default=512, type=int, help='Number of units in hidden layers')
    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate for behavior cloning')
    parser.add_argument('--n_epochs', default=10000, type=int, help='Number of training epochs')
    parser.add_argument('--dir_demo', default='./demos_heuristic', type=str, help='Demonstrations directory')
    parser.add_argument('--dir_trained', default='./trained_agents', type=str, help='Trained agents bunker')
    parser.add_argument('--dir_results', default='./results', type=str, help='Demonstrations directory')
    parser.add_argument('--n_demo_traj', default=10, type=int, help='Number of trajectories in demonstration set ')
    parser.add_argument('--n_eval_traj', default=10, type=int, help='Number of trajectories for evaluation')
    parser.add_argument('--screen_update_freq', default=10, type=int, help='How often training loss is shown')
    parser.add_argument('--min_score', default=200., type=float, \
                        help='Minimum score for solving the game')
    args = parser.parse_args()

    print(f'{"  Starting behavior cloning  ":#^60} ')

    # Instantiates environment, expert and clone agent
    env = gym.make(args.env_name)

    policy = Policy(input_dims=env.observation_space.shape[0],
                    hidden_dims=args.hidden_dims,
                    n_actions=env.action_space.shape[0])

    expert = ExpertAgent(env=env,
                         n_demo_traj=args.n_demo_traj,
                         dir_demo=args.dir_demo,
                         min_score=args.min_score)

    clone_agent = CloneAgent(env=args.env_name,
                             policy=policy,
                             lr=args.lr,
                             dir_demo=args.dir_demo,
                             dir_trained=args.dir_trained,
                             dir_results=args.dir_results,
                             n_epochs=args.n_epochs,
                             n_demo_traj=args.n_demo_traj)

    # Generates expert trajectories data and saves demonstrations to file
    path_states, path_actions = expert.generate_trajectories()
    clone_agent.load_heuristics(path_states, path_actions)

    # Trains behavior clone
    path_agent = clone_agent.train()

    # Saves trained model to file
    print(f'--> Saving agent to\n {path_agent}')
    torch.save(clone_agent, path_agent)
    clone_agent = torch.load(path_agent)

    # Performs a sanity check on training data, training loss and saves to file
    clone_agent.check_against_traindata(path_agent, args)

    # Evaluates trained agent and saves results to file
    evaluate(env=env,
             path_agent=path_agent,
             args=args)

    env.close()

    print(f'{"  Finishing behavior cloning  ":#^60} ')


if __name__ == '__main__':
    main()
