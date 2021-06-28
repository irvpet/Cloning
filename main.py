import gym
import torch
from policy import Policy
from agents import ExpertAgent, CloneAgent
from evaluate import evaluate
from arguments import parse_arguments


def main():

    print(f'{"  Starting behavior cloning  ":#^60} ')

    # Instantiates environment, expert and clone agent
    args = parse_arguments()

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
