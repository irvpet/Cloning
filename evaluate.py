# Helper function to evaluate trained behavior cloning model

import torch
import matplotlib.pyplot as plt
import numpy as np


def evaluate(env, path_agent, args):
    """ Evaluates model on new, unseen states generated sequentially by the environment"""

    clone_agent = torch.load(path_agent)
    score_history = []
    victories = 0

    print(f'\n... Starting {args.n_eval_traj} episodes for evaluation of model... \n model={path_agent}')
    for i in range(args.n_eval_traj):
        state = env.reset()
        done = False
        score = 0

        while not done:
            state_pth = torch.Tensor(state).unsqueeze(0).to(clone_agent.policy.device)
            action_pth = clone_agent.policy(state_pth)

            action = action_pth.squeeze(0).cpu().detach().numpy()
            state_, reward, done, info = env.step(action)
            score += reward
            state = state_
            #env.render()

        victories += 1 if score > args.min_score else 0

        print(f'Episode {i:2}, Reward: {score:.1f}')
        score_history.append(score)

    threshold = np.ones_like(score_history)*args.min_score

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(f'Average victory rate: {victories/args.n_eval_traj*100}%\n Model:{path_agent}')
    ax.plot(score_history, marker='o', linestyle='', label='Episode score')
    ax.plot(threshold, linestyle='-', label='Threshold for wining')
    ax.set_ylabel('Score')
    ax.set_xlabel('Test episode')
    ax.legend()

    fig_name = 'Evaluation_bc_' + str(args.env_name) + '_nepochs=' + str(args.n_epochs) + '_lr=' \
                      + str(args.lr) + '_n_traj=' + str(args.n_demo_traj) + '_.png'

    path = args.dir_results + '/' + fig_name
    print(f'\n--> Saving results to \n{path}')
    plt.savefig(path)

    #plt.show()

