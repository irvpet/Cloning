# Expert agent
# Behavior cloning agent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from OpenAI_env_source_code import heuristic
import sys


class ExpertAgent():
    def __init__(self, env, n_demo_traj, dir_demo, min_score):
        self.env = env
        self.n_demo_traj = n_demo_traj
        self.states = []
        self.actions = []
        self.scores = np.zeros(n_demo_traj)
        self.dir_demo = dir_demo
        self.min_score = min_score

    def generate_trajectories(self):
        """Expert generated trajectories"""
        print(f'\n... Generating expert trajectories ...')
        demo_quality_reached = False

        while not demo_quality_reached:
            for i in range(self.n_demo_traj):
                print(f'Traj {i}   ', end='')
                state = self.env.reset()
                done = False
                score = 0
                while not done:
                    action = heuristic(self.env, state)
                    state_, reward, done, info = self.env.step(action)
                    self.states.append(state)
                    self.actions.append(action)
                    score += reward
                    state = state_
                    #self.env.render()

                print(f'Score {score:.0f}')
                self.scores[i] = score

            if np.min(self.scores) >= self.min_score:
                demo_quality_reached = True
                print(f'\n--> Minimum score limit achieved\n')
            else:
                demo_quality_reached = False
                print(f'\n--> Minimum score limit not achieved \n Generating new trajectories...')

        self.env.close()

        states = np.array(self.states)
        actions = np.array(self.actions)

        file_name_obs = 'expert_heuristic_states_ntraj=' + str(self.n_demo_traj) + '.npy'
        file_name_acs = 'expert_heuristic_actions_ntraj=' + str(self.n_demo_traj) + '.npy'

        path_states = self.dir_demo + '/' + file_name_obs
        path_actions = self.dir_demo + '/' + file_name_acs

        print(f'--> Saving state action pairs')
        print(f'{path_states}')
        np.save(path_states, states)
        print(f'{path_actions}')
        np.save(path_actions, actions)

        return path_states, path_actions


class CloneAgent():
    def __init__(self, env, policy, lr, dir_demo, dir_trained, dir_results, n_epochs, n_demo_traj):
        self.env = env
        self.policy = policy
        self.lr = lr
        self.dir_demo = dir_demo
        self.dir_results = dir_results
        self.dir_trained = dir_trained
        self.n_epochs = n_epochs
        self.n_demo_traj = n_demo_traj
        self.optim = optim.Adam(params=policy.parameters(), lr=lr)        
        self.states = None
        self.actions = None
        self.loss_history = []


    def load_heuristics(self, path_states, path_actions):
        """Loads demonstrations generated by heuristics from Open AI original source code """

        try:
            self.states = np.load(path_states, allow_pickle=False)
            self.actions = np.load(path_actions, allow_pickle=False)

        except:
            print(f'Unable to load the following files')
            print(f'{path_states}')
            print(f'{path_actions}')

            sys.exit('... Terminated ...')


    def train(self):
        """Trains behavior clone agent on loaded data"""

        print(f'\n... Starting agent training...')
        screen_update_freq = 100

        agent_file_name = 'bc_' + str(self.env) + '_nepochs=' + str(self.n_epochs) + '_lr=' + str(self.lr) + \
                          '_n_traj=' + str(self.n_demo_traj) + '_model.pth'

        states_pth = torch.Tensor(self.states).to(self.policy.device)
        actions_pth = torch.Tensor(self.actions).to(self.policy.device)

        loss_fn = nn.MSELoss()

        for i in range(self.n_epochs):
            prev_actions = self.policy(states_pth)
            loss = loss_fn(prev_actions, actions_pth)
            self.loss_history.append(loss.item())

            if i % screen_update_freq == 0:
                print(f'Epoch {i}, loss: {loss.item():.6f}')

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        path_agent = self.dir_trained + str("/") + agent_file_name

        return path_agent

    def check_against_traindata(self, path_agent, args):
        """Checks if policy can reproduce training data pattern
           Shows training error evolution"""

        states_pth = torch.Tensor(self.states).to(self.policy.device)
        prev_actions_pth = self.policy(states_pth)

        prev_actions = prev_actions_pth.detach().cpu().numpy()

        n_actions = prev_actions_pth.size()[1]
        fig, ax = plt.subplots(1, n_actions + 1, figsize=(12, 6))
        fig.suptitle('Sanity check on training data & training loss', fontsize=16)

        action_labels = ['Main engine', 'Lateral engines']

        for i in range(n_actions):
            ax[i].plot(self.actions[:, i], label=f'Expert')
            ax[i].plot(prev_actions[:, i], label=f'Clone')
            ax[i].set_title(action_labels[i])
            ax[i].set_ylabel('Level of engine deployment')
            ax[i].set_xlabel('Time step')
            ax[i].legend()

        ax[n_actions].plot(self.loss_history, label='Training loss')
        ax[n_actions].set_xlabel('Epochs')
        ax[n_actions].set_title(f'Learning rate = {self.lr}')
        ax[n_actions].legend()
        plt.tight_layout(pad=1.5, w_pad=1.2, h_pad=2.0)

        fig_name = 'SanityCheck_bc_' + str(args.env_name) + '_nepochs=' + str(args.n_epochs) + '_lr=' \
                   + str(args.lr) + '_n_traj=' + str(args.n_demo_traj) + '_.png'

        fig_path = self.dir_results + '/' + fig_name

        print(f'--> Saving sanity check to: {fig_path}')
        plt.savefig(fig_path)
