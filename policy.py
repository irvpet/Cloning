# Policy for Behavior Cloning

import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, input_dims, hidden_dims, n_actions):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dims, out_features=hidden_dims)
        self.fc2 = nn.Linear(in_features=hidden_dims, out_features=hidden_dims)
        self.fc3 = nn.Linear(in_features=hidden_dims, out_features=n_actions)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, state):
        activation = torch.relu(self.fc1(state))
        activation = torch.relu(self.fc2(activation))
        output = self.fc3(activation)

        return output


def test_policy():
    placeholder_state = torch.Tensor([0., 1., 2., 3., 4., 5., 6., 7.])
    input_dims = 8
    hidden_dims = 64
    n_actions = 2
    policy = Policy(input_dims, hidden_dims, n_actions)
    output = policy(placeholder_state)

    print(f'Input:\n {placeholder_state}\n')
    print(f'Policy:\n{policy}\n')
    print(f'Output:\n {output}')


if __name__ == '__main__':
    #test_policy()
    policy = Policy(8, 64, 2)
    print(policy)