from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical

class SimpleDiscreteMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=128,
                 output_softmax=False,
                 activation=F.relu,
                 ):
        super(SimpleDiscreteMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation
        self.output_softmax = output_softmax
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        logits = self.fc2(x)
        if self.output_softmax:
            logits = F.softmax(logits, dim=-1)
        return logits
    
    def get_distribution(self, x):
        logits = self.forward(x)
        return Categorical(logits=logits)

    def sample_action(self, x):
        distribution = self.get_distribution(x)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.unsqueeze(-1), log_prob.unsqueeze(-1)

class SimpleContinuousMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=128,
                 activation=F.relu,
                 ):
        super(SimpleContinuousMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2) # mean and std
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        return mean, std
    
    def action_distribution(self, x):
        mean, std = self.forward(x)
        return torch.distributions.Normal(mean, std)
    