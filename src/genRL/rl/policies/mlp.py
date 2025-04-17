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
    
    def action_distribution(self, x):
        logits = self.forward(x)
        return Categorical(logits=logits)

    def sample_action(self, s, action=None, eval_entropy=False):
        # assume action is [batch, timesteps, 1]
        distribution = self.action_distribution(s)
        action = distribution.sample() if action is None else action.squeeze(-1)
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy() if eval_entropy else None
        
        return action.unsqueeze(-1), log_prob.unsqueeze(-1), entropy

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
    
    def sample_action(self, s, action=None, eval_entropy=False):
        # assume action is [batch, timesteps, action_dim]
        distribution = self.action_distribution(s)
        action = distribution.rsample() if action is None else action
        log_prob = distribution.log_prob(action)
        log_prob = log_prob.sum(-1, keepdim=True)
        entropy = distribution.entropy() if eval_entropy else None
        return action, log_prob, entropy
    