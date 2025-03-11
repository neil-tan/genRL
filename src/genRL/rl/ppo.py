import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self,
                 learning_rate=0.0005,
                 gamma=0.98,
                 lmbda=0.95,
                 eps_clip=0.1,
                 normalize_advantage=True,
                 K_epoch=3,):
        super(PPO, self).__init__()
        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.normalize_advantage = normalize_advantage
        self.K_epoch = K_epoch
        
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = -1):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    @torch.no_grad()
    def put_data(self, transition):
        self.data.append(transition)
    
    @torch.no_grad()
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done.all() else 1
            done_lst.append([done_mask])
            
        # s,a,r,s_prime,done_mask, prob_a
        ret = torch.stack(s_lst).transpose(0,1), torch.stack(a_lst).transpose(0,1), \
              torch.stack(r_lst).transpose(1,0), torch.stack(s_prime_lst).transpose(0,1), \
              torch.tensor(done_lst).transpose(1,0), torch.stack(prob_a_lst).transpose(1,0)
                                          
        ret = tuple(x.detach() for x in ret)
        self.data = []
        return ret
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K_epoch):
            with torch.no_grad():
                values = self.v(s).squeeze(-1)
                values_prime = self.v(s_prime).squeeze(-1)
                
                td_target = r + self.gamma * values_prime * done_mask
                delta = td_target - values
            
                advantages = torch.zeros_like(delta)
                last_gae = 0
                for t in reversed(range(delta.shape[-1])):
                    last_gae = delta[:, t] + self.gamma * self.lmbda * last_gae * done_mask[:, t]
                    advantages[:, t] = last_gae
            
                advantages = self.get_normalized_advantage(advantages)

                # for boardcasting
                advantages = advantages.unsqueeze(-1)
                td_target = td_target.unsqueeze(-1)

            pi = self.pi(s, softmax_dim=-1)
            pi_a = pi.gather(-1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def get_normalized_advantage(self, advantages):
        if self.normalize_advantage and advantages.shape[0] > 1:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
