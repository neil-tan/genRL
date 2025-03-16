import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from genRL.utils import masked_mean, masked_sum, masked_var, masked_std, normalize_advantage


class SimpleMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 softmax_output,
                 hidden_dim=128,
                 activation=F.relu,
                 ):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation
        self.softmax_output = softmax_output
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        if self.softmax_output:
            return F.softmax(x, dim=-1)
        return x

class PPO(nn.Module):
    def __init__(self,
                 pi,
                 v,
                 learning_rate=0.0005,
                 gamma=0.98,
                 lmbda=0.95,
                 value_loss_coef=0.5,
                 eps_clip=0.1,
                 max_grad_norm=0.5,
                 normalize_advantage=True,
                 K_epoch=3,
                 wandb_run=None,
                 **kwargs,):
        super(PPO, self).__init__()
        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.value_loss_coef = value_loss_coef
        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.K_epoch = K_epoch
        
        self.pi = pi
        self.v = v
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        if wandb_run:
            self.wandb_run = wandb_run
    
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
            done_lst.append(done) # invert done_mask
            
        # s,a,r,s_prime,done_mask, prob_a
        
        done_mask = torch.stack(done_lst).transpose(1,0)
        # done_mask = torch.cat([torch.zeros_like(done_mask[:, 0:1]), done_mask[:, :-1]], dim=1) # right shift
        done_mask = ~done_mask # invert done_mask
        
        s = torch.stack(s_lst).transpose(0,1)# * done_mask
        a = torch.stack(a_lst).transpose(0,1)# * done_mask
        r = torch.stack(r_lst).transpose(1,0) * done_mask.squeeze(-1)
        s_prime = torch.stack(s_prime_lst).transpose(0,1)# * done_mask
        prob_a = torch.stack(prob_a_lst).transpose(1,0)# * done_mask

        ret = (s, a, r, s_prime, done_mask, prob_a)
        ret = (x.detach() for x in ret)
                                          
        self.data = []
        
        return ret
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K_epoch):
            with torch.no_grad():
                values = self.v(s)
                values_prime = self.v(s_prime)
                
                td_target = r.unsqueeze(-1) + self.gamma * values_prime * done_mask
                delta = td_target - values
            
                advantages = torch.zeros_like(delta)
                last_gae = 0
                for t in reversed(range(delta.shape[1])):
                    last_gae = delta[:, t] + self.gamma * self.lmbda * last_gae * done_mask[:, t]
                    advantages[:, t] = last_gae

                if self.normalize_advantage and advantages.shape[0] > 1:
                    advantages = normalize_advantage(advantages, done_mask)

            pi = self.pi(s)
            pi_a = pi.gather(-1,a)
            ratio = torch.exp(torch.log(pi_a  + 1e-10) - torch.log(prob_a + 1e-10))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).masked_select(done_mask).mean()
                        
            value_loss = F.smooth_l1_loss(self.v(s), td_target)

            loss = policy_loss + self.value_loss_coef * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            self.log("loss", loss)
            self.log("policy_loss", policy_loss)
            self.log("value_loss", value_loss)
            self.log("advantages", advantages)
            self.log("values", values)
            self.log("ratio", ratio)
            self.log("done mask sum", done_mask.sum())
    
    def log(self, name, value):
        if self.wandb_run:
            self.wandb_run.log({name: value})
