import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from genRL.utils import normalize_advantage, mask_right_shift, masked_std, masked_mean
from tqdm import trange
from genRL.configs import GRPOConfig
import wandb

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

class GRPO(nn.Module):
    def __init__(self,
                 pi,
                 config:GRPOConfig,
                 wandb_run=None,
                 **kwargs,):
        super(GRPO, self).__init__()
        self.data = []
        self.config = config
        
        self.pi = pi
        self.optimizer = optim.Adam(pi.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
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
        
        s = torch.stack(s_lst).transpose(0,1)# * done_mask
        a = torch.stack(a_lst).transpose(0,1)# * done_mask
        # r = torch.stack(r_lst).transpose(1,0) * rshift_mask.squeeze(-1)
        r = torch.stack(r_lst).transpose(1,0)
        s_prime = torch.stack(s_prime_lst).transpose(0,1)# * done_mask
        prob_a = torch.stack(prob_a_lst).transpose(1,0)# * done_mask
        
        # reward is still defined at the first done timestep
        # but anything more than that is invalid
        if torch.count_nonzero(r[:,1:].unsqueeze(-1).masked_select(done_mask[:,0:-1,:])) > 0:
            print("\033[33mwarning: Detected rewards for invalid timesteps\033[0m")

        ret = (s, a, r, s_prime, done_mask, prob_a)
        ret = (x.detach() for x in ret)
                                          
        self.data = []
        
        return ret
        
    def train_net(self):
        cfg = self.config
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        
        valid_mask = ~mask_right_shift(done_mask)
        
        with torch.no_grad():
            # unlike PPO, which considers step-wise rewards
            # here, we need to calculate the outcome supervision reward
            reward = torch.sum(r.unsqueeze(-1).masked_fill(~valid_mask, 0), dim=1, keepdim=True)
            reward_mean = torch.mean(reward)
            reward_std = torch.std(reward)
            reward = reward.expand(-1, valid_mask.shape[1], -1)

            advantages = reward - reward_mean
            advantages = advantages / (reward_std + 1e-8)
            advantages = advantages.masked_fill(~valid_mask, 0.0)

        for i in trange(cfg.K_epoch, desc="ppo", leave=False):
            pi = self.pi(s)
            pi_a = pi.gather(-1,a)

            log_prob_old = torch.log(prob_a + 1e-10)
            log_prob = torch.log(pi_a + 1e-10)

            ratio = torch.exp(log_prob - log_prob_old)
            approx_kl = (pi_a * (log_prob - log_prob_old)).sum(-1).masked_select(valid_mask.squeeze(-1)).mean()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-cfg.eps_clip, 1+cfg.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).masked_select(valid_mask).mean()

            entropy = Categorical(probs=pi).entropy()
            entropy_loss = -entropy.mean()

            loss = policy_loss + cfg.entropy_coef * entropy_loss + cfg.kl_coef * approx_kl

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            
            self.log("advantages", advantages)
            self.log("loss", loss)
            self.log("policy_loss", policy_loss)
            self.log("ratio", ratio)
            self.log("r", r)
            self.log("valid mask sum", valid_mask.sum())
            self.log("entropy", entropy)
            self.log("entropy_loss", entropy_loss)
            self.log("approx_kl", approx_kl)
    def log(self, name, value):
        if self.wandb_run:
            self.wandb_run.log({name: value})
