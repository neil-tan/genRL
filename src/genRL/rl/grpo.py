import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from genRL.utils import normalize_advantage, mask_right_shift, masked_std, masked_mean
from tqdm import trange
from genRL.configs import GRPOConfig


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
        
        self.wandb_run = wandb_run if wandb_run else None

    def train_net(self, buffer):
        cfg = self.config
        s, a, r, s_prime, done_mask, prob_a = buffer.make_batch()
        
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
            
            self.log("reward_mean", reward_mean)
            self.log("reward_std", reward_std)

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

    def set_run(self, run):
        if self.wandb_run is not None and self.wandb_run != run:
            raise ValueError("WandB run already set")
        self.wandb_run = run