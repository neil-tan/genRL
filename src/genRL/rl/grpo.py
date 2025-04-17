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
        s, a, r, s_prime, done_mask, log_prob_a = buffer.make_batch()
        
        valid_mask = ~mask_right_shift(done_mask)
        
        with torch.no_grad():
            # unlike PPO, which agnostic to sparse or dense rewards
            # here, we need to calculate the outcome supervision reward
            reward = torch.sum(r.unsqueeze(-1).masked_fill(~valid_mask, 0), dim=1, keepdim=True)
            reward_mean = torch.mean(reward)
            reward_std = torch.std(reward)
            reward = reward.expand(-1, valid_mask.shape[1], -1)

            advantages = reward - reward_mean
            advantages = advantages / (reward_std + 1e-8)
            advantages = advantages.masked_fill(~valid_mask, 0.0)
            
            self.log("grpo/reward_mean", reward_mean)
            self.log("grpo/reward_std", reward_std)

        for i in trange(cfg.K_epoch, desc="ppo", leave=False):
            # Sample current policy
            _, log_prob, entropy = self.pi.sample_action(s, a, eval_entropy=True)

            ratio = torch.exp(log_prob - log_prob_a)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-cfg.eps_clip, 1+cfg.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).masked_select(valid_mask).mean()

            kl_loss = F.kl_div(log_prob, log_prob_a, log_target=True, reduction="none").masked_select(valid_mask).mean()

            entropy_loss = -entropy.mean()

            loss = policy_loss + cfg.entropy_coef * entropy_loss + cfg.kl_coef * kl_loss

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
            self.log("KL loss", kl_loss)

    def log(self, name, value):
        if self.wandb_run:
            self.wandb_run.log({name: value})

    def set_run(self, run):
        if self.wandb_run is not None and self.wandb_run != run:
            raise ValueError("WandB run already set")
        self.wandb_run = run