import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from genRL.utils import normalize_advantage, mask_right_shift
from tqdm import trange
from genRL.configs import PPOConfig

class PPO(nn.Module):
    def __init__(self,
                 pi,
                 v,
                 config:PPOConfig,
                 wandb_run=None,
                 **kwargs,):
        super(PPO, self).__init__()
        self.data = []
        self.config = config
        
        self.pi = pi
        self.v = v
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        self.wandb_run = wandb_run if wandb_run else None

    def train_net(self, buffer):
        cfg = self.config
        s, a, r, s_prime, done_mask, log_prob_a = buffer.make_batch()
        
        valid_mask = ~mask_right_shift(done_mask)

        for i in trange(cfg.K_epoch, desc="ppo", leave=False):
            # Sample current policy
            _, log_prob, entropy = self.pi.sample_action(s, a, eval_entropy=True)
            values = self.v(s)
            
            with torch.no_grad():
                values_prime = self.v(s_prime)

                # r is [batch, timesteps]
                td_target = r.unsqueeze(-1) + cfg.gamma * values_prime * ~done_mask
                delta = td_target - values
            
                advantages = torch.zeros_like(delta)
                last_gae = torch.zeros_like(delta[:, 0])
                for t in reversed(range(delta.shape[1])):
                    last_gae = delta[:, t] + cfg.gamma * cfg.lmbda * last_gae * ~done_mask[:, t]
                    advantages[:, t] = last_gae

                if cfg.normalize_advantage and advantages.shape[0] > 1:
                    self.log("pre-norm advantages", advantages)
                    advantages = normalize_advantage(advantages, valid_mask)
                
                # advantages = advantages.squeeze(-1)

            ratio = torch.exp(log_prob - log_prob_a)
            # approx_kl = (pi_a * (log_prob - log_prob_old)).sum(-1).masked_select(valid_mask.squeeze(-1)).mean()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-cfg.eps_clip, 1+cfg.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).masked_select(valid_mask).mean()
            value_loss = F.smooth_l1_loss(values, td_target, reduction='none').masked_select(valid_mask).mean()
            kl_loss = F.kl_div(log_prob, log_prob_a, log_target=True, reduction="none").masked_select(valid_mask).mean()

            entropy_loss = -entropy.mean()

            loss = policy_loss + cfg.value_loss_coef * value_loss + cfg.entropy_coef * entropy_loss + cfg.kl_coef * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            
            self.log("advantages", advantages)
            self.log("loss", loss)
            self.log("policy_loss", policy_loss)
            self.log("value_loss", value_loss)
            self.log("values", values)
            self.log("ratio", ratio)
            self.log("td_target", td_target)
            self.log("delta", delta)
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