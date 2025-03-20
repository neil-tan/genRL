import genRL.gym_envs.genesis.cartpole
import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from genRL.rl.ppo import PPO, SimpleMLP
import genesis as gs
import sys
import numpy as np
import wandb

#Hyperparameters
# Hyperparameters as a dictionary
config = {
    "learning_rate": 0.00005,
    "gamma": 0.99,
    "lmbda": 0.95,
    "value_loss_coef": 1,
    "normalize_advantage": False,
    "max_grad_norm": 0.5,
    "eps_clip": 0.1,
    "T_horizon": 1000,
    "random_seed": 42,
    "num_envs": 16,
    "reward_scale": 0.01,
}

np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])


def training_loop(env):
    wandb.login()
    run = wandb.init(
                    project="genRL_cartpole_ppo",
                    name="test_run1",
                    config=config,
                    # mode="disabled", # dev dry-run
                )

    pi = SimpleMLP(softmax_output=True, input_dim=4, hidden_dim=256, output_dim=2, activation=F.tanh)
    v = SimpleMLP(softmax_output=False, input_dim=4, hidden_dim=256, output_dim=1, activation=F.tanh)

    model = PPO(pi=pi, v=v, wandb_run=run, **config)
    
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(config["T_horizon"]):
                prob = model.pi(s)
                m = Categorical(prob)
                a = m.sample().unsqueeze(-1)
                s_prime, r, done, truncated, info = env.step(a)

                prob_a = torch.gather(prob, -1, a)
                model.put_data((s, a.detach(), r*config["reward_scale"], s_prime, prob_a.detach(), done))
                s = s_prime

                score += r
                
                done = done.all() if isinstance(done, torch.Tensor) else done
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            interval_score = score/print_interval
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, interval_score.mean().item()))
            run.log({"rewards histo": wandb.Histogram(interval_score), "mean reward": interval_score.mean()})
            score = 0.0


def main():
    env = gym.make("GenCartPole-v0",
                   render_mode="human" if sys.platform == "darwin" else "ansi",
                   max_force=1000,
                   targetVelocity=10,
                   num_envs=config["num_envs"],
                   return_tensor=True,
                   logging_level="warning", # "info", "warning", "error", "debug"
                   gs_backend=gs.cpu,
                   seed=config["random_seed"],
                   )
    
    env.reset()
    
    if not sys.platform == "linux":
        gs.tools.run_in_another_thread(fn=training_loop, args=(env,))
    else:
        training_loop(env)

    env.render()
    env.close()
    

if __name__ == '__main__':
    main()
