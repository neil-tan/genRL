import genRL.gym_envs.genesis.cartpole
import gymnasium as gym
import torch
from torch.distributions import Categorical
from genRL.rl.ppo import PPO
import genesis as gs
import sys

#Hyperparameters
learning_rate = 0.01
gamma         = 0.99
lmbda         = 0.97
eps_clip      = 0.1
T_horizon     = 1000


def training_loop(env):
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample()
                s_prime, r, done, truncated, info = env.step(a)
                done = torch.all(done) if isinstance(done, torch.Tensor) else done

                prob_a = torch.gather(prob, -1, a.unsqueeze(0)).squeeze(0)
                model.put_data((s, a, r/100.0, s_prime, prob_a, done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0


def main():
    env = gym.make("GenCartPole-v0",
                   render_mode="human" if sys.platform == "darwin" else "ansi",
                   num_envs=5,
                   max_force=1000,
                   targetVelocity=10,
                   logging_level="warning", # "info", "warning", "error", "debug"
                   gs_backend=gs.cpu
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
