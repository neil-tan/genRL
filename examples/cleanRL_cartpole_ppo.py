import genRL.gym_envs.genesis.cartpole as gen_cartpole
import gymnasium as gym
import torch
from torch.distributions import Categorical
from genRL.rl.ppo import PPO
import genesis as gs
import sys
import time

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20


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
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
        
    env.close()

def main():
    env = gym.make("GenCartPole-v0", render_mode="human", max_force=1000, targetVelocity=5)

    if not sys.platform == "linux":
        gs.tools.run_in_another_thread(fn=training_loop, args=(env,))
    else:
        training_loop(env, 300)

    time.sleep(1)
    while True:
        env.render()
        # if env.unwrapped.done == True:
        #     break
        # time.sleep(1)
    

if __name__ == '__main__':
    main()
