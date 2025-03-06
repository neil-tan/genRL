import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
# T_horizon     = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
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

        for i in range(K_epoch):
            with torch.no_grad():
                values = self.v(s).squeeze(-1)
                values_prime = self.v(s_prime).squeeze(-1)
                
                td_target = r + gamma * values_prime * done_mask
                delta = td_target - values
            
                advantages = torch.zeros_like(delta)
                last_gae = 0
                for t in reversed(range(delta.shape[-1])):
                    last_gae = delta[:, t] + gamma * lmbda * last_gae * done_mask[:, t]
                    advantages[:, t] = last_gae

            pi = self.pi(s, softmax_dim=-1)
            pi_a = pi.gather(-1,a).squeeze(-1)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s).squeeze(-1) , td_target)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def main():
    env = gym.make('CartPole-v1')
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

if __name__ == '__main__':
    main()