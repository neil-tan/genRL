import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

T_horizon     = 20,

class PPO(nn.Module):
    def __init__(self,
                 learning_rate = 0.0005,
                 gamma         = 0.98,
                 lmbda         = 0.95,
                 eps_clip      = 0.1,
                 K_epoch       = 3,
                 device = torch.cpu):
        super(PPO, self).__init__()
        self.data = []
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.device = "cpu"
        
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
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
            done_mask = torch.where(torch.tensor(done), torch.tensor(0.0), torch.tensor(1.0))
            done_lst.append(done_mask)
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float, device=self.device).permute((1,0,2)), \
                                          torch.stack(a_lst, dim=-1).to(self.device), \
                                          torch.stack(r_lst, dim=-1).to(self.device), \
                                          torch.tensor(s_prime_lst, dtype=torch.float).permute((1,0,2)).to(self.device), \
                                          torch.stack(done_lst, dim=-1).to(self.device), \
                                          torch.stack(prob_a_lst, dim=-1).to(self.device)
        self.data = [] 
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime).squeeze(-1)
            delta = td_target - self.v(s).squeeze(-1)
            delta = delta.masked_fill(~done_mask.bool(), 0.0).detach()
            # delta = delta.detach().numpy()


            advantage = torch.zeros_like(r)
            advantage_slice = 0.0
            for t in reversed(range(delta.shape[-1])): # reverse iterate
                delta_t = delta[:,t]
                advantage_slice = self.gamma * self.lmbda * advantage_slice + delta_t
                advantage[:,t] = advantage_slice

            # advantage_lst = []
            # advantage = 0.0
            # for delta_t in delta[::-1]:
            #     advantage = self.gamma * self.lmbda * advantage + delta_t[0]
            #     advantage_lst.append([advantage])
            # advantage_lst.reverse()
            # advantage = torch.tensor(advantage_lst, dtype=torch.float)
            advantage = advantage.masked_fill(~done_mask.bool(), 0.0)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(-1,a.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            ratio = ratio.masked_fill(~done_mask.bool(), 1)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s).squeeze(-1) , td_target.detach())

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