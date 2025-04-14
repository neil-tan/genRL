import torch

class SimpleBuffer:
    def __init__(self, buffer_size, device=None):
        self.device = device
        self.buffer_size = buffer_size
        self.buffer = []

    @torch.no_grad()
    def add(self, transition):
        assert len(self.buffer) <= self.buffer_size, "Experience size exceeds buffer size"
        self.buffer.append(transition)
    
    def clear(self):
        self.buffer = []
    
    @torch.no_grad()
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.buffer:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_lst.append(done) # invert done_mask
            
        # s,a,r,s_prime,done_mask, prob_a
        
        done_mask = torch.stack(done_lst).transpose(1,0).unsqueeze(-1)
        
        s = torch.stack(s_lst).transpose(0,1)# * done_mask
        a = torch.stack(a_lst).transpose(0,1)# * done_mask
        # Stack rewards which are now [num_envs] shaped tensors
        r = torch.stack(r_lst).transpose(1,0)
        s_prime = torch.stack(s_prime_lst).transpose(0,1)# * done_mask
        prob_a = torch.stack(prob_a_lst).transpose(1,0)# * done_mask
        
        # reward is still defined at the first done timestep
        # but anything more than that is invalid
        if torch.count_nonzero(r[:,1:].masked_select(done_mask[:,0:-1].squeeze(-1))) > 0:
            print("\033[33mwarning: Detected rewards for invalid timesteps\033[0m")

        ret = (s, a, r, s_prime, done_mask, prob_a)
        ret = (x.detach() for x in ret)
        ret = (x.to(self.device) for x in ret) if self.device else ret

        return ret
