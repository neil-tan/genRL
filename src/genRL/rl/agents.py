
from genRL.rl.policies.mlp import SimpleMLP
from genRL.rl.ppo import PPO
from genRL.rl.grpo import GRPO

def ppo_agent(config):
    pi = SimpleMLP(softmax_output=True, input_dim=4, hidden_dim=256, output_dim=2)
    v = SimpleMLP(softmax_output=False, input_dim=4, hidden_dim=256, output_dim=1)
    model = PPO(pi=pi, v=v, wandb_run=None, config=config)
    
    return model

def grpo_agent(config):
    pi = SimpleMLP(softmax_output=True, input_dim=4, hidden_dim=256, output_dim=2)
    model = GRPO(pi=pi, wandb_run=None, config=config)
    
    return model
