from genRL.rl.policies.mlp import SimpleMLP
from genRL.rl.ppo import PPO
from genRL.rl.grpo import GRPO
import gymnasium as gym # Import gymnasium

def ppo_agent(env: gym.Env, config): # Add env argument
    # Infer dimensions from env
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    pi = SimpleMLP(softmax_output=True, input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
    v = SimpleMLP(softmax_output=False, input_dim=input_dim, hidden_dim=256, output_dim=1)
    model = PPO(pi=pi, v=v, wandb_run=None, config=config)
    
    return model

def grpo_agent(env: gym.Env, config): # Add env argument
    # Infer dimensions from env
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    pi = SimpleMLP(softmax_output=True, input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
    model = GRPO(pi=pi, wandb_run=None, config=config)
    
    return model
