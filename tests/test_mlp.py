import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import pytest

# Assuming the classes are importable like this:
from genRL.rl.policies.mlp import SimpleDiscreteMLP, SimpleContinuousMLP

# Test parameters
BATCH_SIZE = 4
INPUT_DIM = 10
HIDDEN_DIM = 32
OUTPUT_DIM_DISCRETE = 5
OUTPUT_DIM_CONTINUOUS = 3

@pytest.fixture
def sample_state():
    return torch.randn(BATCH_SIZE, INPUT_DIM)

@pytest.fixture
def sample_action_discrete():
    # Shape [batch, 1] as expected by sample_action return
    return torch.randint(0, OUTPUT_DIM_DISCRETE, (BATCH_SIZE, 1))

@pytest.fixture
def sample_action_continuous():
    # Shape [batch, output_dim]
    return torch.randn(BATCH_SIZE, OUTPUT_DIM_CONTINUOUS)

# --- Tests for SimpleDiscreteMLP ---

def test_discrete_mlp_init():
    model = SimpleDiscreteMLP(INPUT_DIM, OUTPUT_DIM_DISCRETE, HIDDEN_DIM)
    assert isinstance(model.fc1, torch.nn.Linear)
    assert isinstance(model.fc2, torch.nn.Linear)
    assert model.fc1.in_features == INPUT_DIM
    assert model.fc1.out_features == HIDDEN_DIM
    assert model.fc2.in_features == HIDDEN_DIM
    assert model.fc2.out_features == OUTPUT_DIM_DISCRETE

def test_discrete_mlp_forward(sample_state):
    model = SimpleDiscreteMLP(INPUT_DIM, OUTPUT_DIM_DISCRETE, HIDDEN_DIM)
    logits = model.forward(sample_state)
    assert logits.shape == (BATCH_SIZE, OUTPUT_DIM_DISCRETE)
    assert logits.dtype == torch.float32

def test_discrete_mlp_forward_softmax(sample_state):
    model = SimpleDiscreteMLP(INPUT_DIM, OUTPUT_DIM_DISCRETE, HIDDEN_DIM, output_softmax=True)
    probs = model.forward(sample_state)
    assert probs.shape == (BATCH_SIZE, OUTPUT_DIM_DISCRETE)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH_SIZE))
    assert torch.all(probs >= 0) and torch.all(probs <= 1)

def test_discrete_mlp_action_distribution(sample_state):
    model = SimpleDiscreteMLP(INPUT_DIM, OUTPUT_DIM_DISCRETE, HIDDEN_DIM)
    dist = model.action_distribution(sample_state)
    assert isinstance(dist, Categorical)
    assert dist.batch_shape == (BATCH_SIZE,)
    assert dist.logits.shape == (BATCH_SIZE, OUTPUT_DIM_DISCRETE)

def test_discrete_mlp_sample_action(sample_state):
    model = SimpleDiscreteMLP(INPUT_DIM, OUTPUT_DIM_DISCRETE, HIDDEN_DIM)
    action, log_prob, entropy = model.sample_action(sample_state, eval_entropy=True)
    assert action.shape == (BATCH_SIZE, 1)
    assert action.dtype == torch.int64
    assert log_prob.shape == (BATCH_SIZE, 1)
    assert log_prob.dtype == torch.float32
    assert entropy.shape == (BATCH_SIZE,) # Entropy is per batch item for Categorical
    assert entropy.dtype == torch.float32

def test_discrete_mlp_sample_action_given_action(sample_state, sample_action_discrete):
    model = SimpleDiscreteMLP(INPUT_DIM, OUTPUT_DIM_DISCRETE, HIDDEN_DIM)
    # Provide action with shape [batch, 1]
    action_in = sample_action_discrete
    action_out, log_prob, entropy = model.sample_action(sample_state, action=action_in, eval_entropy=True)
    
    # Action should be returned as is (but unsqueezed)
    assert torch.equal(action_out, action_in)
    assert action_out.shape == (BATCH_SIZE, 1)
    assert log_prob.shape == (BATCH_SIZE, 1)
    assert entropy is not None


# --- Tests for SimpleContinuousMLP ---

def test_continuous_mlp_init():
    model = SimpleContinuousMLP(INPUT_DIM, OUTPUT_DIM_CONTINUOUS, HIDDEN_DIM)
    assert isinstance(model.fc1, torch.nn.Linear)
    assert isinstance(model.fc2, torch.nn.Linear)
    assert model.fc1.in_features == INPUT_DIM
    assert model.fc1.out_features == HIDDEN_DIM
    assert model.fc2.in_features == HIDDEN_DIM
    assert model.fc2.out_features == OUTPUT_DIM_CONTINUOUS * 2

def test_continuous_mlp_forward(sample_state):
    model = SimpleContinuousMLP(INPUT_DIM, OUTPUT_DIM_CONTINUOUS, HIDDEN_DIM)
    mean, std = model.forward(sample_state)
    assert mean.shape == (BATCH_SIZE, OUTPUT_DIM_CONTINUOUS)
    assert std.shape == (BATCH_SIZE, OUTPUT_DIM_CONTINUOUS)
    assert mean.dtype == torch.float32
    assert std.dtype == torch.float32
    assert torch.all(std > 0) # Check positivity constraint

def test_continuous_mlp_action_distribution(sample_state):
    model = SimpleContinuousMLP(INPUT_DIM, OUTPUT_DIM_CONTINUOUS, HIDDEN_DIM)
    dist = model.action_distribution(sample_state)
    assert isinstance(dist, Normal)
    assert dist.batch_shape == (BATCH_SIZE, OUTPUT_DIM_CONTINUOUS)
    assert dist.event_shape == torch.Size([])

def test_continuous_mlp_sample_action(sample_state):
    model = SimpleContinuousMLP(INPUT_DIM, OUTPUT_DIM_CONTINUOUS, HIDDEN_DIM)
    action, log_prob, entropy = model.sample_action(sample_state, eval_entropy=True)
    assert action.shape == (BATCH_SIZE, OUTPUT_DIM_CONTINUOUS)
    assert action.dtype == torch.float32
    assert log_prob.shape == (BATCH_SIZE, 1) # Summed over action dim
    assert log_prob.dtype == torch.float32
    assert entropy is not None
    assert entropy.shape == (BATCH_SIZE, OUTPUT_DIM_CONTINUOUS) # Entropy is per dimension for Normal
    assert entropy.dtype == torch.float32

def test_continuous_mlp_sample_action_given_action(sample_state, sample_action_continuous):
    model = SimpleContinuousMLP(INPUT_DIM, OUTPUT_DIM_CONTINUOUS, HIDDEN_DIM)
    action_in = sample_action_continuous
    action_out, log_prob, entropy = model.sample_action(sample_state, action=action_in, eval_entropy=True)
    
    # Action should be returned as is
    assert torch.equal(action_out, action_in)
    assert action_out.shape == (BATCH_SIZE, OUTPUT_DIM_CONTINUOUS)
    assert log_prob.shape == (BATCH_SIZE, 1)
    assert entropy is not None

def test_continuous_mlp_rsample_grad(sample_state):
    """Check if gradients flow back through rsample."""
    model = SimpleContinuousMLP(INPUT_DIM, OUTPUT_DIM_CONTINUOUS, HIDDEN_DIM)
    model.train() # Ensure model is in training mode

    state = sample_state.clone().requires_grad_(True) # Input needs grad
    
    # Need to ensure parameters require grad (usually true by default)
    for param in model.parameters():
        param.requires_grad_(True)

    action, log_prob, _ = model.sample_action(state)
    
    # Create a dummy loss that depends on the action
    dummy_loss = action.mean() 
    
    # Backpropagate
    dummy_loss.backward()

    # Check if gradients exist for model parameters (specifically fc2 weights)
    assert model.fc2.weight.grad is not None
    assert model.fc2.weight.grad.shape == model.fc2.weight.shape
    assert not torch.all(model.fc2.weight.grad == 0)

    # Check if gradients exist for the input state (if needed)
    assert state.grad is not None
    assert not torch.all(state.grad == 0)
