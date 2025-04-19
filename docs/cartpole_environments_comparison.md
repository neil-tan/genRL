# Comparison of Genesis and MuJoCo CartPole Environments

## Overview

This document compares the Genesis (`GenCartPole-v0`) and MuJoCo (`MujocoCartPole-v0`) implementations of the CartPole environment in the genRL library. This comparison helps explain why the MuJoCo implementation is easier to train with reinforcement learning algorithms like PPO and GRPO.

## Testing Methodology

Comprehensive tests were added to both environments to verify:
1. Basic functionality (initialization, stepping, resetting, rendering)
2. Reward consistency over time
3. Vectorized environment capabilities
4. Direct behavior comparison between the two implementations

## Key Findings

### Observation Space Differences

Initial observations from the same seed are notably different:
- Genesis: `[-0.00547759 -0.01217138  0.00504477  0.01468207]`
- MuJoCo: `[ 0.0273956   0.03585979 -0.00611216  0.0197368 ]`

### Movement Range Differences

Under identical action patterns:
- **Cart Position Range**:
  - Genesis: `[-0.0057, 0.2123]` (range ~0.218)
  - MuJoCo: `[0.0275, 0.0718]` (range ~0.044)
  - **The Genesis cart moves approximately 5x more than MuJoCo's cart with the same force**

- **Pole Angle Range**:
  - Genesis: `[-0.0530, 0.0055]` (range ~0.058 radians)
  - MuJoCo: `[-0.0151, -0.0059]` (range ~0.009 radians)
  - **The Genesis pole oscillates approximately 6x more than MuJoCo's pole**

### Action Interpretation

- Genesis maps actions from `[0,1]` to a target velocity, with 0.5 being neutral
- MuJoCo maps actions from `[-1,1]` directly to force

### Default Parameters

- Genesis has a default `targetVelocity` of 0.1, which can be increased
- MuJoCo has a constant force scaling mechanism

## Why MuJoCo's Implementation Is Easier to Train

1. **Lower Sensitivity**: MuJoCo's cart moves less for the same actions, creating a more stable training environment with less extreme state transitions.

2. **Smaller State Space**: The reduced pole angle oscillations in MuJoCo mean the agent has to learn a narrower range of states.

3. **Different Physics Parameters**: The physics models may have different damping, friction, or mass parameters.

4. **Action Range Consistency**: MuJoCo's `[-1,1]` mapping to direct force may be more intuitive for optimization algorithms compared to Genesis's velocity-based control.

## Recommendations for Improving Genesis CartPole Training

1. **Reduce Target Velocity**: Lower the `targetVelocity` parameter (default: 0.1, often set to higher values like 5.0 in tests).

2. **Match Physics Parameters**: Review the URDF files to ensure mass, friction, and other parameters match between implementations.

3. **Normalize Observations**: Add observation normalization to stabilize training.

4. **Adjust Termination Thresholds**: Verify that both implementations use consistent termination criteria for pole angles.

## Testing Evidence

The comprehensive testing suite provides:
- Verification that rewards are consistently 1.0 while the pole remains upright
- Confirmation that cart positions change predictably when forces are applied
- Demonstration that vectorized environments work properly for both implementations
- Direct comparisons showing the differences in dynamics between the two implementations

## Conclusion

The Genesis CartPole environment is more sensitive to actions and has wider state ranges, making it more challenging for reinforcement learning algorithms to train effectively. By adjusting parameters to more closely match MuJoCo's implementation, training performance can likely be improved.