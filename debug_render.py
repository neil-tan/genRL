#!/usr/bin/env python3
"""
Simple debug script to test MujocoCartPole environment rendering.
"""
import os
import sys
print("1. Setting environment variables...")
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering
os.environ['PYTHONDONTWRITEBYTECODE'] = '1' # Prevent .pyc file creation
print(f"   MUJOCO_GL = {os.environ.get('MUJOCO_GL')}")
print(f"   PYTHONDONTWRITEBYTECODE = {os.environ.get('PYTHONDONTWRITEBYTECODE')}")

print("2. Importing modules...")
try:
    import numpy as np
    print("   NumPy imported successfully.")
except ImportError:
    print("   ERROR: NumPy import failed.")
    sys.exit(1)

try:
    import gymnasium as gym
    print("   Gymnasium imported successfully.")
except ImportError:
    print("   ERROR: Gymnasium import failed.")
    sys.exit(1)

# Add analysis code to find problematic lines
print("3. Analyzing cartpole.py file...")
# Construct expected path relative to this script
expected_module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src/genRL/gym_envs/mujoco"))
module_path = os.path.join(expected_module_dir, "cartpole.py")
print(f"   Expected module path: {module_path}")

if not os.path.exists(module_path):
    print(f"   ERROR: Module path does not exist!")
else:
    try:
        with open(module_path, 'r') as f:
            lines = f.readlines()
        print(f"   Read {len(lines)} lines from {module_path}")
        # Look for problematic lines
        problematic_terms = ["mjtVisFlag", "self.option", "option.flags", "mjVIS_GLOBAL"]
        found_problem = False
        for i, line in enumerate(lines):
            for term in problematic_terms:
                if term in line and not line.strip().startswith('#'): # Ignore comments
                    print(f"   WARNING: Found active term '{term}' on Line {i+1}: {line.strip()}")
                    found_problem = True
        if not found_problem:
            print("   No active problematic terms found in source file.")
        
        print("   Analysis complete.")
    except Exception as e:
        print(f"   ERROR analyzing file: {e}")

# Clear any potentially cached modules
sys.modules.pop('genRL.gym_envs.mujoco.cartpole', None)
sys.modules.pop('genRL.gym_envs.mujoco', None)
sys.modules.pop('genRL.gym_envs', None)
sys.modules.pop('genRL', None)
print("   Cleared potentially cached modules from sys.modules")

print("4. Importing environment module...")
try:
    # Ensure the source directory is in the Python path
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        print(f"   Added {src_dir} to sys.path")
        
    import genRL.gym_envs.mujoco.cartpole
    # Check the actual path used for the import
    imported_path = genRL.gym_envs.mujoco.cartpole.__file__
    print(f"   Environment module imported successfully.")
    print(f"   Actual imported path: {imported_path}")
    if imported_path != module_path:
         print("   WARNING: Imported path differs from expected path!")
except ImportError as e:
    print(f"   ERROR: Import failed: {e}")
    print(f"   Current sys.path: {sys.path}")
    sys.exit(1)
except Exception as e:
    print(f"   ERROR during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def main():
    print("\n5. Creating environment...")
    try:
        env = gym.make('MujocoCartPole-v0', render_mode='rgb_array')
        print("   Environment created successfully")
    except Exception as e:
        print(f"   ERROR: Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n6. Resetting environment...")
    try:
        obs, info = env.reset(seed=42)  # Use a fixed seed for reproducibility
        print(f"   Reset successful. Observation shape: {obs.shape}")
    except Exception as e:
        print(f"   ERROR: Failed to reset environment: {e}")
        env.close()
        return
    
    print("\n7. Taking steps...")
    try:
        for i in range(5):
            action = env.action_space.sample()
            print(f"   Step {i+1}: action={action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   -> reward={reward:.2f}, terminated={terminated}")
    except Exception as e:
        print(f"   ERROR: Failed during step: {e}")
        env.close()
        return
    
    print("\n8. Rendering...")
    try:
        frame = env.render()
        if frame is None:
            # This might happen if the renderer failed to initialize correctly
            print("   ERROR: render() returned None! Check renderer initialization logs.")
        else:
            print(f"   Rendered successfully! Frame shape: {frame.shape}, dtype: {frame.dtype}")
            print(f"   Frame stats - min: {frame.min()}, max: {frame.max()}, mean: {frame.mean():.2f}")
            
            # Check if frame is blank (all zeros or very dark)
            if frame.mean() < 1.0:
                print("   WARNING: Frame appears to be blank/black! Check for GL/driver issues.")
            
            # Save frame data
            print("\n9. Saving frame...")
            try:
                # Save as PNG if we have matplotlib
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 8))
                    plt.imshow(frame)
                    plt.title("MuJoCo Rendering")
                    plt.axis('off')
                    plt.savefig("mujoco_render_test.png")
                    plt.close()
                    print("   Saved frame as mujoco_render_test.png")
                except ImportError:
                    print("   Matplotlib not available, using pickle instead.")
                    
                # Always save as pickle for backup
                import pickle
                with open("mujoco_frame.pkl", "wb") as f:
                    pickle.dump(frame, f)
                print("   Saved raw frame data to mujoco_frame.pkl")
            except Exception as e:
                print(f"   ERROR saving frame: {e}")
    except Exception as e:
        print(f"   ERROR during rendering: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n10. Closing environment...")
    try:
        env.close()
        print("   Environment closed successfully")
    except Exception as e:
        print(f"   ERROR closing environment: {e}")
    
    print("\nDebug script completed.")

if __name__ == "__main__":
    print(f"Running debug script with Python {sys.version}")
    main() 