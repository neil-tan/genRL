# tests/test_wandb_logging.py
import subprocess
# import wandb # Keep for potential future local file parsing, but API won't be used # Commented out as API object not needed
import time
import pytest
import re
import os
import glob
import json # For potentially reading summary files

# Define the path to the training script relative to the workspace root
# TRAIN_SCRIPT_PATH = "examples/train.py" # No longer needed
TRAIN_SCRIPT_MODULE = "examples.train" # Define module path
# Use a specific project name for testing (less relevant in offline mode, but good practice)
# TEST_PROJECT_NAME = "genrl-test-logging" # Keep original or use offline specific? Let's keep original for now.
TEST_PROJECT_NAME = "genrl-test-logging" 
TEST_RUN_NAME_BASE = "test-log-run" # Keep original base name

# @pytest.mark.skipif(os.getenv("WANDB_API_KEY") is None, reason="Requires WANDB_API_KEY environment variable") # REMOVED this line
def test_train_script_logs_metrics_and_video(): # Original function name, now runs offline
    """
    Tests if the train.py script logs expected metrics (implicitly by checking summary) 
    and a video file to a local offline Wandb directory when --record-video is enabled.
    """
    run_name = f"{TEST_RUN_NAME_BASE}-offline-{int(time.time())}" # Adjusted name slightly for clarity
    n_epi = 5 # Keep low for quick test
    report_interval = 2 # Ensure at least one report happens
    start_time = time.time() # Record time before running the script
    
    # Command to run the training script as a module
    command = [
        "python", "-m", TRAIN_SCRIPT_MODULE, # Use -m and module path
        "--project-name", TEST_PROJECT_NAME, # Use original project name
        "--run-name", run_name,
        "--env-id", "GenCartPole-v0", 
        "algo:ppo-config", 
        "--algo.n-epi", str(n_epi),
        "--algo.report-interval", str(report_interval),
        "--record-video",
        "--wandb-mode", "offline" # Run in offline mode
    ]

    run_id = None
    process_output = ""
    try:
        print(f"\nRunning command: {' '.join(command)}")
        
        # --- Set PYTHONPATH for subprocess ---
        env = os.environ.copy()
        # Construct absolute path to src directory relative to this test file
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')) 
        env['PYTHONPATH'] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
        print(f"Setting PYTHONPATH for subprocess: {env['PYTHONPATH']}")
        # --- End Set PYTHONPATH ---

        # Run the training script as a subprocess with modified environment
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            timeout=180,
            env=env # Pass the modified environment
        ) 
        process_output = result.stdout + "\n" + result.stderr
        print("--- Script Output ---")
        print(process_output)
        print("--- End Script Output ---")

        # Extract the run ID printed by train.py (useful for finding the directory)
        # Assuming train.py still prints this, even in offline mode
        match = re.search(r"WANDB_RUN_ID:(\w+)", process_output)
        if match:
            run_id = match.group(1)
            print(f"Extracted WANDB_RUN_ID: {run_id}")
        else:
            # Fallback: try to find the latest offline directory if ID isn't printed
            print("WANDB_RUN_ID not found in script output. Will try to find the latest offline directory.")
            
    except subprocess.CalledProcessError as e:
        print("--- Subprocess Error Output ---")
        print(e.stdout)
        print(e.stderr)
        print("--- End Subprocess Error Output ---")
        pytest.fail(f"Training script failed with error: {e}")
    except subprocess.TimeoutExpired as e:
        print("--- Subprocess Timeout Output ---")
        if e.stdout: print(e.stdout.decode())
        if e.stderr: print(e.stderr.decode())
        print("--- End Subprocess Timeout Output ---")
        pytest.fail(f"Training script timed out after 180 seconds.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during script execution: {e}")

    # Verify logging by checking the local offline directory
    
    # Find the correct offline directory
    offline_run_dir = None
    wandb_base_dir = "wandb" # Assuming default wandb directory
    if run_id:
        # Try finding the directory using the run_id
        possible_dirs = glob.glob(os.path.join(wandb_base_dir, f"offline-run-*-{run_id}"))
        if possible_dirs:
            # Sort by modification time just in case there are multiple matches (shouldn't happen often)
            possible_dirs.sort(key=os.path.getmtime, reverse=True)
            offline_run_dir = possible_dirs[0] 
            print(f"Found offline directory using run_id: {offline_run_dir}")
        else:
            print(f"Could not find offline directory for run_id {run_id}. Will try finding the latest.")

    if not offline_run_dir:
        # Find the latest offline directory created after the test started
        all_offline_dirs = glob.glob(os.path.join(wandb_base_dir, "offline-run-*"))
        latest_dir = None
        latest_mtime = start_time 
        for d in all_offline_dirs:
            try:
                mtime = os.path.getmtime(d)
                if mtime >= latest_mtime: # Use >= to catch runs started exactly at start_time
                    latest_mtime = mtime
                    latest_dir = d
            except OSError:
                continue # Ignore if directory disappears or permissions issue
        
        if latest_dir:
            offline_run_dir = latest_dir
            print(f"Found latest offline directory (created after test start): {offline_run_dir}")
        else:
             pytest.fail("Could not find a suitable offline Wandb run directory.")

    assert offline_run_dir is not None, "Failed to identify the offline run directory."
    assert os.path.isdir(offline_run_dir), f"Offline run directory {offline_run_dir} does not exist or is not a directory."

    # Verify video logging
    print(f"Checking for video file in {offline_run_dir}...")
    video_dir = os.path.join(offline_run_dir, "files", "videos")
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    assert video_files, f"No .mp4 video file found in {video_dir}."
    print(f"Successfully verified video logging. Found: {video_files[0]}")

    # Optional: Verify metric logging by checking for summary file
    summary_file_path = os.path.join(offline_run_dir, "files", "wandb-summary.json")
    print(f"Checking for summary file: {summary_file_path}...")
    assert os.path.isfile(summary_file_path), f"Summary file not found at {summary_file_path}."
    
    # Optionally, check content:
    try:
        with open(summary_file_path, 'r') as f:
            summary_data = json.load(f)
        # Check if a key typically logged exists (e.g., related to reward or steps)
        # Note: Exact keys depend on what train.py logs to summary
        assert any(k.endswith("reward") or k.endswith("step") or k.endswith("loss") or k == "_timestamp" for k in summary_data.keys()), \
            f"No expected metric keys (like reward, step, loss) found in summary file: {summary_file_path}"
        print(f"Successfully verified summary file exists and contains expected key patterns.")
    except (json.JSONDecodeError, FileNotFoundError, AssertionError) as e:
         pytest.fail(f"Failed to verify summary file content: {e}")

    print(f"Successfully verified offline logging for run in directory: {offline_run_dir}")

    # No cleanup needed for offline runs unless explicitly desired
