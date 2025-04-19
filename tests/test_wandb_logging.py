# tests/test_wandb_logging.py
import subprocess
import wandb
import time
import pytest
import re
import os

# Set MUJOCO_GL environment variable to ensure proper rendering
os.environ['MUJOCO_GL'] = 'egl'

# Define constants for test configuration
TRAIN_SCRIPT_PATH = "examples/train.py"
TEST_PROJECT_NAME = "genrl-test-logging" 
TEST_RUN_NAME_BASE = "test-log-run"
API_RETRY_COUNT = 5
API_RETRY_DELAY = 10
TEST_TIMEOUT = 180

def fetch_wandb_runs_with_retries(project=TEST_PROJECT_NAME, entity=None, max_retries=API_RETRY_COUNT, delay=API_RETRY_DELAY):
    """
    Helper function to fetch recent wandb runs for a project with retries.
    Returns list of run objects or raises exception if all retries fail.
    """
    api = wandb.Api(timeout=60)
    entity = entity or api.default_entity
    project_path = f"{entity}/{project}"
    
    for attempt in range(max_retries):
        try:
            # Get the most recent runs from the project
            runs = api.runs(project_path, per_page=10)
            if runs:
                return runs
            
            print(f"No runs found in project. Retrying... ({attempt+1}/{max_retries})")
                
        except Exception as e:
            print(f"Error fetching wandb runs (attempt {attempt+1}): {e}")
        
        # Sleep before retry if not the last attempt
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    # If we get here, all retries failed
    raise RuntimeError(f"Failed to fetch runs after {max_retries} attempts")

def fetch_run_data_with_retries(run, keys=None, max_retries=API_RETRY_COUNT, delay=API_RETRY_DELAY):
    """
    Helper function to fetch history data for a wandb run with retries.
    Returns the run history or raises exception if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            history = list(run.scan_history(keys=keys)) if keys else list(run.scan_history())
            
            # Return immediately if we find the data we're looking for
            if history and (not keys or all(any(k in record for record in history) for k in keys)):
                return history
                
            # If we have history but not all expected keys, log and continue retrying
            if history:
                print(f"Run data found but missing required keys. Retrying... ({attempt+1}/{max_retries})")
            else:
                print(f"Run history is empty. Retrying... ({attempt+1}/{max_retries})")
                
        except Exception as e:
            print(f"Error fetching history for run {run.id} (attempt {attempt+1}): {e}")
        
        # Sleep before retry if not the last attempt
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    # If we get here, all retries failed
    raise RuntimeError(f"Failed to fetch history data after {max_retries} attempts")

def check_for_video_content(run, max_retries=API_RETRY_COUNT, delay=API_RETRY_DELAY):
    """
    Check if a W&B run has video content, either as files or media table entries.
    Returns True if videos are found, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            # Get run summary to check for media tables
            summary_dict = run.summary._json_dict
            
            # Check for media tables that might contain videos
            media_keys = [k for k, v in summary_dict.items() 
                         if isinstance(v, dict) and v.get('_type') in ['video-file', 'table-file']]
                         
            # Check for direct video references in summary (using the new key "video")
            video_keys = [k for k, v in summary_dict.items() 
                         if k == "video" and isinstance(v, dict) and v.get('_type') == 'video-file']
            
            # Check for media logs in the history (using the new key "video")
            history = list(run.scan_history(keys=["video"]))
            has_video_in_history = any("video" in record for record in history)
            
            # Check for actual files with video extensions
            files = run.files()
            video_files = [f for f in files if f.name.endswith(('.mp4', '.avi', '.mov', '.webm'))]
            
            if video_keys or media_keys or has_video_in_history or video_files:
                # Print what we found for debugging
                if video_keys:
                    print(f"Found {len(video_keys)} video keys in summary: {video_keys}")
                if media_keys:
                    print(f"Found {len(media_keys)} media keys in summary: {media_keys}")
                if has_video_in_history:
                    print("Found video references in history")
                if video_files:
                    print(f"Found {len(video_files)} video files: {[f.name for f in video_files]}")
                return True
                
            print(f"No video content found in run {run.id}. Retrying... ({attempt+1}/{max_retries})")
            
        except Exception as e:
            print(f"Error checking for video content in run {run.id} (attempt {attempt+1}): {e}")
        
        # Sleep before retry if not the last attempt
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    # If we get here, all retries failed or no videos found
    return False

def extract_run_id_from_output(output):
    """Extract the wandb run ID from process output"""
    # Look for multiple possible formats of wandb ID in output
    patterns = [
        r"WANDB_RUN_ID:(\w+)",  # Explicit WANDB_RUN_ID format
        r"wandb: Run data is saved locally in ([^\s]+)",  # Path format
        r"wandb: Synced [^:]+ https://wandb.ai/[^/]+/[^/]+/runs/([^/]+)",  # URL format
        r"run-(\w+)"  # Common run ID pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            # If the match is a path containing a run ID, extract just the ID
            run_id = match.group(1)
            if "/" in run_id:
                # Extract run ID from path if needed
                path_match = re.search(r"run-(\w+)", run_id)
                if path_match:
                    return path_match.group(1)
            return run_id
    
    return None

@pytest.mark.skipif(os.getenv("WANDB_API_KEY") is None, reason="Requires WANDB_API_KEY environment variable")
def test_train_script_logs_metrics_and_videos():
    """Tests if the train.py script logs expected metrics and videos to Wandb."""
    # Setup test parameters with timestamp for uniqueness
    timestamp = int(time.time())
    run_name = f"{TEST_RUN_NAME_BASE}-{timestamp}"
    env_id = "GenCartPole-v0"  # Use Genesis environment by default
    
    # Use small wandb_video_episodes to ensure videos are captured within our short test
    video_episodes = 5  # Small number to trigger videos frequently
    command = [
        "python", TRAIN_SCRIPT_PATH,
        "--project-name", TEST_PROJECT_NAME,
        "--run-name", run_name,
        "--env-id", env_id,
        # Use the new argument name
        "--wandb-video-episodes", str(video_episodes),
        # Remove --record-video flag
        "algo:ppo-config",
        "--algo.n-epi", "10",  # Run for enough episodes to generate videos
        "--algo.report-interval", "2"  # Ensure frequent reporting
    ]

    # Run the training script and capture output
    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            timeout=TEST_TIMEOUT
        )
        process_output = result.stdout + "\n" + result.stderr
        print(f"Training completed. Output excerpt: {process_output[:200]}...")
    except subprocess.CalledProcessError as e:
        process_output = f"{e.stdout or ''}\n{e.stderr or ''}"
        pytest.fail(f"Training script failed with error: {e}\nOutput: {process_output}")
    except subprocess.TimeoutExpired as e:
        process_output = f"{e.stdout.decode() if e.stdout else ''}\n{e.stderr.decode() if e.stderr else ''}"
        pytest.fail(f"Training script timed out after {TEST_TIMEOUT} seconds.\nOutput: {process_output}")
    
    # Extract run ID from output
    run_id = extract_run_id_from_output(process_output)
    assert run_id, "Failed to extract WANDB_RUN_ID from script output"
    print(f"Extracted W&B run ID: {run_id}")
    
    # Connect to the run directly using its ID
    try:
        api = wandb.Api(timeout=60)
        entity = api.default_entity
        run_path = f"{entity}/{TEST_PROJECT_NAME}/{run_id}"
        run = api.run(run_path)
        
        # Verify wandb metrics were logged
        history = fetch_run_data_with_retries(run, keys=["mean reward", "_step"])
        
        # Verify expected metrics exist
        assert any("mean reward" in record for record in history), \
            f"Metric 'mean reward' not found in Wandb run history for run {run_id}"
            
        print(f"Successfully verified 'mean reward' logging for run: {run_id}")
        
        # Allow some time for videos to be processed by W&B
        print("Waiting for W&B to process videos...")
        time.sleep(15)
        
        # Verify video files were uploaded
        print(f"Checking for video content in run: {run_id}")
        # Update check_for_video_content to look for the new wandb key "video"
        has_videos = check_for_video_content(run, max_retries=8, delay=10)
        
        assert has_videos, f"No video content found in Wandb run {run_id}"
        print(f"Successfully verified video uploads for run: {run_id}")
        
        # Clean up test run (optional - uncomment if needed)
        # run.delete()
        
    except Exception as e:
        pytest.fail(f"Error verifying wandb metrics or videos: {e}")

@pytest.mark.skipif(os.getenv("WANDB_API_KEY") is None, reason="Requires WANDB_API_KEY environment variable")
def test_mujoco_video_logging():
    """Test that videos are properly logged with Mujoco environment."""
    # Setup test parameters with timestamp for uniqueness
    timestamp = int(time.time())
    run_name = f"{TEST_RUN_NAME_BASE}-mujoco-{timestamp}"
    env_id = "MujocoCartPole-v0"  # Use Mujoco environment specifically
    
    # Use small wandb_video_episodes to ensure videos are captured within our short test
    video_episodes = 5  # Small number to trigger videos frequently
    command = [
        "python", TRAIN_SCRIPT_PATH,
        "--project-name", TEST_PROJECT_NAME,
        "--run-name", run_name,
        "--env-id", env_id,
        # Use the new argument name
        "--wandb-video-episodes", str(video_episodes),
        # Remove --record-video flag
        "algo:ppo-config",
        "--algo.n-epi", "10",  # Run for enough episodes to generate videos
        "--algo.report-interval", "2"  # Ensure frequent reporting
    ]

    # Run the training script and capture output
    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            timeout=TEST_TIMEOUT
        )
        process_output = result.stdout + "\n" + result.stderr
        print(f"Training completed. Output excerpt: {process_output[:200]}...")
    except subprocess.CalledProcessError as e:
        process_output = f"{e.stdout or ''}\n{e.stderr or ''}"
        pytest.fail(f"Training script failed with error: {e}\nOutput: {process_output}")
    except subprocess.TimeoutExpired as e:
        process_output = f"{e.stdout.decode() if e.stdout else ''}\n{e.stderr.decode() if e.stderr else ''}"
        pytest.fail(f"Training script timed out after {TEST_TIMEOUT} seconds.\nOutput: {process_output}")
    
    # Extract run ID from output
    run_id = extract_run_id_from_output(process_output)
    assert run_id, "Failed to extract WANDB_RUN_ID from script output"
    print(f"Extracted W&B run ID: {run_id}")
    
    # Connect to the run directly using its ID
    try:
        api = wandb.Api(timeout=60)
        entity = api.default_entity
        run_path = f"{entity}/{TEST_PROJECT_NAME}/{run_id}"
        run = api.run(run_path)
        
        # Verify wandb metrics were logged
        history = fetch_run_data_with_retries(run, keys=["mean reward", "_step"])
        
        # Verify expected metrics exist
        assert any("mean reward" in record for record in history), \
            f"Metric 'mean reward' not found in Wandb run history for run {run_id}"
            
        print(f"Successfully verified 'mean reward' logging for run: {run_id}")
        
        # Allow some time for videos to be processed by W&B
        print("Waiting for W&B to process videos...")
        time.sleep(15)
        
        # Verify video files were uploaded
        print(f"Checking for video content in run: {run_id}")
        # Update check_for_video_content to look for the new wandb key "video"
        has_videos = check_for_video_content(run, max_retries=8, delay=10)
        
        assert has_videos, f"No video content found in Wandb run {run_id}"
        print(f"Successfully verified video uploads for run: {run_id}")
        
    except Exception as e:
        pytest.fail(f"Error verifying wandb metrics or videos: {e}")

# Add a similar specific test for Genesis environments
@pytest.mark.skipif(os.getenv("WANDB_API_KEY") is None, reason="Requires WANDB_API_KEY environment variable")
def test_genesis_video_logging():
    """Test that videos are properly logged with Genesis environment."""
    # Setup test parameters with timestamp for uniqueness
    timestamp = int(time.time())
    run_name = f"{TEST_RUN_NAME_BASE}-genesis-{timestamp}"
    env_id = "GenCartPole-v0"  # Use Genesis environment specifically
    
    # Use small wandb_video_episodes to ensure videos are captured within our short test
    video_episodes = 5  # Small number to trigger videos frequently
    command = [
        "python", TRAIN_SCRIPT_PATH,
        "--project-name", TEST_PROJECT_NAME,
        "--run-name", run_name,
        "--env-id", env_id,
        # Use the new argument name
        "--wandb-video-episodes", str(video_episodes),
        # Remove --record-video flag
        "algo:ppo-config",
        "--algo.n-epi", "10",  # Run for enough episodes to generate videos
        "--algo.report-interval", "2"  # Ensure frequent reporting
    ]

    # Run the training script and capture output
    try:
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, 
            timeout=TEST_TIMEOUT
        )
        process_output = result.stdout + "\n" + result.stderr
        print(f"Training completed. Output excerpt: {process_output[:200]}...")
    except subprocess.CalledProcessError as e:
        process_output = f"{e.stdout or ''}\n{e.stderr or ''}"
        pytest.fail(f"Training script failed with error: {e}\nOutput: {process_output}")
    except subprocess.TimeoutExpired as e:
        process_output = f"{e.stdout.decode() if e.stdout else ''}\n{e.stderr.decode() if e.stderr else ''}"
        pytest.fail(f"Training script timed out after {TEST_TIMEOUT} seconds.\nOutput: {process_output}")
    
    # Extract run ID from output
    run_id = extract_run_id_from_output(process_output)
    assert run_id, "Failed to extract WANDB_RUN_ID from script output"
    print(f"Extracted W&B run ID: {run_id}")
    
    # Connect to the run directly using its ID
    try:
        api = wandb.Api(timeout=60)
        entity = api.default_entity
        run_path = f"{entity}/{TEST_PROJECT_NAME}/{run_id}"
        run = api.run(run_path)
        
        # Verify wandb metrics were logged
        history = fetch_run_data_with_retries(run, keys=["mean reward", "_step"])
        
        # Verify expected metrics exist
        assert any("mean reward" in record for record in history), \
            f"Metric 'mean reward' not found in Wandb run history for run {run_id}"
            
        print(f"Successfully verified 'mean reward' logging for run: {run_id}")
        
        # Allow some time for videos to be processed by W&B
        print("Waiting for W&B to process videos...")
        time.sleep(15)
        
        # Verify video files were uploaded
        print(f"Checking for video content in run: {run_id}")
        # Update check_for_video_content to look for the new wandb key "video"
        has_videos = check_for_video_content(run, max_retries=8, delay=10)
        
        assert has_videos, f"No video content found in Wandb run {run_id}"
        print(f"Successfully verified video uploads for run: {run_id}")
        
    except Exception as e:
        pytest.fail(f"Error verifying wandb metrics or videos: {e}")
