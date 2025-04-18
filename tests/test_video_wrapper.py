import pytest
import os
import numpy as np
import gymnasium as gym
import multiprocessing as mp
import tempfile
import imageio
from pathlib import Path

# Set MUJOCO_GL environment variable to ensure proper rendering
os.environ['MUJOCO_GL'] = 'egl'

# Import environments to register them
import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.mujoco.cartpole
from genRL.wrappers.record_single_env_video import RecordSingleEnvVideo

class TestRender:
    """Test the render method of environments to ensure it works consistently."""
    
    @pytest.mark.parametrize("env_id", ["GenCartPole-v0", "MujocoCartPole-v0"])
    def test_env_render(self, env_id):
        """Test that environments can render rgb_array frames."""
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset()
        
        # Test direct render call without mode
        frame = env.render()
        assert frame is not None, f"{env_id} render() returned None"
        assert isinstance(frame, np.ndarray), f"{env_id} render() didn't return numpy array"
        assert len(frame.shape) == 3, f"{env_id} frame has wrong shape: {frame.shape}"
        assert frame.shape[2] == 3, f"{env_id} frame doesn't have RGB channels"
        
        # Check frame has reasonable content (not all zeros or ones)
        assert np.min(frame) < 10, f"{env_id} frame min value is too high: {np.min(frame)}"
        assert np.max(frame) > 200, f"{env_id} frame max value is too low: {np.max(frame)}"
        
        env.close()


class TestVideoWrapper:
    """Test the RecordSingleEnvVideo wrapper."""
    
    @pytest.mark.parametrize("env_id", ["GenCartPole-v0", "MujocoCartPole-v0"])
    def test_wrapper_recording(self, env_id):
        """Test that the video wrapper can record frames and save a video file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create environment
            env = gym.make(env_id, render_mode="rgb_array")
            
            # Apply wrapper with no lock (so this instance will be the recorder)
            wrapped_env = RecordSingleEnvVideo(
                env=env,
                recorder_lock=None,
                recorder_flag=None,
                name_prefix=f"test-{env_id}",
                video_length=10,  # Short video for testing
                fps=10,
                record_video=True
            )
            
            # Check wrapper state
            assert wrapped_env.is_recording_worker, "Wrapper should be designated as recorder"
            assert wrapped_env.video_folder is not None, "Temp folder should be created"
            
            # Run environment for a few steps to generate video
            wrapped_env.reset()
            
            # Step and verify recording starts
            for i in range(15):  # More than video_length to test completing a video
                action = wrapped_env.action_space.sample()
                _, _, done, truncated, _ = wrapped_env.step(action)
                
                if done or truncated:
                    wrapped_env.reset()
            
            # Close to ensure video is finalized
            wrapped_env.close()
            
            # Check that a video file was created in the wrapper's temp directory
            video_files = list(Path(wrapped_env.video_folder).glob("*.mp4"))
            assert len(video_files) > 0, f"No video files created for {env_id}"
            
            # Verify video file is valid
            video_path = str(video_files[0])
            assert os.path.getsize(video_path) > 0, f"Video file is empty: {video_path}"
            
            # Try to read the video file to ensure it's a valid format
            reader = imageio.get_reader(video_path)
            frames = list(reader)
            assert len(frames) > 0, f"Video has no frames: {video_path}"
    
    @pytest.mark.parametrize("env_id", ["GenCartPole-v0", "MujocoCartPole-v0"])
    def test_wrapper_sync_primitives(self, env_id):
        """Test that synchronization primitives work to designate only one recorder."""
        recorder_lock = mp.Lock()
        recorder_flag = mp.Value('i', 0)
        
        # Create multiple wrappers with the same lock and flag
        env1 = gym.make(env_id, render_mode="rgb_array")
        wrapped_env1 = RecordSingleEnvVideo(
            env=env1,
            recorder_lock=recorder_lock,
            recorder_flag=recorder_flag,
            name_prefix=f"test-{env_id}-1",
            video_length=10,
            record_video=True
        )
        
        env2 = gym.make(env_id, render_mode="rgb_array")
        wrapped_env2 = RecordSingleEnvVideo(
            env=env2,
            recorder_lock=recorder_lock,
            recorder_flag=recorder_flag,
            name_prefix=f"test-{env_id}-2",
            video_length=10,
            record_video=True
        )
        
        # Only one should be designated as recorder
        assert wrapped_env1.is_recording_worker != wrapped_env2.is_recording_worker, \
            "Only one wrapper should be designated as recorder"
        
        # The flag should be set
        assert recorder_flag.value == 1, "Recorder flag should be set to 1"
        
        # Clean up
        wrapped_env1.close()
        wrapped_env2.close()