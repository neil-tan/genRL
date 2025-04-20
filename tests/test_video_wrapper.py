import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import tempfile
import time
import multiprocessing as mp
from unittest.mock import patch, MagicMock, ANY, call

# Import the wrapper
from genRL.wrappers.record_video import RecordVideoWrapper

# --- Mock Environment --- #
class MockEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    observation_space = spaces.Box(0, 255, (64, 64, 3), np.uint8)
    action_space = spaces.Discrete(2)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.episode_step = 0
        self.observation = self.observation_space.sample()

    def step(self, action):
        self.episode_step += 1
        terminated = self.episode_step >= 10 # End after 10 steps
        truncated = False
        reward = 1.0
        self.observation = self.observation_space.sample()
        info = {}
        # Simulate returning info from underlying env if needed
        return self.observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        self.observation = self.observation_space.sample()
        info = {}
        return self.observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            # Return a frame with changing color based on step
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, self.episode_step % 3] = 255 # Cycle color
            return frame
        else:
            return None

    def close(self):
        pass

# --- Fixtures --- #
@pytest.fixture
def mock_env():
    "Provides a fresh instance of the MockEnv."
    return MockEnv()

@pytest.fixture
def temp_video_dir():
    "Creates a temporary directory for video output."
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

# --- Tests --- #

def test_single_process_init(mock_env):
    "Test initialization in a single-process scenario."
    wrapper = RecordVideoWrapper(mock_env)
    assert wrapper.is_recording_worker is True
    assert wrapper.recording_enabled is True
    assert wrapper._temp_dir_handle is not None
    assert wrapper.video_folder is not None
    wrapper.close() # Cleanup temp dir

@patch('genRL.wrappers.record_video.imageio.get_writer')
def test_single_process_recording(mock_writer_factory, mock_env, temp_video_dir):
    "Test basic recording flow in single process."
    # Mock the writer object itself
    mock_writer = MagicMock()
    mock_writer_factory.return_value = mock_writer

    # Wrap the env, trigger first episode
    wrapper = RecordVideoWrapper(mock_env, episode_trigger=lambda ep_id: ep_id == 0)
    # Override video folder to use our fixture temp dir for easier checks if needed
    # (though mocking imageio makes file check less critical)
    wrapper.video_folder = temp_video_dir

    assert wrapper.is_recording_worker is True

    # Episode 0: Should record
    obs, info = wrapper.reset()
    assert wrapper.recording is True
    assert mock_writer_factory.call_count == 1 # Writer created on first reset
    start_call_args = mock_writer_factory.call_args
    assert temp_video_dir in start_call_args[0][0] # Path includes temp dir
    assert "episode0" in start_call_args[0][0] # Filename includes episode id

    mock_append_calls = 0
    for i in range(5):
        action = wrapper.action_space.sample()
        obs, reward, term, trunc, info = wrapper.step(action)
        mock_append_calls += 1
        # First frame captured on reset, subsequent on step
        assert mock_writer.append_data.call_count == (i + 1) + 1

    assert wrapper.recording is True # Still recording
    wrapper.close() # Should close writer
    assert wrapper.recording is False
    mock_writer.close.assert_called_once()

    # Episode 1: Should NOT record
    mock_writer_factory.reset_mock()
    mock_writer.reset_mock()
    obs, info = wrapper.reset()
    assert wrapper.recording is False
    mock_writer_factory.assert_not_called() # No new writer
    for i in range(5):
        action = wrapper.action_space.sample()
        obs, reward, term, trunc, info = wrapper.step(action)
    mock_writer.append_data.assert_not_called() # No frames added
    wrapper.close()

def test_multi_process_designation(mock_env):
    "Test that only one wrapper becomes the recorder in simulated multi-process."
    num_wrappers = 5
    lock = mp.Lock()
    flag = mp.Value('i', 0) # Initial value 0

    wrappers = []
    for _ in range(num_wrappers):
        # Pass the *same* lock and flag to each instance
        wrapper = RecordVideoWrapper(mock_env, recorder_lock=lock, recorder_flag=flag)
        wrappers.append(wrapper)

    recorder_count = 0
    for wrapper in wrappers:
        if wrapper.is_recording_worker:
            recorder_count += 1
        # Clean up potential temp dirs created by the recorder
        wrapper.close()

    assert recorder_count == 1, "Exactly one wrapper should be designated as recorder"
    assert flag.value == 1, "Recorder flag should be set to 1"

@patch('genRL.wrappers.record_video.imageio.get_writer')
def test_episode_trigger(mock_writer_factory, mock_env, temp_video_dir):
    "Test custom episode trigger functionality."
    mock_writer = MagicMock()
    mock_writer_factory.return_value = mock_writer

    # Trigger recording every 2nd episode (episodes 0, 2, 4, ...)
    trigger = lambda ep_id: ep_id % 2 == 0
    wrapper = RecordVideoWrapper(mock_env, episode_trigger=trigger, video_folder=temp_video_dir)

    # Episode 0 (should record)
    wrapper.reset()
    assert wrapper.recording is True
    mock_writer_factory.assert_called_once()
    wrapper.close()

    # Episode 1 (should NOT record)
    mock_writer_factory.reset_mock()
    wrapper.reset()
    assert wrapper.recording is False
    mock_writer_factory.assert_not_called()
    wrapper.close()

    # Episode 2 (should record)
    mock_writer_factory.reset_mock()
    wrapper.reset()
    assert wrapper.recording is True
    mock_writer_factory.assert_called_once()
    wrapper.close()

@patch('genRL.wrappers.record_video.imageio.get_writer')
def test_frame_capture(mock_writer_factory, mock_env, temp_video_dir):
    "Test that _capture_frame calls env.render and appends data."
    mock_writer = MagicMock()
    mock_writer_factory.return_value = mock_writer
    mock_env.render = MagicMock(return_value=np.array([[[0, 0, 0]]], dtype=np.uint8)) # Mock render

    wrapper = RecordVideoWrapper(mock_env, video_folder=temp_video_dir)

    # Start recording (default trigger records episode 0)
    wrapper.reset()
    assert wrapper.recording is True
    mock_writer_factory.assert_called_once()
    mock_env.render.assert_called_once() # Render called during reset capture

    # Step and capture frames
    mock_env.render.reset_mock()
    mock_frame1 = np.array([[[1, 1, 1]]], dtype=np.uint8)
    mock_frame2 = np.array([[[2, 2, 2]]], dtype=np.uint8)
    mock_env.render.side_effect = [mock_frame1, mock_frame2]

    wrapper.step(0) # Step 1
    wrapper.step(1) # Step 2

    # Assert render was called twice during steps
    assert mock_env.render.call_count == 2
    # Assert append_data was called with the correct frames
    mock_writer.append_data.assert_has_calls([
        call(mock_frame1),
        call(mock_frame2)
    ])

    # Close the wrapper and writer
    wrapper.close()
    mock_writer.close.assert_called_once()

# Add more tests below...
