import gymnasium as gym
import numpy as np
import os
import imageio
import wandb
import time
from gymnasium.core import Wrapper, Env
import tempfile
import multiprocessing as mp
from typing import Optional, Callable
import torch
import cv2 # Import OpenCV for resizing

# Rename the class
class RecordVideoWrapper(Wrapper):
    """Wraps an environment to record video frames, handling both single and multi-process scenarios.

    In multi-process scenarios (e.g., using SyncVectorEnv), multiprocessing 
    synchronization primitives (Lock, Value) must be provided to designate a 
    single instance across processes as the recorder. 
    
    If synchronization primitives are not provided, the wrapper assumes it's running 
    in a single-process context (e.g., with a Genesis environment) and designates 
    itself as the recorder.

    Args:
        env (gym.Env): The base environment instance to wrap.
        recorder_lock (Optional[mp.Lock]): A multiprocessing lock (required for multi-process).
        recorder_flag (Optional[mp.Value]): A multiprocessing Value('i', 0) (required for multi-process).
        name_prefix (str): Prefix for the video file name.
        video_length (int): The maximum number of frames to record in one video.
        # Change step_trigger to episode_trigger
        episode_trigger (Optional[Callable[[int], bool]]): A function that takes the episode count 
                                                        and returns true if recording should be enabled 
                                                        for that episode. If None, records first episode.
        fps (int): Frames per second for the output video.
        record_video (bool): Whether to enable video recording. Defaults to True.
        frame_width (Optional[int]): Optional width to resize frames to.
        frame_height (Optional[int]): Optional height to resize frames to.
    """
    def __init__(
        self,
        env: Env,
        recorder_lock: Optional[mp.Lock] = None,
        recorder_flag: Optional[mp.Value] = None,
        name_prefix: str = "rl-video",
        video_length: int = 1000, # Default to longer length
        # Rename trigger
        episode_trigger: Optional[Callable[[int], bool]] = None,
        fps: int = 30,
        record_video: bool = True, # Add enable/disable flag
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ):
        super().__init__(env)
        
        # Initialize attributes
        self.is_recording_worker = False
        self._temp_dir_handle = None # Use a different name to avoid confusion
        self.video_folder = None 
        self.recording_enabled = record_video
        self.recording = False
        self.video_writer = None
        self.latest_video_path = None
        self.recorded_frames = 0
        self.episode_id = -1
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.name_prefix = name_prefix
        self.video_length = video_length
        self.fps = fps

        # --- Recorder Designation Logic --- 
        designated_recorder = False # Temporary flag
        if not self.recording_enabled:
            print(f"[RecordVideoWrapper] Recording disabled globally.")
        elif recorder_lock is not None and recorder_flag is not None:
            # Multi-process scenario: Use primitives
            with recorder_lock:
                if recorder_flag.value == 0:
                    designated_recorder = True
                    recorder_flag.value = 1 
                    print(f"[RecordVideoWrapper] Designated as recorder (multi-process attempt).")
        else:
            # Single-process scenario: Assume this is the recorder
            designated_recorder = True
            print(f"[RecordVideoWrapper] Designated as recorder (single-process attempt).")
        # --- End Recorder Designation Logic ---

        # --- Setup Recorder if Designated --- 
        if designated_recorder:
            # Check render mode support first
            if "rgb_array" not in self.env.unwrapped.metadata.get("render_modes", []):
                print(f"[RecordVideoWrapper recorder] Warning: Wrapped env {env.unwrapped.spec.id if env.unwrapped.spec else env} does not list 'rgb_array' in metadata. Recording might fail.")
                # Proceed anyway, but it might fail later

            # Try creating the temporary directory
            try:
                self._temp_dir_handle = tempfile.TemporaryDirectory()
                self.video_folder = self._temp_dir_handle.name
                self.is_recording_worker = True # Set flag ONLY if temp dir is successful
                print(f"[RecordVideoWrapper recorder] Successfully created temp video folder: {self.video_folder}")
            except Exception as e:
                 print(f"[RecordVideoWrapper recorder] Failed to create temp directory: {e}")
                 self.is_recording_worker = False # Ensure flag is false if temp dir fails
                 self.video_folder = None
                 # Clean up handle if it exists but assignment failed somehow
                 if self._temp_dir_handle:
                     try: self._temp_dir_handle.cleanup() 
                     except: pass
                 self._temp_dir_handle = None

        # --- End Setup Recorder --- 

        # Set episode trigger based on final recorder status
        if self.is_recording_worker:
            # Default trigger: record the first episode (episode_id == 0)
            if episode_trigger is None:
                self.episode_trigger = lambda ep_id: ep_id == 0
            else:
                self.episode_trigger = episode_trigger
        else:
             # If not the recorder, ensure trigger always returns False
             self.episode_trigger = lambda ep_id: False

    def reset(self, **kwargs):
        """Reset the environment, starting video recording if triggered."""
        obs, info = self.env.reset(**kwargs)
        
        # Always close recorder on reset if it was running
        if self.recording:
            self.close_video_recorder()

        self.episode_id += 1 # Increment episode counter

        if self.is_recording_worker and self.episode_trigger(self.episode_id):
            self.start_video_recorder()
            if self.recording: # Check if recorder started successfully
                self._capture_frame() # Capture the first frame after reset

        return obs, info

    def step(self, action):
        """Step the environment, capturing a frame if recording."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        done = torch.logical_or(torch.as_tensor(terminated), torch.as_tensor(truncated)).all()
        
        if self.recording: # Only capture if currently recording
            self._capture_frame()
            
            # Check conditions to stop recording
            if self.recorded_frames >= self.video_length:
                 print(f"[RecordVideoWrapper recorder] Reached video length limit ({self.video_length}).")
                 self.close_video_recorder()
            elif done: # Stop recording at the end of the episode
                 print(f"[RecordVideoWrapper recorder] Episode ended (terminated={terminated}, truncated={truncated}).")
                 self.close_video_recorder()

        return obs, reward, terminated, truncated, info

    def _capture_frame(self):
        """Render the frame, optionally resize, and write it to the video recorder."""
        if not self.is_recording_worker or self.video_writer is None:
            return
        
        try:
            # --- Debug: Check render mode before rendering --- #
            if self.recorded_frames == 0: # Only print once per recording
                mode_to_log = getattr(self.env, 'render_mode', 'AttributeNotFound')
                print(f"[RecordVideoWrapper recorder] Capturing frame. Env render mode: {mode_to_log}")
            # --- End Debug --- #

            # Use the render method of the wrapped environment
            frame = self.env.render() # Assumes render_mode='rgb_array' was set correctly
            
            if frame is None:
                 print(f"[RecordVideoWrapper recorder] Warning: env.render() returned None.")
                 return
            if not isinstance(frame, np.ndarray):
                 print(f"[RecordVideoWrapper recorder] Warning: env.render() did not return a numpy array (type: {type(frame)}). Skipping frame.")
                 return
            if frame.ndim != 3 or frame.shape[2] != 3:
                 print(f"[RecordVideoWrapper recorder] Warning: Frame has unexpected shape {frame.shape}. Skipping frame.")
                 return
                 
            # Resize frame if dimensions are specified
            if self.frame_width is not None and self.frame_height is not None:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)

            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                # print(f"[RecordVideoWrapper recorder] Warning: Frame dtype is {frame.dtype}, converting to uint8.")
                frame = frame.astype(np.uint8)

            self.video_writer.append_data(frame)
            self.recorded_frames += 1

        except Exception as e:
            print(f"[RecordVideoWrapper recorder] Error capturing frame: {e}")
            # Optionally close recorder on error?
            # self.close_video_recorder() 

    def start_video_recorder(self):
        """Start the video recorder if designated and conditions met."""
        if not self.is_recording_worker or self.recording or self.video_folder is None:
            return
            
        self.close_video_recorder() # Ensure any previous writer is closed

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Include episode ID in filename
        video_filename = f"{self.name_prefix}-episode{self.episode_id}-{timestamp}.mp4"
        self.latest_video_path = os.path.join(self.video_folder, video_filename)
        
        print(f"[RecordVideoWrapper recorder] Starting video recording to {self.latest_video_path} (episode={self.episode_id}, length={self.video_length}, fps={self.fps})")
        try:
            # Use macro_block_size=None for broader compatibility, let imageio choose
            self.video_writer = imageio.get_writer(self.latest_video_path, fps=self.fps, macro_block_size=None) 
            self.recording = True
            self.recorded_frames = 0
        except Exception as e:
             print(f"[RecordVideoWrapper recorder] Failed to start video writer: {e}")
             self.video_writer = None
             self.recording = False
             self.latest_video_path = None

    def close_video_recorder(self):
        """Close the video recorder and optionally log to wandb."""
        if not self.is_recording_worker or not self.recording:
            return

        print(f"[RecordVideoWrapper recorder] Stopping video recording. Frames captured: {self.recorded_frames}")
        if self.video_writer:
            try:
                self.video_writer.close()
            except Exception as e:
                 print(f"[RecordVideoWrapper recorder] Error closing video writer: {e}")

            # Log to wandb if conditions met
            if wandb.run is not None and self.latest_video_path and self.recorded_frames > 0:
                try:
                    print(f"[RecordVideoWrapper recorder] Logging video to wandb: {self.latest_video_path}")
                    # Use a consistent key, e.g., "video"
                    wandb.log({"video": wandb.Video(self.latest_video_path, format="mp4")})
                    
                except Exception as e:
                    print(f"[RecordVideoWrapper recorder] Error logging video to wandb: {e}")
            elif not wandb.run:
                 print(f"[RecordVideoWrapper recorder] wandb run not active, cannot log video.")

        self.recording = False
        self.video_writer = None
        # Keep recorded_frames count until next recording starts
        # self.recorded_frames = 0 # Don't reset here, reset in start_video_recorder

    def close(self):
        """Close the wrapper, the base environment, and clean up the temp directory if owner."""
        if self.is_recording_worker:
            self.close_video_recorder() # Ensure video file is closed and logged
            
            # Clean up the temporary directory using the handle
            if self._temp_dir_handle:
                try:
                    print(f"[RecordVideoWrapper recorder] Cleaning up temp video directory: {self.video_folder}")
                    self._temp_dir_handle.cleanup()
                except Exception as e:
                    print(f"[RecordVideoWrapper recorder] Error cleaning up temp video dir {self.video_folder}: {e}")
                self._temp_dir_handle = None
                self.video_folder = None # Clear the path after cleanup
                
        super().close() # Call the close method of the wrapped env

