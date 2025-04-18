import gymnasium as gym
import numpy as np
import torch
import wandb
import tempfile
import os
import imageio # Add imageio import for saving video locally if needed
import inspect # Add inspect import
from genRL.utils import downsample_list_image_to_video_array

class RecordVideo(gym.Wrapper):
    """Wraps a single environment to record video and log to wandb.

    Handles vectorized environments by only recording from the worker with index 0.
    Assumes the underlying environment supports 'rgb_array' rendering.
    Logs the video to wandb based on a step interval relative to wandb steps.
    """
    def __init__(self, env, wandb_video_steps, video_key="env_video", fps=None):
        super().__init__(env)

        # Determine if this is the primary worker (index 0) or a single env
        self.worker_index = getattr(env, 'worker_index', 0) # Default to 0 if not vectorized or index not passed
        self.is_primary_worker = (self.worker_index == 0)

        if not self.is_primary_worker:
            # Non-primary workers don't need recording logic
            self.wandb_video_steps = None
            self._is_recording = False
            self._recorded_frames = []
            self._temp_video_dir = None
            self.video_key = None
            self.fps = None
            return # Skip rest of init for non-primary workers

        # --- Initialization only for primary worker (or single env) --- 
        if wandb_video_steps is None or wandb_video_steps <= 0:
            print("[RecordVideo] Warning: wandb_video_steps not set or invalid. Disabling video recording.")
            self.wandb_video_steps = None
        else:
            self.wandb_video_steps = wandb_video_steps

        self.video_key = video_key
        
        # Try to get FPS from env metadata, default to 30
        self.fps = fps
        if self.fps is None:
             try:
                 self.fps = self.env.metadata.get('render_fps', 30)
             except AttributeError:
                 self.fps = 30

        self._last_video_log_step = 0
        self._is_recording = False
        self._recorded_frames = []
        self._temp_video_dir = None

        # Check if underlying env supports rgb_array rendering
        try:
            if 'rgb_array' not in self.env.metadata.get('render_modes', []):
                 print(f"[RecordVideo Worker {self.worker_index}] Warning: Underlying env does not list 'rgb_array' in metadata. Recording might fail.")
        except Exception as e:
            print(f"[RecordVideo Worker {self.worker_index}] Warning: Could not verify 'rgb_array' support in underlying env: {e}")

        if self.wandb_video_steps is not None:
            self._temp_video_dir = tempfile.TemporaryDirectory()
            print(f"[RecordVideo Worker {self.worker_index}] Temp video dir: {self._temp_video_dir.name}")
            # Ensure the env's render mode is set correctly if possible (might be immutable)
            try:
                if self.env.render_mode != 'rgb_array':
                    print(f"[RecordVideo Worker {self.worker_index}] Warning: Env render_mode is '{self.env.render_mode}'. Recording requires 'rgb_array'. Attempting to render with mode override.")
            except AttributeError:
                 print(f"[RecordVideo Worker {self.worker_index}] Warning: Could not check/set underlying env render_mode.")


    def step(self, action):
        """Steps the environment and captures a frame if recording and is primary worker."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Only record if this is the primary worker and recording is active
        if self.is_primary_worker and self._is_recording:
            try:
                # --- Debugging --- 
                print(f"[RecordVideo Worker {self.worker_index} DEBUG] self.env type: {type(self.env)}")
                try:
                    render_sig = inspect.signature(self.env.render)
                    print(f"[RecordVideo Worker {self.worker_index} DEBUG] self.env.render signature: {render_sig}")
                except Exception as sig_e:
                    print(f"[RecordVideo Worker {self.worker_index} DEBUG] Error inspecting self.env.render signature: {sig_e}")
                # --- End Debugging ---
                frame = self.env.render(mode="rgb_array")
                if frame is not None:
                    self._recorded_frames.append(frame)
            except Exception as e:
                print(f"[RecordVideo Worker {self.worker_index}] Error rendering frame: {e}")

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Resets the environment, stops/logs previous recording, starts new one (only for primary worker)."""
        # Only primary worker handles recording logic
        if self.is_primary_worker:
            self._stop_recording()

        obs, info = self.env.reset(seed=seed, options=options)

        if self.is_primary_worker:
            self._start_recording()

        return obs, info

    # Add a render method to handle the mode argument
    def render(self, mode=None):
        """Renders the environment, passing the mode argument down."""
        # Use the mode passed to this call, or the wrapper's default if None
        # Note: The base env might have its own default render_mode, 
        # but the wrapper needs to decide which mode to request.
        # We prioritize the mode passed to this function call.
        render_mode_to_use = mode if mode is not None else self.render_mode 
        
        # If this instance is the primary worker and is recording, 
        # it might have already rendered in the step method. 
        # However, calling render explicitly might be for visualization 
        # or getting a specific frame outside the step loop.
        # We should just pass the call down.
        
        # Ensure the underlying env's render method is called with the correct mode
        return self.env.render(mode=render_mode_to_use)

    def _start_recording(self):
        # This method should only be called by the primary worker instance
        if self.wandb_video_steps is None:
            return

        try:
            wandb_step = wandb.run.step if wandb.run else 0
        except AttributeError:
            wandb_step = 0

        if ((wandb_step - self._last_video_log_step) >= self.wandb_video_steps or \
                self._last_video_log_step == 0) and not self._is_recording:

            self._recorded_frames = []
            self._is_recording = True
            self._last_video_log_step = wandb_step
            print(f"[RecordVideo Worker {self.worker_index}] Starting video recording at step {wandb_step}.")

    def _stop_recording(self):
        # This method should only be called by the primary worker instance
        if not self._is_recording or not self._recorded_frames:
            self._is_recording = False
            return

        print(f"[RecordVideo Worker {self.worker_index}] Stopping video recording. Frames captured: {len(self._recorded_frames)}")
        self._is_recording = False

        if self._temp_video_dir is None:
            print(f"[RecordVideo Worker {self.worker_index}] Warning: No temp directory for saving video.")
            self._recorded_frames = []
            return

        try:
            factor = 5 # Adjust downsampling factor as needed
            video_array = downsample_list_image_to_video_array(self._recorded_frames, factor=factor)
            final_fps = self.fps / factor

            if wandb.run is not None:
                wandb.log({self.video_key: wandb.Video(video_array, fps=final_fps, format="mp4")})
                print(f"[RecordVideo Worker {self.worker_index}] Video logged to wandb with key '{self.video_key}'.")
            else:
                print(f"[RecordVideo Worker {self.worker_index}] Wandb not initialized, skipping video log.")

        except Exception as e:
            print(f"[RecordVideo Worker {self.worker_index}] Error processing or logging video: {e}")
        finally:
            self._recorded_frames = []

    def close(self):
        """Closes the environment and cleans up recording resources (only for primary worker)."""
        if self.is_primary_worker:
            self._stop_recording()
            if self._temp_video_dir is not None:
                try:
                    self._temp_video_dir.cleanup()
                    print(f"[RecordVideo Worker {self.worker_index}] Cleaned up temp video directory.")
                except Exception as e:
                    print(f"[RecordVideo Worker {self.worker_index}] Error cleaning up temp video directory: {e}")
        # Always close the underlying environment
        self.env.close()

    def __del__(self):
        # Ensure close is called, especially for the primary worker's cleanup
        self.close()
