import gymnasium as gym
import numpy as np
import torch
import wandb
import tempfile
import os
import imageio # Add imageio import for saving video locally if needed
from gymnasium.vector import VectorWrapper
from genRL.utils import downsample_list_image_to_video_array

class RecordVectorizedVideo(VectorWrapper):
    """Wraps a vectorized environment to record video from a designated worker (env 0).

    Assumes the underlying vectorized environment's workers support 'rgb_array' rendering.
    Logs the video to wandb based on a step interval.
    """
    def __init__(self, env, wandb_video_steps, video_key="vector_video", fps=30):
        super().__init__(env)

        if not isinstance(env, gym.vector.VectorEnv):
             raise TypeError("RecordVectorizedVideo wrapper requires a VectorEnv.")

        if wandb_video_steps is None or wandb_video_steps <= 0:
            print("[RecordVideoWrapper] Warning: wandb_video_steps not set or invalid. Disabling video recording.")
            self.wandb_video_steps = None
        else:
            self.wandb_video_steps = wandb_video_steps

        self.video_key = video_key
        self.fps = fps # Base FPS, will be adjusted by downsampling

        self._last_video_log_step = 0
        self._is_recording = False
        self._recorded_frames = []
        self._temp_video_dir = None

        # Check if underlying envs support rgb_array rendering
        try:
            if hasattr(self.env, 'single_metadata') and 'rgb_array' not in self.env.single_metadata.get('render_modes', []):
                 print(f"[RecordVideoWrapper] Warning: Underlying env does not list 'rgb_array' in metadata. Recording might fail.")
            elif hasattr(self.env.unwrapped, 'envs') and hasattr(self.env.unwrapped.envs[0], 'metadata') and 'rgb_array' not in self.env.unwrapped.envs[0].metadata.get('render_modes', []):
                 print(f"[RecordVideoWrapper] Warning: Underlying env worker 0 does not list 'rgb_array' in metadata. Recording might fail.")
        except Exception as e:
            print(f"[RecordVideoWrapper] Warning: Could not verify 'rgb_array' support in underlying env: {e}")

        if self.wandb_video_steps is not None:
            self._temp_video_dir = tempfile.TemporaryDirectory()
            print(f"[RecordVideoWrapper] Temp video dir: {self._temp_video_dir.name}")

    def step(self, actions):
        """Steps the environment and captures a frame from worker 0 if recording."""
        obs, reward, terminated, truncated, info = self.env.step(actions)

        if self._is_recording:
            try:
                # Render frame from worker 0 using underlying env's call method
                frames = self.env.unwrapped.call("render")
                if frames and frames[0] is not None:
                    self._recorded_frames.append(frames[0])
            except Exception as e:
                print(f"[RecordVideoWrapper] Error rendering frame from worker 0: {e}")

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Resets the environment, stops/logs previous recording, starts new one."""
        self._stop_recording()
        obs, info = self.env.reset(seed=seed, options=options)
        self._start_recording()
        return obs, info

    def _start_recording(self):
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
            print(f"[RecordVideoWrapper] Starting video recording at step {wandb_step}.")

    def _stop_recording(self):
        if not self._is_recording or not self._recorded_frames:
            self._is_recording = False
            return
        print(f"[RecordVideoWrapper] Stopping video recording. Frames captured: {len(self._recorded_frames)}")
        self._is_recording = False
        if self._temp_video_dir is None:
            print("[RecordVideoWrapper] Warning: No temp directory for saving video.")
            self._recorded_frames = []
            return
        try:
            factor = 5
            video_array = downsample_list_image_to_video_array(self._recorded_frames, factor=factor)
            base_fps = 30
            try:
                 if hasattr(self.env, 'single_metadata'):
                     base_fps = self.env.single_metadata.get('render_fps', 30)
                 elif hasattr(self.env.unwrapped, 'envs') and hasattr(self.env.unwrapped.envs[0], 'metadata'):
                     base_fps = self.env.unwrapped.envs[0].metadata.get('render_fps', 30)
            except Exception:
                 pass
            final_fps = base_fps / factor
            if wandb.run is not None:
                wandb.log({self.video_key: wandb.Video(video_array, fps=final_fps, format="mp4")})
                print(f"[RecordVideoWrapper] Video logged to wandb with key '{self.video_key}'.")
            else:
                print("[RecordVideoWrapper] Wandb not initialized, skipping video log.")
        except Exception as e:
            print(f"[RecordVideoWrapper] Error processing or logging video: {e}")
        finally:
            self._recorded_frames = []

    def close(self):
        """Closes the environment and cleans up recording resources."""
        self._stop_recording()
        self.env.close()
        if self._temp_video_dir is not None:
            try:
                self._temp_video_dir.cleanup()
                print("[RecordVideoWrapper] Cleaned up temp video directory.")
            except Exception as e:
                print(f"[RecordVideoWrapper] Error cleaning up temp video directory: {e}")

    def __del__(self):
        self.close()
