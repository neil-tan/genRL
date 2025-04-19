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
    """Wraps a vectorized environment to record video from worker 0 based on episode count.

    Assumes the underlying vectorized environment's workers support 'rgb_array' rendering.
    Logs the video to wandb based on an episode interval.
    """
    # Rename wandb_video_episodes to wandb_video_episodes
    def __init__(self, env, wandb_video_episodes, video_key="vector_video", fps=30):
        super().__init__(env)

        if not isinstance(env, gym.vector.VectorEnv):
             raise TypeError("RecordVectorizedVideo wrapper requires a VectorEnv.")

        # Use episode count
        if wandb_video_episodes is None or wandb_video_episodes <= 0:
            print("[RecordVideoWrapper] Warning: wandb_video_episodes not set or invalid. Disabling video recording.")
            self.wandb_video_episodes = None
        else:
            self.wandb_video_episodes = wandb_video_episodes

        self.video_key = video_key
        self.fps = fps # Base FPS, will be adjusted by downsampling

        # Rename step counter to episode counter
        self._last_video_log_episode = 0
        self._current_episode = 0 # Add episode counter
        self._is_recording = False
        self._recorded_frames = []
        self._temp_video_dir = None

        # Check if underlying envs support rgb_array rendering
        try:
            # Access the metadata of the base single environment
            if hasattr(self.env, 'single_metadata') and 'rgb_array' not in self.env.single_metadata.get('render_modes', []):
                 print(f"[RecordVideoWrapper] Warning: Underlying env does not list 'rgb_array' in metadata. Recording might fail.")
            # Check metadata from worker 0 if single_metadata is not available (covers more vec env types)
            elif hasattr(self.env.unwrapped, 'envs') and hasattr(self.env.unwrapped.envs[0], 'metadata') and 'rgb_array' not in self.env.unwrapped.envs[0].metadata.get('render_modes', []): # Fix: this.env -> self.env
                 print(f"[RecordVideoWrapper] Warning: Underlying env worker 0 does not list 'rgb_array' in metadata. Recording might fail.")

        except Exception as e:
            print(f"[RecordVideoWrapper] Warning: Could not verify 'rgb_array' support in underlying env: {e}")


        if self.wandb_video_episodes is not None:
            self._temp_video_dir = tempfile.TemporaryDirectory()
            print(f"[RecordVideoWrapper] Temp video dir: {self._temp_video_dir.name}")

    def step(self, actions):
        """Steps the environment and captures a frame from worker 0 if recording."""
        obs, reward, terminated, truncated, info = self.env.step(actions)

        if self._is_recording:
            try:
                # Render frame from worker 0
                # Access the underlying vector environment's 'call' method
                frames = self.env.unwrapped.call("render")
                # Assuming render returns a list of frames, one per worker
                if frames and frames[0] is not None:
                    self._recorded_frames.append(frames[0])
            except Exception as e:
                print(f"[RecordVideoWrapper] Error rendering frame from worker 0: {e}")
                # Optionally stop recording if rendering fails consistently
                # self._stop_recording()

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Resets the environment, increments episode count, stops/logs previous recording, starts new one."""
        # Stop and log the previous recording before reset potentially starts a new one
        self._stop_recording()

        # Increment episode counter *before* reset potentially starts a new recording
        self._current_episode += 1 

        obs, info = self.env.reset(seed=seed, options=options)

        # Start new recording if conditions met (based on the *new* episode count)
        self._start_recording()

        return obs, info

    def _start_recording(self):
        # Use episode count for triggering
        if self.wandb_video_episodes is None:
            return

        # Check episode interval
        # Start recording if interval passed or first episode, and not already recording
        # Note: _current_episode is 1-based after the first reset
        if ((self._current_episode - self._last_video_log_episode) >= self.wandb_video_episodes or \
                self._last_video_log_episode == 0) and not self._is_recording:

            self._recorded_frames = []
            self._is_recording = True
            # Record the episode number when recording *starts*
            self._last_video_log_episode = self._current_episode 
            print(f"[RecordVideoWrapper] Starting video recording at episode {self._current_episode}.")

    def _stop_recording(self):
        if not self._is_recording or not self._recorded_frames:
            self._is_recording = False
            return

        print(f"[RecordVideoWrapper] Stopping video recording for episode {self._last_video_log_episode}. Frames captured: {len(self._recorded_frames)}")
        self._is_recording = False

        if self._temp_video_dir is None:
            print("[RecordVideoWrapper] Warning: No temp directory for saving video.")
            self._recorded_frames = []
            return

        try:
            # Downsample and create video array
            factor = 5 # Adjust downsampling factor as needed
            video_array = downsample_list_image_to_video_array(self._recorded_frames, factor=factor)

            # Adjust FPS based on downsampling
            # Need to get base FPS from underlying env metadata if possible
            base_fps = self.fps # Use the fps passed during init as default
            try:
                 if hasattr(self.env, 'single_metadata'):
                     base_fps = self.env.single_metadata.get('render_fps', self.fps)
                 elif hasattr(self.env.unwrapped, 'envs') and hasattr(self.env.unwrapped.envs[0], 'metadata'):
                     base_fps = self.env.unwrapped.envs[0].metadata.get('render_fps', self.fps)
            except Exception:
                 pass # Use default fps

            final_fps = base_fps / factor

            # Log to wandb if available
            if wandb.run is not None:
                # Log with episode number in the key for clarity, if desired
                # video_log_key = f"{self.video_key}_ep{self._last_video_log_episode}"
                video_log_key = self.video_key # Or keep the original key
                wandb.log({video_log_key: wandb.Video(video_array, fps=final_fps, format="mp4")})
                print(f"[RecordVideoWrapper] Video logged to wandb with key '{video_log_key}'.")
            else:
                print("[RecordVideoWrapper] Wandb not initialized, skipping video log.")
                # Optionally save locally
                # video_path = os.path.join(self._temp_video_dir.name, f"{self.video_key}_{self._last_video_log_step}.mp4")
                # imageio.mimsave(video_path, video_array, fps=final_fps)
                # print(f"[RecordVideoWrapper] Video saved locally to {video_path}")

        except Exception as e:
            print(f"[RecordVideoWrapper] Error processing or logging video: {e}")
        finally:
            self._recorded_frames = [] # Clear frames regardless of success

    def close(self):
        """Closes the environment and cleans up recording resources."""
        # Stop recording and log any remaining frames
        self._stop_recording()

        # Close the underlying environment
        self.env.close()

        # Clean up temp directory
        if self._temp_video_dir is not None:
            try:
                self._temp_video_dir.cleanup()
                print("[RecordVideoWrapper] Cleaned up temp video directory.")
            except Exception as e:
                print(f"[RecordVideoWrapper] Error cleaning up temp video directory: {e}")

    def __del__(self):
        # Ensure cleanup happens even if close() isn't explicitly called
        # Note: __del__ can be unreliable; explicit close() is preferred.
        try:
            self.close()
        except Exception:
            # Avoid errors during garbage collection if close fails or already called
            pass
