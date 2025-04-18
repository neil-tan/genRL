import gymnasium as gym
import numpy as np
import os
import imageio
import wandb
import time
from gymnasium.core import Wrapper
import multiprocessing as mp 
from functools import partial 

class RecordSingleEnvVideo(Wrapper):
    """Wraps a single environment instance to record video frames.

    Uses multiprocessing synchronization primitives (Lock, Value) passed during
    initialization to designate a single instance across processes as the recorder.
    Expects the video folder path to be provided.

    Args:
        env (gym.Env): The base environment instance to wrap.
        video_folder (str): The folder to save the video frames to.
        recorder_lock (mp.Lock): A multiprocessing lock.
        recorder_flag (mp.Value): A multiprocessing Value('i', 0).
        name_prefix (str): Prefix for the video file name.
        video_length (int): The maximum number of frames to record in one video.
        step_trigger (callable): A function that takes the step count and returns true if recording should be enabled.
        fps (int): Frames per second for the output video.
    """
    def __init__(
        self,
        env: gym.Env,
        video_folder: str, 
        recorder_lock: mp.Lock,
        recorder_flag: mp.Value,
        name_prefix: str = "rl-video",
        video_length: int = 200,
        step_trigger = None, 
        fps: int = 30,
    ):
        super().__init__(env)
        
        self.is_recording_worker = False
        self.video_folder = None 
        self.recording = False
        self.video_writer = None
        self.latest_video_path = None
        self.recorded_frames = 0
        self.step_id = 0 

        # --- Recorder Designation Logic ---
        with recorder_lock:
            if recorder_flag.value == 0:
                self.is_recording_worker = True
                recorder_flag.value = 1 
                print(f"[RecordSingleEnvVideo] Designated as recorder.")
        # --- End Recorder Designation Logic ---

        if self.is_recording_worker:
            if video_folder is None:
                 print("[RecordSingleEnvVideo recorder] Error: video_folder cannot be None for the recording worker.")
                 self.is_recording_worker = False 
            else:
                self.video_folder = os.path.abspath(video_folder)
                print(f"[RecordSingleEnvVideo recorder] Using video folder: {self.video_folder}")

            if self.is_recording_worker: 
                if step_trigger is None:
                    self.step_trigger = lambda step: step == 0 
                else:
                    self.step_trigger = step_trigger

                self.name_prefix = name_prefix
                self.video_length = video_length
                self.fps = fps

                if "rgb_array" not in self.env.metadata.get("render_modes", []):
                    print(f"[RecordSingleEnvVideo recorder] Warning: Wrapped env {env} does not list 'rgb_array' in metadata. Recording might fail.")
        
        if not self.is_recording_worker:
             self.step_trigger = lambda step: False

    def reset(self, **kwargs):
        """Reset the environment, starting video recording if triggered and is target worker."""
        obs, info = self.env.reset(**kwargs)
        
        if self.is_recording_worker:
            self.step_id = 0 
            if self.recording:
                self.close_video_recorder()
            
            if self.step_trigger(self.step_id): 
                self.start_video_recorder()
            
            if self.recording:
                self._capture_frame()

        return obs, info

    def step(self, action):
        """Step the environment, capturing a frame if recording."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.is_recording_worker:
            self.step_id += 1 
            
            if self.recording and (terminated or truncated):
                self.close_video_recorder()
            elif self.recording and self.recorded_frames >= self.video_length:
                 print(f"[RecordSingleEnvVideo recorder] Reached video length limit ({self.video_length}).")
                 self.close_video_recorder()
            
            if self.recording:
                self._capture_frame()

        return obs, reward, terminated, truncated, info

    def _capture_frame(self):
        """Render the frame and write it to the video recorder."""
        if not self.is_recording_worker or self.video_writer is None:
            return
        
        try:
            frame = self.env.render(mode="rgb_array") 
            if frame is None:
                 print(f"[RecordSingleEnvVideo recorder] Warning: env.render('rgb_array') returned None.")
                 return
            if not isinstance(frame, np.ndarray):
                 print(f"[RecordSingleEnvVideo recorder] Warning: env.render('rgb_array') did not return a numpy array (type: {type(frame)}). Skipping frame.")
                 return
                 
            self.video_writer.append_data(frame)
            self.recorded_frames += 1

        except Exception as e:
            print(f"[RecordSingleEnvVideo recorder] Error capturing frame: {e}")

    def start_video_recorder(self):
        """Start the video recorder if not already recording and is the target worker."""
        if not self.is_recording_worker or self.recording or self.video_folder is None:
            return
        
        if not os.path.isdir(self.video_folder):
            print(f"[RecordSingleEnvVideo recorder] Error: Video folder {self.video_folder} does not exist. Cannot start recording.")
            return
            
        self.close_video_recorder() 

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_filename = f"{self.name_prefix}-step{self.step_id}-{timestamp}.mp4"
        self.latest_video_path = os.path.join(self.video_folder, video_filename)
        
        print(f"[RecordSingleEnvVideo recorder] Starting video recording to {self.latest_video_path} (length={self.video_length}, fps={self.fps})")
        try:
            self.video_writer = imageio.get_writer(self.latest_video_path, fps=self.fps, macro_block_size=1)
            self.recording = True
            self.recorded_frames = 0
        except Exception as e:
             print(f"[RecordSingleEnvVideo recorder] Failed to start video writer: {e}")
             self.video_writer = None
             self.recording = False
             self.latest_video_path = None

    def close_video_recorder(self):
        """Close the video recorder and optionally log to wandb."""
        if not self.is_recording_worker or not self.recording:
            return

        print(f"[RecordSingleEnvVideo recorder] Stopping video recording. Frames captured: {self.recorded_frames}")
        if self.video_writer:
            try:
                self.video_writer.close()
            except Exception as e:
                 print(f"[RecordSingleEnvVideo recorder] Error closing video writer: {e}")

            if wandb.run is not None and self.latest_video_path and self.recorded_frames > 0:
                try:
                    print(f"[RecordSingleEnvVideo recorder] Logging video to wandb: {self.latest_video_path}")
                    wandb.log({"single_env_video": wandb.Video(self.latest_video_path, fps=self.fps, format="mp4")})
                except Exception as e:
                    print(f"[RecordSingleEnvVideo recorder] Error logging video to wandb: {e}")
            elif not wandb.run:
                 print(f"[RecordSingleEnvVideo recorder] wandb run not active, cannot log video.")

        self.recording = False
        self.video_writer = None
        self.recorded_frames = 0

    def close(self):
        """Close the wrapper and the base environment. Video dir cleanup is external."""
        if self.is_recording_worker:
            self.close_video_recorder()
                
        super().close() 

