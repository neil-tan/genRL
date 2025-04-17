import torch
import functools
import numpy as np
from typing import List
import wandb
import tempfile
import pickle
import os
import re
import optuna
import functools
import genesis as gs

# --- Debugging Utility ---
_DEBUG_ENABLED = os.environ.get('GENRL_DEBUG', '0') == '1'

def debug_print(*args, **kwargs):
    """Prints messages only if the GENRL_DEBUG environment variable is set to '1'."""
    if _DEBUG_ENABLED:
        print("[DEBUG]", *args, **kwargs)

def masked_mean(x, mask):
    return (x * mask).sum() / max(mask.sum(), 1)

def masked_sum(x, mask):
    return (x * mask).sum()

def masked_var(x, mask):
    N = max(mask.sum(), 1)
    mean = masked_mean(x, mask)
    return ((x - mean) ** 2 * mask).sum() / N

def masked_std(x, mask):
    return torch.sqrt(masked_var(x, mask) + 1e-8)

def normalize_advantage(advantages, valid_mask):
    assert advantages.shape[0] > 1
    mean = masked_mean(advantages, valid_mask)
    std = masked_std(advantages, valid_mask)
    return (advantages - mean) * valid_mask / (std + 1e-8)

# input: (batch_size, seq_len)
@functools.lru_cache(maxsize=8)
def mask_right_shift(mask):
    return torch.cat([torch.zeros_like(mask[:, 0:1]), mask[:, :-1]], dim=1)


def downsample_list_image_to_video_array(images:List[np.array], factor:int):
    # output: (T, C, H, W)
    assert len(images) > 0 and isinstance(images[0], np.ndarray)
    assert len(images[0].shape) == 3

    images = np.stack(images[::factor], axis=0)
    images = np.transpose(images, (0, 3, 1, 2))
    return images

def save_tune_session(study, project_name, study_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        pickle.dump(study, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.flush()
        os.fsync(tmp.fileno())
        
        # Verify the file was written correctly
        tmp.seek(0)
        try:
            pickle.load(tmp)
        except Exception as e:
            print(f"Error verifying saved study: {e}")
            return
        
        run = wandb.init(project=project_name, job_type="job-type")
        artifact = wandb.Artifact(f"{study_name}_session_info", type="optuna_session")
        artifact.add_file(tmp.name, name="session_info.pkl")
        run.log_artifact(artifact)
        run.finish()

def load_tune_session(project_name, study_name, version="latest"):
    api = wandb.Api()

    try:
        artifact = api.artifact(f"{project_name}/{study_name}_session_info:{version}")
    except wandb.errors.CommError as e:
        if re.search(r'project.*not found', str(e).lower())\
            or re.search(r'artifact.*not found', str(e).lower()):
            print("Project not found")
            return None
        raise e
    except Exception as e:
        print(f"Failed to load artifact: {e}")
        raise e
    
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact.download(tmpdir)
        file_path = os.path.join(tmpdir, "session_info.pkl")
        
        if not os.path.exists(file_path):
            print(f"session_info.pkl not found in the artifact")
            return None
        
        with open(file_path, 'rb') as f:
            try:
                session_info = pickle.load(f)
            except EOFError as e:
                print("Corrupted session file. Unable to load session info.")
                raise e
            except Exception as e:
                print(f"Error loading session info: {e}")
                raise e
    
    return session_info

def wandb_load_study(project_name,
                     study_name,
                     resume_from_version=None,
                     **kwargs):

    session_info = load_tune_session(project_name, study_name, version=resume_from_version)
    total_trials_completed = 0
    if session_info is not None:
        print("Resuming existing study session")
        study = session_info["study"]
        total_trials_completed = session_info["total_trials_completed"]
    else:
        # If the study doesn't exist, create a new one
        study = optuna.create_study(**kwargs)
        print("Created new study")
    
    return study, total_trials_completed

def wandb_save_study(study, total_trials_completed, project_name, study_name):
    session_info = {
        "study" : study,
        "total_trials_completed" : total_trials_completed,
    }
    save_tune_session(session_info, project_name, study_name)

def is_cuda_available():
    if torch.cuda.is_available():
        return True
    else:
        return False

def auto_pytorch_device(gs_backend=None):
    """
    Automatically selects the device based on availability.
    
    Returns:
        str: 'cuda' if a GPU is available, otherwise 'cpu'
    """
    
    if gs_backend == gs.cpu:
        return 'cpu'
    
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    
    assert gs_backend is not None, "No Specified device and no GPU available"

def to_device(device):
    """
    Decorator that moves the return value of a function to a specific device.
    
    Args:
        device: The target device (e.g., 'cuda', 'cuda:0', 'cpu')
    
    Returns:
        Decorator function that wraps the original function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return move_to_device(result, device)
        return wrapper
    return decorator

def move_to_device(obj, device):
    """
    Recursively moves an object and all its tensor contents to the specified device.
    
    Args:
        obj: The object to move (tensor, module, list, tuple, dict, etc.)
        device: The target device
        
    Returns:
        The same object structure with all tensors moved to the target device
    """
    if hasattr(obj, 'to') and callable(obj.to):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj
