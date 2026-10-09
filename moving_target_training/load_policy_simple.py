import pathlib

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import numpy as np
import glob
from pathlib import Path
from ray.rllib.utils.torch_utils import FLOAT_MIN
from typing import Callable, Union
import json

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def _checkpoint_number(checkpoint_path: Path) -> int:
    return int(checkpoint_path.name.split("_")[-1])


def _rl_module_path(checkpoint_path: Path) -> Path:
    return (
        checkpoint_path
        / "learner_group"
        / "learner"
        / "rl_module"
        / DEFAULT_MODULE_ID
    )


def _rl_module_metadata_path(checkpoint_path: Path) -> Path:
    return _rl_module_path(checkpoint_path) / RLModule.CLASS_AND_CTOR_ARGS_FILE_NAME


def find_checkpoints_newest_first(checkpoint_path_dir: Path) -> list[Path]:
    checkpoints = [
        Path(checkpoint)
        for checkpoint in glob.glob(
            str(checkpoint_path_dir) + "/**/checkpoint_*", recursive=True
        )
        if Path(checkpoint).is_dir()
    ]
    if len(checkpoints) == 0:
        raise ValueError("No model to re-load and continue training")

    return sorted(checkpoints, key=_checkpoint_number, reverse=True)


def find_latest_checkpoint(checkpoint_path_dir: Path) -> Path:
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_path_dir_path: The path to the directory containing checkpoints.

    Returns:
        The path to the latest checkpoint directory.

    Raises:
        ValueError: If no checkpoints are found in the directory.
    """

    latest_checkpoint = find_checkpoints_newest_first(checkpoint_path_dir)[0]
    print(f"Found checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def load_policy(policy_path_general: Path, return_logits=False) -> Callable:
    """Load a PyTorch policy from a saved model.

    Args:
        policy_path_general: The path to the saved model.
        return_logits: Whether to return logits instead of probabilities.
    Returns:
        A function that takes observations and returns actions.
    """

    try:
        checkpoints = find_checkpoints_newest_first(policy_path_general)
    except ValueError:
        loading_torch = True
        print(
            f"No RLlib checkpoint found in {policy_path_general}. Attempting to load PyTorch model."
        )
    else:
        loading_torch = False
    if loading_torch:
        raise ValueError(f"No RLlib checkpoint found in {policy_path_general}")

    rl_module = None
    checkpoint_errors = []
    for path_checkpoint in checkpoints:
        metadata_path = _rl_module_metadata_path(path_checkpoint)
        if not metadata_path.is_file():
            print(
                f"Skipping checkpoint without learner model metadata: "
                f"{path_checkpoint}"
            )
            checkpoint_errors.append(f"{path_checkpoint}: missing {metadata_path.name}")
            continue

        try:
            print(f"Attempting to load policy from checkpoint: {path_checkpoint}")
            rl_module = RLModule.from_checkpoint(_rl_module_path(path_checkpoint))
            print(f"Loaded policy from checkpoint: {path_checkpoint}")
            break
        except FileNotFoundError as exc:
            print(f"Skipping incomplete checkpoint {path_checkpoint}: {exc}")
            checkpoint_errors.append(f"{path_checkpoint}: {exc}")

    if rl_module is None:
        attempted = "\n".join(checkpoint_errors)
        raise FileNotFoundError(
            f"No checkpoint with a loadable learner model was found in "
            f"{policy_path_general}. Attempted checkpoints:\n{attempted}"
        )

    rl_module.eval()

    def policy(
        obs: list[float],
        deterministic: bool = True,
    ) -> Union[int, tuple[int, np.ndarray]]:
        """Policy function that takes observations and returns actions.

        Args:
            obs: A list of observations.
            deterministic: If True, use argmax for action selection; otherwise, sample from the action distribution.
        Returns:
            An integer representing the selected action.
        """
        if isinstance(obs, dict):
            obs_vec = obs["observations"]
        else:
            obs_vec = obs
        obs_vec = np.array(obs_vec, dtype=np.float32)

        input_dict = {Columns.OBS: torch.from_numpy(obs_vec).unsqueeze(0)}

        with torch.no_grad():
            rl_module_out = rl_module.forward_inference(input_dict)
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])

        if deterministic:
            action = np.argmax(logits[0])  # Use argmax for deterministic action
        else:
            action = np.random.choice(len(logits[0]), p=softmax(logits[0]))

        # A few of the old models with the new architecture did not have the padding for the action charge. If that is the case, uncomment here
        # if len(logits.shape) > 1 and logits.shape[1] < 33:
        #     action = action + 1  # Account for missing "no-op" action in some models
        #     # print(
        #     #     "Warning: Model was trained without a no-op action. Action indices have been shifted by +1 to account for this."
        #     # )

        if return_logits:
            return action, logits
        else:
            return action

    return policy
