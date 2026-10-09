import glob
import os
import shutil
from pathlib import Path

import torch
# torch.set_num_threads(11)
# os.environ["MKL_NUM_THREADS"] = "11"
import ray
from avs_rl_tools.utils import get_available_cores
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from bsk_rl.utils.rllib.discounting import \
    TimeDiscountedGAEPPOTorchLearner  # EpisodeDataCallbacks,
# from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from configurations import T_ORBIT, U_NORM
from gym_uncertainty_charge import Env_wrapped
from RL_module_complete import GATModule
from wandb_config import WandbLogger

# os.environ["RAY_DEDUP_LOGS"] = "0"

def train_model(
    model_name,
    output_directory,
    env_args={},
    n_envs=1,
    checkpoint_frequency=1,
    checkpoints_to_keep=2,
    reload_frequency=1_000_000,
    total_timesteps=1_000_000,
    training_args={},
    rl_module_args={},
    temp_dir="/tmp",
    n_steps_switch=None,
    continue_previous=False,
    wandb_logger=None,
):
    os.environ["RAY_TMPDIR"] = os.environ["TMPDIR"] = temp_dir
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    # register_env("SatelliteUncertaintyTargets-RLlib", _satellite_tasking_env_creator)

    ray.init(
        ignore_reinit_error=True,
        num_cpus=get_available_cores(),
        object_store_memory=2_000_000_000,  # 2 GB
        _temp_dir=temp_dir,
    )
    config = (
        PPOConfig()
        .training(**training_args)
        .env_runners(
            num_env_runners=n_envs + 1,
            sample_timeout_s=1000.0,
            # module_to_env_connector=lambda env: (ContinuePreviousAction(),),
        )
        .environment(
            env=Env_wrapped,
            env_config=env_args,
        )
        .callbacks(WrappedEpisodeDataCallbacks)
        .reporting(
            metrics_num_episodes_for_smoothing=30,
            metrics_episode_collection_timeout_s=180,
        )
        .checkpointing(export_native_model_files=True)
        .framework(framework="torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .resources(num_gpus=0)
    )
    config = config.training(
        **training_args,
        # learner_connector=lambda obs_space, act_space: (
        #     MakeAddedStepActionValid(expected_train_batch_size=config.train_batch_size),
        #     CondenseMultiStepActions(),
        # ),
        learner_class=TimeDiscountedGAEPPOTorchLearner,
    )
    config.rl_module(**rl_module_args)
    config.logger_config = dict(
        type=UnifiedLogger, logdir=output_directory / model_name
    )

    start_from_zero = False
    if continue_previous:
        checkpoint_path_dir = (
            output_directory / model_name
        )
        checkpoints = glob.glob(str(checkpoint_path_dir) + "/checkpoint_*")
        if len(checkpoints) == 0:
            print("No model to re-load and continue training")
            start_from_zero = True
        
        if start_from_zero is False:
            checkpoint_number_str = ""
            checkpoint_number_int = 0
            #Figure out the most recent checkpoint
            for checkpoint_number in checkpoints:
                checkpoint_folder_i = checkpoint_number.split("/")[-1]
                checkpoint_number_i = int(checkpoint_folder_i.split("_")[-1])
                if checkpoint_number_i > checkpoint_number_int:
                    checkpoint_number_int = checkpoint_number_i
                    checkpoint_number_str = checkpoint_folder_i.split("_")[-1]

            latest_checkpoint_dir = checkpoint_path_dir / f"checkpoint_{checkpoint_number_str}"
            if not os.path.exists(latest_checkpoint_dir):
                print(f"Selected path does not exist: {latest_checkpoint_dir}")
                exit()
            
            ppo = PPO(config)
            print(str(latest_checkpoint_dir))
            ppo.restore(str(latest_checkpoint_dir))
            iter = checkpoint_number_int + 1
            results = ppo.train()
            step = results["num_env_steps_trained_lifetime"]
            if n_steps_switch is None:
                cond_switch = True
                checkpoint_saved = True
            elif step >= n_steps_switch:
                cond_switch = True
                checkpoint_saved = True
            else:
                cond_switch = False
                checkpoint_saved = False
    
    if not continue_previous or start_from_zero:
        ppo = PPO(config)
        iter = 0
        step = 0
        if n_steps_switch is not None:
            cond_switch = False
            checkpoint_saved = False
        else:
            cond_switch = True
            checkpoint_saved = True

    while True:
        prev_step = step
        
        results = ppo.train()
        step = results["num_env_steps_trained_lifetime"]

        if wandb_logger is not None:
            wandb_logger.log(results)

        checkpoint_path = (
            output_directory / model_name / f"checkpoint_{str(iter).zfill(6)}"
        )
        if iter % checkpoint_frequency == 0:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(checkpoint_path)

        if step > total_timesteps:
            break

        if cond_switch is False and (step + reload_frequency) > n_steps_switch:
            cond_switch = True

        if step % reload_frequency < prev_step % reload_frequency:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(checkpoint_path)
            if cond_switch and checkpoint_saved is False:
                checkpoint_saved = True
                shutil.copytree(output_directory / model_name, output_directory / f"{model_name}_switch")
            ray.shutdown()
            # register_env("SatelliteUncertaintyTargets-RLlib", _satellite_tasking_env_creator)
            ray.init(
                ignore_reinit_error=True,
                num_cpus=get_available_cores(),
                object_store_memory=2_000_000_000,  # 2 GB
                _temp_dir=temp_dir,
            )
            ppo = PPO.from_checkpoint(checkpoint_path)

        if iter > checkpoints_to_keep * checkpoint_frequency - 1:
            for i in range(checkpoint_frequency):
                remove_dir = (
                    output_directory
                    / model_name
                    / f"checkpoint_{str(iter - checkpoints_to_keep * checkpoint_frequency - i).zfill(6)}"
                )
                try:
                    shutil.rmtree(remove_dir)
                except FileNotFoundError:
                    pass

        iter += 1

    if wandb_logger is not None:
        wandb_logger.finish()

def episode_data_callback(env):
    reward = env.rewarder.cum_reward
    reward = sum(reward.values()) / len(reward)
    orbits = env.simulator.sim_time / (T_ORBIT)

    targets = getattr(env.satellite, "_imaged_targets_all", None)
    if targets is None or len(targets) == 0:
        targets = env.rewarder.data.imaged or []
    if targets is None:
        targets = []
    elif isinstance(targets, (set, tuple)):
        targets = list(targets)
    elif not isinstance(targets, list):
        targets = [targets]
    max_area = max((t.search_area for t in targets), default=0.0)
    max_uncertainty = max(((t.search_area ** 2) / (U_NORM ** 2) for t in targets), default=0.0)
    mean_priority = sum((t.priority for t in targets), 0.0) / len(targets) if targets else 0.0
    mean_uncertainty = sum(((t.search_area ** 2) / (U_NORM ** 2) for t in targets), 0.0) / len(targets) if targets else 0.0
    steps = int(getattr(env.satellite, "_episode_step_count", 0))
    battery_sum = float(getattr(env.satellite, "_battery_sum", 0.0))
    battery_mean = battery_sum / steps if steps > 0 else -1.0

    data = dict(
        reward=reward,
        alive=float(env.satellite.is_alive()),
        rw_status_valid=float(env.satellite.dynamics.rw_speeds_valid()),
        battery_status_valid=float(env.satellite.dynamics.battery_valid()),
        battery_min=float(getattr(env.satellite, "_battery_min", -1.0)),
        battery_mean=battery_mean,
        orbits_complete=orbits,
        imaged = env.satellite.imaged,
        missed = env.satellite.missed,
        duplicates = env.rewarder.data.duplicates,
        image_action = env.satellite.image_count,
        charge_action = env.satellite.charge_count,
        repeat_after_prev_success = int(
            getattr(env.satellite, "_repeat_after_prev_success_count", -1)
        ),
        repeat_after_prev_success_zero_reward = int(
            getattr(env.satellite, "_repeat_after_prev_success_zero_reward_count", -1)
        ),
        filterd_targets = int(
            getattr(env.satellite, "_filtered_total", -1)
        ),
        max_area = max_area / 1e3,  # to km
        mean_priority = mean_priority,
        mean_uncertainty = mean_uncertainty,
        max_uncertainty = max_uncertainty,
    )
    if orbits > 0:
        data["reward_per_orbit"] = reward / orbits
    if not env.satellite.is_alive():
        data["orbits_complete_partial_only"] = orbits
        
    return data

if __name__ == "__main__":
    import argparse
    import importlib
    import importlib.util
    import yaml
    from avs_rl_tools.utils import build_job_array, sanitize_np
    import flexible_spec_options as spec_options

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cases", type=int, default=1)
    parser.add_argument(
        "--setup-env-module",
        type=str,
        default="dynamic_imaging_rewardFlexible_flexible_specs_charge",
        help="Module name or .py file path that defines setup_env",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=30,
        help="Number of environment runners to use for training",
    )
    parser.add_argument(
        "--reward-function",
        type=str,
        default="multiply",
        help=(
            "Reward formula registered in envs_common.REWARD_FUNCTIONS "
            "(for example: multiply, weighted_multiply, linear, square, cube)"
        ),
    )
    parser.add_argument(
        "--reward-weight",
        type=float,
        default=1.0,
        help="Weight passed to the selected flexible reward function",
    )
    parser.add_argument(
        "--filter-value",
        type=float,
        default=10_000.0,
        help="Search-area threshold passed to setup_env as threshold",
    )
    parser.add_argument(
        "--observation-spec",
        type=str,
        default="area2_prob_32",
        choices=spec_options.observation_spec_names(),
        help="Named observation_spec from flexible_spec_options.py",
    )
    parser.add_argument(
        "--action-spec",
        type=str,
        default="image_32",
        choices=spec_options.action_spec_names(),
        help="Named action_spec from flexible_spec_options.py",
    )
    args = parser.parse_args()

    def load_setup_env(import_target):
        if import_target.endswith(".py") or os.path.sep in import_target:
            module_path = Path(import_target).expanduser().resolve()
            spec = importlib.util.spec_from_file_location(
                module_path.stem, str(module_path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from file: {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(import_target)
        if not hasattr(module, "setup_env"):
            raise AttributeError(
                f"{import_target} does not define a setup_env function"
            )
        return module.setup_env

    setup_env = load_setup_env(args.setup_env_module)

    reward_function_label = args.reward_function.replace(os.path.sep, "_")
    base_model_name = (
        f"simple_model_charge_reward_"
        f"{reward_function_label}_w{args.reward_weight}_"
        f"filter{args.filter_value}_"
        f"obsSpec_{args.observation_spec}_actSpec_{args.action_spec}"
    )
    n_envs = max(1, args.n_envs)
    output_dir = Path(
        "/scratch/alpine/chbe6195/simple_safety_models_desat_test1"   #CHANGE PATH HERE
        # "./simple_models"
        # Should be in /scratch/alpine/[user] when running on the cluster
    )

    os.makedirs(output_dir, exist_ok=True)

    observation_spec = spec_options.get_observation_spec(
        args.observation_spec,
        reward_function=args.reward_function,
        reward_weight=args.reward_weight,
    )
    action_spec = spec_options.get_action_spec(args.action_spec)
    model_config_overrides = spec_options.get_model_config_overrides(
        args.observation_spec,
        args.action_spec,
    )

    env, satellites, env_args = setup_env(
        test=False,
        reward_function=args.reward_function,
        weight=args.reward_weight,
        threshold=args.filter_value,
        observation_spec=observation_spec,
        action_spec=action_spec,
    )

    env_args["episode_data_callback"] = episode_data_callback

    #We need to check which parameters we are going to use for training (gamma etc)
    jobs = build_job_array(
        training_args = dict(
            lr=[[
                [0, 0.00033003435881682255],
                [40000, 0.00033003435881682255 / 16.749479444886223],
            ],],
            gamma=[0.9997],  # - CHECKED
            train_batch_size=[int(300 * 10 * 3)],  # - CHECKED
            num_sgd_iter=[20],  # - CHECKED
            lambda_=[0.8713548569911232],  # - CHECKED
            use_kl_loss=[False],
            clip_param=[0.14701727973480344],  # - CHECKED
            grad_clip=[0.3104924935285628],  # - CHECKED
            entropy_coeff=[[
                [0, 0.023694512589767867],
                [3000000, 0],
            ],], # should be 0 at the end of training
        ),
        env_args={k:[v] for k,v in env_args.items()},
    )

    n_cases = max(1, args.n_cases)
    n_to_run = min(n_cases, len(jobs))

    for N in range(n_to_run):
        print(f"Running job {N}: {N+1} of {len(jobs)}")
        job_args = jobs[N]
        model_name = f"{base_model_name}_case_{N}"

        with open(output_dir / f"{model_name}_params.txt", "w") as file:
            yaml.dump(sanitize_np(job_args), file)

        wandb_logger = WandbLogger(
            project_name="desat_test1",
            run_name=f"{model_name}_RND",
            key_path=Path(
                "/projects/chbe6195/DLA-Project-AVS/moving_target_training/wandb_key.txt"
            ),
            config={},
        )

        train_model(
            model_name=model_name,
            output_directory=output_dir,
            checkpoint_frequency=2,
            checkpoints_to_keep=2,
            total_timesteps=20_000_000,
            reload_frequency=1_000_000,
            n_envs=n_envs,
            rl_module_args = dict(
                model_config_dict={
                    "n_targets": 32,
                    "obs_sat": 41,
                    "non_imaging_actions": 1,
                    "width_f": 256,
                    "depth_f": 2,
                    "width_g": 128,
                    "depth_g": 4,
                    "tgt_encoded_dim": 128,
                    "attention_depth": 1,
                    "num_heads": 2,
                    "attention_dim": 128,
                    "width_f_sat": 256,
                    "depth_f_sat": 2,
                    "width_g_sat": 128,
                    "depth_g_sat": 4,
                    "sat_attention_dim": 128,
                    "sat_attention_heads": 2,
                    "sat_encoded_dim": 128,
                    "act_function": "ReLU",
                    "critic_tgt_encoded_dim": 128,
                    "critic_width_f": 256,
                    "critic_depth_f": 2,
                    "critic_width_g": 64,
                    "critic_depth_g": 3,
                    "dropout": 0.1,
                    "post_self_attention": True,
                },
                rl_module_spec=RLModuleSpec(module_class=GATModule),
            ),
            temp_dir="/scratch/alpine/chbe6195/temp_dir",
            continue_previous=True,
            wandb_logger=wandb_logger,
            **job_args,
        )
