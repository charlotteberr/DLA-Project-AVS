from importlib.metadata import version
from typing import ClassVar

version("ray")  # Parent package of RLlib

import numpy as np

from bsk_rl import act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw


class ScanningDownlinkDynModel(dyn.ContinuousImagingDynModel, dyn.GroundStationDynModel):
    # Define some custom properties to be accessed in the state
    @property
    def instrument_pointing_error(self) -> float:
        r_BN_P_unit = self.r_BN_P/np.linalg.norm(self.r_BN_P) 
        c_hat_P = self.satellite.fsw.c_hat_P
        return np.arccos(np.dot(-r_BN_P_unit, c_hat_P))
    
    @property
    def solar_pointing_error(self) -> float:
        a = self.world.gravFactory.spiceObject.planetStateOutMsgs[
            self.world.sun_index
        ].read().PositionVector
        a_hat_N = a / np.linalg.norm(a)
        nHat_B = self.satellite.sat_args["nHat_B"]
        NB = np.transpose(self.BN)
        nHat_N = NB @ nHat_B
        return np.arccos(np.dot(nHat_N, a_hat_N))

class ScanningSatellite(sats.AccessSatellite):
    observation_spec: ClassVar[list[obs.Observation]] = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction"),
            dict(prop="wheel_speeds_fraction"),
            dict(prop="instrument_pointing_error", norm=np.pi),
            dict(prop="solar_pointing_error", norm=np.pi)
        ),
        obs.OpportunityProperties(
            dict(prop="opportunity_open", norm=5700),
            dict(prop="opportunity_close", norm=5700),
            type="ground_station",
            n_ahead_observe=1,
        ),
        obs.Eclipse(norm=5700),
        obs.Time(),
    ]
    action_spec: ClassVar[list[act.Action]] = [
        act.Scan(duration=180.0),
        act.Charge(duration=120.0),
        act.Downlink(duration=60.0),
        act.Desat(duration=60.0),
    ]
    dyn_type = ScanningDownlinkDynModel
    fsw_type = fsw.ContinuousImagingFSWModel

sat = ScanningSatellite(
    "Scanner-1",
    sat_args=dict(
        # Data
        dataStorageCapacity=5000 * 8e6,  # bits
        storageInit=lambda: np.random.uniform(0.0, 0.8) * 5000 * 8e6,
        instrumentBaudRate=0.5 * 8e6,
        transmitterBaudRate=-50 * 8e6,
        # Power
        batteryStorageCapacity=200 * 3600,  # W*s
        storedCharge_Init=lambda: np.random.uniform(0.3, 1.0) * 200 * 3600,
        basePowerDraw=-10.0,  # W
        instrumentPowerDraw=-30.0,  # W
        transmitterPowerDraw=-25.0,  # W
        thrusterPowerDraw=-80.0,  # W
        panelArea=0.25,
        # Attitude
        imageAttErrorRequirement=0.1,
        imageRateErrorRequirement=0.1,
        disturbance_vector=lambda: np.random.normal(scale=0.0001, size=3),  # N*m
        maxWheelSpeed=6000.0,  # RPM
        wheelSpeeds=lambda: np.random.uniform(-3000, 3000, 3),
        desatAttitude="nadir",
    )
)

duration = 5 * 5700.0  # About 5 orbits
env_args = dict(
    satellite=sat,
    scenario=scene.UniformNadirScanning(value_per_second=1/duration),
    rewarder=data.ScanningTimeReward(),
    time_limit=duration,
    failure_penalty=-1.0,
    terminate_on_time_limit=True,
)

def episode_data_callback(env):
    reward = env.rewarder.cum_reward
    reward = sum(reward.values()) / len(reward)
    orbits = env.simulator.sim_time / (95 * 60)

    data = dict(
        reward=reward,
        # Are satellites dying, and how and when?
        alive=float(env.satellite.is_alive()),
        rw_status_valid=float(env.satellite.dynamics.rw_speeds_valid()),
        battery_status_valid=float(env.satellite.dynamics.battery_valid()),
        orbits_complete=orbits,
    )
    if orbits > 0:
        data["reward_per_orbit"] = reward / orbits
    if not env.satellite.is_alive():
        data["orbits_complete_partial_only"] = orbits
        
    return data

import bsk_rl.utils.rllib  # noqa To access "SatelliteTasking-RLlib"
from ray.rllib.algorithms.ppo import PPOConfig
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks

N_CPUS = 3

training_args = dict(
    lr=0.00003,
    gamma=0.999,
    train_batch_size=250,  # usually a larger number, like 2500
    num_sgd_iter=10,
    model=dict(fcnet_hiddens=[512, 512], vf_share_layers=False),
    lambda_=0.95,
    use_kl_loss=False,
    clip_param=0.1,
    grad_clip=0.5,
)

config = (
    PPOConfig()
    .training(**training_args)
    .env_runners(num_env_runners=N_CPUS-1, sample_timeout_s=1000.0)
    .environment(
        env="SatelliteTasking-RLlib",
        env_config=dict(**env_args, episode_data_callback=episode_data_callback),
    )
    .reporting(
        metrics_num_episodes_for_smoothing=1,
        metrics_episode_collection_timeout_s=180,
    )
    .checkpointing(export_native_model_files=True)
    .framework(framework="torch")
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .callbacks(WrappedEpisodeDataCallbacks)
)

import math
from pathlib import Path

import ray
from ray import train, tune

STORAGE_PATH = "/home/charlotte/ray_results/bsk_rl"
EXPERIMENT_NAME = "scanning_satellite_100iter_1000bs"
EXPERIMENT_DIR = Path(STORAGE_PATH) / EXPERIMENT_NAME
TRAINING_ITERATIONS = 1000  # Adjust the number of iterations as needed
METRIC = "env_runners/episode_return_mean"
MODE = "max"

ray.init(
    ignore_reinit_error=True,
    num_cpus=N_CPUS,
    object_store_memory=2_000_000_000,  # 2 GB
)

if tune.Tuner.can_restore(EXPERIMENT_DIR):
    # Subsequent runs: continue the trial from its latest saved checkpoint.
    print(f"Restoring experiment from {EXPERIMENT_DIR}")
    tuner = tune.Tuner.restore(
        str(EXPERIMENT_DIR),
        trainable="PPO",
        param_space=config.to_dict(),
        resume_unfinished=True,
        resume_errored=True,  # errored trials pick up from their last checkpoint
    )
else:
    # First run: nothing to restore from.
    print(f"No restorable experiment at {EXPERIMENT_DIR}; starting a new run.")
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(metric=METRIC, mode=MODE),
        run_config=train.RunConfig(
            name=EXPERIMENT_NAME,
            storage_path=STORAGE_PATH,
            stop={"training_iteration": TRAINING_ITERATIONS},
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=None,  # None = never delete a checkpoint; keep them all
                checkpoint_frequency=10,
                checkpoint_at_end=True,
                checkpoint_score_attribute=METRIC,
                checkpoint_score_order=MODE,
            ),
        ),
    )

results = tuner.fit()

# Shutdown Ray
ray.shutdown()

from bsk_rl import SatelliteTasking
from bsk_rl.utils.rllib import load_torch_mlp_policy


def lookup_metric(metrics: dict, key: str):
    """Read a "a/b/c" metric out of RLlib's nested result dict, or None if absent."""
    node = metrics
    for part in key.split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def select_checkpoint(result):
    """Best kept checkpoint by METRIC, falling back to the latest one."""
    best_checkpoint = None
    best_score = None
    for checkpoint, metrics in result.best_checkpoints or []:
        score = lookup_metric(metrics, METRIC)
        if score is None or (isinstance(score, float) and math.isnan(score)):
            continue  # no completed episode by the time this checkpoint was taken
        if best_score is None or score > best_score:
            best_checkpoint, best_score = checkpoint, score
    if best_checkpoint is None:
        # Never scored (e.g. training stopped before the first episode finished).
        best_checkpoint = result.checkpoint
    return best_checkpoint, best_score


best_result = results.get_best_result(metric=METRIC, mode=MODE)
best_checkpoint, best_score = select_checkpoint(best_result)
if best_checkpoint is None:
    raise RuntimeError(
        f"No checkpoints were saved under {EXPERIMENT_DIR}; nothing to evaluate."
    )
print(f"Evaluating checkpoint {best_checkpoint.path} ({METRIC}={best_score})")

policy = load_torch_mlp_policy(best_checkpoint.path, policy_name="default_policy")

# Deterministic check
EVAL_SEEDS = [0]

env = SatelliteTasking(**env_args, log_level="INFO")
for seed in EVAL_SEEDS:
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = policy(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    print(
        f"seed={seed} reward={total_reward:.4f} "
        f"alive={env.satellite.is_alive()} "
        f"sim_time={env.simulator.sim_time:.1f}s"
    )
env.close()