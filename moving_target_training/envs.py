import logging
from typing import Any

import numpy as np

from envs_common import (
    AreaTargets,
    DynamicImage,
    FlexibleUncertainReward,
    REWARD_FUNCTIONS,
    TotalUncertaintyReward,
    UncertainCubeReward,
    UncertainMultiplyReward,
    UncertainSquareReward,
    UncertainWeightReward,
    UniqueUncertainMultiplyReward,
    UniqueUncertainWeightReward,
    r_LP_P_uncertain,
    target_angle_uncertain,
)

logger = logging.getLogger(__name__)


try:
    # Optional: only needed for the Vizard-enabled environments.
    from envs_viz import UniqueImageStoreViz, UncertainWeightRewardViz  # noqa: F401
except Exception:  # pragma: no cover
    pass


############
# SHIELD
# Make sure the observation number is correct!!!

def power_shielding_function(
    obs,
    act,
    min_power=0.25,
    charge_rate=1.0,
    discharge_rate=1.0,
    **kwargs,
):
    """Force charging if not in eclipse and below min_power or the time.

    Returns:
        Return None if shield not activated, else return shielded action.
    """
    current_power = obs[12]
    eclipse_start = obs[19]
    eclipse_end = obs[20]

    in_eclipse = eclipse_end < eclipse_start

    if not in_eclipse and current_power < _power_requirement(
        eclipse_start, eclipse_end, min_power, charge_rate, discharge_rate
    ):
        return 0

    return None


def _power_requirement(
    eclipse_start, eclipse_end, min_power, charge_rate, discharge_rate, time=0.0
):
    time = time % 1
    eclipse_end = (eclipse_end - time) % 1
    eclipse_start = (eclipse_start - time) % 1
    eclipse_duration = (eclipse_end - eclipse_start) % 1
    in_eclipse = eclipse_end < eclipse_start
    if in_eclipse:
        return min_power + eclipse_end * discharge_rate
    eclipse_draw = eclipse_duration * discharge_rate
    charge_time = eclipse_draw / charge_rate
    if charge_time < eclipse_start:
        return min_power
    return min_power + (charge_time - eclipse_start) * charge_rate


class logInfos:
    def __init__(self, sats, obs_to_log=None):
        if obs_to_log is None:
            obs_to_log = {}

        self.actions = {}
        self.interference = {}
        self.actions_time = {}
        self.actions_target = {}
        self.obs_sats = {}
        self.storage_level = {}
        self.rw_speeds = {}
        self.obs_to_log = obs_to_log
        for sat in sats:
            self.actions[sat.id] = []
            self.interference[sat.id] = []
            self.actions_time[sat.id] = []
            self.actions_target[sat.id] = []
            self.obs_sats[sat.id] = {}
            self.storage_level[sat.id] = []
            self.rw_speeds[sat.id] = []
            for obs_key in self.obs_to_log:
                self.obs_sats[sat.id][obs_key] = []

    def log(self, action, interference, obs, sim_time, sat_id, target_name):
        self.actions[sat_id].append(action)
        self.interference[sat_id].append(interference)
        self.actions_time[sat_id].append(sim_time)
        self.actions_target[sat_id].append(target_name)
        for obs_key in self.obs_to_log:
            self.obs_sats[sat_id][obs_key].append(self.obs_to_log[obs_key])

    def get_logs(self) -> dict[str, Any]:
        vec_actions = [0, 0, 0, 0]
        log_obs = {}
        log_obs_vec = {}
        for obs in self.obs_to_log:
            log_obs_vec[obs] = []
            log_obs[obs] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        for sat_id in self.actions:
            for action in self.actions[sat_id]:
                if action == 0:
                    vec_actions[0] += 1
                elif action == 1:
                    vec_actions[1] += 1
                elif action == 2:
                    vec_actions[2] += 1
                else:
                    vec_actions[3] += 1

            for obs in self.obs_to_log:
                log_obs_vec[obs] = log_obs_vec[obs] + self.obs_sats[sat_id][obs]

        for obs in self.obs_to_log:
            obs_vector = np.array(log_obs_vec[obs])
            log_obs[obs]["min"] = float(np.min(obs_vector))
            log_obs[obs]["max"] = float(np.max(obs_vector))
            log_obs[obs]["mean"] = float(np.mean(obs_vector))
            log_obs[obs]["std"] = float(np.std(obs_vector))

        total_actions = sum(vec_actions)

        return {
            "log_actions": vec_actions,
            "log_total_actions": total_actions,
            "log_obs": log_obs,
            "actions_list": self.actions,
            "actions_list_time": self.actions_time,
            "obs_sats": self.obs_sats,
            "storage_level": self.storage_level,
            "interference": self.interference,
        }
