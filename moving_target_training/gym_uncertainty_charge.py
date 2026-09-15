import logging
import numpy as np
from typing import Any
from bsk_rl.gym import SatelliteTasking
from bsk_rl.utils.rllib.callbacks import EpisodeDataWrapper, EpisodeDataLogger
from bsk_rl.utils import vizard
from gymnasium import Wrapper

logger = logging.getLogger(__name__)


class SatelliteUncertaintyTargets(SatelliteTasking):
    @staticmethod
    def _target_key(target):
        if target is None:
            return None
        target_id = getattr(target, "id", None)
        if target_id is not None:
            return str(target_id)
        target_name = getattr(target, "name", None)
        if target_name is not None:
            return str(target_name)
        return str(id(target))

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        sat = self.satellites[0]
        sat._pending_prev_success_target_key = None
        sat._repeat_after_prev_success_count = 0
        sat._repeat_after_prev_success_zero_reward_count = 0
        sat._imaged_targets_all = []
        sat._episode_reward_sum = 0.0
        sat._episode_step_count = 0
        sat._battery_min = 1.0
        sat._battery_sum = 0.0
        sat._filtered_this_step = 0
        sat._filtered_total = 0
        return obs, info

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Take a single-satellite action, run parent step, and update target uncertainties."""
        sat = self.satellites[0]
        sat._filtered_this_step = 0
        target_selected = None
        selected_target_key = None
        if action > 0:
            target_selected = sat.parse_target_selection(action-1)
            selected_target_key = self._target_key(target_selected)
        pending_prev_success_target_key = getattr(
            sat, "_pending_prev_success_target_key", None
        )
        repeat_after_prev_success = (
            pending_prev_success_target_key is not None
            and selected_target_key == pending_prev_success_target_key
        )
        sat._pending_prev_success_target_key = None
        old_stored_data = np.array(
            sat.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData,
            dtype=float,
        )
        obs, reward, terminated, truncated, info = super().step(action)
        d_ts = info["d_ts"]
        if isinstance(reward, dict):
            step_reward = float(sum(reward.values()))
        else:
            step_reward = float(np.sum(np.asarray(reward)))
        sat._episode_reward_sum = float(getattr(sat, "_episode_reward_sum", 0.0)) + step_reward
        sat._episode_step_count = int(getattr(sat, "_episode_step_count", 0)) + 1
        new_stored_data = np.array(
            sat.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData,
            dtype=float,
        )
        batt = sat.dynamics.battery_charge_fraction
        sat._battery_min = min(getattr(sat, "_battery_min", 1.0), batt)
        sat._battery_sum = getattr(sat, "_battery_sum", 0.0) + batt
        imaged_this_step = np.any(new_stored_data - old_stored_data > 0.0)
        reward_is_positive = float(np.sum(np.asarray(reward))) > 0.0
        sat._repeat_after_prev_success_count = int(
            getattr(sat, "_repeat_after_prev_success_count", 0)
            + int(repeat_after_prev_success)
        )
        sat._repeat_after_prev_success_zero_reward_count = int(
            getattr(sat, "_repeat_after_prev_success_zero_reward_count", 0)
            + int(repeat_after_prev_success and (not reward_is_positive))
        )

        targets = sat.data_store.data.known
        area_factor = 20e3

        # Compute all new values in vectorized form
        search_areas = np.array([t.search_area for t in targets])
        acquiring_speeds = np.array([t.acquiring_speed for t in targets])
        speeds = np.array([t.speed for t in targets])

        new_search_areas = search_areas + speeds * d_ts
        new_image_durations = new_search_areas * np.ceil(new_search_areas / area_factor) / acquiring_speeds

        means = new_image_durations / 2.0
        stds = new_image_durations * 0.3
        new_image_times = np.maximum(0.0, np.random.normal(loc=means, scale=stds))

        # Bulk update the objects (loop still needed for assignment)
        targets_to_update = []
        for t, old_area, area, dur, time in zip(
            targets, search_areas, new_search_areas, new_image_durations, new_image_times
        ):
            t.search_area = area
            t.image_duration = dur
            t.image_time = time

            last_area = getattr(t, "_last_area_updated", float(old_area))
            if area > (1.0 + 0.0) * last_area: # Change threshold
                targets_to_update.append(t)

        # Reset uncertainty whenever an image was actually taken, even if reward was 0.
        if imaged_this_step:
            if target_selected is not None:
                sat._imaged_targets_all.append(target_selected)
                target_selected.search_area = 0.0
                target_selected.image_duration = 0.0
                target_selected.image_time = 0.0
                targets_to_update.append(target_selected)
                sat._pending_prev_success_target_key = selected_target_key

        # Only update and store last_area for targets that exceeded threshold
        if targets_to_update:
            self.update_target_sizes(targets_to_update)
            print("Number of targets updated:", len(targets_to_update))
            for t in targets_to_update:
                t._last_area_updated = t.search_area  # update baseline only when updated

        threshold = float(getattr(sat, "_search_area_threshold", 0.0))
        filtered_now = sum(1 for t in sat.data_store.data.known if t.search_area <= threshold)
        sat._filtered_this_step = filtered_now
        sat._filtered_total = int(getattr(sat, "_filtered_total", 0)) + filtered_now

        # The parent step already computed/cached observation at this sim_time.
        # We changed target states after that, so invalidate cache and rebuild obs now.
        sat.observation_builder.obs_dict_cache = None
        obs = sat.get_obs()

        return obs, reward, terminated, truncated, info
    
    @vizard.visualize
    def update_target_sizes(self, targets, vizInstance=None, vizSupport=None):
        """Update target colors in Vizard."""
        def compute_area_vertices(r_LP_P, area_size):
            r = np.array(r_LP_P).flatten()

            # Half size
            d = area_size / 2.0

            # Normalize center vector
            r_hat = r / np.linalg.norm(r)

            # Earth rotation axis
            k_hat = np.array([0.0, 0.0, 1.0])

            # East unit vector
            e_hat = np.cross(k_hat, r_hat)
            e_hat /= np.linalg.norm(e_hat)

            # North unit vector
            n_hat = np.cross(r_hat, e_hat)

            # Corner positions
            r_top_left     = r + d * n_hat - d * e_hat
            r_bottom_left  = r - d * n_hat - d * e_hat
            r_top_right    = r + d * n_hat + d * e_hat
            r_bottom_right = r - d * n_hat + d * e_hat

            vertices = list(r_top_left) + list(r_top_right) + list(r_bottom_right) + list(r_bottom_left)
            return vertices
        
        for target in targets:
            vizSupport.addQuadMap(vizInstance,
                                ID=int(target.name.split("-")[-1]),
                                parentBodyName="earth", 
                                vertices=compute_area_vertices(target.r_LP_P, target.search_area),
                                color=[255, 255, 255, int(target.priority * 100)],
                                )

def _satellite_tasking_env_creator(env_config):
    if "episode_data_callback" in env_config:
        episode_data_callback = env_config.pop("episode_data_callback")
    else:
        episode_data_callback = None
    if "satellite_data_callback" in env_config:
        satellite_data_callback = env_config.pop("satellite_data_callback")
    else:
        satellite_data_callback = None

    return EpisodeDataWrapper(
        SatelliteUncertaintyTargets(**env_config),
        episode_data_callback=episode_data_callback,
        satellite_data_callback=satellite_data_callback,
    )

def env_creation(**env_config):
        env = SatelliteUncertaintyTargets(**env_config)
        return env

class Env_wrapped(EpisodeDataLogger, Wrapper):
    def __init__(self, env_config):
        episode_data_callback = env_config.pop("episode_data_callback", None)
        satellite_data_callback = env_config.pop("satellite_data_callback", None)
        env = env_creation(**env_config)
        EpisodeDataLogger.__init__(self, episode_data_callback, satellite_data_callback)
        Wrapper.__init__(self, env)
