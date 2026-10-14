import os
import re
from copy import copy

import h5py
import numpy as np
from scipy.stats import norm
from avs_rl_tools.utils import build_job_array
from Basilisk.architecture import bskLogging
from envs import logInfos, power_shielding_function
import dynamic_imaging_rewardFlexible_flexible_specs_charge
import flexible_spec_options
from configurations import T_ORBIT, U_NORM

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

class Colors:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        RESET = '\033[0m'

class WastedTime:
    def __init__(self, satellites, max_wasted_time=10000, verbose=False):
        self.sat_order = [satellite.name for satellite in satellites]
        self.begin_current_action = {satellite.name: 0 for satellite in satellites}
        self.previous_action = {satellite.name: None for satellite in satellites}
        self.previous_target = {satellite.name: None for satellite in satellites}
        self.verbose = verbose
        self.max_wasted_time = max_wasted_time

    def calc_wasted_time(self, actions, current_time, info) -> dict:
        """Calculate the wasted time between actions

        Args:
            actions: List of actions
            current_time: Current simulation time
            info: Dictionary of info from the environment

        Returns:
            wasted_time: Time wasted between actions
        """
        wasted_time = {}
        target_info = self.get_target_info(info)
        for satellite in self.previous_action.keys():
            temp_wasted_time = 0
            action = actions[satellite]

            if self.previous_target[satellite] is not None:
                if action is None:
                    if self.previous_target[satellite] in target_info[satellite].keys():
                        if (
                            target_info[satellite][self.previous_target[satellite]]
                            == "Imaged"
                            or target_info[satellite][self.previous_target[satellite]]
                            == "Timed"
                        ):
                            self.begin_current_action[satellite] = current_time
                            action = self.previous_action[satellite]
                            self.previous_target[satellite] = None
                    else:
                        action = self.previous_action[satellite]
                else:
                    if target_info[satellite].keys() == 0:
                        pass

                    elif (
                        self.previous_target[satellite]
                        not in target_info[satellite].keys()
                    ):
                        temp_wasted_time += (
                            current_time - self.begin_current_action[satellite]
                        )
                        self.begin_current_action[satellite] = current_time
                        for target in target_info[satellite].keys():
                            if target_info[satellite][target] == "Tasked":
                                self.previous_target[satellite] = target
                                self.begin_current_action[satellite] = current_time
                    else:
                        if len(target_info[satellite].keys()) == 1:
                            if (
                                target_info[satellite][self.previous_target[satellite]]
                                == "None"
                            ):
                                pass
                            else:
                                self.previous_target[satellite] = None
                                self.begin_current_action[satellite] = current_time
                                self.previous_action[satellite] = action
                        else:
                            for target in target_info[satellite].keys():
                                if target != self.previous_target[satellite]:
                                    if target_info[satellite][target] == "Tasked":
                                        self.previous_target[satellite] = target
                                        self.begin_current_action[
                                            satellite
                                        ] = current_time
                                        self.previous_action[satellite] = action
                                    else:
                                        self.previous_target[satellite] = None
                                        self.begin_current_action[
                                            satellite
                                        ] = current_time
                                        self.previous_action[satellite] = action
            else:
                if action is not None:
                    for target in target_info[satellite].keys():
                        if target_info[satellite][target] == "Tasked":
                            self.previous_target[satellite] = target
                            self.begin_current_action[satellite] = current_time
                            self.previous_action[satellite] = action
                        else:
                            self.previous_target[satellite] = None
                            self.begin_current_action[satellite] = current_time

            self.previous_action[satellite] = action

            if temp_wasted_time >= self.max_wasted_time:
                temp_wasted_time = 0

            wasted_time[satellite] = temp_wasted_time

        if self.verbose:
            print("Begin current action")
            print(self.begin_current_action)
            print("Previous action")
            print(self.previous_action)
            print("Previous target")
            print(self.previous_target)

        return wasted_time

    def get_target_info(self, info) -> dict:
        """Get the target info from the info dictionary

        Args:
            info: Dictionary of info from the environment

        Returns:
            target_info: Dictionary of target info
        """
        target_info = {}
        for satellite in self.previous_target.keys():
            target_temp = None
            final_dict = {}
            target_temp_list_dict = {}
            case = None
            str_temp = info[satellite]
            for str_temp_i_list in str_temp:
                str_temp_i = str_temp_i_list[1]
                if self.verbose:
                    print(f"{Colors.BLUE}Str_temp_i: {str_temp_i}{Colors.RESET}")
                if re.search(r"\(.*?\)", str_temp_i):
                    if self.verbose:
                        print(f"{Colors.CYAN}Target found in string{Colors.RESET}")
                    target_temp = re.search(r"\(.*?\)", str_temp_i).group(0)[1:-1]
                    if re.search(r"imaged", str_temp_i):
                        case = "Imaged"
                    elif re.search(r"timed termination", str_temp_i):
                        case = "Timed"
                    elif re.search(r"tasked for imaging", str_temp_i):
                        case = "Tasked"
                    if target_temp in target_temp_list_dict.keys():
                        temp_list = target_temp_list_dict[target_temp]
                        temp_list.append(case)
                        target_temp_list_dict[target_temp] = temp_list
                    else:
                        target_temp_list_dict[target_temp] = [case]

            for target_temp_i in target_temp_list_dict.keys():
                if target_temp_list_dict[target_temp_i] is not None:
                    targ_temp = target_temp_list_dict[target_temp_i]
                    if "Imaged" in target_temp_list_dict[target_temp_i]:
                        case_temp = "Imaged"
                    elif "Tasked" in targ_temp and "Timed" in targ_temp:
                        case_temp = "Imaged"
                    elif "Timed" in target_temp_list_dict[target_temp_i]:
                        case_temp = "Timed"
                    elif "Tasked" in target_temp_list_dict[target_temp_i]:
                        case_temp = "Tasked"
                    else:
                        case_temp = "None"
                    final_dict[target_temp_i] = case_temp

            target_info[satellite] = final_dict

        if self.verbose:
            for sat in target_info.keys():
                print(sat)
                for target in target_info[sat].keys():
                    print(target, " --- ", target_info[sat][target])

        return target_info
    
def target_angle(sat_r_BN_P, sat_c_hat_P, opp_r_LP_P):
    vector_target_spacecraft_P = opp_r_LP_P - sat_r_BN_P
    vector_target_spacecraft_P_hat = vector_target_spacecraft_P / np.linalg.norm(
        vector_target_spacecraft_P
    )
    return np.arccos(np.dot(vector_target_spacecraft_P_hat, sat_c_hat_P))


def target_angle_rate(sat_r_BN_P, sat_v_BN_P, sat_omega_BP_P, opp_r_LP_P):
    r_BN_P = sat_r_BN_P
    v_BN_P = sat_v_BN_P
    r_LP_P = opp_r_LP_P
    omega_BP_P = sat_omega_BP_P
    omega_CP_ref = (
        omega_BP_P
        - np.cross(v_BN_P, r_LP_P - r_BN_P) / np.linalg.norm(r_LP_P - r_BN_P) ** 2
    )
    return np.linalg.norm(omega_CP_ref)


def regression_model(
    tgt_angle,
    tgt_angle_rate,
    x_0=0.4320,
    x_1=101.1396,
    x_2=641.8838,
    x_3=-45.9458,
    x_4=-7681.1881,
    x_5=-610.2113,
):
    slew_time = (
        x_0
        + x_1 * tgt_angle
        + x_2 * tgt_angle_rate
        + x_3 * tgt_angle**2
        + x_4 * tgt_angle_rate**2
        + x_5 * tgt_angle * tgt_angle_rate
    )
    return np.clip(slew_time, 1e-3, None)

def completion_probability(
    t_now,
    t_open,
    t_close,
    slew_time,
    image_duration
):
    t_start = max(t_now + slew_time, t_open)
    t_avail = t_close - t_start

    if t_avail <= 0:
        return 0.0

    # Degenerate case: already resolved target
    if image_duration <= 0:
        return 1.0

    mu = image_duration / 2.0
    sigma = image_duration * 0.3

    return norm.cdf(t_avail, loc=mu, scale=sigma)

def policy_function_regression(sat, lookahead):
    upcoming = sat.find_next_opportunities(n=32, types=sat.target_types)
    candidate_actions = []
    candidate_scores = []
    for action_idx, opp in enumerate(upcoming, start=1):
        tgt_i = opp["object"]
        opening_window, closing_window = opp["window"]
        # if tgt_i in sat.data_store.data.imaged:
        #     continue
        tgt_angle = target_angle(
            sat.dynamics.r_BN_P,
            sat.fsw.c_hat_P,
            tgt_i.r_LP_P,
        )
        tgt_angle_rate = target_angle_rate(
            sat.dynamics.r_BN_P,
            sat.dynamics.v_BN_P,
            sat.dynamics.omega_BP_P,
            tgt_i.r_LP_P,
        )
        slew_time = regression_model(tgt_angle, tgt_angle_rate)
        if slew_time > closing_window - sat.simulator.sim_time:
            continue
        min_time = max(slew_time, opening_window - sat.simulator.sim_time)
        uncertainty = tgt_i.search_area / U_NORM
        completionP = completion_probability(sat.simulator.sim_time, opening_window, closing_window, slew_time, tgt_i.image_duration)
        score = tgt_i.priority * uncertainty * completionP / min_time
        candidate_scores.append(score)
        candidate_actions.append(action_idx)

    if len(candidate_actions) == 0:
        return 0

    n_ahead = lookahead
    if len(candidate_actions) < lookahead:
        n_ahead = len(candidate_actions)

    best_target = np.argmax(candidate_scores[:n_ahead])
    action = candidate_actions[best_target]

    return action

def test_policy(
    n_targets=3000,
    horizon=60 * 95 * 3,
    target_distribution="uniform",
    seed=None,
    verbose=False,
    policies_path=None,
    policy_name=None,
    shielded=True,
    filter=0.0,
    min_power=0.25,
    lookahead=32,
    reward_function="multiply",
    reward_weight=1.0,
    observation_spec_name=None,
    action_spec_name=None,
):
    env_kwargs = dict(
        n_targets=n_targets,
        horizon=horizon,
        target_distribution=target_distribution,
        threshold=filter,
        reward_function=reward_function,
        weight=reward_weight,
    )

    env_kwargs["observation_spec"] = flexible_spec_options.get_observation_spec(
        observation_spec_name,
        reward_function=reward_function,
        reward_weight=reward_weight,
    )
    env_kwargs["action_spec"] = flexible_spec_options.get_action_spec(action_spec_name)
    
    env, satellites, _ = dynamic_imaging_rewardFlexible_flexible_specs_charge.setup_env(test=True, **env_kwargs)

    stats = {}
    reward_cum_env = 0.0
    reward_cum_manual = 0.0

    observation, info = env.reset(seed=seed)

    log_info = logInfos(satellites)

    targets_with_opportunities_init = set(
        sum(
            [
                [
                    opp["object"]
                    for opp in sat.opportunities
                    if opp["type"] == "target"
                    and opp["window"][1] > 0
                    and opp["window"][0] < 3 * T_ORBIT
                ]
                for sat in env.satellites
            ],
            [],
        )
    )
    max_reward_init = sum([target.priority for target in targets_with_opportunities_init])
    total_uncertain_init = sum([target.search_area / U_NORM for target in targets_with_opportunities_init])
    priority_uncertain_init = sum([target.priority * target.search_area / U_NORM for target in targets_with_opportunities_init])

    satellites = env.unwrapped.satellites  # store once
    target_dict = {satellite.id: {"target_name": [], "target_pri": [], "target_area": [], "target_ang": [], "target_open": [], "target_close": [], "action_success" : []} for satellite in satellites}

    while True:
        print(f"<time: {env.unwrapped.simulator.sim_time:.1f}s>")

        # actions = []
        for satellite in satellites:
            # action = np.argmax(policy_net.predict(satellite.get_obs()[None, :])[0][0])
            obs_dict = satellite.observation_builder.obs_dict()
            sat_props = obs_dict['sat_props']
            priorities = [t['priority'] for t in obs_dict['target'].values()]
            uncertainties = [t['search_area_normd'] for t in obs_dict['target'].values()]
            close_times = [t['opportunity_close_normd'] for t in obs_dict['target'].values()]
            action = policy_function_regression(satellite, lookahead)
            interference = False
            if shielded:
                shielded_action = power_shielding_function(
                    satellite.get_obs(),
                    action,
                    min_power=min_power,
                )
                if shielded_action is not None:
                    interference = shielded_action != action
                    action = shielded_action
            # actions.append(action)
            target = satellite.parse_target_selection(action)
            target_obs = satellite.observation_builder.obs_dict()
            target_obs = target_obs.get("target", {})
            target_key = f"target_{action}"  # adjust offset as needed
            obs = target_obs[target_key] if target_key in target_obs else None

            name_str = str(target.name) if target is not None else "charge"
            if action > 0:
                target_dict[satellite.id]["target_name"].append(name_str)
                target_dict[satellite.id]["target_pri"].append(target.priority if target else None)
                target_dict[satellite.id]["target_area"].append(target.search_area if target else None)

                if obs is not None:
                    target_dict[satellite.id]["target_ang"].append(obs["target_angle_normd"])
                    target_dict[satellite.id]["target_open"].append(obs["opportunity_open_normd"])
                    target_dict[satellite.id]["target_close"].append(obs["opportunity_close_normd"])
                else:
                    target_dict[satellite.id]["target_ang"].append(None)
                    target_dict[satellite.id]["target_open"].append(None)
                    target_dict[satellite.id]["target_close"].append(None)

            log_info.log(action, interference, satellite.get_obs(), env.unwrapped.simulator.sim_time, satellite.id, name_str)

        # Detect actual imaging success using storage delta (reward can be 0 even when an image is taken).
        sat0 = satellites[0]
        old_stored_data = np.array(
            sat0.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData, dtype=float
        )

        print("\tAction:", action)
        observation, reward, terminated, truncated, info = env.step(action)

        new_stored_data = np.array(
            sat0.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData, dtype=float
        )
        imaged_this_step = bool(np.any(new_stored_data - old_stored_data > 0.0))

        if isinstance(reward, dict):
            step_reward_env = float(sum(reward.values()))
        else:
            step_reward_env = float(np.sum(np.asarray(reward)))
        reward_cum_env += step_reward_env

        # Record if action succeeded (actual image taken).
        for satellite in env.unwrapped.satellites:
            if action > 0:
                target_dict[satellite.id]["action_success"].append(imaged_this_step)

        # Manual reward for the selected target at decision time (priority * uncertainty),
        # but only when an image was actually taken.
        if action > 0 and imaged_this_step and target is not None:
            uncertainty_now = float(target.search_area) / U_NORM
            reward_cum_manual += float(target.priority) * uncertainty_now

        for sat, msgs in info.items():
            if isinstance(msgs, list):
                for msg in msgs:
                    if len(msg) == 2:
                        # print(f"\t{sat} at {msg[0]:.1f}: {msg[1]}")
                        print(f"\t{sat} at {msg[0]}: {msg[1]}")

        print(
            f"\tReward(env): {step_reward_env:.3f} ({reward_cum_env:.3f} cumulative) | "
            f"Reward(manual): {reward_cum_manual:.3f}"
        )

        if terminated or truncated:
            break

    dict_log = log_info.get_logs()

    total_uncertainty = 0.0
    num_targets = len(env.unwrapped.rewarder.data.known)
    for target in env.unwrapped.rewarder.data.known:
        # if target not in env.unwrapped.rewarder.data.imaged:
        uncertainty = target.search_area / U_NORM
        total_uncertainty += target.priority * uncertainty
    total_uncertainty_norm = total_uncertainty / num_targets

    targets_with_opportunities_end = set(
        sum(
            [
                [
                    opp["object"]
                    for opp in sat.opportunities
                    if opp["type"] == "target"
                    and opp["window"][1] > 0
                    and opp["window"][0] < 3 * T_ORBIT
                ]
                for sat in env.unwrapped.satellites
            ],
            [],
        )
    )

    stats["exec"] = dict(
        reward_cum=reward_cum_env,
        reward_cum_env=reward_cum_env,
        reward_cum_manual=reward_cum_manual,
        unique=len(env.unwrapped.rewarder.data.imaged),
        duplicates=env.unwrapped.rewarder.data.duplicates,
        alive=(env.satellite.is_alive()),
        target_names=[target.name for target in env.unwrapped.rewarder.data.imaged],
        priorities=[target.priority for target in env.unwrapped.rewarder.data.imaged],
        areas=[target.search_area for target in env.unwrapped.rewarder.data.imaged],
        orbits_complete=env.unwrapped.simulator.sim_time / (T_ORBIT),
        missed_targets=env.unwrapped.satellites[0].missed,
        total_uncertainty=total_uncertainty,
        total_uncertainty_norm=total_uncertainty_norm,
        log_info=dict_log,
        target_dict=target_dict,
        max_reward_init = max_reward_init,
        max_reward_end = sum([target.priority for target in targets_with_opportunities_end]),
        total_uncertain_init = total_uncertain_init,
        total_uncertain_end = sum([target.search_area / U_NORM for target in targets_with_opportunities_end]),
        priority_uncertain_init = priority_uncertain_init,
        priority_uncertain_end = sum([target.priority *  target.search_area / U_NORM for target in targets_with_opportunities_end]),
    )
    return stats


if __name__ == "__main__":
    from pathlib import Path
    from avs_rl_tools.utils import build_job_array
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    t_orb = T_ORBIT
    cases = build_job_array(
        seed = [i for i in range(50)],
        n_targets=[1000, 2000, 3000, 4000, 6000, 8000, 10000],
        horizon=[3 * t_orb],
        target_distribution=["uniform"],
        shielded=[True],
        lookahead=[32],
        filter=[1e4],
        min_power=[0.3, 0.4, 0.5], # Tune for better performance/alive, 0.25 is the default value for the model
        reward_function=["multiply"],
        reward_weight=[1.0],
        observation_spec_name=["charge_area2_reward_32"],
        action_spec_name=["charge_image_32"],
    )

    save_path = Path(__file__).resolve().parent.parent / "data" / "charge_heuristic_slew_filter_higherMinPower_1050_try3" 
    os.makedirs(save_path, exist_ok=True)

    max_jobs = 11
    job_count = 0
    start_from = 0
    for j, case in enumerate(cases):
        i = j + start_from
        if (
            i >= 0
            and not os.path.isfile(save_path / f"case_{i}.h5")
            and job_count < max_jobs
        ):
            print(f"========= CASE {i+1}/{(len(cases) + start_from)} =========")
            file_h5 = h5py.File(save_path / f"case_{i}.h5", "w")
            file_h5.close()

            stats = test_policy(
                # seed=i,
                **case,
            )
            stats["case"] = case

            # To save the data in a h5 file
            file_h5 = h5py.File(save_path / f"case_{i}.h5", "a")
            for key, value in stats["exec"].items():
                if key == "reward_per_sat_cum":
                    file_reward = file_h5.create_group("reward_per_sat")
                    for key_sat, value_sat in value.items():
                        file_reward.create_dataset(key_sat, data=value_sat)
                elif key == "wasted_time_sat_cum":
                    file_time = file_h5.create_group("wasted_time_sat")
                    for key_sat, value_sat in value.items():
                        file_time.create_dataset(key_sat, data=value_sat)
                elif key == "log_info":
                    for key_obs, value_obs in value.items():
                        if isinstance(value_obs, dict):
                            file_obs = file_h5.create_group(key_obs)
                            for key_obs_sat, value_obs_sat in value_obs.items():
                                if isinstance(value_obs_sat, dict):
                                    file_obs_sat = file_obs.create_group(key_obs_sat)
                                    for key_obs_sat, value_obs_sat in value_obs_sat.items():
                                        file_obs_sat.create_dataset(key_obs_sat, data=value_obs_sat)
                                else:
                                    file_obs.create_dataset(key_obs_sat, data=value_obs_sat)
                        else:
                            file_h5.create_dataset(key_obs, data=value_obs)
                elif key == "target_dict":
                    grp_target = file_h5.create_group("target_dict")
                    for sat_id, sat_data in value.items():
                        grp_sat = grp_target.create_group(str(sat_id))
                        for subkey, subval in sat_data.items():
                            try:
                                if isinstance(subval, list):
                                    if len(subval) == 0:
                                        data_array = np.array([])
                                    elif isinstance(subval[0], np.ndarray):
                                        data_array = np.stack(subval)
                                    elif isinstance(subval[0], str):  # <-- handle list of strings
                                        dt = h5py.string_dtype(encoding='utf-8')
                                        data_array = np.array(subval, dtype=dt)
                                    else:
                                        data_array = np.array(subval)
                                else:
                                    data_array = np.array(subval)
                                grp_sat.create_dataset(subkey, data=data_array)
                            except Exception as e:
                                print(f"Failed to save {subkey} for {sat_id}: {e}")
                else:
                    file_h5.create_dataset(key, data=value)

            for key, value in stats["case"].items():
                try:
                    file_h5.create_dataset(key, data=value)
                except TypeError:
                    print(f"Could not save {key}, casting {value} to string")
                    file_h5.create_dataset(key, data=str(value))

            file_h5.close()

            job_count += 1
