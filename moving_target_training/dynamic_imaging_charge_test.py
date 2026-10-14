import os
import re
from copy import copy
from functools import partial

import h5py
import numpy as np
from load_policy_simple import load_policy
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


class ShieldedPolicy:
    def __init__(self, policy_path, env_args=None, shield_function=None, zero_element=None) -> None:
        # self.policy_net = tf.keras.models.load_model(policy_path)
        self.zero_element = zero_element
        self.shield_function = shield_function
        self.policy_function = None
        if policy_path is not None:
            self.policy_function = load_policy(policy_path)

    def act(self, observation, extra_states=[]):
        interference = False
        if self.zero_element is not None:
            observation[self.zero_element] = 0.0
        shielded_action = None
        # action = np.argmax(self.policy_net.predict(observation[None, :])[0][0])
        if self.policy_function is not None:
            action = self.policy_function(observation)
        else:
            action = None
        if self.shield_function is not None:
            shielded_action = self.shield_function(observation, action)
        if shielded_action is not None:
            if action != shielded_action:
                interference = True
            action = shielded_action
        return action, interference
    
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
    _, _, env_args = dynamic_imaging_rewardFlexible_flexible_specs_charge.setup_env(test=False)
    env_args_list = build_job_array(env_args={k:[v] for k,v in env_args.items()},)
    if shielded:
        shield_function = partial(power_shielding_function, min_power=min_power)
        policy = ShieldedPolicy(policies_path[policy_name], env_args=env_args_list[0]['env_args'], shield_function=shield_function, zero_element=None)
    else:
        policy = ShieldedPolicy(policies_path[policy_name], env_args=env_args_list[0]['env_args'], shield_function=None, zero_element=None)

    stats = {}
    reward_cum = 0

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
            action, interference = policy.act(satellite.get_obs())
            # actions.append(action)
            target = satellite.parse_target_selection(action)
            target_obs = satellite.observation_builder.obs_dict()
            target_obs = target_obs.get("target", {})
            target_key = f"target_{action}"  # adjust offset as needed
            obs = target_obs[target_key] if target_key in target_obs else None

            if action > 0:
                name_str = str(target.name)
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

            log_info.log(action, interference, satellite.get_obs(), env.unwrapped.simulator.sim_time, satellite.id, str(target.name))

        print("\tAction:", action)
        observation, reward, terminated, truncated, info = env.step(action)

        # Record if action succeeded (reward > 0)
        for satellite in env.unwrapped.satellites:
            # Assume reward is per satellite; adjust if reward is a scalar for all
            if action > 0:
                action_success = reward > 0
                target_dict[satellite.id]["action_success"].append(action_success)

        for sat, msgs in info.items():
            if isinstance(msgs, list):
                for msg in msgs:
                    if len(msg) == 2:
                        # print(f"\t{sat} at {msg[0]:.1f}: {msg[1]}")
                        print(f"\t{sat} at {msg[0]}: {msg[1]}")

        reward_cum += reward
        print(f"\tReward: {reward:.3f} ({reward_cum:.3f} cumulative)")

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
        reward_cum=reward_cum,
        unique=len(env.rewarder.data.imaged),
        duplicates=env.rewarder.data.duplicates,
        alive=(env.satellite.is_alive()),
        target_names=[target.name for target in env.rewarder.data.imaged],
        priorities=[target.priority for target in env.rewarder.data.imaged],
        areas=[target.search_area for target in env.rewarder.data.imaged],
        orbits_complete=env.simulator.sim_time / (T_ORBIT),
        missed_targets=env.satellites[0].missed,
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

    policy_path = Path("/projects/chbe6195/saved_policies/best_run_num_sgd20")

    policies_path = {"RL_base": policy_path} 

    policy_name = ["RL_base"]
    
    t_orb = T_ORBIT
    cases = build_job_array(
        seed = [i for i in range(50)],
        n_targets=[1000, 2000, 3000, 4000, 6000, 8000, 10000],
        horizon=[3 * t_orb],
        target_distribution=["uniform"],
        policy_name=policy_name,
        shielded=[True],
        filter=[1e4],
        min_power=[0.3, 0.4, 0.5], # Tune for better performance/alive, 0.25 is the default value for the model
        reward_function=["multiply"],
        reward_weight=[1.0],
        observation_spec_name=["charge_area2_reward_32"],
        action_spec_name=["charge_image_32"],
    )

    save_path = Path(__file__).resolve().parent.parent / "data" / "charge_multiply_filter_higherMinpower_1050_try2" 
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
                policies_path=policies_path,
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
