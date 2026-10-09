import numpy as np
from Basilisk.utilities import orbitalMotion
from bsk_rl import act, obs

from dynamic_imaging_rewardFlexible_flexible_specs_charge import (
    Density,
    RetaskImage,
    opportunity_completion_probability,
    s_hat_H,
)
from envs_common import REWARD_FUNCTIONS
from configurations import T_ORBIT, U_NORM


def _reward_value_factory(reward_function="multiply", reward_weight=1.0):
    use_target_reward = reward_function is None or reward_weight is None

    def reward_value(sat, opp):
        target = opp["object"]
        active_reward_function = (
            getattr(target, "random_reward_function", None)
            if use_target_reward
            else reward_function
        )
        active_reward_weight = (
            getattr(target, "random_reward_weight", None)
            if use_target_reward
            else reward_weight
        )
        if active_reward_function is None or active_reward_weight is None:
            active_reward_function = "multiply"
            active_reward_weight = 1.0
        reward_formula = REWARD_FUNCTIONS[active_reward_function]
        uncertainty = (target.search_area ** 2) / (U_NORM ** 2)
        return float(reward_formula(target.priority, uncertainty, active_reward_weight))

    return reward_value


def _target_properties(
    include_priority=True,
    include_search_area=True,
    include_probability=True,
    include_reward_value=False,
    reward_function="multiply",
    reward_weight=1.0,
):
    properties = [
        dict(prop="r_LB_H", norm=orbitalMotion.REQ_EARTH * 1e3),
        dict(prop="target_angle", norm=np.pi / 2),
        dict(prop="target_angle_rate", norm=0.03),
        dict(prop="opportunity_open", norm=300.0),
        dict(prop="opportunity_close", norm=300.0),
    ]
    if include_priority:
        properties.insert(0, dict(prop="priority"))
    if include_search_area:
        properties.append(
            dict(
                prop="search_area",
                fn=lambda sat, opp: opp["object"].search_area ** 2,
                norm=(U_NORM) ** 2,
            )
        )
    if include_probability:
        properties.append(
            dict(prop="completion_probability", fn=opportunity_completion_probability)
        )
    if include_reward_value:
        properties.append(
            dict(
                prop="reward_value",
                fn=_reward_value_factory(
                    reward_function=reward_function,
                    reward_weight=reward_weight,
                ),
            )
        )
    return properties


def area2_prob_32(**kwargs):
    n_ahead = 32
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        Density(intervals=20, norm=5),
        obs.OpportunityProperties(
            *_target_properties(include_search_area=True, include_probability=True),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]


def area2_prob_16(**kwargs):
    n_ahead = 16
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        Density(intervals=20, norm=5),
        obs.OpportunityProperties(
            *_target_properties(include_search_area=True, include_probability=True),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]


def area2_basic_32(**kwargs):
    n_ahead = 32
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        Density(intervals=20, norm=5),
        obs.OpportunityProperties(
            *_target_properties(include_search_area=True, include_probability=False),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]


def no_density_prob_32(**kwargs):
    n_ahead = 32
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        obs.OpportunityProperties(
            *_target_properties(include_search_area=True, include_probability=True),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]


def area2_reward_32(reward_function="multiply", reward_weight=1.0):
    n_ahead = 32
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        Density(intervals=20, norm=5),
        obs.OpportunityProperties(
            *_target_properties(
                include_search_area=True,
                include_probability=True,
                include_reward_value=True,
                reward_function=reward_function,
                reward_weight=reward_weight,
            ),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]


def area2_reward_16(reward_function="multiply", reward_weight=1.0):
    n_ahead = 16
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        Density(intervals=20, norm=5),
        obs.OpportunityProperties(
            *_target_properties(
                include_search_area=True,
                include_probability=True,
                include_reward_value=True,
                reward_function=reward_function,
                reward_weight=reward_weight,
            ),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]


def reward_only_32(reward_function="multiply", reward_weight=1.0):
    n_ahead = 32
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        Density(intervals=20, norm=5),
        obs.OpportunityProperties(
            *_target_properties(
                include_priority=False,
                include_search_area=False,
                include_probability=True,
                include_reward_value=True,
                reward_function=reward_function,
                reward_weight=reward_weight,
            ),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]


def reward_only_16(reward_function="multiply", reward_weight=1.0):
    n_ahead = 16
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        Density(intervals=20, norm=5),
        obs.OpportunityProperties(
            *_target_properties(
                include_priority=False,
                include_search_area=False,
                include_probability=True,
                include_reward_value=True,
                reward_function=reward_function,
                reward_weight=reward_weight,
            ),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]


def charge_area2_reward_32(reward_function="multiply", reward_weight=1.0):
    n_ahead = 32
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
            dict(prop="battery_charge_fraction"),
            dict(prop="s_hat_H", fn=s_hat_H),
        ),
        obs.Eclipse(norm=T_ORBIT),
        Density(intervals=20, norm=5),
        obs.OpportunityProperties(
            *_target_properties(
                include_search_area=True,
                include_probability=True,
                include_reward_value=True,
                reward_function=reward_function,
                reward_weight=reward_weight,
            ),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]

def charge_desat_area2_reward_32(reward_function="multiply", reward_weight=1.0):
    n_ahead = 32
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
            dict(prop="battery_charge_fraction"),
            dict(prop="wheel_speeds_fraction"),
            dict(prop="s_hat_H", fn=s_hat_H),
        ),
        obs.Eclipse(norm=T_ORBIT),
        Density(intervals=20, norm=5),
        obs.OpportunityProperties(
            *_target_properties(
                include_search_area=True,
                include_probability=True,
                include_reward_value=True,
                reward_function=reward_function,
                reward_weight=reward_weight,
            ),
            type="target",
            n_ahead_observe=n_ahead,
        ),
    ]
    
OBSERVATION_SPECS = {
    "area2_prob_32": {
        "factory": area2_prob_32,
        "n_targets": 32,
        "obs_sat": 32,
    },
    "area2_prob_16": {
        "factory": area2_prob_16,
        "n_targets": 16,
        "obs_sat": 32,
    },
    "area2_basic_32": {
        "factory": area2_basic_32,
        "n_targets": 32,
        "obs_sat": 32,
    },
    "no_density_prob_32": {
        "factory": no_density_prob_32,
        "n_targets": 32,
        "obs_sat": 12,
    },
    "area2_reward_32": {
        "factory": area2_reward_32,
        "n_targets": 32,
        "obs_sat": 32,
    },
    "area2_reward_16": {
        "factory": area2_reward_16,
        "n_targets": 16,
        "obs_sat": 32,
    },
    "reward_only_32": {
        "factory": reward_only_32,
        "n_targets": 32,
        "obs_sat": 32,
    },
    "reward_only_16": {
        "factory": reward_only_16,
        "n_targets": 16,
        "obs_sat": 32,
    },
    "charge_area2_reward_32": {
        "factory": charge_area2_reward_32,
        "n_targets": 32,
        "obs_sat": 32,
    },
     "charge_desat_area2_reward_32": {
        "factory": charge_desat_area2_reward_32,
        "n_targets": 32,
        "obs_sat": 41,
    },
}


def image_32():
    return [RetaskImage(n_ahead_image=32)]


def image_16():
    return [RetaskImage(n_ahead_image=16)]


def charge_image_32():
    return [
        act.Charge(duration=60.0),
        RetaskImage(n_ahead_image=32),
    ]

def charge_desat_image_32():
    return [
        act.Charge(duration=60.0),
        act.Desat(duration=60.0),
        RetaskImage(n_ahead_image=32),
    ]

ACTION_SPECS = {
    "image_32": {
        "factory": image_32,
        "n_targets": 32,
        "non_imaging_actions": 0,
    },
    "image_16": {
        "factory": image_16,
        "n_targets": 16,
        "non_imaging_actions": 0,
    },
    "charge_image_32": {
        "factory": charge_image_32,
        "n_targets": 32,
        "non_imaging_actions": 1,
    },
     "charge_desat_image_32": {
        "factory": charge_desat_image_32,
        "n_targets": 32,
        "non_imaging_actions": 2,
    },
}


def observation_spec_names():
    return tuple(OBSERVATION_SPECS)


def action_spec_names():
    return tuple(ACTION_SPECS)


def get_observation_spec(name, reward_function="multiply", reward_weight=1.0):
    return OBSERVATION_SPECS[name]["factory"](
        reward_function=reward_function,
        reward_weight=reward_weight,
    )


def get_action_spec(name):
    return ACTION_SPECS[name]["factory"]()


def get_model_config_overrides(observation_name, action_name):
    observation_config = OBSERVATION_SPECS[observation_name]
    action_config = ACTION_SPECS[action_name]
    if observation_config["n_targets"] != action_config["n_targets"]:
        raise ValueError(
            "Observation/action spec mismatch: "
            f"{observation_name} observes {observation_config['n_targets']} targets, "
            f"but {action_name} images {action_config['n_targets']} targets."
        )
    return {
        "n_targets": observation_config["n_targets"],
        "obs_sat": observation_config["obs_sat"],
        "non_imaging_actions": action_config["non_imaging_actions"],
    }
