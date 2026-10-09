import gymnasium as gym
import numpy as np
from scipy.stats import norm

from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion, macros
from bsk_rl import act, data, obs, scene, sats
from bsk_rl.scene.targets import Target
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.sim.fsw import action
from bsk_rl.utils.functional import aliveness_checker
from typing import Union, Iterable, Optional

from bsk_rl.utils.orbital import random_orbit
import configurations as config
from configurations import T_ORBIT
from envs import AreaTargets, FlexibleUncertainReward

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

class RetaskImage(act.Image):
    """Image action that always reissues the task, even for same target consecutively."""

    def image(
        self, target: Union[int, "Target", str], prev_action_key: Optional[str] = None
    ) -> str:
        target = self.satellite.parse_target_selection(target)
        self.satellite.task_target_for_imaging(target, max_duration=self.max_duration)
        return target.id
    
def s_hat_H(sat):
    r_SN_N = (
        sat.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
            sat.simulator.world.sun_index
        ]
        .read()
        .PositionVector
    )
    r_BN_N = sat.dynamics.r_BN_N
    r_SB_N = np.array(r_SN_N) - np.array(r_BN_N)
    r_SB_H = sat.dynamics.HN @ r_SB_N
    return r_SB_H / np.linalg.norm(r_SB_H)


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
        + x_3 * tgt_angle ** 2
        + x_4 * tgt_angle_rate ** 2
        + x_5 * tgt_angle * tgt_angle_rate
    )
    return np.clip(slew_time, 1e-3, None)


def completion_probability(t_now, t_open, t_close, slew_time, image_duration):
    t_start = max(t_now + slew_time, t_open)
    t_avail = t_close - t_start

    if t_avail <= 0:
        return 0.0

    if image_duration <= 0:
        return 1.0

    mu = image_duration / 2.0
    sigma = max(image_duration * 0.3, 1e-6)

    return float(np.clip(norm.cdf(t_avail, loc=mu, scale=sigma), 0.0, 1.0))


def opportunity_completion_probability(sat, opp):
    target = opp["object"]
    r_LP_P = target.r_LP_P
    tgt_angle = target_angle(sat.dynamics.r_BN_P, sat.fsw.c_hat_P, r_LP_P)
    tgt_angle_rate = target_angle_rate(
        sat.dynamics.r_BN_P,
        sat.dynamics.v_BN_P,
        sat.dynamics.omega_BP_P,
        r_LP_P,
    )
    slew_time = regression_model(tgt_angle, tgt_angle_rate)

    t_open, t_close = opp["window"]
    return completion_probability(
        sat.simulator.sim_time, t_open, t_close, slew_time, target.image_duration
    )

class Density(obs.Observation):
    def __init__(
        self,
        interval_duration=60 * 3,
        intervals=10,
        norm=3,
    ):
        self.satellite: "sats.ImagingSatellite"
        super().__init__()
        self.interval_duration = interval_duration
        self.intervals = intervals
        self.norm = norm

    def get_obs(self):
        if self.intervals == 0:
            return []

        self.satellite.calculate_additional_windows(
            self.simulator.sim_time
            + (self.intervals + 1) * self.interval_duration
            - self.satellite.window_calculation_time
        )
        soonest = self.satellite.upcoming_opportunities_dict(types="target")
        rewards = np.array([opportunity.priority for opportunity in soonest])
        times = np.array([opportunities[0][1] for opportunities in soonest.values()])
        time_bins = np.floor((times - self.simulator.sim_time) / self.interval_duration)
        densities = [sum(rewards[time_bins == i]) for i in range(self.intervals)]
        return np.array(densities) / self.norm

class CustomDynModel(dyn.FullFeaturedDynModel):
    # @aliveness_checker
    def rw_speeds_valid(self) -> bool:
        """Check if any wheel speed exceeds the ``maxWheelSpeed``."""
        valid = all(
            abs(speed) < self.maxWheelSpeed * macros.rpm2radsec
            for speed in self.wheel_speeds
        )
        return valid
    
class CustomFswModel(fsw.SteeringImagerFSWModel):
    
    @action
    def action_image(self, r_LP_P: Iterable[float], data_name: str, acquisitionTime: float, allowedTime: float) -> None:
        """Attempt to image a target at a location.

        This action sets the target attitude to one tracking a ground location. If the
        target is within the imaging constraints, an image will be taken and stored in
        the data buffer. The instrument power sink will be active as long as the task is
        enabled.

        Args:
            r_LP_P: [m] Planet-fixed planet relative target location.
            data_name: Data buffer to store image data to.
        """
        self.insControl.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.imagingTarget.r_LP_P_Init = r_LP_P
        self.dynamics.instrument.nodeDataName = data_name
        self.insControl.imaged = 0
        self.insControl.useDurationImaging = 1
        self.insControl.acquisitionTime = macros.sec2nano(acquisitionTime)
        self.insControl.allowedTime = macros.sec2nano(allowedTime)
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.name)

def default_observation_spec(n_ahead_observe=32):
    return [
        obs.SatProperties(
            dict(prop="omega_BN_B", norm=0.03),  # 0,1,2 - CHECKED
            dict(prop="c_hat_H"),  # 3,4,5 - CHECKED
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),  # 6,7,8 - CHECKED
            dict(prop="v_BN_P", norm=7616.5),  # 9,10,11 - CHECKED
            dict(prop="battery_charge_fraction"),  # 12 - CHECKED
            # dict(prop="storage_level_fraction"),  # 13
            dict(prop="wheel_speeds_fraction"),  # 14,15,16
            dict(prop="s_hat_H", fn=s_hat_H),  # 17, 18, 19 - CHECKED
        ),
        # obs.OpportunityProperties(
        #     dict(prop="opportunity_open", norm=T_ORBIT),
        #     dict(prop="opportunity_close", norm=T_ORBIT),
        #     type="ground_station",
        #     n_ahead_observe=1,
        # ),  # 20, 21
        obs.Eclipse(norm=T_ORBIT),  # 22,23 - CHECKED
        Density(intervals=20, norm=5),  # 24 - CHECKED
        obs.OpportunityProperties(
            dict(prop="priority"),  # - CHECKED
            dict(prop="r_LB_H", norm=orbitalMotion.REQ_EARTH * 1e3),  # - CHECKED
            dict(
                prop="target_angle", norm=np.pi / 2
            ),  # - CHECKED (LOOK IF THIS IS 1 OR 3)
            dict(prop="target_angle_rate", norm=0.03),  
            dict(prop="opportunity_open", norm=300.0),  # - CHECKED
            dict(prop="opportunity_close", norm=300.0),  # - CHECKED
            dict(prop="search_area", fn=lambda sat, opp: opp["object"].search_area ** 2, norm=(100.0 * 1e3) ** 2), 
            dict(prop="completion_probability", fn=opportunity_completion_probability),
            # dict(prop="image_duration", fn=lambda sat, opp: opp["object"].image_duration, norm=200.0),  
            type="target",
            n_ahead_observe=n_ahead_observe,
        ),
    ]

def default_action_spec(n_ahead_image=32):
    return [
        act.Charge(duration=60.0),  # 0
        # act.Downlink(),  # 1
        act.Desat(duration=60.0),  # 2
        RetaskImage(n_ahead_image=n_ahead_image),
    ]

class CustomSatComposed(sats.ImagingSatellite):

    observation_spec = default_observation_spec()
    action_spec = default_action_spec()

    dyn_type = CustomDynModel
    fsw_type = CustomFswModel

    def __init__(self, *args, image_count=0, charge_count=0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_count = image_count
        self.charge_count = charge_count

    def set_action(self, action: Union[int, Target, str]):
        super().set_action(action)
        if action == 0:
            self.charge_count += 1
        elif action > 0:
            self.image_count += 1


    def reset_post_sim_init(self):
        super().reset_post_sim_init()
        self.image_count = 0
        self.charge_count = 0

    def task_target_for_imaging(
        self, target: "Target", max_duration: Optional[float] = None
    ):
        """Task the satellite to image a target.

        Args:
            target: Selected target
            max_duration: [s] Maximum duration to wait for imaging. If None, wait until
                the end of the target's access window.
        """
        msg = f"{target} tasked for imaging, image_time: {target.image_time:.1f}, duration: {target.image_duration:.1f}"
        self.logger.info(msg)
        self.fsw.action_image(target.r_LP_P, self.buffer_name, target.image_time, target.image_duration) # times in seconds
        self.enable_target_window(target, max_duration=max_duration)
        self.draw_imaging_line(target)
        self.latest_target = target


class CustomSatComposedFlexible(CustomSatComposed):
    """Importable flexible satellite class for Ray worker deserialization."""


CustomSatComposedFlexible1 = CustomSatComposedFlexible


def make_satellite_class(observation_spec=None, action_spec=None, class_name=None):
    """Configure and return the importable flexible satellite class.

    BSK-RL reads specs from the satellite class while constructing its builders,
    so the class attributes are set before satellite instantiation. Ray workers
    need this class to exist at module import time for deserialization.
    """
    if observation_spec is None:
        observation_spec = default_observation_spec()
    if action_spec is None:
        action_spec = default_action_spec()
    if not isinstance(observation_spec, list):
        raise TypeError("observation_spec must be a list of bsk_rl observations")
    if not isinstance(action_spec, list):
        raise TypeError("action_spec must be a list of bsk_rl actions")
    if len(action_spec) == 0:
        raise ValueError("action_spec must contain at least one action")

    if class_name not in (None, CustomSatComposedFlexible.__name__):
        raise ValueError(
            "Flexible satellite class names must be stable for Ray; "
            f"use {CustomSatComposedFlexible.__name__!r}."
        )

    CustomSatComposedFlexible.observation_spec = observation_spec
    CustomSatComposedFlexible.action_spec = action_spec
    CustomSatComposedFlexible1.observation_spec = observation_spec
    CustomSatComposedFlexible1.action_spec = action_spec
    return CustomSatComposedFlexible


def setup_env(
        test=True,
        horizon=None,
        n_targets=None, 
        target_distribution=None, 
        reward_function="multiply",
        weight=1.0,
        threshold=10_000.0,
        observation_spec=None,
        action_spec=None,
        failure_penalty=config.TRAINING_PARAMS["failure_penalty"],
    ):
    # This function sets up the environment and satellites for the simulation
    # It pulls in the configuration parameters from the configurations.py file
    # Return the gym environment and the satellites list

    if n_targets is None:
        n_targets = config.SIM_PARAMS["n_targets"]
    if target_distribution is None:
        target_distribution = config.SIM_PARAMS["target_distribution"]
    if horizon is None:
        horizon = config.SIM_PARAMS["horizon"] * T_ORBIT
    
    # communication = config.SIM_PARAMS["communication"] %We are always assuming full communication
    data_storage_capacity = config.STEERING_2Hz_POWER["sat_args"]["dataStorageCapacity"]
    satellite_cls = (
        CustomSatComposed
        if observation_spec is None and action_spec is None
        else make_satellite_class(
            observation_spec=observation_spec,
            action_spec=action_spec,
        )
    )

    if target_distribution == "uniform":
        env_features = AreaTargets(n_targets=n_targets)
    elif target_distribution == "cities":
        env_features = scene.CityTargets(n_targets=n_targets)
    else:
        raise (ValueError("Invalid distribution type"))

    satellites = []
    sat_args = satellite_cls.default_sat_args(
        **config.STEERING_2Hz_POWER["sat_args"],
        storageInit=lambda: np.random.randint(
            config.RANDOM_INIT["storage_low_percent"] * data_storage_capacity,
            config.RANDOM_INIT["storage_high_percent"] * data_storage_capacity,
        ),
        wheelSpeeds=lambda: np.random.uniform(
            config.RANDOM_INIT["rw_speed_low"],
            config.RANDOM_INIT["rw_speed_high"],
            3,
        ),
    )
    satellites.append(
        satellite_cls(
            "EO",
            sat_args,
        )
    )

    for sat in satellites:
        sat._search_area_threshold = threshold

    for sat in satellites:
        def search_area_filter(opportunity, threshold=threshold):
            if opportunity["type"] == "target":
                return opportunity["object"].search_area > threshold
            return True

        sat.add_access_filter(search_area_filter, types="target")

    #print("Opp dictionary", satellites[0].observation_space)

    if test:
        from gym_uncertainty_charge import SatelliteUncertaintyTargets
        # Make the environment with Gymnasium
        env = SatelliteUncertaintyTargets(
            satellite=satellites[0],
            terminate_on_time_limit=True,
            # Select an EnvironmentModel compatible with the models in the satellite
            # world_type=world.GroundStationWorldModel,
            # world_args=world.GroundStationWorldModel.default_world_args(),
            scenario=env_features,
            #scenario=scene.UniformTargets(1000),
            rewarder=FlexibleUncertainReward(
                reward_function=reward_function,
                weight=weight,
            ),
            sim_rate=config.STEERING_2Hz_POWER["sim_rate"],
            max_step_duration=T_ORBIT,
            time_limit=horizon,
            log_level="WARNING",
            failure_penalty=failure_penalty,
            #vizard_dir="/tmp/vizard",
            # disable_env_checker=True,
        )
        env_args_dict = None
    else:
        env = None
        env_args_dict = dict(
            satellite = satellites[0],
            # world_type = world.GroundStationWorldModel,
            # world_args = world.GroundStationWorldModel.default_world_args(),
            scenario = env_features,
            rewarder = FlexibleUncertainReward(
                reward_function=reward_function,
                weight=weight,
            ),
            sim_rate = config.STEERING_2Hz_POWER["sim_rate"],
            max_step_duration = T_ORBIT,
            time_limit = horizon,
            log_level="WARNING",
            failure_penalty=failure_penalty,
        )

    return env, satellites, env_args_dict


if __name__ == "__main__":
    import flexible_spec_options as spec_options

    observation_spec_name = "charge_area2_reward_32"
    action_spec_name = "charge_image_32"
    observation_spec = spec_options.get_observation_spec(observation_spec_name)
    action_spec = spec_options.get_action_spec(action_spec_name)

    env, satellites, env_args_dict = setup_env(
        n_targets=1000,
        observation_spec=observation_spec,
        action_spec=action_spec,
    )

    # Run the simulation until timeout or agent failure
    total_reward = 0.0
    observation, info = env.reset(seed=None)
    # The composed satellite action space returns a human-readable action map
    # Each satellite has the same action space and similar observation space
    print("Actions:", satellites[0].action_description)
    print("States:", env.satellites[0].observation_description)
    # print("TEST BATTERY: ", observation[0][12])
    # print("TEST ECLIPSE START: ", observation[0][16])
    # print("TEST ECLIPSE END: ", observation[0][17])
    
    obs_desc = env.satellites[0].observation_description
    obs_values = observation[0]  # assuming a single satellite

    # # Extract indices for target-related features
    # target_indices = [i for i, name in enumerate(obs_desc) if name.startswith("target.")]
    # target_obs = [obs_values[i] for i in target_indices]

    # # Optionally, pair with names for readability
    # target_info = [(obs_desc[i], obs_values[i]) for i in target_indices]
    # for name, val in target_info:
    #     print(f"{name}: {val}")

    # # Using the composed satellite features also provides a human-readable state:
    print("Observation:", env.satellites[0].observation_builder.obs_dict())
    for satellite in env.satellites:
        for k, v in satellite.observation_builder.obs_dict().items():
            print(f"\t\t{k}:  {v}")

    count = 0
    while True:
        print(f"<time: {env.simulator.sim_time:.1f}s>")

        if count == 0:
            # Vector with an action for each satellite (we can pass different actions for each satellite)
            # Tasking all satellites to charge (tasking None as the first action will raise a warning)
            action_vector = [0]
        elif count == 1:
            # None will continue the last action, but will also raise a warning
            action_vector = [None]
        elif count == 2:
            # Tasking different actions for each satellite
            action_vector = [1]
        else:
            # Tasking random actions
            action_vector = env.action_space.sample()
        count += 1

        action_step = np.random.randint(0, 32)
        observation, reward, terminated, truncated, info = env.step(action_step)
        print("Info:", info)

        print(f"\tReward: {reward:.3f}")
        cum_reward = env.rewarder.cum_reward
        cum_reward = sum(cum_reward.values()) / len(cum_reward)
        print(f"\tCumulative reward: {cum_reward:.3f}")
        print(f"Battery charge: {env.satellites[0].dynamics.battery_charge_fraction}")
        if terminated or truncated:
            print("Episode complete.")
            break

    print("Total reward:", cum_reward)
    print("Total image actions taken:", env.satellites[0].image_count)
    print("T_ORBIT:", T_ORBIT)
