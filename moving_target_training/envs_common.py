import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from bsk_rl.act import Image
from bsk_rl.data.base import Data
from bsk_rl.data.unique_image_data import UniqueImageData, UniqueImageReward
from bsk_rl.scene.targets import Target, UniformTargets
from configurations import U_NORM

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite


class DynamicImage(Image):
    def __init__(self, n_ahead_image: int, name: str = "action_image"):
        # Initialize with some default max_duration (will be overwritten dynamically)
        super().__init__(n_ahead_image=n_ahead_image, max_duration=None, name=name)

    def image(
        self, target: Union[int, "Target", str], prev_action_key: Optional[str] = None
    ) -> str:
        target = self.satellite.parse_target_selection(target)
        self.max_duration = target.image_time
        return super().image(target, prev_action_key)


class AreaTargets(UniformTargets):
    def __init__(self, *args, speed_range=None, assign_speed: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.assign_speed = assign_speed
        self.speed_range = (
            (22e3 / 3600, 46e3 / 3600) if speed_range is None else speed_range
        )

    def regenerate_targets(self) -> None:
        super().regenerate_targets()
        for target in self.targets:
            if self.assign_speed:
                target.speed = np.random.uniform(self.speed_range[0], self.speed_range[1])
            search_area = np.random.uniform(10.0, 100.0) * 1e3  # in meters
            target.search_area = search_area
            target.acquiring_speed = np.random.uniform(3.0, 6.0) * 1e3  # m/s
            image_duration = (
                search_area
                * np.ceil(search_area / (20.0 * 1e3))
                / target.acquiring_speed
            )
            target.image_duration = image_duration

            mean_duration = image_duration / 2.0
            std_duration = image_duration * 0.3  # adjust as needed
            image_time = np.random.normal(loc=mean_duration, scale=std_duration)
            target.image_time = max(0.0, float(image_time))

            # Used by some viz helpers; harmless elsewhere.
            target._last_area_updated = target.search_area


#############
# Reward


def _uncertainty(target: Target) -> float:
    return (target.search_area ** 2) / (U_NORM ** 2)


def _reward_multiply(priority: float, uncertainty: float, weight: float) -> float:
    return priority * uncertainty


def _reward_linear(priority: float, uncertainty: float, weight: float) -> float:
    return (1 - weight) * priority + weight * uncertainty


def _reward_square(priority: float, uncertainty: float, weight: float) -> float:
    return (1 - weight) * priority + weight * (uncertainty**2)


def _reward_cube(priority: float, uncertainty: float, weight: float) -> float:
    return (1 - weight) * priority + weight * (uncertainty**3)


REWARD_FUNCTIONS: dict[str, Callable[[float, float, float], float]] = {
    "multiply": _reward_multiply,
    "linear": _reward_linear,
    "square": _reward_square,
    "cube": _reward_cube,
}


class FlexibleUncertainReward(UniqueImageReward):
    """Configurable target reward based on priority, uncertainty, and weight.

    Add new reward formulas by defining a top-level function with signature
    ``(priority, uncertainty, weight)`` and registering it in ``REWARD_FUNCTIONS``.
    """

    def __init__(
        self,
        reward_function: str = "multiply",
        weight: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if reward_function not in REWARD_FUNCTIONS:
            valid = ", ".join(sorted(REWARD_FUNCTIONS))
            raise ValueError(
                f"Invalid reward_function '{reward_function}'. Valid options: {valid}"
            )
        self.reward_function = reward_function
        self.weight = weight

    def create_data_store(self, satellite: "Satellite") -> None:
        satellite.data_store = self.data_store_type(
            satellite,
            initial_data=self.initial_data(satellite),
            **self.data_store_kwargs,
        )
        self.cum_reward[satellite.name] = 0.0

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        reward: dict[str, float] = {}
        imaged_counts: dict[Target, int] = {}
        reward_formula = REWARD_FUNCTIONS[self.reward_function]
        for new_data in new_data_dict.values():
            for target in new_data.imaged:
                imaged_counts[target] = imaged_counts.get(target, 0) + 1

        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                value = reward_formula(
                    target.priority,
                    _uncertainty(target),
                    self.weight,
                )
                reward[sat_id] += self.reward_fn(value) / imaged_counts[target]
        return reward


class UncertainWeightReward(UniqueImageReward):
    def __init__(self, weight: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def create_data_store(self, satellite: "Satellite") -> None:
        satellite.data_store = self.data_store_type(
            satellite,
            initial_data=self.initial_data(satellite),
            **self.data_store_kwargs,
        )
        self.cum_reward[satellite.name] = 0.0

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        reward: dict[str, float] = {}
        imaged_counts: dict[Target, int] = {}
        for new_data in new_data_dict.values():
            for target in new_data.imaged:
                imaged_counts[target] = imaged_counts.get(target, 0) + 1

        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                uncertainty = target.search_area / U_NORM
                reward[sat_id] += (
                    self.reward_fn(target.priority + self.weight * uncertainty)
                    / imaged_counts[target]
                )
        return reward


class UncertainSquareReward(UniqueImageReward):
    def __init__(self, weight: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def create_data_store(self, satellite: "Satellite") -> None:
        satellite.data_store = self.data_store_type(
            satellite,
            initial_data=self.initial_data(satellite),
            **self.data_store_kwargs,
        )
        self.cum_reward[satellite.name] = 0.0

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        reward: dict[str, float] = {}
        imaged_counts: dict[Target, int] = {}
        for new_data in new_data_dict.values():
            for target in new_data.imaged:
                imaged_counts[target] = imaged_counts.get(target, 0) + 1

        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                uncertainty = target.search_area / U_NORM
                reward[sat_id] += (
                    self.reward_fn(target.priority + self.weight * (uncertainty**2))
                    / imaged_counts[target]
                )
        return reward


class UncertainCubeReward(UniqueImageReward):
    def __init__(self, weight: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def create_data_store(self, satellite: "Satellite") -> None:
        satellite.data_store = self.data_store_type(
            satellite,
            initial_data=self.initial_data(satellite),
            **self.data_store_kwargs,
        )
        self.cum_reward[satellite.name] = 0.0

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        reward: dict[str, float] = {}
        imaged_counts: dict[Target, int] = {}
        for new_data in new_data_dict.values():
            for target in new_data.imaged:
                imaged_counts[target] = imaged_counts.get(target, 0) + 1

        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                uncertainty = target.search_area / U_NORM
                reward[sat_id] += (
                    self.reward_fn(target.priority + self.weight * (uncertainty**3))
                    / imaged_counts[target]
                )
        return reward


class UniqueUncertainWeightReward(UniqueImageReward):
    def __init__(self, weight: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        reward: dict[str, float] = {}
        imaged_targets = sum([new_data.imaged for new_data in new_data_dict.values()], [])
        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                if target not in self.data.imaged:
                    uncertainty = target.search_area / U_NORM
                    reward[sat_id] += self.reward_fn(
                        target.priority + self.weight * uncertainty
                    ) / imaged_targets.count(target)
        return reward


class UncertainMultiplyReward(UniqueImageReward):
    def create_data_store(self, satellite: "Satellite") -> None:
        satellite.data_store = self.data_store_type(
            satellite,
            initial_data=self.initial_data(satellite),
            **self.data_store_kwargs,
        )
        self.cum_reward[satellite.name] = 0.0

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        reward: dict[str, float] = {}
        imaged_counts: dict[Target, int] = {}
        for new_data in new_data_dict.values():
            for target in new_data.imaged:
                imaged_counts[target] = imaged_counts.get(target, 0) + 1

        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                uncertainty = target.search_area / U_NORM
                reward[sat_id] += (
                    self.reward_fn(target.priority * uncertainty) / imaged_counts[target]
                )
        return reward


class UniqueUncertainMultiplyReward(UniqueImageReward):
    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        reward: dict[str, float] = {}
        imaged_targets = sum([new_data.imaged for new_data in new_data_dict.values()], [])
        for sat_id, new_data in new_data_dict.items():
            reward[sat_id] = 0.0
            for target in new_data.imaged:
                if target not in self.data.imaged:
                    uncertainty = target.search_area / U_NORM
                    reward[sat_id] += self.reward_fn(target.priority * uncertainty) / imaged_targets.count(
                        target
                    )
        return reward


class TotalUncertaintyReward(UniqueImageReward):
    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        reward: dict[str, float] = {}

        already_imaged = set(self.data.imaged)
        new_imaged = sum([new_data.imaged for new_data in new_data_dict.values()], [])
        all_imaged = already_imaged.union(new_imaged)

        total_uncertainty = 0.0
        num_targets = len(self.data.known)
        for target in self.data.known:
            if target not in all_imaged:
                uncertainty = target.search_area / U_NORM
                total_uncertainty += target.priority * uncertainty
                total_uncertainty_norm = total_uncertainty / num_targets

        for sat_id in new_data_dict.keys():
            reward[sat_id] = -self.reward_fn(total_uncertainty_norm)

        return reward

    def reward(self, new_data_dict: dict[str, Data]) -> dict[str, float]:
        reward = self.calculate_reward(new_data_dict)
        for satellite_id, sat_reward in reward.items():
            self.cum_reward[satellite_id] = sat_reward

        for new_data in new_data_dict.values():
            self.data += new_data

        nonzero_reward = {k: v for k, v in reward.items() if v != 0}
        logger.info(f"Total reward: {nonzero_reward}")
        return reward


#############
# SATELLITE OBSERVATIONS


def r_LP_P_uncertain(sat, opp):
    return (
        opp["r_LP_P"](sat.simulator.sim_time)
        if callable(opp["r_LP_P"])
        else opp["r_LP_P"]
    )


def target_angle_uncertain(sat, opp):
    vector_target_spacecraft_P = opp["r_LP_P"](sat.simulator.sim_time) - sat.dynamics.r_BN_P
    vector_target_spacecraft_P_hat = vector_target_spacecraft_P / np.linalg.norm(
        vector_target_spacecraft_P
    )
    return np.arccos(np.dot(vector_target_spacecraft_P_hat, sat.fsw.c_hat_P))
