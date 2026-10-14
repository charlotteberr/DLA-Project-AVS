import numpy as np
from bsk_rl.utils.orbital import random_orbit
from Basilisk.utilities import orbitalMotion

def random_disturbance_vector(magnitude_disturbance, seed = None):
    disturbance_rand_vector = np.random.normal(size=3)
    disturbance_rand_unit_vector = (
        disturbance_rand_vector/np.linalg.norm(disturbance_rand_vector)
        )
    disturbance_vector = (
        disturbance_rand_unit_vector * magnitude_disturbance
        )
    return disturbance_vector

ALTITUDE = 800  # Altitude in km - CHECKED
T_ORBIT = (
    2
    * np.pi
    * np.sqrt((orbitalMotion.REQ_EARTH + ALTITUDE) ** 3 / orbitalMotion.MU_EARTH)
)

U_NORM = (100.0 * 1e3 + 46e3 / 3600 * T_ORBIT * 3) # max search area in meters, for normalization

STEERING_2Hz_POWER = dict(
    name="steering_2hz_safety",
    sat_args=dict(
        oe=lambda: random_orbit(
            alt=ALTITUDE,  # 800 km altitude - CHECKED
            i=45,  # 45 degrees inclination - CHECKED
        ),
        imageAttErrorRequirement=0.01,
        imageRateErrorRequirement=0.01,
        batteryStorageCapacity=80.0 * 3600,  # turn off aliveness checker or increase capacity
        storedCharge_Init=lambda: np.random.uniform(0.4, 1.0) * 80.0 * 3600,
        u_max=0.4,  # More realistic values than 1.0
        K1=0.25,  # Control gain - CHECKED
        K3=3.0,  # Control gain - CHECKED
        servo_P=150 / 5,
        nHat_B=np.array([0, 0, -1]),
        omega_max=np.radians(
            5.0
        ),  # Maximum rate command in degrees per second - CHECKED
        imageTargetMinimumElevation=np.arctan(
            800 / 500
        ),  # 58 degrees minimum elevation - CHECKED
        rwBasePower=20,
        disturbance_vector=lambda: random_disturbance_vector(0.001),
        maxWheelSpeed=1500,
        dataStorageCapacity=20 * 8e6 * 100,
        transmitterPacketSize=0,
    ),
    sim_rate=0.5,
)

RANDOM_INIT = {
    "storage_low_percent": 0,
    "storage_high_percent": 0.01,
    "rw_speed_low": -900,
    "rw_speed_high": 900,
    "disturbance_magnitude": 0.001, #1mNm
}

SIM_PARAMS = {
    "n_targets": (100, 10000),
    "horizon": 3,
    "target_distribution": "uniform",
    "seed": 0,
    "n_ahead_act": 32,
    "n_ahead_observe": 32,
    "max_step_duration": 300.0,
}

TRAINING_PARAMS = {
    "failure_penalty": -3,
#     "learning_rate": 0.00003,
#     "gamma": 0.999,
#     "train_batch_size": 10000,
#     "minibatch_size": 250,
#     "num_sgd_iter": 50
}
