import argparse
from collections import Counter
import glob
import os
import pathlib
import re
import sys
from typing import Dict, List, Tuple

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configurations import U_NORM


USE_MAIN_CONFIG = True
MAIN_CONFIG = {
    # Folders under data/
    "data_folders": [
        "charge_multiply_filter_higherMinpower_1050_try2",
        "charge_heuristic_slew_filter_higherMinPower_1050_try3",
    ],
    # Policy names to include. Empty list => include all discovered.
    "policies": [],
    # Policy names to exclude.
    "exclude_policies": [],
    # Shielded filter: "any", "true", or "false"
    "shielded": "any",
    # Custom labels per policy name.
    "policy_labels": {
        "RL_base [min_power=0.3]": "RL 0.30",
        "RL_base [min_power=0.4]": "RL 0.40",
        "RL_base [min_power=0.5]": "RL 0.50",
        "32 [min_power=0.3]": "Heur 0.30",
        "32 [min_power=0.4]": "Heur 0.40",
        "32 [min_power=0.5]": "Heur 0.50",
    },
    "target_detail": False,
    "target_heatmap": False,
    # Must contain exactly 2 policies when target_heatmap=True.
    "heatmap_policies": [],
    "output_subdir": "charge_multiply_filter_minpower_try5",
    "debug_df_gaps": False,
    "debug_df_gaps_head": 50,
    # Optional dataset name to append to policy identity, e.g. "filter" or "min_power".
    # Empty string disables splitting policies by extra file content.
    "policy_variant_field": "min_power",
}


def average_df(df: pd.DataFrame, by: str) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    df_mean = df.groupby(by=by).mean().reset_index()
    df_err = df.groupby(by=by).std().reset_index()
    df_err.columns = [f"{col}_err" for col in df_err.columns]
    return pd.concat([df_mean, df_err], axis=1)


def target_uncertainty_from_search_area(search_area):
    search_area = np.asarray(search_area, dtype=float)
    return search_area / U_NORM


def plot_error(ax, x, y, y_err=None, color=None, label=None, **kwargs):
    ax.plot(x, y, label=label, color=color, **kwargs)
    if y_err is not None:
        ax.fill_between(
            x,
            y - y_err,
            y + y_err,
            alpha=0.1,
            zorder=1,
            facecolor=color,
            edgecolor="none",
        )


def decode_if_bytes(value):
    return value.decode() if isinstance(value, bytes) else value


def policy_sort_key(policy: str):
    base_policy, variant_field, variant_value = split_policy_and_variant(policy)
    variant_suffix = "" if variant_value is None else f"{variant_field}={variant_value}"
    base_key = _base_policy_sort_key(base_policy)
    return (*base_key, variant_suffix)


def _base_policy_sort_key(policy: str):
    if policy == "Base_sat":
        return (0, 0)
    reward_match = re.match(r"^Reward_(.+)_sat$", policy)
    if reward_match:
        value = reward_match.group(1)
        reward_map = {
            "01": 0.1,
            "05": 0.5,
            "1": 1.0,
            "2": 2.0,
            "3": 3.0,
            "4": 4.0,
            "5": 5.0,
        }
        return (1, reward_map.get(value, 999))
    if policy in {"Multiply_sat", "Square_sat", "Cube_sat"}:
        nonlinear_order = {"Multiply_sat": 0, "Square_sat": 1, "Cube_sat": 2}
        return (2, nonlinear_order[policy])
    try:
        return (3, float(policy))
    except ValueError:
        return (4, policy)


def policy_label(policy: str) -> str:
    base_policy, variant_field, variant_value = split_policy_and_variant(policy)
    base_label = _base_policy_label(base_policy)
    if variant_value is None:
        return base_label
    return f"{base_label} ({variant_field}={variant_value})"


def _base_policy_label(policy: str) -> str:
    if policy == "Base_sat":
        return r"$\lambda = 0$"
    reward_match = re.match(r"^Reward_(.+)_sat$", policy)
    if reward_match:
        value = reward_match.group(1)
        map_label = {
            "01": "0.1",
            "05": "0.5",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
        }
        if value in map_label:
            return rf"$\lambda = {map_label[value]}$"
    return policy


def policy_label_with_overrides(policy: str, label_overrides: Dict[str, str]) -> str:
    if policy in label_overrides:
        return label_overrides[policy]
    base_policy, variant_field, variant_value = split_policy_and_variant(policy)
    if variant_value is not None and base_policy in label_overrides:
        return f"{label_overrides[base_policy]} ({variant_field}={variant_value})"
    return policy_label(policy)


def decode_scalar(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_scalar(value.item())
        if value.size == 1:
            return decode_scalar(value.reshape(-1)[0])
    return decode_if_bytes(value)


def normalize_variant_value(value) -> str:
    if value is None:
        return ""
    decoded = decode_scalar(value)
    return str(decoded).strip()


def combine_policy_and_variant(policy: str, variant_field: str, variant_value: str) -> str:
    if not variant_field or not variant_value:
        return policy
    return f"{policy} [{variant_field}={variant_value}]"


def split_policy_and_variant(policy: str) -> Tuple[str, str | None, str | None]:
    match = re.match(r"^(.*) \[([^=\]]+)=(.+)\]$", policy)
    if not match:
        return policy, None, None
    return match.group(1), match.group(2), match.group(3)


def get_optional_dataset_value(df_h5, field_name: str):
    if not field_name:
        return None
    if field_name in df_h5:
        return df_h5[field_name][()]
    return None


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip())
    return cleaned.strip("_") or "plot"


def parse_list_args(values: List[str]) -> List[str]:
    parsed = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def parse_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return bool(int(value))
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    return None


def parse_shielded_filter(value) -> str:
    if value is None:
        return "any"
    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value).strip().lower()
    if normalized in {"any", "all", "*", ""}:
        return "any"
    if normalized in {"true", "1", "yes"}:
        return "true"
    if normalized in {"false", "0", "no"}:
        return "false"
    raise ValueError("shielded filter must be one of: any, true, false")


def _to_1d_array(values):
    if values is None:
        return None
    arr = np.array(values).reshape(-1)
    return arr


def _as_float(value):
    if isinstance(value, bytes):
        value = value.decode()
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _is_success_value(value) -> bool:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int, np.floating, float)):
        return float(value) == 1.0
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "success", "s"}
    return False


def _target_name_key(value) -> str:
    decoded = decode_scalar(value)
    if decoded is None:
        return ""
    name = str(decoded).strip()
    if name.lower() in {"", "none", "nan", "charge"}:
        return ""
    return name


def repair_success_from_recorded_images(action_success, action_target_names, recorded_target_names, file_name: str):
    success = np.array([_is_success_value(v) for v in action_success], dtype=bool)
    target_keys = [_target_name_key(v) for v in _to_1d_array(action_target_names)]
    recorded_counts = Counter(
        key for key in (_target_name_key(v) for v in _to_1d_array(recorded_target_names)) if key
    )
    if not recorded_counts:
        return success

    remaining = recorded_counts.copy()
    for is_success, key in zip(success, target_keys):
        if is_success and key and remaining[key] > 0:
            remaining[key] -= 1

    repaired = success.copy()
    repaired_count = 0
    for idx in range(len(target_keys) - 1, -1, -1):
        key = target_keys[idx]
        if not repaired[idx] and key and remaining[key] > 0:
            repaired[idx] = True
            remaining[key] -= 1
            repaired_count += 1

    if repaired_count > 0:
        print(
            f"Info: repaired {repaired_count} action_success entries from recorded target_names "
            f"in {os.path.basename(file_name)}."
        )
    return repaired


def recorded_total_image_count(df_h5):
    if "unique" not in df_h5 or "duplicates" not in df_h5:
        return None
    unique_count = _as_float(df_h5["unique"][()])
    duplicate_count = _as_float(df_h5["duplicates"][()])
    if np.isnan(unique_count) or np.isnan(duplicate_count):
        return None
    return unique_count + duplicate_count


def count_interference_events(df_h5) -> float:
    if "interference" not in df_h5:
        return 0.0

    def count_values(values) -> float:
        arr = _to_1d_array(values)
        if arr is None:
            return 0.0
        return float(np.sum([parse_optional_bool(v) is True for v in arr]))

    node = df_h5["interference"]
    if isinstance(node, h5py.Group):
        total = 0.0
        for key in node.keys():
            total += count_values(node[key])
        return total

    return count_values(node)


def extract_action_codes(df_h5, target_dict, dataset_name):
    if dataset_name in target_dict and isinstance(target_dict[dataset_name], h5py.Group):
        if "action_list" in target_dict[dataset_name]:
            return _to_1d_array(target_dict[dataset_name]["action_list"])

    if f"{dataset_name}/action_list" in target_dict:
        return _to_1d_array(target_dict[f"{dataset_name}/action_list"])

    if "action_list" in df_h5:
        return _to_1d_array(df_h5["action_list"])

    if "actions_list" in df_h5:
        actions_group = df_h5["actions_list"]
        action_keys = list(actions_group.keys())
        if action_keys:
            return _to_1d_array(actions_group[action_keys[0]])

    return None


def align_action_codes_to_target_log(action_codes, target_len: int, file_name: str):
    if action_codes is None:
        return None

    numeric_codes = np.array([_as_float(v) for v in _to_1d_array(action_codes)], dtype=float)
    if len(numeric_codes) == target_len:
        return numeric_codes

    valid = ~np.isnan(numeric_codes)
    image_codes = numeric_codes[valid & (numeric_codes != 0.0)]
    if len(image_codes) == target_len:
        # Full episode action logs include charge action 0; target logs usually store only image actions.
        return image_codes

    print(
        f"Warning: action-code alignment skipped in {os.path.basename(file_name)} "
        f"(action_codes={len(numeric_codes)}, image_actions={len(image_codes)}, target_log={target_len})."
    )
    return None


def extract_episode_actions_and_times(df_h5):
    action_times = None
    actions = None

    if "actions_list_time" in df_h5:
        times_node = df_h5["actions_list_time"]
        if isinstance(times_node, h5py.Group):
            keys = list(times_node.keys())
            if keys:
                action_times = _to_1d_array(times_node[keys[0]])
        else:
            action_times = _to_1d_array(times_node)

    if "actions_list" in df_h5:
        actions_node = df_h5["actions_list"]
        if isinstance(actions_node, h5py.Group):
            keys = list(actions_node.keys())
            if keys:
                actions = _to_1d_array(actions_node[keys[0]])
        else:
            actions = _to_1d_array(actions_node)
    elif "action_list" in df_h5:
        actions = _to_1d_array(df_h5["action_list"])

    if action_times is None or actions is None:
        return None, None

    n = min(len(action_times), len(actions))
    if n == 0:
        return None, None
    return np.array(action_times[:n], dtype=float), np.array(actions[:n])


def infer_target_times_from_actions(df_h5, expected_len: int):
    ep_times, ep_actions = extract_episode_actions_and_times(df_h5)
    if ep_times is None or ep_actions is None:
        return None
    action_codes = np.array([_as_float(v) for v in ep_actions], dtype=float)
    valid = ~np.isnan(action_codes)
    # Charge action is 0. Non-zero actions are treated as imaging actions.
    image_mask = valid & (action_codes != 0.0)
    inferred = ep_times[image_mask]
    if len(inferred) == 0:
        return None
    if expected_len > 0 and len(inferred) != expected_len:
        print(
            f"Warning: inferred imaging actions/time mismatch in {os.path.basename(df_h5.filename)} "
            f"(imaging_actions={len(inferred)} vs target_log={expected_len})."
        )
    return np.array(inferred, dtype=float)


def get_target_dict_array(target_dict, dataset_name: str, field_name: str, file_name: str, required: bool = True):
    direct_key = f"{dataset_name}/{field_name}"
    if direct_key in target_dict:
        return np.array(target_dict[direct_key])

    if dataset_name in target_dict and isinstance(target_dict[dataset_name], h5py.Group):
        subgroup = target_dict[dataset_name]
        if field_name in subgroup:
            return np.array(subgroup[field_name])

    for group_name in target_dict.keys():
        node = target_dict[group_name]
        if isinstance(node, h5py.Group) and field_name in node:
            return np.array(node[field_name])

    found = []
    target_dict.visititems(
        lambda name, obj: found.append(np.array(obj))
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == field_name
        else None
    )
    if found:
        return found[0]

    if required:
        available = []
        target_dict.visititems(
            lambda name, obj: available.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        raise KeyError(
            f"Missing '{field_name}' in target_dict for file {os.path.basename(file_name)}. "
            f"Available datasets: {available[:20]}"
        )
    return None


def get_target_dict_time_array(target_dict, dataset_name: str, file_name: str, expected_len: int):
    time_field_candidates = [
        "action_time",
        "actions_time",
        "target_time",
        "image_time",
        "imaging_time",
        "time",
    ]
    for field in time_field_candidates:
        arr = get_target_dict_array(target_dict, dataset_name, field, file_name, required=False)
        if arr is None:
            continue
        arr = _to_1d_array(arr)
        if arr is not None and len(arr) == expected_len:
            return np.array(arr, dtype=float)
    return None


def get_image_action_count(temp_actions, charge_mode: bool) -> float:
    arr = np.array(temp_actions, dtype=float)
    if charge_mode:
        # In charge mode, index 0 is charge action. Image actions shift by +1.
        return float(np.sum(arr[1:5]))
    return float(np.sum(arr[0:4]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flexible plotting for uncertain reward experiments without case_X blocks."
    )
    parser.add_argument("legacy_target_detail", nargs="?", default=None)
    parser.add_argument("legacy_target_heatmap", nargs="?", default=None)
    parser.add_argument(
        "--target-detail",
        action="store_true",
        help="Generate per-policy target detail plots.",
    )
    parser.add_argument(
        "--target-heatmap",
        action="store_true",
        help="Generate side-by-side KDE heatmap for two selected policies.",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=[],
        help="Policies to plot (space or comma separated). Default: all discovered policies.",
    )
    parser.add_argument(
        "--exclude-policies",
        nargs="*",
        default=[],
        help="Policies to skip (space or comma separated).",
    )
    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="List discovered policies and exit.",
    )
    parser.add_argument(
        "--heatmap-policies",
        nargs="*",
        default=[],
        help="Exactly two policies for --target-heatmap (space or comma separated).",
    )
    parser.add_argument(
        "--data-folders",
        nargs="*",
        default=[
            "dynamic_model_uncertain_speedFix",
            "dynamic_model_uncertain_polynomial",
            "dynamic_model_uncertain_heuristic_time",
            "dynamic_model_uncertain_heuristic_scale",
        ],
        help="Data folder names under data/ to load .h5 files from.",
    )
    parser.add_argument(
        "--output-subdir",
        default="heuristic_scale_flexible",
        help="Output subdir under figures/.",
    )
    parser.add_argument(
        "--shielded",
        default="any",
        help="Filter shielded data: any, true, or false.",
    )
    parser.add_argument(
        "--debug-df-gaps",
        action="store_true",
        help="Print df_gaps debug summaries and head rows.",
    )
    parser.add_argument(
        "--policy-variant-field",
        default="filter",
        help="Optional dataset name to append to the policy identity. Empty disables it.",
    )
    return parser


def resolve_bool_flag(flag: bool, legacy_value) -> bool:
    if flag:
        return True
    if legacy_value is None:
        return False
    return bool(int(legacy_value))


def load_data(
    files: List[str], charge_mode: bool = False, policy_variant_field: str = "filter"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, np.ndarray]]]:
    n_targets = []
    reward = []
    target_distribution = []
    target_type = []
    imaged = []
    policy_name = []
    log_action_image = []
    list_position = []
    total_uncertainty = []
    total_uncertainty_norm = []
    uncertainty = []
    uncertainty_norms = []
    duplicate = []
    pri_area_dot_sum = []
    max_reward = []
    uncertain_init = []
    uncertain_end = []
    pri_unc_init = []
    pri_unc_end = []
    action_time = []
    target_names = []
    target_success = []
    action_times_success = []
    target_names_success = []
    target_status_success = []
    seeds = []
    unique_imaged = []
    orbits_complete = []
    shielded = []
    policy_variants = []
    charge_action_percent = []
    charge_time_percent = []
    interference_count = []

    count_empty = 0
    position_counter = 0
    dict_data: Dict[int, Dict[str, np.ndarray]] = {}

    for file in files:
        try:
            df_h5 = h5py.File(file, "r")
        except OSError:
            count_empty += 1
            continue
        if len(df_h5.keys()) == 0:
            count_empty += 1
            df_h5.close()
            continue

        n_targets.append(df_h5["n_targets"][()])
        if "reward_cum_env" in df_h5:
            reward.append(df_h5["reward_cum_env"][()])
        else:
            reward.append(df_h5["reward_cum"][()])
        target_distribution.append(df_h5["target_distribution"][()])
        duplicate.append(df_h5["duplicates"][()])
        seeds.append(df_h5["seed"][()])
        target_type.append(get_optional_dataset_value(df_h5, "use_moving_targets"))
        unique_imaged.append(df_h5["unique"][()])
        if "policy_name" in df_h5:
            policy_name.append(df_h5["policy_name"][()])
        elif "lookahead" in df_h5:
            policy_name.append(df_h5['lookahead'][()])
        elif "k" in df_h5:
            policy_name.append(df_h5["k"][()])
        else:
            policy_name.append(df_h5["min_time"][()])
        if "orbits_complete" in df_h5:
            orbits_complete.append(float(df_h5["orbits_complete"][()]))
        else:
            orbits_complete.append(np.nan)
        if "shielded" in df_h5:
            shielded.append(parse_optional_bool(df_h5["shielded"][()]))
        else:
            shielded.append(None)
        policy_variants.append(
            normalize_variant_value(get_optional_dataset_value(df_h5, policy_variant_field))
        )
        interference_count.append(count_interference_events(df_h5))

        temp_actions = df_h5["log_actions"]
        log_action_image.append(get_image_action_count(temp_actions, charge_mode))
        total_uncertainty.append(df_h5["total_uncertainty"][()])
        total_uncertainty_norm.append(df_h5["total_uncertainty_norm"][()])
        recorded_target_names = df_h5["target_names"][:]
        target_names.append(recorded_target_names)

        max_reward.append(df_h5["max_reward_end"][()])
        uncertain_init.append(df_h5["total_uncertain_init"][()])
        uncertain_end.append(df_h5["total_uncertain_end"][()])
        pri_unc_init.append(df_h5["priority_uncertain_init"][()])
        pri_unc_end.append(df_h5["priority_uncertain_end"][()])

        action_times_group = df_h5["actions_list_time"]
        data_name = list(action_times_group.keys())[0]
        act_times = np.array(action_times_group[data_name])
        action_time.append(act_times)
        ep_times, ep_actions = extract_episode_actions_and_times(df_h5)
        if ep_times is None or ep_actions is None:
            charge_action_percent.append(np.nan)
            charge_time_percent.append(np.nan)
        else:
            ep_action_codes = np.array([_as_float(v) for v in ep_actions], dtype=float)
            valid_actions = ~np.isnan(ep_action_codes)
            total_action_count = np.sum(valid_actions)
            charge_action_count = np.sum(ep_action_codes[valid_actions] == 0.0)
            if total_action_count > 0:
                charge_action_percent.append(100.0 * charge_action_count / total_action_count)
            else:
                charge_action_percent.append(np.nan)

            if len(ep_times) >= 2:
                sort_idx = np.argsort(ep_times)
                t_sorted = ep_times[sort_idx]
                a_sorted = ep_action_codes[sort_idx]
                dt = np.diff(t_sorted)
                valid_dt = dt > 0
                total_time = np.sum(dt[valid_dt])
                if total_time > 0:
                    charge_time = np.sum(dt[(a_sorted[:-1] == 0.0) & valid_dt])
                    charge_time_percent.append(100.0 * charge_time / total_time)
                else:
                    charge_time_percent.append(np.nan)
            else:
                charge_time_percent.append(np.nan)

        dict_data[position_counter] = {}
        dict_data[position_counter]["target_priorities"] = df_h5["priorities"][:]
        dict_data[position_counter]["target_areas"] = df_h5["areas"][:] / 1e3

        target_dict = df_h5["target_dict"]
        dataset_name = list(target_dict.keys())[0]
        act_target_priorities = get_target_dict_array(target_dict, dataset_name, "target_pri", file)
        act_target_areas = get_target_dict_array(target_dict, dataset_name, "target_area", file)
        act_target_opens = get_target_dict_array(target_dict, dataset_name, "target_open", file, required=False)
        act_target_closes = get_target_dict_array(target_dict, dataset_name, "target_close", file, required=False)
        act_target_success = get_target_dict_array(target_dict, dataset_name, "action_success", file, required=False)
        act_target_names = get_target_dict_array(target_dict, dataset_name, "target_name", file)
        target_action_times = get_target_dict_time_array(
            target_dict, dataset_name, file, expected_len=len(act_target_names)
        )
        if target_action_times is None:
            target_action_times = infer_target_times_from_actions(df_h5, expected_len=len(act_target_names))
        action_codes = extract_action_codes(df_h5, target_dict, dataset_name)
        if act_target_success is None:
            # Target log may contain only imaged actions; treat all target entries as successful image actions.
            act_target_success = np.ones(len(act_target_names), dtype=bool)
        if act_target_opens is None:
            act_target_opens = np.zeros(len(act_target_names), dtype=float)
        if act_target_closes is None:
            act_target_closes = np.zeros(len(act_target_names), dtype=float)

        # Keep target-log arrays aligned (some schemas can differ in length).
        n_target = min(
            len(act_target_priorities),
            len(act_target_areas),
            len(act_target_success),
            len(act_target_names),
            len(act_target_opens),
            len(act_target_closes),
        )
        act_target_priorities = act_target_priorities[:n_target]
        act_target_areas = act_target_areas[:n_target]
        act_target_success = act_target_success[:n_target]
        act_target_names = act_target_names[:n_target]
        act_target_opens = act_target_opens[:n_target]
        act_target_closes = act_target_closes[:n_target]
        if target_action_times is not None:
            n_time = min(len(target_action_times), n_target)
            if len(target_action_times) != n_target:
                print(
                    f"Warning: target action-time alignment adjusted in {os.path.basename(file)} "
                    f"(target_times={len(target_action_times)} vs target_log={n_target})."
                )
            target_action_times = np.array(target_action_times[:n_time], dtype=float)
            act_target_priorities = act_target_priorities[:n_time]
            act_target_areas = act_target_areas[:n_time]
            act_target_success = act_target_success[:n_time]
            act_target_names = act_target_names[:n_time]
            act_target_opens = act_target_opens[:n_time]
            act_target_closes = act_target_closes[:n_time]
        dict_data[position_counter]["act_target_priorities"] = act_target_priorities
        dict_data[position_counter]["act_target_areas"] = act_target_areas / 1e3
        dict_data[position_counter]["act_target_uncertainty"] = target_uncertainty_from_search_area(
            act_target_areas
        )
        dict_data[position_counter]["act_target_opens"] = act_target_opens * 300
        dict_data[position_counter]["act_target_closes"] = act_target_closes * 300
        dict_data[position_counter]["act_target_names"] = act_target_names

        bool_success = repair_success_from_recorded_images(
            act_target_success,
            act_target_names,
            recorded_target_names,
            file,
        )
        charge_mask = np.zeros_like(bool_success, dtype=bool)
        aligned_action_codes = align_action_codes_to_target_log(action_codes, len(charge_mask), file)
        if aligned_action_codes is not None:
            charge_mask = aligned_action_codes == 0.0

        act_target_status = np.where(charge_mask, "charge", np.where(bool_success, "success", "fail"))
        dict_data[position_counter]["act_target_success_raw"] = act_target_success
        dict_data[position_counter]["act_target_success"] = act_target_status
        dict_data[position_counter]["act_target_charge"] = charge_mask

        target_success.append(act_target_status)
        success_mask = act_target_status == "success"
        success_count = float(np.sum(success_mask))
        total_recorded_images = recorded_total_image_count(df_h5)
        if total_recorded_images is not None and total_recorded_images > success_count:
            success_count = total_recorded_images
        imaged.append(success_count)
        priorities = act_target_priorities[success_mask]
        if target_action_times is None:
            target_action_times = np.full(len(success_mask), np.nan)
        action_times_success.append(target_action_times[success_mask])
        target_names_success.append(act_target_names[success_mask])
        target_status_success.append(act_target_status[success_mask])
        uncertainty.append(np.sum(act_target_areas[success_mask] / 1e3))
        uncertainty_norm = target_uncertainty_from_search_area(act_target_areas[success_mask])
        uncertainty_norms.append(np.sum(uncertainty_norm))
        pri_area_dot = [np.dot(a, b) for a, b in zip(priorities, uncertainty_norm)]
        pri_area_dot_sum.append(np.sum(pri_area_dot))

        list_position.append(position_counter)
        position_counter += 1
        df_h5.close()

    print("Number of empty files:", count_empty)

    df = pd.DataFrame(
        {
            "n_targets": n_targets,
            "reward": reward,
            "target_distribution": target_distribution,
            "duplicate": duplicate,
            "seeds": seeds,
            "target_type": target_type,
            "imaged": imaged,
            "unique_imaged": unique_imaged,
            "uncertainty": uncertainty,
            "policy_name": policy_name,
            "log_action_image": log_action_image,
            "total_uncertainty": total_uncertainty,
            "total_uncertainty_norm": total_uncertainty_norm,
            "uncertainty_norms": uncertainty_norms,
            "position": list_position,
            "pri_area_dot_sum": pri_area_dot_sum,
            "max_reward": max_reward,
            "uncertain_init": uncertain_init,
            "uncertain_end": uncertain_end,
            "pri_unc_init": pri_unc_init,
            "pri_unc_end": pri_unc_end,
            "action_time": action_time,
            "target_names": target_names,
            "target_success": target_success,
            "action_times_success": action_times_success,
            "target_names_success": target_names_success,
            "target_status_success": target_status_success,
            "orbits_complete": orbits_complete,
            "shielded": shielded,
            "policy_variant": policy_variants,
            "charge_action_percent": charge_action_percent,
            "charge_time_percent": charge_time_percent,
            "interference_count": interference_count,
        }
    )
    df["orbits_complete"] = pd.to_numeric(df["orbits_complete"], errors="coerce").clip(upper=3)
    df["alive"] = df["orbits_complete"] / 3
    df["image_success"] = df["imaged"] / df["log_action_image"]
    df["policy_name"] = df["policy_name"].apply(lambda x: str(decode_if_bytes(x)))
    df["policy_variant"] = df["policy_variant"].apply(normalize_variant_value)
    df["policy_base"] = df["policy_name"]
    df["policy_name"] = df.apply(
        lambda row: combine_policy_and_variant(
            row["policy_name"], policy_variant_field, row["policy_variant"]
        ),
        axis=1,
    )
    print("Discovered policies:", sorted(pd.unique(df["policy_name"]), key=policy_sort_key))

    df_exploded = df.explode(
        ["target_names_success", "action_times_success", "target_status_success"],
        ignore_index=True,
    )
    df_exploded["policy_name"] = df_exploded["policy_name"].astype(str)
    df_exploded["target_names_success"] = df_exploded["target_names_success"].apply(decode_if_bytes)
    df_exploded["action_times_success"] = df_exploded["action_times_success"].astype(float)
    df_exploded["target_status_success"] = df_exploded["target_status_success"].astype(str)
    df_exploded = df_exploded[df_exploded["target_status_success"] == "success"].copy()
    df_exploded = df_exploded.sort_values(
        by=[
            "policy_name",
            "shielded",
            "seeds",
            "n_targets",
            "target_names_success",
            "action_times_success",
        ]
    )
    df_exploded["time_gap"] = df_exploded.groupby(
        ["policy_name", "shielded", "seeds", "n_targets", "target_names_success"],
        dropna=False,
    )["action_times_success"].diff()
    df_gaps = df_exploded.dropna(subset=["time_gap"])

    df["uncertain_diff"] = df["uncertain_end"] - df["uncertain_init"]
    df["pri_unc_diff"] = df["pri_unc_end"] - df["pri_unc_init"]
    df["uncertain_diff_norm"] = df["uncertain_diff"] / df["uncertain_init"]
    df["pri_unc_diff_norm"] = df["pri_unc_diff"] / df["pri_unc_init"]
    df = df.sort_values(by="n_targets")

    return df, df_gaps, dict_data


def main():
    parser = build_parser()
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 9})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linewidth"] = 0.2
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["legend.framealpha"] = 0.8

    legend_kwargs = dict(
        ncol=3,
        handlelength=1,
        borderpad=0.4,
        labelspacing=0.2,
        columnspacing=0.6,
        handletextpad=0.4,
    )

    target_detail_cli = resolve_bool_flag(args.target_detail, args.legacy_target_detail)
    target_heatmap_cli = resolve_bool_flag(args.target_heatmap, args.legacy_target_heatmap)

    if USE_MAIN_CONFIG:
        data_folders = MAIN_CONFIG.get("data_folders", args.data_folders)
        requested_policies_cfg = MAIN_CONFIG.get("policies", [])
        excluded_policies_cfg = MAIN_CONFIG.get("exclude_policies", [])
        label_overrides = MAIN_CONFIG.get("policy_labels", {})
        target_detail = MAIN_CONFIG.get("target_detail", target_detail_cli)
        target_heatmap = MAIN_CONFIG.get("target_heatmap", target_heatmap_cli)
        heatmap_policies_cfg = MAIN_CONFIG.get("heatmap_policies", [])
        output_subdir = MAIN_CONFIG.get("output_subdir", args.output_subdir)
        shielded_filter = parse_shielded_filter(MAIN_CONFIG.get("shielded", "any"))
        charge_mode = True
        debug_df_gaps = bool(MAIN_CONFIG.get("debug_df_gaps", False))
        debug_df_gaps_head = int(MAIN_CONFIG.get("debug_df_gaps_head", 50))
        policy_variant_field = str(MAIN_CONFIG.get("policy_variant_field", args.policy_variant_field)).strip()
    else:
        data_folders = args.data_folders
        requested_policies_cfg = parse_list_args(args.policies)
        excluded_policies_cfg = parse_list_args(args.exclude_policies)
        label_overrides = {}
        target_detail = target_detail_cli
        target_heatmap = target_heatmap_cli
        heatmap_policies_cfg = parse_list_args(args.heatmap_policies)
        output_subdir = args.output_subdir
        shielded_filter = parse_shielded_filter(args.shielded)
        charge_mode = True
        debug_df_gaps = bool(args.debug_df_gaps)
        debug_df_gaps_head = 50
        policy_variant_field = str(args.policy_variant_field).strip()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    data_root = repo_root / "data"
    figure_root = repo_root / "figures" / output_subdir

    files: List[str] = []
    for folder in data_folders:
        files += glob.glob(str(data_root / folder / "*.h5"))
    if not files:
        raise RuntimeError(f"No .h5 files found in folders: {data_folders}")

    print(f"\nCharge mode: {charge_mode}")
    print(f"Policy variant field: {policy_variant_field or 'disabled'}")
    df, df_gaps, dict_data = load_data(
        files, charge_mode=charge_mode, policy_variant_field=policy_variant_field
    )

    if shielded_filter != "any":
        shielded_bool = shielded_filter == "true"
        before_n = len(df)
        df = df[df["shielded"] == shielded_bool].copy()
        df_gaps = df_gaps[df_gaps["shielded"] == shielded_bool].copy()
        print(
            f"\nApplied shielded filter={shielded_filter}: "
            f"{before_n} -> {len(df)} rows (missing shielded excluded)"
        )
        if df.empty:
            raise RuntimeError(f"No rows left after applying shielded={shielded_filter} filter.")

    if debug_df_gaps:
        print("\n================ df_gaps DEBUG ================")
        print("df_gaps rows:", len(df_gaps))
        print("df_gaps columns:", list(df_gaps.columns))
        if len(df_gaps) > 0:
            print(
                "target_status_success unique values:",
                sorted(df_gaps["target_status_success"].dropna().unique().tolist()),
            )
            summary = (
                df_gaps.groupby(["policy_name", "shielded", "n_targets", "seeds"])
                .size()
                .reset_index(name="gap_rows")
                .sort_values(["policy_name", "shielded", "n_targets", "seeds"])
            )
            print("\nGap-row counts by policy/shielded/n_targets/seeds:")
            print(summary.to_string(index=False))
            df_gaps_sorted = df_gaps.sort_values(
                by=[
                    "policy_name",
                    "shielded",
                    "n_targets",
                    "seeds",
                    "target_names_success",
                    "action_times_success",
                ]
            )
            debug_cols = [
                "policy_name",
                "shielded",
                "seeds",
                "n_targets",
                "target_names_success",
                "target_status_success",
                "action_times_success",
                "time_gap",
                "duplicate",
                "imaged",
                "log_action_image",
            ]
            debug_cols = [c for c in debug_cols if c in df_gaps_sorted.columns]
            df_gaps_debug = df_gaps_sorted[debug_cols].copy()
            print(f"\nFirst {debug_df_gaps_head} rows:")
            print(df_gaps_debug.head(debug_df_gaps_head).to_string(index=False))
            debug_path = figure_root / "debug_df_gaps.csv"
            os.makedirs(debug_path.parent, exist_ok=True)
            df_gaps_debug.to_csv(debug_path, index=False)
            print(f"\nSaved full df_gaps debug CSV: {debug_path}")
        print("==============================================\n")

    available_policies = sorted(pd.unique(df["policy_name"]), key=policy_sort_key)
    requested_policies = requested_policies_cfg
    excluded_policies = set(excluded_policies_cfg)

    if args.list_policies:
        print("\nAvailable policies:")
        for p in available_policies:
            print(f"  - {p}")
        return

    if requested_policies:
        selected_policies = [p for p in requested_policies if p in available_policies]
        missing = [p for p in requested_policies if p not in available_policies]
        if missing:
            print("Requested policies not found:", missing)
    else:
        selected_policies = available_policies

    selected_policies = [p for p in selected_policies if p not in excluded_policies]
    if not selected_policies:
        raise RuntimeError("No policies selected after applying include/exclude filters.")

    print("\nSelected policies:")
    for p in selected_policies:
        print(f"  - {p} ({policy_label_with_overrides(p, label_overrides)})")

    dict_cases_average = {}
    dict_cases_gaps = {}
    dict_cases_gaps_raw = {}
    dict_cases = {}
    dict_cases_label = {}
    dict_cases_style = {}
    dict_cases_policy = {}
    dict_policy_label = {}

    for policy in selected_policies:
        case_df = df[df["policy_name"] == policy].copy()
        if case_df.empty:
            continue
        case_gaps_df = df_gaps[df_gaps["policy_name"] == policy].copy()
        policy_label = policy_label_with_overrides(policy, label_overrides)
        dict_policy_label[policy] = policy_label
        dict_cases[policy] = case_df
        print("------------------------------------------------------")
        print(f"Policy {policy} ({policy_label}): {len(case_df)} cases")
        print(case_df.groupby("n_targets")["alive"].describe())

        shielded_splits = [
            (True, "Shielded", "-"),
            (False, "Non-shielded", "--"),
        ]
        for shielded_value, shielded_text, line_style in shielded_splits:
            split_df = case_df[case_df["shielded"] == shielded_value].copy()
            if split_df.empty:
                continue
            case_key = f"{policy}__shielded_{int(shielded_value)}"
            dict_cases_average[case_key] = average_df(split_df, by="n_targets")
            dict_cases_label[case_key] = f"{policy_label} ({shielded_text})"
            dict_cases_style[case_key] = line_style
            dict_cases_policy[case_key] = policy

            split_gaps_df = case_gaps_df[case_gaps_df["shielded"] == shielded_value].copy()
            if not split_gaps_df.empty:
                dict_cases_gaps[case_key] = average_df(split_gaps_df, by="n_targets")
                dict_cases_gaps_raw[case_key] = split_gaps_df

        unknown_df = case_df[case_df["shielded"].isna()].copy()
        if not unknown_df.empty:
            case_key = f"{policy}__shielded_unknown"
            dict_cases_average[case_key] = average_df(unknown_df, by="n_targets")
            dict_cases_label[case_key] = f"{policy_label} (Shielded Unknown)"
            dict_cases_style[case_key] = ":"
            dict_cases_policy[case_key] = policy

            unknown_gaps_df = case_gaps_df[case_gaps_df["shielded"].isna()].copy()
            if not unknown_gaps_df.empty:
                dict_cases_gaps[case_key] = average_df(unknown_gaps_df, by="n_targets")
                dict_cases_gaps_raw[case_key] = unknown_gaps_df

    plot_cases = [c for c in dict_cases_average.keys()]
    gap_plot_cases = [c for c in dict_cases_gaps.keys()]
    plot_policies = [p for p in selected_policies if p in dict_cases]

    if not plot_cases:
        raise RuntimeError("No policies with usable data to plot.")

    base_colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [base_colors[i % len(base_colors)] for i in range(len(plot_policies))]
    color_by_policy = dict(zip(plot_policies, colors))

    def save_line_plot(filename: str, y_col: str, y_label: str, figsize=(6, 3.5)):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2), constrained_layout=True)
        for case in plot_cases:
            avg_df = dict_cases_average[case]
            yerr = f"{y_col}_err" if f"{y_col}_err" in avg_df else None
            plot_error(
                ax,
                avg_df["n_targets"],
                avg_df[y_col],
                y_err=avg_df[yerr] if yerr else None,
                label=dict_cases_label[case],
                color=color_by_policy[dict_cases_policy[case]],
                linestyle=dict_cases_style[case],
            )
        ax.set_ylabel(y_label)
        ax.set_xlabel("Requests |R|")
        ax.legend(loc="lower center", ncol=2, fontsize=6, framealpha=0.9)
        path_figure = figure_root / filename
        os.makedirs(path_figure.parent, exist_ok=True)
        fig.savefig(path_figure)
        plt.close(fig)

    save_line_plot("Metrics.pdf", "reward", "Reward")
    save_line_plot("Imaged.pdf", "imaged", "Imaged")
    save_line_plot("Unique_Imaged.pdf", "unique_imaged", "Imaged")
    save_line_plot("LogActionImage.pdf", "log_action_image", "Image Action")
    save_line_plot("ImageSuccess.pdf", "image_success", "Image Success Rate")
    save_line_plot("TotalUncertaintyNorm.pdf", "total_uncertainty_norm", "Total Uncertainty Norm")
    save_line_plot("Uncertainty.pdf", "uncertainty", "Uncertainty")
    save_line_plot("Duplicates.pdf", "duplicate", "Duplicates", figsize=(3, 2))
    save_line_plot("PriUnc_cum.pdf", "pri_area_dot_sum", "Priority-Uncertainty Cumulative")
    save_line_plot("Uncertainty_diff.pdf", "uncertain_diff", "Uncertainty Increase for Accessed Targets")
    save_line_plot("Weighted_uncertainty.pdf", "pri_unc_diff", "Weighted Uncertainty Increase")
    save_line_plot("Uncertainty_norm.pdf", "uncertain_diff_norm", "Normalized Uncertainty Increase")
    save_line_plot("Weighted_uncertainty_norm.pdf", "pri_unc_diff_norm", "Normalized Weighted Uncertainty Increase")
    save_line_plot("Alive.pdf", "alive", "Fraction of 3 Orbits Alive")
    save_line_plot("ChargeActionPercent.pdf", "charge_action_percent", "Charge Actions [% of total]")
    save_line_plot("ChargeTimePercent.pdf", "charge_time_percent", "Charge Time [% of action timeline]")
    save_line_plot("InterferenceCount.pdf", "interference_count", "Interference Count")

    # Survival-style orbit completion curve (alive percentage vs completed orbits)
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    for case in plot_cases:
        policy = dict_cases_policy[case]
        shielded_value = None
        if case.endswith("__shielded_1"):
            shielded_value = True
        elif case.endswith("__shielded_0"):
            shielded_value = False
        orbits_df = dict_cases[policy]
        if shielded_value is True:
            orbits_df = orbits_df[orbits_df["shielded"] == True]
        elif shielded_value is False:
            orbits_df = orbits_df[orbits_df["shielded"] == False]
        else:
            orbits_df = orbits_df[orbits_df["shielded"].isna()]
        orbits = orbits_df["orbits_complete"].dropna().to_numpy()
        if len(orbits) == 0:
            continue
        sorted_data = np.sort(orbits)
        total_cases = len(sorted_data)
        aliveness_percentage = 100 * (np.arange(total_cases, 0, -1) / total_cases)
        x_vals = np.concatenate(([0.0], sorted_data))
        y_vals = np.concatenate(([100.0], aliveness_percentage))
        ax.step(
            x_vals,
            y_vals,
            where="post",
            label=dict_cases_label[case],
            color=color_by_policy[policy],
            linestyle=dict_cases_style[case],
        )
    ax.set_xlim(0, 3)
    ax.set_xlabel("Orbits completed")
    ax.set_ylabel("Alive percentage (%)")
    ax.legend(loc="best")
    path_figure = figure_root / "Alive_orbit.pdf"
    os.makedirs(path_figure.parent, exist_ok=True)
    fig.savefig(path_figure)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 0.38))
    handles, labels = plt.gca().get_legend_handles_labels()
    ref_fig, ref_ax = plt.subplots(1, 1, figsize=(6, 3.5))
    for case in plot_cases:
        ref_ax.plot(
            [],
            [],
            label=dict_cases_label[case],
            color=color_by_policy[dict_cases_policy[case]],
            linestyle=dict_cases_style[case],
        )
    ax.legend(*ref_ax.get_legend_handles_labels(), loc="center", **legend_kwargs)
    ax.axis("off")
    path_figure = figure_root / "legend.pdf"
    os.makedirs(path_figure.parent, exist_ok=True)
    fig.savefig(path_figure)
    plt.close(ref_fig)
    plt.close(fig)

    fig, ax_arr = plt.subplots(1, 2, figsize=(6, 2), constrained_layout=True)
    for case in plot_cases:
        avg_df = dict_cases_average[case]
        color = color_by_policy[dict_cases_policy[case]]
        linestyle = dict_cases_style[case]
        plot_error(
            ax_arr[0],
            avg_df["n_targets"],
            avg_df["unique_imaged"],
            y_err=avg_df["unique_imaged_err"],
            label=dict_cases_label[case],
            color=color,
            linestyle=linestyle,
        )
        plot_error(
            ax_arr[1],
            avg_df["n_targets"],
            avg_df["duplicate"],
            y_err=avg_df["duplicate_err"],
            label=dict_cases_label[case],
            color=color,
            linestyle=linestyle,
        )
    ax_arr[0].set_ylabel("Unique Imaged Targets")
    ax_arr[0].set_xlabel("Requests |R|")
    ax_arr[1].set_ylabel("Duplicated Targets")
    ax_arr[1].set_xlabel("Requests |R|")
    path_figure = figure_root / "Unique_Duplicate.pdf"
    os.makedirs(path_figure.parent, exist_ok=True)
    fig.savefig(path_figure)
    plt.close(fig)

    fig, ax_arr = plt.subplots(2, 2, figsize=(6, 4), constrained_layout=True)
    for case in plot_cases:
        avg_df = dict_cases_average[case]
        color = color_by_policy[dict_cases_policy[case]]
        label = dict_cases_label[case]
        linestyle = dict_cases_style[case]
        plot_error(ax_arr[0][0], avg_df["n_targets"], avg_df["pri_area_dot_sum"], avg_df["pri_area_dot_sum_err"], color, label, linestyle=linestyle)
        plot_error(ax_arr[0][1], avg_df["n_targets"], avg_df["reward"], avg_df["reward_err"], color, label, linestyle=linestyle)
        plot_error(ax_arr[1][0], avg_df["n_targets"], avg_df["uncertainty_norms"], avg_df["uncertainty_norms_err"], color, label, linestyle=linestyle)
        plot_error(ax_arr[1][1], avg_df["n_targets"], avg_df["imaged"], avg_df["imaged_err"], color, label, linestyle=linestyle)
    ax_arr[0][0].set_ylabel("Weighted Uncertainty")
    ax_arr[0][1].set_ylabel("Priority")
    ax_arr[1][0].set_ylabel("Uncertainty")
    ax_arr[1][1].set_ylabel("Imaged Targets")
    for row in ax_arr:
        for ax in row:
            ax.set_xlabel("Requests |R|")
    path_figure = figure_root / "Combined.pdf"
    os.makedirs(path_figure.parent, exist_ok=True)
    fig.savefig(path_figure)
    plt.close(fig)

    if gap_plot_cases:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)
        for case in gap_plot_cases:
            avg_df = dict_cases_gaps[case]
            plot_error(
                ax,
                avg_df["n_targets"],
                avg_df["time_gap"],
                y_err=avg_df["time_gap_err"] if "time_gap_err" in avg_df else None,
                label=dict_cases_label[case],
                color=color_by_policy[dict_cases_policy[case]],
                linestyle=dict_cases_style[case],
            )
        ax.set_ylabel("Time gaps for re-imaged targets [s]")
        ax.set_xlabel("Requests |R|")
        path_figure = figure_root / "Time_gaps.pdf"
        os.makedirs(path_figure.parent, exist_ok=True)
        fig.savefig(path_figure)
        plt.close(fig)

        bin_edges = [0, 5000, 10000, 15000]
        for case in gap_plot_cases:
            df_policy = dict_cases_gaps_raw[case]
            counts, edges = np.histogram(df_policy["time_gap"], bins=bin_edges)
            total_counts = counts.sum()
            percentages = counts / total_counts * 100 if total_counts > 0 else np.zeros_like(counts, dtype=float)
            df_hist = pd.DataFrame(
                {
                    "bin_start": edges[:-1],
                    "bin_end": edges[1:],
                    "count": counts,
                    "percentage": percentages,
                }
            )
            case_name = sanitize_filename(dict_cases_label[case])
            csv_path = figure_root / f"time_gap_{case_name}.csv"
            os.makedirs(csv_path.parent, exist_ok=True)
            df_hist.to_csv(csv_path, index=False)

            plt.figure(figsize=(5, 3))
            plt.hist(df_policy["time_gap"], bins=bin_edges, color="blue", alpha=0.6)
            plt.xlabel("Time gap between repeated images [s]")
            plt.ylabel("Count")
            plt.title(dict_cases_label[case])
            plt.xlim(0, 13000)
            plt.tight_layout()
            plt.savefig(figure_root / f"time_gap_{case_name}.pdf")
            plt.close()

    if target_detail:
        for policy in plot_policies:
            pri = []
            area = []
            for pos in dict_cases[policy]["position"]:
                pri += list(dict_data[pos]["target_priorities"])
                area += list(dict_data[pos]["target_areas"])
            label = dict_policy_label[policy]
            policy_name = sanitize_filename(policy)

            plt.figure(figsize=(8, 6))
            sns.kdeplot(x=pri, y=area, cmap="viridis", fill=True)
            plt.title(f"Case {label}")
            plt.xlabel("Target priority")
            plt.ylabel("Target length [km]")
            path_figure = figure_root / "Target_details" / f"Pri_area_{policy_name}.pdf"
            os.makedirs(path_figure.parent, exist_ok=True)
            plt.savefig(path_figure)
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.scatter(pri, area, alpha=0.5, s=10)
            plt.title(f"Case {label}")
            plt.xlabel("Target priority")
            plt.ylabel("Target length [km]")
            path_figure = figure_root / "Target_details" / f"Pri_area_{policy_name}_scatter.pdf"
            os.makedirs(path_figure.parent, exist_ok=True)
            plt.savefig(path_figure)
            plt.close()

            act_pri = []
            act_area = []
            act_close = []
            for pos in dict_cases[policy]["position"]:
                act_pri += list(dict_data[pos]["act_target_priorities"])
                act_area += list(dict_data[pos]["act_target_areas"])
                act_close += list(dict_data[pos]["act_target_closes"])

            plt.figure(figsize=(8, 6))
            sns.kdeplot(x=act_pri, y=act_area, cmap="viridis", fill=True)
            plt.title(f"Case {label}")
            plt.xlabel("Target priority")
            plt.ylabel("Target length [km]")
            path_figure = figure_root / "Target_details" / f"Pri_area_act_{policy_name}.pdf"
            os.makedirs(path_figure.parent, exist_ok=True)
            plt.savefig(path_figure)
            plt.close()

            plt.figure(figsize=(8, 6))
            sns.kdeplot(x=act_pri, y=act_close, cmap="viridis", fill=True)
            plt.title(f"Case {label}")
            plt.xlabel("Target priority")
            plt.ylabel("Target close window [sec]")
            plt.ylim(0, 300)
            path_figure = figure_root / "Target_details" / f"Pri_close_act_{policy_name}.pdf"
            os.makedirs(path_figure.parent, exist_ok=True)
            plt.savefig(path_figure)
            plt.close()

    if target_heatmap:
        heatmap_policies = heatmap_policies_cfg
        if not heatmap_policies:
            heatmap_policies = plot_policies[:2]
        if len(heatmap_policies) != 2:
            raise RuntimeError("Heatmap requires exactly 2 policies. Use --heatmap-policies p1 p2.")
        for p in heatmap_policies:
            if p not in dict_cases:
                raise RuntimeError(f"Heatmap policy not found in selected data: {p}")

        fig, axes = plt.subplots(1, 2, figsize=(6, 2), constrained_layout=True, sharey=True)
        case_max_densities = []
        for i, policy in enumerate(heatmap_policies):
            act_pri = []
            act_area = []
            for pos in dict_cases[policy]["position"]:
                act_pri += list(dict_data[pos]["act_target_priorities"])
                act_area += list(dict_data[pos]["act_target_uncertainty"])

            kde = sns.kdeplot(x=act_pri, y=act_area, cmap="viridis", fill=True, ax=axes[i])
            max_density = max([c.get_array().max() for c in kde.collections if c.get_array() is not None])
            case_max_densities.append(max_density)

            axes[i].set_xlabel("Target Priority")
            axes[i].set_ylim(-0.2, 2.5)
            axes[i].set_title(dict_policy_label[policy])
            axes[i].set_xticks(np.arange(0.0, 1.01, 0.25))
            axes[i].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        axes[0].set_ylabel("Target Uncertainty")

        last_case_max = case_max_densities[-1]
        sm = mpl.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0, vmax=last_case_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.8)
        cbar.set_label("Density (% of case max)")
        tick_percent = np.arange(0, 101, 20)
        tick_values = last_case_max * tick_percent / 100
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{p:.0f}%" for p in tick_percent])

        p1 = sanitize_filename(heatmap_policies[0])
        p2 = sanitize_filename(heatmap_policies[1])
        path_figure = figure_root / "Target_details" / f"Pri_unc_heatmap_{p1}_vs_{p2}.pdf"
        os.makedirs(path_figure.parent, exist_ok=True)
        plt.savefig(path_figure)
        plt.close(fig)


if __name__ == "__main__":
    main()
