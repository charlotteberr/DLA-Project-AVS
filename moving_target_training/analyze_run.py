#!/usr/bin/env python3
"""Summarize an RLlib training run from result.json or progress.csv.

    python3 analyze_run.py <path> [--window 10] [--every 20] [--out clean.csv]
                                  [--metrics alive,imaged,desat_action]

<path> may be a run directory, a result.json, or a progress.csv.

result.json is preferred and is picked automatically when present. progress.csv
is fragile for two reasons found the hard way:

  1. Every time Ray reloads the algorithm, the key order changes, so rows stop
     lining up with the header written at iteration 1.
  2. reward_per_orbit and orbits_complete_partial_only are written conditionally
     by the training callback, so rows can have fewer fields than the header.

For progress.csv this script tries each plausible key order against each set of
missing optional keys and keeps the assignment that satisfies known invariants
(the mean return is logged three times under different names and must agree;
survival fractions must sit in [0, 1]; mean episode length cannot exceed max).
Rows it cannot resolve are counted and reported rather than silently misread.

Standard library only, so it runs without activating the venv.
"""

import argparse
import csv
import json
import os
import sys
from itertools import combinations

# Written only on some iterations by the metrics callback in
# dynamic_targets_time_uncertain_charge_train.py.
OPTIONAL_KEYS = (
    "env_runners/reward_per_orbit",
    "env_runners/orbits_complete_partial_only",
)

# label, key, format
HEADLINE = [
    ("Survived", "env_runners/alive", "{:.1%}"),
    ("Mean return", "env_runners/episode_return_mean", "{:+.3f}"),
    ("Best return", "env_runners/episode_return_max", "{:+.3f}"),
    ("Orbits completed", "env_runners/orbits_complete", "{:.2f}"),
    ("Targets imaged", "env_runners/imaged", "{:.1f}"),
    ("Mean priority", "env_runners/mean_priority", "{:.3f}"),
    ("Duplicates", "env_runners/duplicates", "{:.2f}"),
    ("Missed", "env_runners/missed", "{:.1f}"),
    ("Image actions", "env_runners/image_action", "{:.1f}"),
    ("Charge actions", "env_runners/charge_action", "{:.1f}"),
    ("Desat actions", "env_runners/desat_action", "{:.1f}"),
    ("Peak wheel load", "env_runners/rw_speeds_max_fraction", "{:.3f}"),
    ("Battery check OK", "env_runners/battery_status_valid", "{:.1%}"),
    ("Wheel check OK", "env_runners/rw_status_valid", "{:.1%}"),
    ("Mean battery", "env_runners/battery_mean", "{:.3f}"),
    ("Min battery", "env_runners/battery_min", "{:.3f}"),
    ("Episode length", "env_runners/episode_len_mean", "{:.1f}"),
    ("Entropy", "learners/default_policy/entropy", "{:.3f}"),
    ("VF explained var", "learners/default_policy/vf_explained_var", "{:.3f}"),
    ("Grad norm", "learners/default_policy/gradients_default_optimizer_global_norm", "{:.3f}"),
]

ITER = "training_iteration"
RETURN_MIRRORS = (
    "env_runners/episode_return_mean",
    "env_runners/agent_episode_returns_mean/default_agent",
    "env_runners/module_episode_returns_mean/default_policy",
)


# ---------------------------------------------------------------- loading

def flatten(obj, prefix=""):
    """Flatten nested dicts with '/' separators, the way Ray writes them."""
    out = {}
    for key, value in obj.items():
        full = prefix + key
        if isinstance(value, dict):
            out.update(flatten(value, full + "/"))
        else:
            out[full] = value
    return out


def coerce(text):
    """Turn a CSV field into a float, bool or string."""
    if text in ("", "nan", "None"):
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


def load_json(path):
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(flatten(json.loads(line)))
                except json.JSONDecodeError:
                    pass  # a truncated final line while the job is still running
    return records, []


def candidate_orders(header):
    """Key orders Ray has actually been observed to write."""
    orders = [list(header)]
    groups, seen = {}, []
    for key in header:
        group = key.split("/")[0] if "/" in key else ""
        if group not in groups:
            groups[group] = []
            seen.append(group)
        groups[group].append(key)
    orders.append([k for g in seen for k in sorted(groups[g])])
    orders.append(sorted(header))
    deduped = []
    for order in orders:
        if order not in deduped:
            deduped.append(order)
    return deduped


def plausible(record):
    """Reject an assignment that breaks something we know must hold."""
    try:
        iteration = int(float(record[ITER]))
    except (KeyError, TypeError, ValueError):
        return False
    if not 0 < iteration < 10 ** 6:
        return False

    mirrors = [record.get(k) for k in RETURN_MIRRORS]
    mirrors = [v for v in mirrors if isinstance(v, float)]
    if len(mirrors) >= 2 and max(abs(v - mirrors[0]) for v in mirrors) > 1e-6:
        return False

    for key in ("env_runners/alive", "env_runners/rw_status_valid",
                "env_runners/battery_status_valid"):
        value = record.get(key)
        if isinstance(value, float) and not -1e-9 <= value <= 1 + 1e-9:
            return False

    mean = record.get("env_runners/episode_len_mean")
    top = record.get("env_runners/episode_len_max")
    if isinstance(mean, float) and isinstance(top, float) and mean > top + 1e-6:
        return False
    return True


def load_csv(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], ["progress.csv is empty"]

    header, body = rows[0], rows[1:]
    orders = candidate_orders(header)
    present = [k for k in OPTIONAL_KEYS if k in header]
    # Each way the optional keys could be missing, fewest-missing first.
    drops = [set()]
    for size in range(1, len(present) + 1):
        drops.extend(set(combo) for combo in combinations(present, size))

    records, unresolved, reordered = [], 0, 0
    for row in body:
        values = [coerce(cell) for cell in row]
        best = None
        for index, order in enumerate(orders):
            for drop in drops:
                keys = [k for k in order if k not in drop]
                if len(keys) != len(values):
                    continue
                candidate = dict(zip(keys, values))
                if plausible(candidate):
                    best = (candidate, index)
                    break
            if best:
                break
        if best is None:
            unresolved += 1
            continue
        records.append(best[0])
        if best[1] != 0:
            reordered += 1

    notes = []
    if reordered:
        notes.append(
            "%d of %d rows needed a reordered key list (Ray reloaded mid-run); "
            "read result.json instead if you can" % (reordered, len(body))
        )
    if unresolved:
        notes.append("%d rows could not be aligned and were skipped" % unresolved)
    return records, notes


def resolve(path):
    """Accept a directory (searched downward), a result.json or a progress.csv."""
    if os.path.isfile(path):
        directory = os.path.dirname(os.path.abspath(path))
        preferred = os.path.join(directory, "result.json")
        if path.endswith("progress.csv") and os.path.isfile(preferred):
            print("note: using result.json instead of progress.csv (better keyed)\n")
            return preferred
        return path
    if not os.path.isdir(path):
        sys.exit("No such file or directory: %s" % path)

    # RLlib writes the metric files inside a trial directory, not at the top of
    # the experiment directory, so search downward. Sort result.json ahead of
    # progress.csv, and newest ahead of oldest.
    found = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith("checkpoint_")]
        for name in ("result.json", "progress.csv"):
            if name in files:
                full = os.path.join(root, name)
                found.append((name == "progress.csv", -os.path.getmtime(full), full))
    if not found:
        sys.exit("No result.json or progress.csv anywhere under %s" % path)
    found.sort()
    trials = {os.path.dirname(f) for _, _, f in found}
    if len(trials) > 1:
        print("note: %d trial directories found, using the most recent:\n      %s\n"
              % (len(trials), found[0][2]))
    return found[0][2]


# ---------------------------------------------------------------- reporting

def series(records, key):
    return [(int(float(r[ITER])), r[key]) for r in records
            if key in r and isinstance(r.get(key), float)]


def mean(values):
    return sum(values) / len(values) if values else None


def section(title):
    print("\n" + title)
    print("-" * len(title))


def report_run(records, notes, path, window, every, metrics):
    records.sort(key=lambda r: int(float(r[ITER])))
    iters = [int(float(r[ITER])) for r in records]

    section("RUN")
    print("file              %s" % path)
    print("iterations        %d  (%d to %d)" % (len(records), iters[0], iters[-1]))

    dates = [r["date"] for r in records if isinstance(r.get("date"), str)]
    if dates:
        print("started / ended   %s  ->  %s" % (dates[0], dates[-1]))
    per_iter = [r["time_this_iter_s"] for r in records
                if isinstance(r.get("time_this_iter_s"), float)]
    if per_iter:
        total = sum(per_iter)
        print("wall time         %.2f h   (%.0f s/iteration)"
              % (total / 3600.0, total / len(per_iter)))
    for label, key in (("env steps", "num_env_steps_sampled_lifetime"),
                       ("episodes", "num_episodes_lifetime"),
                       ("env runners", "fault_tolerance/num_healthy_workers"),
                       ("host", "hostname")):
        value = records[-1].get(key)
        if value is not None:
            print("%-17s %s" % (label, int(value) if isinstance(value, float) else value))

    reloads = [int(float(r[ITER])) for r in records
               if r.get("iterations_since_restore") == 1.0 and int(float(r[ITER])) > 1]
    if reloads:
        print("ray reloads at    %s" % ", ".join("iter %d" % i for i in reloads))

    section("CONFIGURATION (inferred from the metrics)")
    worst = series(records, "env_runners/episode_return_min")
    if worst:
        print("failure penalty   ~%.0f  (worst episode return seen: %.3f)"
              % (round(min(v for _, v in worst)), min(v for _, v in worst)))
    wheels = series(records, "env_runners/rw_status_valid")
    if not wheels:
        print("wheel check       not logged")
    elif min(v for _, v in wheels) > 1 - 1e-9:
        print("wheel check       DISABLED - rw_status_valid is 1.0 in every "
              "iteration, so saturation never killed an episode")
    else:
        print("wheel check       live (minimum %.3f)" % min(v for _, v in wheels))
    print("desat logged      %s"
          % ("yes" if series(records, "env_runners/desat_action") else "no"))

    for label, key in (("learning rate", "learners/default_policy/default_optimizer_learning_rate"),
                       ("entropy coeff", "learners/default_policy/curr_entropy_coeff")):
        data = series(records, key)
        if not data:
            continue
        first, last = data[0][1], data[-1][1]
        settled = data[-1][0]
        for iteration, value in data:
            if abs(value - last) < 1e-12:
                settled = iteration
                break
        print("%-17s %.3g -> %.3g  (stopped changing at iteration %d)"
              % (label, first, last, settled))

    # Keep the two ends disjoint on short runs, or "first" and "last" are the same rows.
    window = min(window, max(1, len(records) // 2))
    head = records[:window]
    tail = records[-window:]
    section("START vs END  (mean of first %d and last %d iterations)" % (len(head), len(tail)))
    print("%-20s %12s %12s %12s" % ("metric", "first", "last", "change"))
    for label, key, fmt in HEADLINE:
        before = mean([v for _, v in series(head, key)])
        after = mean([v for _, v in series(tail, key)])
        if before is None or after is None:
            continue
        print("%-20s %12s %12s %12s"
              % (label, fmt.format(before), fmt.format(after),
                 fmt.format(after - before) if "%" not in fmt else
                 "%+.1f pt" % (100 * (after - before))))

    alive = mean([v for _, v in series(tail, "env_runners/alive")])
    rw = mean([v for _, v in series(tail, "env_runners/rw_status_valid")])
    battery = mean([v for _, v in series(tail, "env_runners/battery_status_valid")])
    if alive is not None and (rw is not None or battery is not None):
        section("HOW EPISODES ENDED  (last %d iterations)" % len(tail))
        print("survived          %.1f%%" % (100 * alive))
        if battery is not None:
            print("battery death     %.1f%%" % (100 * (1 - battery)))
        if rw is not None:
            print("wheel death       %.1f%%" % (100 * (1 - rw)))
        if rw is not None and battery is not None:
            residual = alive - (1 - ((1 - rw) + (1 - battery)))
            if abs(residual) > 0.02:
                print("note              %+.1f pt unaccounted for - some episodes "
                      "fail both checks at once" % (100 * residual))

    if every:
        keys = metrics or ["env_runners/alive", "env_runners/episode_return_mean",
                           "env_runners/imaged", "env_runners/orbits_complete"]
        keys = [k for k in keys if series(records, k)]
        section("TRAJECTORY  (every %d iterations)" % every)
        print("%6s %s" % ("iter", " ".join("%16s" % k.split("/")[-1][:16] for k in keys)))
        sampled = records[::every]
        if records[-1] is not sampled[-1]:
            sampled.append(records[-1])  # always show where the run finished
        for record in sampled:
            row = ["%6d" % int(float(record[ITER]))]
            for key in keys:
                value = record.get(key)
                row.append("%16.3f" % value if isinstance(value, float) else "%16s" % "-")
            print(" ".join(row))

    if notes:
        section("WARNINGS")
        for note in notes:
            print("- %s" % note)


def write_clean(records, path):
    """Emit a correctly aligned CSV, which is what you want for plotting."""
    records.sort(key=lambda r: int(float(r[ITER])))
    keys = [ITER] + sorted({k for r in records for k in r} - {ITER})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print("\nwrote %d aligned rows x %d columns to %s" % (len(records), len(keys), path))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="run directory, result.json, or progress.csv")
    parser.add_argument("--window", type=int, default=10,
                        help="iterations to average at each end (default 10)")
    parser.add_argument("--every", type=int, default=0,
                        help="also print a trajectory table every N iterations")
    parser.add_argument("--metrics", default="",
                        help="comma-separated metrics for the trajectory table; "
                             "bare names are matched against env_runners/ first")
    parser.add_argument("--out", default="", help="write an aligned CSV here")
    args = parser.parse_args()

    path = resolve(args.path)
    loader = load_json if path.endswith(".json") else load_csv
    records, notes = loader(path)
    records = [r for r in records if ITER in r]
    if not records:
        sys.exit("No usable iterations found in %s" % path)

    known = {k for r in records for k in r}
    wanted = []
    for name in (m.strip() for m in args.metrics.split(",") if m.strip()):
        for candidate in (name, "env_runners/" + name, "learners/default_policy/" + name):
            if candidate in known:
                wanted.append(candidate)
                break
        else:
            print("note: no metric matching %r; ignoring" % name)

    report_run(records, notes, path, max(1, args.window),
               args.every or (20 if wanted else 0), wanted)
    if args.out:
        write_clean(records, args.out)


if __name__ == "__main__":
    main()
