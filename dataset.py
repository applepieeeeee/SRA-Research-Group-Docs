import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .simulation import ( #pulls from the Ising model file
    DirectedBoundaryStrip,
    PartitionSpec,
    build_strip_static_features,
    enumerate_directed_strips,
    random_bimodal_instance,
    simulate_monolithic,
)

@dataclass(frozen=True)
class BoundaryStripDatasetConfig:
    rows: int = 32
    cols: int = 32 #grid size 32 by 32
    partition_rows: int = 2 
    partition_cols: int = 2 #grid size for each partition
    beta_values: tuple[float, ...] = (0.55,) #some random constant that is mildy important
    num_instances: int = 50 #since we want good dataset we spin up multiple individual instances for robustness yk
    burn_in_steps: int = 128 #ising model is stupid in the begenning so we have to wait for it to become better
    trajectory_steps: int = 5000
    history_length: int = 16
    horizon: int = 16
    window_stride: int = 8 #we use a sliding window to generate lots of examples from a smaller smaple
    include_remote_interior: bool = False
    seed: int = 0
    field_scale: float = 0.0 #the electromagnetic bias thingy that affects

@dataclass
class BoundaryDatasetBundle:
    arrays: dict[str, np.ndarray] #holds the final dataset
    metadata: dict[str, object]

def _split_counts(num_instances: int):
#train, validation, and test splits. 70%, 15% and 15%
    train = max(1, int(num_instances * 0.7))
    val = max(1, int(num_instances * 0.15))

    if train + val >= num_instances:
        val = 1
    
    test = num_instances - train - val
    return (train, val, test) #returns a tuple with all three forms


def _assign_instance_splits(config: BoundaryStripDatasetConfig) -> dict[int, str]:
    #this function is needed due to the sliding window strategy we used to generate our dataset
    #since we may have many samples (due to sliding window) this splitting strategy ensures tha tthey remain in the same split (train, test, or validation)
    rng = np.random.default_rng(config.seed)
    instance_ids = list(range(config.num_instances))
    shuffled = list(rng.permutation(instance_ids))
    train_count, val_count, _ = _split_counts(config.num_instances)

    splits: dict[int, str] = {}
    for position, instance_id in enumerate(shuffled):
        if position < train_count:
            splits[int(instance_id)] = "train"
        elif position < train_count + val_count:
            splits[int(instance_id)] = "val"
        else:
            splits[int(instance_id)] = "test"
    return splits

#calculates how many vlaid windows can be extracted for a cetain edge
def _windows_per_edge(trajectory_steps: int, history_length: int, horizon: int, window_stride: int) -> int:
    usable_starts = trajectory_steps - (history_length + horizon) + 1 #calculates how many positions we can start the window from
    return (usable_starts - 1) // window_stride + 1 #for the default settings theoretically this should return 622 windows

def _allocate_strip_split_arrays(count: int,history_length: int,horizon: int,dynamic_feature_dim: int,strip_width: int,static_dim: int):
    count = max(0, count)
    return {
        "x_dynamic": np.empty((count, history_length, dynamic_feature_dim), dtype=np.int8),
        "x_static": np.empty((count, static_dim), dtype=np.float32),
        "y_next": np.empty((count, strip_width), dtype=np.uint8),
        "y_future": np.empty((count, horizon, strip_width), dtype=np.uint8),
        "local_future": np.empty((count, horizon, strip_width), dtype=np.int8),
        "instance_ids": np.empty((count,), dtype=np.int32),
        "beta_values": np.empty((count,), dtype=np.float32),
        "strip_ids": np.empty((count,), dtype=np.int32),
    } #just makes a bunch of empty arrays for each piece of information of the appropriate length to make informaiton management easier later on


#for one given strip the function is given all relevent information and should output the ML training examples
def _strip_windows(
    history: np.ndarray,
    strip: DirectedBoundaryStrip,
    history_length: int,
    horizon: int,
    window_stride: int,
    beta: float,
    include_remote_interior: bool,
):
    total_window = history_length + horizon #since our model is expected to predict the next x values given the previous x values for a certain edge
                                            #our total window would have to be the addition of the history and the horizon in the future

    local_series = history[:, strip.local_sites[:, 0], strip.local_sites[:, 1]].astype(np.int8) #stores spin history over time for a given boundary strip
    remote_series = history[:, strip.remote_sites[:, 0], strip.remote_sites[:, 1]].astype(np.int8) #same as the previous function but for the neighboring strip

    #So I added this to do some testing but it should be toggled off
    #Essentially in addition to just the boundayr strip we will also save and give the model information one layer deeper
    #wanted to see if performance improved but it did not really seem to make that large of a difference for increase in compute overhead
    remote_interior_series = history[:, strip.remote_interior_sites[:, 0], strip.remote_interior_sites[:, 1]].astype(np.int8)

    local_windows = np.lib.stride_tricks.sliding_window_view(local_series, total_window, axis=0).transpose(0, 2, 1)
    remote_windows = np.lib.stride_tricks.sliding_window_view(remote_series, total_window, axis=0).transpose(0, 2, 1)
    #These two functions actually create the sliding windows for local and remote strips

    #same thing, but for the interior strip (TOGGLED OFF)
    remote_interior_windows = (np.lib.stride_tricks.sliding_window_view(remote_interior_series, total_window, axis=0).transpose(0, 2, 1))

    #strips the windows down to only every window_strideTH window. So like every 8th window for example or every 10th window
    #we do this to reduce the overlap that our model will see in all the windows and to prevent overfitting
    local_windows = local_windows[::window_stride]
    remote_windows = remote_windows[::window_stride]
    remote_interior_windows = remote_interior_windows[::window_stride]

    #splits the local and remote into input and output values. HISTORY and FUTURE. Model should predict FUTURE given HISTORY
    local_history = local_windows[:, :history_length, :]
    remote_history = remote_windows[:, :history_length, :]
    local_future = local_windows[:, history_length:, :]
    remote_future = remote_windows[:, history_length:, :]


    remote_interior_history = remote_interior_windows[:, :history_length, :]

    dynamic_groups = [remote_history, local_history] #just creates a list with remote and local history

    if include_remote_interior:
        dynamic_groups.append(remote_interior_history) #optionally (CURRENLTY TOGGLED OFF) we add the interior history too

    x_dynamic = np.concatenate(dynamic_groups, axis=2).astype(np.int8) #concatinates both arrays that were added to dynamic groups into one large input array

    x_static = build_strip_static_features(strip=strip, beta=beta, include_remote_interior=include_remote_interior) #contains all the features that do not change over time (basically everything but SPIN values)
    x_static = np.repeat(x_static[None, :], repeats=x_dynamic.shape[0], axis=0).astype(np.float32)
    #The second line just duplicates the constant information for every example case we have in x_dynamic


    y_next = ((remote_future[:, 0, :] + 1) // 2).astype(np.uint8) #next step prediction targets for each given input in X
    y_future = ((remote_future + 1) // 2).astype(np.uint8) #Instead of just the next step, this holds the entire future history that we attempt to predict

    return x_dynamic, x_static, y_next, y_future, local_future.astype(np.int8)








def build_boundary_strip_dataset(config: BoundaryStripDatasetConfig):
    if config.trajectory_steps <= config.history_length + config.horizon:
        raise ValueError("you need to either reduce how history + horizon or increase number of steps")

    split_lookup = _assign_instance_splits(config)  #each instance is assigned a split so each split has its own sliding window
                                                    #In the default case there are 50 instances, so that would mean 50 sliding windows

    #A random Ising model is generated to populate the configuration of each instance
    #just for housekeeping purposes and in reality we will later use the function I wrote in the Ising simulation file to actually initialize each instance
    prototype_instance = random_bimodal_instance(
        rows=config.rows,
        cols=config.cols,
        rng=np.random.default_rng(config.seed),
        field_scale=config.field_scale,
    )

    prototype_strips = enumerate_directed_strips( #finds all boundary strips between partitions for each instance
        instance=prototype_instance,
        spec=PartitionSpec(rows=config.partition_rows, cols=config.partition_cols),
    )
    strip_width = prototype_strips[0].strip_width #the width of each boundary strip (16)


    dynamic_feature_dim = strip_width * 2 #multiply by 3 if using the additional internal boundary layer (default is 2 though)

    '''static dim composes all of these together:
        cross-boundary couplings
        local along-strip couplings
        remote along-strip couplings
        optional remote interior couplings
        beta
        orientation one-hot'''
    static_dim = (
        prototype_strips[0].cross_couplings.shape[0]
        + prototype_strips[0].local_along_couplings.shape[0]
        + prototype_strips[0].remote_along_couplings.shape[0]
        + (prototype_strips[0].remote_interior_couplings.shape[0] if config.include_remote_interior else 0)
        + 1
        + prototype_strips[0].orientation_one_hot.shape[0]
    )

    directed_strip_count = len(prototype_strips)
    raw_windows_per_strip = config.trajectory_steps - (config.history_length + config.horizon) + 1


    retained_windows_per_strip = _windows_per_edge(
        trajectory_steps=config.trajectory_steps,
        history_length=config.history_length,
        horizon=config.horizon,
        window_stride=config.window_stride,
    )#the number of windows after our "strides" have been added to reduce duplication


    samples_per_run = directed_strip_count * retained_windows_per_strip #number of samples we get per instance

    #counts how many we have per set
    #kinda unecessary but I was facing issues with distributing our samples according to our splits for some reason
    split_instance_counts = {
        "train": sum(1 for split in split_lookup.values() if split == "train"),
        "val": sum(1 for split in split_lookup.values() if split == "val"),
        "test": sum(1 for split in split_lookup.values() if split == "test"),
    }

    split_total_counts = {
        split_name: split_instance_counts[split_name] * len(config.beta_values) * samples_per_run
        for split_name in ("train", "val", "test")
    } #calculates how many samples each split will contain


    #creates empty arrays with the right sizes that will be filled later
    split_arrays = {
        split_name: _allocate_strip_split_arrays(
            count=split_total_counts[split_name],
            history_length=config.history_length,
            horizon=config.horizon,
            dynamic_feature_dim=dynamic_feature_dim,
            strip_width=strip_width,
            static_dim=static_dim,
        )
        for split_name in ("train", "val", "test")
    }
    #basically just an iterator to fill our arrays
    split_cursors = {split_name: 0 for split_name in ("train", "val", "test")}


    #used for metadata (for debuggin etc.)
    run_summaries: list[dict[str, object]] = []
    total_samples = 0
    raw_total_samples = 0




    #loops over each random ising instance
    for instance_id in range(config.num_instances):
        coupling_rng = np.random.default_rng(config.seed + instance_id) #randomd generator for the initial coupling values
        #uses code from our simulation file to actually create the ising problems
        instance = random_bimodal_instance(
            rows=config.rows,
            cols=config.cols,
            rng=coupling_rng,
            field_scale=config.field_scale,
        )
        #finds the strips and partitions for each instance
        directed_strips = enumerate_directed_strips(
            instance=instance,
            spec=PartitionSpec(rows=config.partition_rows, cols=config.partition_cols),
        )
        split_name = split_lookup[instance_id] #decides which split this particulat Ising instance needs to go to (train, validation, test)

        #loops over all the beta values we want to test for each Ising instance
        #REMINDER: Beta values are jsut inverse temperature. With low beta values meaning there is less importance on the couplings and vice versa
        for beta in config.beta_values:
            run_seed = config.seed + 10_000 * instance_id + int(round(beta * 1_000))

            #run the simulation. It runs monolithic so we have the actual true values
            result = simulate_monolithic(
                instance=instance,
                beta=beta,
                steps=config.burn_in_steps + config.trajectory_steps,
                seed=run_seed,
                record_history=True,
            )
            history = result.history[config.burn_in_steps :]  #storing history not including the burn in steps

            samples_this_run = 0
            for strip in directed_strips: #loop over each boundary strip
                x_dynamic, x_static, y_next, y_future, local_future = _strip_windows(history=history, strip=strip, history_length=config.history_length, horizon=config.horizon, window_stride=config.window_stride, beta=beta, include_remote_interior=config.include_remote_interior)
                #for each boundary strip it creates the Machine Learning examples we will use to train a model

                count = x_dynamic.shape[0] #number of samples
        
                samples_this_run += count
                total_samples += count
                raw_total_samples += raw_windows_per_strip

                cursor = split_cursors[split_name]
                next_cursor = cursor + count

                #store everything to their respective arrays using our cursor(essentially our iterator)
                split_arrays[split_name]["x_dynamic"][cursor:next_cursor] = x_dynamic
                split_arrays[split_name]["x_static"][cursor:next_cursor] = x_static
                split_arrays[split_name]["y_next"][cursor:next_cursor] = y_next
                split_arrays[split_name]["y_future"][cursor:next_cursor] = y_future
                split_arrays[split_name]["local_future"][cursor:next_cursor] = local_future
                split_arrays[split_name]["instance_ids"][cursor:next_cursor] = instance_id
                split_arrays[split_name]["beta_values"][cursor:next_cursor] = beta
                split_arrays[split_name]["strip_ids"][cursor:next_cursor] = strip.strip_id
                split_cursors[split_name] = next_cursor

            run_summaries.append(
                {
                    "instance_id": instance_id,
                    "split": split_name,
                    "beta": beta,
                    "num_directed_strips": len(directed_strips),
                    "samples": samples_this_run,
                    "best_energy": float(np.min(result.energies)),
                }
            )

    arrays = {}
    for split_name in ("train", "val", "test"):
        for key, value in split_arrays[split_name].items():
            arrays[f"{split_name}_{key}"] = value

    metadata = {
        "config": {
            "rows": config.rows,
            "cols": config.cols,
            "partition_rows": config.partition_rows,
            "partition_cols": config.partition_cols,
            "beta_values": list(config.beta_values),
            "num_instances": config.num_instances,
            "burn_in_steps": config.burn_in_steps,
            "trajectory_steps": config.trajectory_steps,
            "history_length": config.history_length,
            "horizon": config.horizon,
            "window_stride": config.window_stride,
            "include_remote_interior": config.include_remote_interior,
            "seed": config.seed,
            "field_scale": config.field_scale,
        },
        "dynamic_feature_groups": (
            ["remote_boundary_strip", "local_boundary_strip", "remote_first_interior_strip"]
            if config.include_remote_interior
            else ["remote_boundary_strip", "local_boundary_strip"]
        ),
        "static_feature_groups": (
            [
                "cross_boundary_couplings",
                "local_along_strip_couplings",
                "remote_along_strip_couplings",
                "remote_boundary_to_interior_couplings",
                "beta",
                "orientation_one_hot",
            ]
            if config.include_remote_interior
            else [
                "cross_boundary_couplings",
                "local_along_strip_couplings",
                "remote_along_strip_couplings",
                "beta",
                "orientation_one_hot",
            ]
        ),
        "target_encoding": {"negative_one": 0, "positive_one": 1},
        "splits": split_lookup,
        "strip_width": strip_width,
        "x_dynamic_dim": dynamic_feature_dim,
        "x_static_dim": static_dim,
        "num_directed_boundary_strips": directed_strip_count,
        "raw_windows_per_strip": raw_windows_per_strip,
        "retained_windows_per_strip": retained_windows_per_strip,
        "samples_per_run": samples_per_run,
        "raw_total_samples": raw_total_samples,
        "total_samples": total_samples,
        "split_total_counts": split_total_counts,
        "run_summaries": run_summaries,
    }

    return BoundaryDatasetBundle(arrays=arrays, metadata=metadata)













#saving and exporting dataset
#Chose to use the .npz file type so if you guys want to work with something other than numpy then you will have to convert it
def save_boundary_dataset(bundle: BoundaryDatasetBundle, output_dir: str | Path, stem: str = "boundary_dataset") -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    npz_path = output_path / f"{stem}.npz"
    metadata_path = output_path / f"{stem}_metadata.json"

    np.savez_compressed(npz_path, **bundle.arrays)
    metadata_path.write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")
    return npz_path, metadata_path
