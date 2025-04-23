# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from argparse import ArgumentParser
from multiprocessing import Process
from datetime import datetime, timedelta
from trainer.cf3dgs_trainer import CFGaussianTrainer
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch


def construct_pose(poses):
    n_trgt = poses.shape[0]
    for i in range(n_trgt - 1, 0, -1):
        poses = torch.cat((poses[:i], poses[[i - 1]] @ poses[i:]), 0)
    return poses


def set_gpu(gpu_id):
    """
    Set the GPU for the current process using CUDA_VISIBLE_DEVICES.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Using GPU: {gpu_id}")


def parse_gpu_list(gpu_list_str):
    """
    Parse a GPU list string into a list of GPU IDs.

    Args:
        gpu_list_str (str): A string representing GPU IDs, e.g., "0,2-5,8".

    Returns:
        List[int]: A list of GPU IDs.
    """
    gpu_ids = []
    for part in gpu_list_str.split(','):
        if '-' in part:  # Handle ranges like "2-5"
            start, end = map(int, part.split('-'))
            gpu_ids.extend(range(start, end + 1))
        else:  # Handle single GPU IDs like "0" or "8"
            gpu_ids.append(int(part))
    return gpu_ids


def split_time_range(start_time, end_time, num_splits):
    """
    Split the given time range into `num_splits` equal intervals.

    Args:
        start_time (str): The start time in "YYYY-MM-DD" format.
        end_time (str): The end time in "YYYY-MM-DD" format.
        num_splits (int): The number of intervals to split into.

    Returns:
        List[Tuple[str, str]]: A list of (start_time, end_time) tuples for each split.
    """
    start = datetime.strptime(start_time, "%Y-%m-%d")
    end = datetime.strptime(end_time, "%Y-%m-%d")
    total_duration = (end - start).days

    if total_duration < num_splits:
        raise ValueError("Time range is too short to be split into the requested number of GPUs.")

    split_duration = total_duration // num_splits
    time_ranges = []

    for i in range(num_splits):
        split_start = start + timedelta(days=i * split_duration)
        split_end = split_start + timedelta(days=split_duration)
        if i == num_splits - 1:  # Ensure the last split ends exactly at `end_time`
            split_end = end
        time_ranges.append((split_start.strftime("%Y-%m-%d"), split_end.strftime("%Y-%m-%d")))

    return time_ranges


def run_on_gpu(gpu_id, time_range, model_cfg, pipe_cfg, optim_cfg):
    """
    Function to run a single process on a specific GPU with a given time range.
    """
    set_gpu(gpu_id)

    # Override start_time and end_time in the configuration
    start_time, end_time = time_range
    model_cfg.start_time = start_time
    model_cfg.end_time = end_time

    # Create a trainer instance
    data_path = model_cfg.source_path
    trainer = CFGaussianTrainer(data_path, model_cfg, pipe_cfg, optim_cfg)

    # Run the desired mode
    start_time_process = datetime.now()
    print(f"Process started on GPU {gpu_id} with range {start_time} to {end_time} at {start_time_process}")

    if model_cfg.mode == "train":
        trainer.train_from_progressive()
    elif model_cfg.mode == "render":
        trainer.render_nvs(traj_opt=model_cfg.traj_opt)
    elif model_cfg.mode == "eval_nvs":
        trainer.eval_nvs()
    elif model_cfg.mode == "eval_pose":
        trainer.eval_pose()

    end_time_process = datetime.now()
    print(f"Process completed on GPU {gpu_id} at {end_time_process}")
    print(f"Duration on GPU {gpu_id}: {end_time_process - start_time_process}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Distributed Training Script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--gpus", type=str, required=True, help="Comma-separated list of GPUs to use (e.g., '0,2-5,8')")
    parser.add_argument("--gl_start_time", type=str, required=True, help="Start time for the data in YYYY-MM-DD format")
    parser.add_argument("--gl_end_time", type=str, required=True, help="End time for the data in YYYY-MM-DD format")
    args = parser.parse_args()

    # Extract model, pipeline, and optimization configurations
    model_cfg = lp.extract(args)
    pipe_cfg = pp.extract(args)
    optim_cfg = op.extract(args)

    # Parse GPU list from the argument
    gpu_ids = parse_gpu_list(args.gpus)
    print(f"Using GPUs: {gpu_ids}")

    # Split the time range into intervals for each GPU
    time_ranges = split_time_range(args.gl_start_time, args.gl_end_time, len(gpu_ids))
    print(f"Time ranges for GPUs: {time_ranges}")

    # Create processes for each GPU
    processes = []
    for gpu_id, time_range in zip(gpu_ids, time_ranges):
        process = Process(
            target=run_on_gpu,
            args=(gpu_id, time_range, model_cfg, pipe_cfg, optim_cfg),
        )
        processes.append(process)

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("All processes have completed.")