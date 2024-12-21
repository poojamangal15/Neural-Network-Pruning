import torch
from torch.profiler import profile, ProfilerActivity
import os
from datetime import datetime

def configure_profiler(output_path="profiler_logs", profile_memory=True, with_flops=False):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    elif torch.backends.mps.is_available():
        activities.append(ProfilerActivity.MPS)

    return profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_path),
        record_shapes=True,
        with_stack=True,
        profile_memory=profile_memory,
        with_flops=with_flops
    )

def log_profiler_summary(profiler, sort_by="cpu_time_total", row_limit=10, output_file=None):
    summary = profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    print(summary)

    if output_file:
        with open(output_file, "w") as f:
            f.write(summary)
        print(f"Profiler summary saved to {output_file}")

def configure_filtered_profiler(output_path="filtered_profiler_logs", filter_ops=None):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    elif torch.backends.mps.is_available():
        activities.append(ProfilerActivity.MPS)

    return profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_path),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
        filter_ops=filter_ops
    )

def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def prepare_trace_dir(base_dir="profiler_logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = os.path.join(base_dir, f"trace_{timestamp}")
    os.makedirs(trace_dir, exist_ok=True)
    return trace_dir

def detect_sort_key():
    return "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
