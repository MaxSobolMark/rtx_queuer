# RTX Queuer

SLURM GPU Reservation Manager for shared lab partitions.

## Problem

Lab members share access to a partition on a SLURM cluster. If the partition is idle, outside users can claim the GPUs, making it hard for lab members to reclaim them. This tool maintains placeholder jobs that reserve GPUs, yielding them only when lab members submit real work.

## Features

- Maintains a pool of placeholder jobs (default: 8 GPUs per instance)
- Automatically detects when lab members need GPUs and deallocates
- Multiple queuers coordinate via shared job prefix and index-based priority
- Handles job expiration (2-day limit) by keeping replacement jobs queued

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit:

```yaml
queuer_index: 0              # This instance's index (0, 1, 2, ...)
script_path: /path/to/script.sh
partition: rl
gpu_type: RTX_PRO_6000
gpus_per_job: 1
target_jobs: 8
time_limit: "2-00:00:00"
poll_interval: 30
job_prefix: rtx_queuer
```

## Usage

```bash
# Run with default config.yaml
python -m rtx_queuer.main

# Run with custom config
python -m rtx_queuer.main -c /path/to/config.yaml
```

## Multi-Queuer Coordination

Multiple queuer instances can run simultaneously, each maintaining its own pool of GPUs. Coordination works via:

1. **Job naming**: Jobs are named `{prefix}_{index}_{uuid}` (e.g., `rtx_queuer_0_a1b2c3d4`)
2. **Index priority**: Lower-index queuers deallocate first
3. **Cascade**: If queuer 0 can't satisfy demand, queuer 1 handles the remainder

Example with 2 queuers (16 total GPUs):
- Lab member requests 4 GPUs
- Queuer 0 deallocates 4 of its 8 jobs
- Queuer 1 remains at 8 jobs
- Lab member requests 10 more GPUs
- Queuer 0 deallocates remaining 4, queuer 1 deallocates 6

## Placeholder Script

Create a simple placeholder script:

```bash
#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Sleep for the duration of the job
sleep infinity
```
