"""Entry point and daemon loop."""

import argparse
import signal
import sys
import time
import uuid
from datetime import datetime

from .config import Config, load_config
from .coordinator import (
    get_my_jobs,
    get_pending_external_jobs,
    select_jobs_to_cancel,
    select_pending_jobs_to_cancel,
)
from .slurm import cancel_job, get_queue_status, submit_job


def generate_job_name(prefix: str, index: int) -> str:
    """Generate a unique job name."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}_{index}_{short_uuid}"


def log(message: str) -> None:
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


class Queuer:
    def __init__(self, config: Config):
        self.config = config
        self.running = True

    def get_my_job_count(self, jobs: list) -> tuple[int, int]:
        """Return (running_count, pending_count) for this queuer's jobs."""
        my_jobs = get_my_jobs(jobs, self.config.job_prefix, self.config.queuer_index)
        running = sum(1 for j in my_jobs if j.is_running)
        pending = sum(1 for j in my_jobs if j.is_pending)
        return running, pending

    def submit_placeholder_jobs(self, count: int) -> int:
        """Submit placeholder jobs. Returns number successfully submitted."""
        submitted = 0
        for _ in range(count):
            job_name = generate_job_name(
                self.config.job_prefix,
                self.config.queuer_index,
            )
            job_id = submit_job(
                script_path=self.config.script_path,
                job_name=job_name,
                partition=self.config.partition,
                gpu_type=self.config.gpu_type,
                gpus=self.config.gpus_per_job,
                time_limit=self.config.time_limit,
                qos=self.config.qos,
            )
            if job_id:
                log(f"Submitted job {job_id} ({job_name})")
                submitted += 1
        return submitted

    def handle_deallocation(self, jobs: list, effective_target: int) -> int:
        """Cancel jobs to reach effective target.

        Returns number of jobs cancelled.
        """
        my_jobs = get_my_jobs(jobs, self.config.job_prefix, self.config.queuer_index)
        my_running = [j for j in my_jobs if j.is_running]
        my_pending = [j for j in my_jobs if j.is_pending]

        cancelled = 0

        # Cancel running jobs if over effective target
        if len(my_running) > effective_target:
            gpus_to_free = len(my_running) - effective_target
            pending_external = get_pending_external_jobs(jobs, self.config.job_prefix)

            # Log details about who is requesting GPUs
            requesters = []
            for job in pending_external:
                requesters.append(f"{job.user}:{job.job_id}({job.name}, {job.gpus}gpu)")
            log(f"External jobs pending, need to free {gpus_to_free} GPUs. Requested by: {', '.join(requesters)}")

            to_cancel = select_jobs_to_cancel(my_jobs, gpus_to_free)
            for job in to_cancel:
                if cancel_job(job.job_id):
                    log(f"Cancelled job {job.job_id} ({job.name})")
                    cancelled += 1

        # Cancel pending jobs that would exceed effective target
        pending_to_keep = max(0, effective_target - len(my_running) + cancelled)
        pending_to_cancel = select_pending_jobs_to_cancel(my_jobs, pending_to_keep)
        for job in pending_to_cancel:
            if cancel_job(job.job_id):
                log(f"Cancelled pending job {job.job_id} ({job.name})")
                cancelled += 1

        return cancelled

    def run_once(self) -> None:
        """Run a single iteration of the daemon loop."""
        jobs = get_queue_status(self.config.partition)

        # Check current job counts
        running, pending = self.get_my_job_count(jobs)
        total = running + pending

        # Calculate effective target (reserve capacity for external pending jobs)
        pending_external = get_pending_external_jobs(jobs, self.config.job_prefix)
        external_gpus_needed = sum(j.gpus for j in pending_external)
        effective_target = max(0, self.config.target_jobs - external_gpus_needed)

        log(f"Status: {running} running, {pending} pending, target={self.config.target_jobs}, effective={effective_target}")

        # Handle deallocation if we're over the effective target
        if running > effective_target or pending > 0:
            cancelled = self.handle_deallocation(jobs, effective_target)
            if cancelled > 0:
                return  # Don't submit new jobs this cycle

        # Submit jobs up to effective target
        if total < effective_target:
            to_submit = effective_target - total
            log(f"Under effective target, submitting {to_submit} jobs")
            self.submit_placeholder_jobs(to_submit)

    def run(self) -> None:
        """Run the daemon loop."""
        log(f"Starting queuer (index={self.config.queuer_index})")

        while self.running:
            try:
                self.run_once()
            except Exception as e:
                log(f"Error in daemon loop: {e}")

            time.sleep(self.config.poll_interval)

        log("Queuer stopped")

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self.running = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RTX Queuer - SLURM GPU Reservation Manager"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    queuer = Queuer(config)

    # Handle signals for graceful shutdown
    def handle_signal(signum, frame):
        log(f"Received signal {signum}, shutting down...")
        queuer.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    queuer.run()


if __name__ == "__main__":
    main()
