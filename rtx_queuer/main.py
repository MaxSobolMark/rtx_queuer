"""Entry point and daemon loop."""

import argparse
import signal
import sys
import time
import uuid
from datetime import datetime

from .config import Config, load_config
from .coordinator import (
    calculate_gpus_to_deallocate,
    get_all_queuer_jobs,
    get_my_jobs,
    get_pending_external_jobs,
    select_jobs_to_cancel,
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
            )
            if job_id:
                log(f"Submitted job {job_id} ({job_name})")
                submitted += 1
        return submitted

    def handle_deallocation(self, jobs: list) -> int:
        """Check and handle deallocation for external pending jobs.

        Returns number of jobs cancelled.
        """
        pending_external = get_pending_external_jobs(jobs, self.config.job_prefix)
        if not pending_external:
            return 0

        all_queuer_jobs = get_all_queuer_jobs(jobs, self.config.job_prefix)
        gpus_to_free = calculate_gpus_to_deallocate(
            pending_external,
            self.config.queuer_index,
            all_queuer_jobs,
        )

        if gpus_to_free <= 0:
            return 0

        log(f"External jobs pending, need to free {gpus_to_free} GPUs")

        my_jobs = get_my_jobs(jobs, self.config.job_prefix, self.config.queuer_index)
        to_cancel = select_jobs_to_cancel(my_jobs, gpus_to_free)

        cancelled = 0
        for job in to_cancel:
            if cancel_job(job.job_id):
                log(f"Cancelled job {job.job_id} ({job.name})")
                cancelled += 1

        return cancelled

    def run_once(self) -> None:
        """Run a single iteration of the daemon loop."""
        jobs = get_queue_status(self.config.partition)

        # Check current job counts
        running, pending = self.get_my_job_count(jobs)
        total = running + pending

        log(f"Status: {running} running, {pending} pending, target={self.config.target_jobs}")

        # Submit jobs if under target
        if total < self.config.target_jobs:
            to_submit = self.config.target_jobs - total
            log(f"Under target, submitting {to_submit} jobs")
            self.submit_placeholder_jobs(to_submit)

        # Handle deallocation for external jobs
        self.handle_deallocation(jobs)

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
