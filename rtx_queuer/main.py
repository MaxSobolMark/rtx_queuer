"""Entry point and daemon loop."""

import argparse
import signal
import sys
import time
import uuid
from datetime import datetime

from .config import Config, load_config
from .coordinator import (
    get_external_jobs_blocked_on_resources,
    get_my_jobs,
    select_jobs_to_cancel,
)
from .slurm import Job, cancel_job, get_queue_status, submit_job


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

    def cancel_jobs(self, jobs_to_cancel: list[Job], reason: str) -> int:
        """Cancel a list of jobs. Returns number cancelled."""
        cancelled = 0
        for job in jobs_to_cancel:
            if cancel_job(job.job_id):
                log(f"Cancelled {job.job_id} ({job.name}) - {reason}")
                cancelled += 1
        return cancelled

    def run_once(self) -> None:
        """Run a single iteration of the daemon loop."""
        jobs = get_queue_status(self.config.partition)
        my_jobs = get_my_jobs(jobs, self.config.job_prefix, self.config.queuer_index)
        my_running = [j for j in my_jobs if j.is_running]
        my_pending = [j for j in my_jobs if j.is_pending]
        total = len(my_running) + len(my_pending)

        log(f"Status: {len(my_running)} running, {len(my_pending)} pending, target={self.config.target_jobs}")

        blocked_external = get_external_jobs_blocked_on_resources(
            jobs, self.config.job_prefix, self.config.partition
        )

        if not blocked_external:
            # Normal operation: maintain target count
            if total < self.config.target_jobs:
                to_submit = self.config.target_jobs - total
                log(f"Under target, submitting {to_submit} jobs")
                self.submit_placeholder_jobs(to_submit)
            return

        # External jobs are blocked - need to yield
        # Separate by blocking reason
        qos_blocked = [j for j in blocked_external if j.pending_reason == "QOSMaxJobsPerUserLimit"]
        resource_blocked = [j for j in blocked_external if j.pending_reason in ("Resources", "Priority")]

        # Handle QOSMaxJobsPerUserLimit: reduce total job count (no replacement)
        if qos_blocked:
            ext = qos_blocked[0]
            log(f"External job {ext.user}:{ext.job_id} blocked by QOS limit")
            if my_running:
                # Cancel one running job to free a slot
                self.cancel_jobs([my_running[-1]], "freeing QOS slot")
            elif my_pending:
                # Cancel one pending job
                self.cancel_jobs([my_pending[-1]], "freeing QOS slot")
            return

        # Handle Resources/Priority: free GPUs while maintaining queue presence
        if not resource_blocked:
            return

        gpus_needed = sum(j.gpus for j in resource_blocked)
        gpus_available = sum(j.gpus for j in my_running)

        if gpus_available == 0:
            # No running jobs to cancel
            return

        # Determine which running jobs to cancel
        running_to_cancel = select_jobs_to_cancel(my_running, gpus_needed)
        if not running_to_cancel:
            return

        # Calculate job counts after cancellation
        running_after = len(my_running) - len(running_to_cancel)
        pending_needed = self.config.target_jobs - running_after
        to_submit = max(0, pending_needed - len(my_pending))

        # Ensure at least 1 pending job so we never have zero queue presence
        if len(my_pending) + to_submit < 1:
            to_submit = 1

        # Submit replacement jobs first
        if to_submit > 0:
            log(f"Submitting {to_submit} replacement jobs before yielding")
            self.submit_placeholder_jobs(to_submit)

            # Confirm we have pending jobs before cancelling
            fresh_jobs = get_queue_status(self.config.partition)
            fresh_my = get_my_jobs(fresh_jobs, self.config.job_prefix, self.config.queuer_index)
            if not any(j.is_pending for j in fresh_my):
                log("WARNING: No pending jobs confirmed in queue, skipping cancellation")
                return
        elif len(my_pending) == 0:
            # We have no pending and aren't submitting any - unsafe to cancel
            log("WARNING: No pending jobs in queue, skipping cancellation")
            return

        # Now safe to cancel running jobs
        requesters = [f"{j.user}:{j.job_id}" for j in resource_blocked]
        log(f"Freeing GPUs for: {', '.join(requesters)}")
        self.cancel_jobs(running_to_cancel, "freeing GPUs")

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
