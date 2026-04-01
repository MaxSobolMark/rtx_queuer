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

    def handle_deallocation(
        self,
        jobs: list,
        blocked_external: list,
        pending_ids_to_cancel: set[str] | None = None,
    ) -> int:
        """Cancel jobs to free GPUs for blocked external jobs.

        Args:
            jobs: Current queue status
            blocked_external: External jobs that are blocked
            pending_ids_to_cancel: If provided, only cancel pending jobs with these IDs
                (used to avoid cancelling newly submitted replacement jobs)

        Returns number of jobs cancelled.
        """
        if not blocked_external:
            return 0

        my_jobs = get_my_jobs(jobs, self.config.job_prefix, self.config.queuer_index)
        my_running = [j for j in my_jobs if j.is_running]

        # Filter pending jobs to only those we should cancel
        if pending_ids_to_cancel is not None:
            my_pending = [j for j in my_jobs if j.is_pending and j.job_id in pending_ids_to_cancel]
        else:
            my_pending = [j for j in my_jobs if j.is_pending]

        cancelled = 0

        # Cancel pending queuer jobs only if they have higher priority (lower job_id)
        # than blocked external jobs - jobs submitted later don't block earlier ones
        if my_pending and blocked_external:
            min_external_job_id = min(int(j.job_id) for j in blocked_external)
            blocking_pending = [j for j in my_pending if int(j.job_id) < min_external_job_id]

            if blocking_pending:
                yielding_to = [f"{j.user}:{j.job_id}" for j in blocked_external]
                log(f"Cancelling {len(blocking_pending)} pending jobs (higher priority than external) to yield to: {', '.join(yielding_to)}")
                for job in blocking_pending:
                    if cancel_job(job.job_id):
                        log(f"Cancelled pending job {job.job_id} ({job.name})")
                        cancelled += 1

        # Cancel running jobs to free GPUs for jobs blocked on Resources
        blocked_on_resources = [j for j in blocked_external if j.pending_reason == "Resources"]
        if blocked_on_resources and my_running:
            gpus_needed = sum(j.gpus for j in blocked_on_resources)
            gpus_to_free = min(gpus_needed, sum(j.gpus for j in my_running))

            requesters = [f"{j.user}:{j.job_id}({j.name}, {j.gpus}gpu)" for j in blocked_on_resources]
            log(f"Freeing {gpus_to_free} GPUs for: {', '.join(requesters)}")

            to_cancel = select_jobs_to_cancel(my_running, gpus_to_free)
            for job in to_cancel:
                if cancel_job(job.job_id):
                    log(f"Cancelled running job {job.job_id} ({job.name})")
                    cancelled += 1

        return cancelled

    def run_once(self) -> None:
        """Run a single iteration of the daemon loop."""
        jobs = get_queue_status(self.config.partition)

        # Check current job counts
        running, pending = self.get_my_job_count(jobs)
        total = running + pending

        log(f"Status: {running} running, {pending} pending, target={self.config.target_jobs}")

        # Check for blocked external jobs
        blocked_external = get_external_jobs_blocked_on_resources(jobs, self.config.job_prefix, self.config.partition)

        if blocked_external:
            # Get list of current pending job IDs
            my_jobs = get_my_jobs(jobs, self.config.job_prefix, self.config.queuer_index)
            old_pending = [j for j in my_jobs if j.is_pending]
            old_pending_ids = {j.job_id for j in old_pending}

            # Only cancel pending jobs that have higher priority (lower job_id) than external jobs
            min_external_job_id = min(int(j.job_id) for j in blocked_external)
            pending_to_cancel = [j for j in old_pending if int(j.job_id) < min_external_job_id]

            if not pending_to_cancel:
                # Our pending jobs have lower priority - they're not blocking external jobs
                # Just maintain target count like normal
                if total < self.config.target_jobs:
                    to_submit = self.config.target_jobs - total
                    log(f"Under target, submitting {to_submit} jobs")
                    self.submit_placeholder_jobs(to_submit)
                return

            pending_ids_to_cancel = {j.job_id for j in pending_to_cancel}

            # Submit replacement jobs first (enough to maintain target after cancellation)
            to_submit = max(len(pending_to_cancel), self.config.target_jobs - running)
            if to_submit > 0:
                log(f"Submitting {to_submit} jobs before yielding")
                self.submit_placeholder_jobs(to_submit)

            # Re-query queue and only cancel if new jobs are confirmed in queue
            fresh_jobs = get_queue_status(self.config.partition)
            fresh_my_jobs = get_my_jobs(fresh_jobs, self.config.job_prefix, self.config.queuer_index)
            new_pending = [j for j in fresh_my_jobs if j.is_pending and j.job_id not in old_pending_ids]

            if new_pending:
                log(f"Confirmed {len(new_pending)} new jobs in queue, proceeding with cancellation")
                self.handle_deallocation(fresh_jobs, blocked_external, pending_ids_to_cancel)
            else:
                log("WARNING: New jobs not yet visible in queue, skipping cancellation this cycle")
        else:
            # No external jobs blocked - just maintain target count
            if total < self.config.target_jobs:
                to_submit = self.config.target_jobs - total
                log(f"Under target, submitting {to_submit} jobs")
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
