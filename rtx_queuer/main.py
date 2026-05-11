"""Entry point and daemon loop."""

import argparse
import getpass
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

        my_user = getpass.getuser()
        # QOSMaxJobsPerUserLimit is a per-user cap, so cancelling our placeholders
        # only helps when the blocked job belongs to our own user.
        qos_blocked = [
            j for j in blocked_external
            if j.pending_reason == "QOSMaxJobsPerUserLimit" and j.user == my_user
        ]
        resource_blocked = [
            j for j in blocked_external
            if j.pending_reason in ("Resources", "Priority")
        ]

        effective_target = max(1, self.config.target_jobs - len(qos_blocked))
        if qos_blocked:
            log(f"Same-user QOS limit hit ({qos_blocked[0].job_id}), effective target {effective_target}")

        if resource_blocked and my_running:
            gpus_needed = sum(j.gpus for j in resource_blocked)
            running_to_cancel = select_jobs_to_cancel(my_running, gpus_needed)
            if running_to_cancel:
                running_after = len(my_running) - len(running_to_cancel)
                pending_needed = effective_target - running_after
                to_submit = max(0, pending_needed - len(my_pending))

                if len(my_pending) + to_submit < 1:
                    to_submit = 1

                if to_submit > 0:
                    log(f"Submitting {to_submit} replacement jobs before yielding")
                    self.submit_placeholder_jobs(to_submit)

                    fresh_jobs = get_queue_status(self.config.partition)
                    fresh_my = get_my_jobs(fresh_jobs, self.config.job_prefix, self.config.queuer_index)
                    if not any(j.is_pending for j in fresh_my):
                        log("WARNING: No pending jobs confirmed in queue, skipping cancellation")
                        return
                elif len(my_pending) == 0:
                    log("WARNING: No pending jobs in queue, skipping cancellation")
                    return

                requesters = [f"{j.user}:{j.job_id}" for j in resource_blocked]
                log(f"Freeing GPUs for: {', '.join(requesters)}")
                self.cancel_jobs(running_to_cancel, "freeing GPUs")
                return

        if total > effective_target:
            excess = total - effective_target
            to_cancel: list[Job] = []
            if my_pending:
                to_cancel = my_pending[-excess:]
            elif my_running:
                to_cancel = my_running[-excess:]
            if to_cancel:
                self.cancel_jobs(to_cancel, "freeing QOS slot")
            return

        if total < effective_target:
            to_submit = effective_target - total
            log(f"Under target ({effective_target}), submitting {to_submit} jobs")
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
