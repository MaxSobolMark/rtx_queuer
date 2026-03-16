"""SLURM command interface."""

import subprocess
from dataclasses import dataclass


@dataclass
class Job:
    job_id: str
    name: str
    user: str
    state: str
    partition: str
    gpus: int

    @property
    def is_running(self) -> bool:
        return self.state == "RUNNING"

    @property
    def is_pending(self) -> bool:
        return self.state == "PENDING"


def run_command(cmd: list[str]) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def get_queue_status(partition: str) -> list[Job]:
    """Get all jobs in the partition using squeue."""
    # Format: JobID|Name|User|State|Partition|Gres
    cmd = [
        "squeue",
        "-p", partition,
        "-o", "%A|%j|%u|%T|%P|%b",
        "--noheader",
    ]
    try:
        output = run_command(cmd)
    except subprocess.CalledProcessError:
        return []

    jobs = []
    for line in output.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 6:
            continue

        job_id, name, user, state, part, gres = parts[:6]

        # Parse GPU count from gres (e.g., "gpu:RTX_PRO_6000:1")
        gpus = 0
        if gres and "gpu" in gres.lower():
            try:
                gpus = int(gres.split(":")[-1])
            except (ValueError, IndexError):
                gpus = 1

        jobs.append(Job(
            job_id=job_id.strip(),
            name=name.strip(),
            user=user.strip(),
            state=state.strip(),
            partition=part.strip(),
            gpus=gpus,
        ))

    return jobs


def submit_job(
    script_path: str,
    job_name: str,
    partition: str,
    gpu_type: str,
    gpus: int,
    time_limit: str,
    qos: str | None = None,
) -> str | None:
    """Submit a job using sbatch. Returns job ID or None on failure."""
    cmd = [
        "sbatch",
        "--job-name", job_name,
        "--partition", partition,
        "--gres", f"gpu:{gpu_type}:{gpus}",
        "--time", time_limit,
    ]
    if qos:
        cmd.extend(["--qos", qos])
    cmd.append(script_path)
    try:
        output = run_command(cmd)
        # Output: "Submitted batch job 12345"
        return output.strip().split()[-1]
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e}")
        return None


def cancel_job(job_id: str) -> bool:
    """Cancel a job using scancel. Returns True on success."""
    cmd = ["scancel", job_id]
    try:
        run_command(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to cancel job {job_id}: {e}")
        return False
