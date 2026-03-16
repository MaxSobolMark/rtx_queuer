"""Multi-queuer coordination logic."""

from .slurm import Job


def parse_job_index(job_name: str, prefix: str) -> int | None:
    """Extract queuer index from job name.

    Job name format: {prefix}_{index}_{uuid}
    Example: rtx_queuer_0_a1b2c3d4 -> 0
    """
    if not job_name.startswith(prefix + "_"):
        return None
    try:
        parts = job_name.split("_")
        # prefix_index_uuid -> index is at position after prefix parts
        prefix_parts = len(prefix.split("_"))
        return int(parts[prefix_parts])
    except (IndexError, ValueError):
        return None


def get_my_jobs(jobs: list[Job], prefix: str, my_index: int) -> list[Job]:
    """Get jobs belonging to this queuer instance."""
    return [j for j in jobs if parse_job_index(j.name, prefix) == my_index]


def get_all_queuer_jobs(jobs: list[Job], prefix: str) -> dict[int, list[Job]]:
    """Group all queuer jobs by their index."""
    result: dict[int, list[Job]] = {}
    for job in jobs:
        idx = parse_job_index(job.name, prefix)
        if idx is not None:
            if idx not in result:
                result[idx] = []
            result[idx].append(job)
    return result


def get_pending_external_jobs(jobs: list[Job], prefix: str) -> list[Job]:
    """Get pending jobs NOT from any queuer instance."""
    return [
        j for j in jobs
        if j.is_pending and parse_job_index(j.name, prefix) is None
    ]


def get_external_jobs_blocked_on_resources(jobs: list[Job], prefix: str) -> list[Job]:
    """Get external jobs that are actually blocked waiting for GPUs.

    Only these jobs need us to deallocate - jobs pending for other reasons
    (Priority, QOS limits, etc.) are not blocked by our jobs.
    """
    return [
        j for j in jobs
        if j.is_blocked_on_resources and parse_job_index(j.name, prefix) is None
    ]


def calculate_gpus_to_deallocate(
    pending_external: list[Job],
    my_index: int,
    all_queuer_jobs: dict[int, list[Job]],
) -> int:
    """Calculate how many GPUs this queuer should deallocate.

    Lower-index queuers deallocate first. This queuer only deallocates
    if lower-index queuers can't satisfy the full demand.
    """
    total_gpus_needed = sum(job.gpus for job in pending_external)
    if total_gpus_needed == 0:
        return 0

    # Sort queuers by index
    sorted_indices = sorted(all_queuer_jobs.keys())

    gpus_handled = 0
    for queuer_idx in sorted_indices:
        queuer_jobs = all_queuer_jobs[queuer_idx]
        running_gpus = sum(j.gpus for j in queuer_jobs if j.is_running)

        if queuer_idx == my_index:
            # How many GPUs should this queuer deallocate?
            remaining_needed = total_gpus_needed - gpus_handled
            return min(remaining_needed, running_gpus)

        gpus_handled += running_gpus
        if gpus_handled >= total_gpus_needed:
            return 0  # Lower-index queuers handled it

    return 0


def select_jobs_to_cancel(jobs: list[Job], gpus_needed: int) -> list[Job]:
    """Select running jobs to cancel to free the required GPUs."""
    if gpus_needed <= 0:
        return []

    to_cancel = []
    gpus_freed = 0

    # Cancel running jobs to free GPUs
    running = [j for j in jobs if j.is_running]
    for job in running:
        if gpus_freed >= gpus_needed:
            break
        to_cancel.append(job)
        gpus_freed += job.gpus

    return to_cancel


def select_pending_jobs_to_cancel(jobs: list[Job], max_to_keep: int) -> list[Job]:
    """Select pending jobs to cancel, keeping at most max_to_keep."""
    pending = [j for j in jobs if j.is_pending]
    if len(pending) <= max_to_keep:
        return []
    # Cancel excess pending jobs
    return pending[max_to_keep:]
