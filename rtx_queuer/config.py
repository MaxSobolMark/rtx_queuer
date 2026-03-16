"""Configuration loading and validation."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    queuer_index: int
    script_path: str
    partition: str
    gpu_type: str
    gpus_per_job: int
    target_jobs: int
    time_limit: str
    poll_interval: int
    job_prefix: str
    qos: str | None

    def __post_init__(self):
        if self.queuer_index < 0:
            raise ValueError("queuer_index must be non-negative")
        if self.gpus_per_job < 1:
            raise ValueError("gpus_per_job must be at least 1")
        if self.target_jobs < 1:
            raise ValueError("target_jobs must be at least 1")
        if self.poll_interval < 1:
            raise ValueError("poll_interval must be at least 1 second")


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(
        queuer_index=data.get("queuer_index", 0),
        script_path=data["script_path"],
        partition=data.get("partition", "rl"),
        gpu_type=data.get("gpu_type", "RTX_PRO_6000"),
        gpus_per_job=data.get("gpus_per_job", 1),
        target_jobs=data.get("target_jobs", 24),
        time_limit=data.get("time_limit", "2-00:00:00"),
        poll_interval=data.get("poll_interval", 30),
        job_prefix=data.get("job_prefix", "rtx_queuer"),
        qos=data.get("qos"),
    )
