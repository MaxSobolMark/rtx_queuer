#!/usr/bin/env python3
"""GPU placeholder job that maintains high utilization without disk I/O.

Runs a training loop on randomly generated data to keep the GPU active.
All data is generated in memory - no disk access.
"""

import argparse
import time

import torch
import torch.nn as nn


class LargeModel(nn.Module):
    """A reasonably large model to keep GPU busy."""

    def __init__(self, hidden_size: int = 2048, num_layers: int = 12):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            ])
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.layers(x))


def run_training_loop(
    batch_size: int = 64,
    hidden_size: int = 2048,
    num_layers: int = 12,
    steps: int = 1_000_000_000,
    log_interval: int = 100,
) -> None:
    """Run a training loop on random data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = LargeModel(hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Starting training for {steps:,} steps...")
    print()

    start_time = time.time()
    total_loss = 0.0

    for step in range(1, steps + 1):
        # Generate random data on GPU directly (no disk I/O)
        x = torch.randn(batch_size, hidden_size, device=device)
        target = torch.randn(batch_size, hidden_size, device=device)

        # Forward pass
        output = model(x)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / log_interval
            steps_per_sec = step / elapsed

            if device.type == "cuda":
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(
                    f"Step {step:,} | Loss: {avg_loss:.4f} | "
                    f"Steps/s: {steps_per_sec:.1f} | "
                    f"Mem: {mem_used:.1f}/{mem_reserved:.1f} GB"
                )
            else:
                print(
                    f"Step {step:,} | Loss: {avg_loss:.4f} | "
                    f"Steps/s: {steps_per_sec:.1f}"
                )

            total_loss = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU placeholder training job")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--steps", type=int, default=1_000_000_000, help="Training steps")
    parser.add_argument("--log-interval", type=int, default=100, help="Log every N steps")
    args = parser.parse_args()

    run_training_loop(
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        steps=args.steps,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
