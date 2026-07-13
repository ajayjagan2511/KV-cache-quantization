"""
main.py — Entry point for the KV-Cache Quantization benchmark suite.

Run both benchmark settings sequentially and display interactive Plotly charts.

Usage:
    python main.py                     # runs both settings
    python main.py --setting original  # runs only the O(N^2) baseline comparison
    python main.py --setting flash     # runs only the Flash Attention comparison
"""

import argparse
import gc
import torch

from src.benchmark.run_benchmark import benchmark_attention, plot_results


def parse_args():
    parser = argparse.ArgumentParser(description="KV-Cache INT8 Quantization Benchmark")
    parser.add_argument(
        "--setting",
        choices=["original", "flash", "both"],
        default="both",
        help="Which benchmark to run (default: both)",
    )
    return parser.parse_args()


def run(setting: str):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    df = benchmark_attention(setting=setting)
    plot_results(df, setting=setting)

    pivot = df.pivot(index="Seq_Label", columns="Model", values=["Latency (ms)", "Peak Memory (MB)"])
    print(pivot.to_string())
    return df


def main():
    args = parse_args()

    settings = ["original", "flash_int8_kv"] if args.setting == "both" else [
        "flash_int8_kv" if args.setting == "flash" else "original"
    ]

    for s in settings:
        run(s)


if __name__ == "__main__":
    main()
