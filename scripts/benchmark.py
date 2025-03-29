#!/usr/bin/env python
"""
Benchmark script for Macrodata Refinement (MDR).

This script measures the performance of MDR's core functions with
different data sizes and configurations.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from tabulate import tabulate

from mdr.core.refinement import (
    RefinementConfig,
    smooth_data,
    remove_outliers,
    impute_missing_values,
    refine_data,
    apply_refinement_pipeline
)
from mdr.core.validation import validate_data
from mdr.core.transformation import transform_data
from mdr.io.readers import read_csv
from mdr.io.writers import write_csv
from mdr.utils.logging import setup_logger, get_logger, LogLevel


@dataclass
class BenchmarkResult:
    """Results of a benchmark run."""
    
    name: str
    data_size: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    
    def __post_init__(self) -> None:
        """Validate benchmark result data."""
        assert isinstance(self.name, str), "name must be a string"
        assert isinstance(self.data_size, int), "data_size must be an integer"
        assert self.data_size > 0, "data_size must be positive"
        assert isinstance(self.mean_time, float), "mean_time must be a floating-point number"
        assert self.mean_time >= 0.0, "mean_time must be non-negative"
        assert isinstance(self.std_time, float), "std_time must be a floating-point number"
        assert self.std_time >= 0.0, "std_time must be non-negative"
        assert isinstance(self.min_time, float), "min_time must be a floating-point number"
        assert self.min_time >= 0.0, "min_time must be non-negative"
        assert isinstance(self.max_time, float), "max_time must be a floating-point number"
        assert self.max_time >= 0.0, "max_time must be non-negative"


def generate_benchmark_data(
    size: int, 
    num_variables: int = 3,
    outlier_rate: float = 0.05,
    missing_rate: float = 0.05,
    noise_level: float = 0.2
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data for benchmarking.
    
    Args:
        size: Number of data points per variable
        num_variables: Number of variables to generate
        outlier_rate: Fraction of data points that will be outliers
        missing_rate: Fraction of data points that will be missing
        noise_level: Standard deviation of the noise
        
    Returns:
        Dictionary mapping variable names to data arrays
    """
    assert isinstance(size, int), "size must be an integer"
    assert size > 0, "size must be positive"
    assert isinstance(num_variables, int), "num_variables must be an integer"
    assert num_variables > 0, "num_variables must be positive"
    assert isinstance(outlier_rate, float), "outlier_rate must be a floating-point number"
    assert 0.0 <= outlier_rate <= 1.0, "outlier_rate must be between 0 and 1"
    assert isinstance(missing_rate, float), "missing_rate must be a floating-point number"
    assert 0.0 <= missing_rate <= 1.0, "missing_rate must be between 0 and 1"
    assert isinstance(noise_level, float), "noise_level must be a floating-point number"
    assert noise_level >= 0.0, "noise_level must be non-negative"
    
    # Generate data dictionary
    data_dict = {}
    
    # Generate time points
    x = np.linspace(0, 10, size)
    
    # Generate variables with different patterns
    for i in range(num_variables):
        # Generate base pattern
        if i % 3 == 0:
            # Sinusoidal pattern
            y = np.sin(x + i/2) + noise_level * np.random.randn(size)
        elif i % 3 == 1:
            # Linear trend with noise
            y = 0.5 * x + i + noise_level * np.random.randn(size)
        else:
            # Exponential pattern
            y = np.exp(x/10) + i + noise_level * np.random.randn(size)
        
        # Add outliers
        num_outliers = int(size * outlier_rate)
        if num_outliers > 0:
            outlier_indices = np.random.choice(size, num_outliers, replace=False)
            y[outlier_indices] = y[outlier_indices] * 5  # Exaggerate values
        
        # Add missing values
        num_missing = int(size * missing_rate)
        if num_missing > 0:
            missing_indices = np.random.choice(
                size, num_missing, replace=False
            )
            y[missing_indices] = np.nan
        
        # Add to data dictionary
        var_name = f"var_{i+1}"
        data_dict[var_name] = y
    
    return data_dict


def benchmark_function(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    num_runs: int = 5
) -> BenchmarkResult:
    """
    Benchmark a function by running it multiple times and measuring execution time.
    
    Args:
        func: Function to benchmark
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        num_runs: Number of times to run the function
        
    Returns:
        BenchmarkResult with timing statistics
    """
    assert callable(func), "func must be callable"
    assert isinstance(args, tuple), "args must be a tuple"
    assert isinstance(kwargs, dict), "kwargs must be a dictionary"
    assert isinstance(num_runs, int), "num_runs must be an integer"
    assert num_runs > 0, "num_runs must be positive"
    
    # Measure execution time for each run
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    times_array = np.array(times)
    mean_time = float(np.mean(times_array))
    std_time = float(np.std(times_array))
    min_time = float(np.min(times_array))
    max_time = float(np.max(times_array))
    
    # Determine data size from the first array argument, if present
    data_size = 0
    for arg in args:
        if isinstance(arg, np.ndarray):
            data_size = len(arg)
            break
        elif isinstance(arg, dict) and all(isinstance(v, np.ndarray) for v in arg.values()):
            data_size = len(next(iter(arg.values())))
            break
    
    # Create and return benchmark result
    return BenchmarkResult(
        name=func.__name__,
        data_size=data_size,
        mean_time=mean_time,
        std_time=std_time,
        min_time=min_time,
        max_time=max_time
    )


def run_benchmarks(
    data_sizes: List[int],
    num_runs: int = 5,
    output_dir: Optional[str] = None
) -> Dict[str, List[BenchmarkResult]]:
    """
    Run benchmarks for different MDR functions with varying data sizes.
    
    Args:
        data_sizes: List of data sizes to test
        num_runs: Number of runs for each benchmark
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping function names to lists of benchmark results
    """
    assert isinstance(data_sizes, list), "data_sizes must be a list"
    assert all(isinstance(size, int) and size > 0 for size in data_sizes), \
        "all data sizes must be positive integers"
    assert isinstance(num_runs, int), "num_runs must be an integer"
    assert num_runs > 0, "num_runs must be positive"
    if output_dir is not None:
        assert isinstance(output_dir, str), "output_dir must be a string"
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    setup_logger(level=LogLevel.INFO)
    logger = get_logger()
    
    # Create default refinement config
    config = RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=2.5,
        imputation_method="linear",
        normalization_type="minmax"
    )
    
    # Initialize results dictionary
    results = {
        "smooth_data": [],
        "remove_outliers": [],
        "impute_missing_values": [],
        "refine_data": [],
        "apply_refinement_pipeline": [],
        "validate_data": [],
        "transform_data": []
    }
    
    # Run benchmarks for each data size
    for size in data_sizes:
        logger.info(f"Running benchmarks for data size {size}...")
        
        # Generate benchmark data
        single_var_data = np.sin(np.linspace(0, 10, size)) + 0.2 * np.random.randn(size)
        single_var_data_with_outliers = single_var_data.copy()
        single_var_data_with_outliers[np.random.choice(size, size//20)] = 10.0
        
        single_var_data_with_missing = single_var_data.copy()
        single_var_data_with_missing[np.random.choice(size, size//20)] = np.nan
        
        single_var_data_with_both = single_var_data.copy()
        single_var_data_with_both[np.random.choice(size, size//20)] = 10.0
        single_var_data_with_both[np.random.choice(size, size//20)] = np.nan
        
        multi_var_data = generate_benchmark_data(
            size=size,
            num_variables=5,
            outlier_rate=0.05,
            missing_rate=0.05
        )
        
        # Benchmark smooth_data
        logger.info("  Benchmarking smooth_data...")
        results["smooth_data"].append(
            benchmark_function(
                smooth_data,
                args=(single_var_data, 0.2),
                kwargs={},
                num_runs=num_runs
            )
        )
        
        # Benchmark remove_outliers
        logger.info("  Benchmarking remove_outliers...")
        results["remove_outliers"].append(
            benchmark_function(
                remove_outliers,
                args=(single_var_data_with_outliers, 2.5),
                kwargs={},
                num_runs=num_runs
            )
        )
        
        # Benchmark impute_missing_values
        logger.info("  Benchmarking impute_missing_values...")
        results["impute_missing_values"].append(
            benchmark_function(
                impute_missing_values,
                args=(single_var_data_with_missing,),
                kwargs={"method": "linear"},
                num_runs=num_runs
            )
        )
        
        # Benchmark refine_data
        logger.info("  Benchmarking refine_data...")
        results["refine_data"].append(
            benchmark_function(
                refine_data,
                args=(single_var_data_with_both, config),
                kwargs={},
                num_runs=num_runs
            )
        )
        
        # Benchmark apply_refinement_pipeline
        logger.info("  Benchmarking apply_refinement_pipeline...")
        results["apply_refinement_pipeline"].append(
            benchmark_function(
                apply_refinement_pipeline,
                args=(multi_var_data, config),
                kwargs={},
                num_runs=num_runs
            )
        )
        
        # Benchmark validate_data
        logger.info("  Benchmarking validate_data...")
        results["validate_data"].append(
            benchmark_function(
                validate_data,
                args=(multi_var_data,),
                kwargs={"checks": ["range", "missing", "outliers"]},
                num_runs=num_runs
            )
        )
        
        # Benchmark transform_data
        logger.info("  Benchmarking transform_data...")
        transformations = [
            {"type": "normalize", "method": "minmax"},
            {"type": "scale", "factor": 2.0, "offset": 1.0}
        ]
        results["transform_data"].append(
            benchmark_function(
                transform_data,
                args=(single_var_data, transformations),
                kwargs={},
                num_runs=num_runs
            )
        )
    
    # Save results if output directory is provided
    if output_dir:
        logger.info(f"Saving results to {output_dir}...")
        
        # Save as CSV
        results_df = pd.DataFrame([
            {
                "function": result.name,
                "data_size": result.data_size,
                "mean_time": result.mean_time,
                "std_time": result.std_time,
                "min_time": result.min_time,
                "max_time": result.max_time
            }
            for func_results in results.values()
            for result in func_results
        ])
        
        results_df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
        
        # Generate plots
        plot_benchmark_results(results, os.path.join(output_dir, "benchmark_plots"))
    
    return results


def plot_benchmark_results(
    results: Dict[str, List[BenchmarkResult]],
    output_dir: Optional[str] = None
) -> None:
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary mapping function names to lists of benchmark results
        output_dir: Directory to save plots
    """
    assert isinstance(results, dict), "results must be a dictionary"
    if output_dir is not None:
        assert isinstance(output_dir, str), "output_dir must be a string"
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure for all functions
    plt.figure(figsize=(12, 8))
    
    for func_name, func_results in results.items():
        # Extract data sizes and mean times
        data_sizes = [result.data_size for result in func_results]
        mean_times = [result.mean_time for result in func_results]
        
        # Plot mean time vs. data size
        plt.plot(data_sizes, mean_times, marker='o', label=func_name)
    
    plt.xlabel("Data Size (number of points)")
    plt.ylabel("Mean Execution Time (seconds)")
    plt.title("MDR Performance Benchmark")
    plt.grid(True)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    
    # Save the plot if output directory is provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, "all_functions.png"), dpi=300)
    
    plt.close()
    
    # Create individual plots for each function
    for func_name, func_results in results.items():
        plt.figure(figsize=(10, 6))
        
        # Extract data sizes and timing statistics
        data_sizes = [result.data_size for result in func_results]
        mean_times = [result.mean_time for result in func_results]
        min_times = [result.min_time for result in func_results]
        max_times = [result.max_time for result in func_results]
        
        # Plot mean time with error bars
        plt.errorbar(
            data_sizes, 
            mean_times, 
            yerr=[[m - min_t for m, min_t in zip(mean_times, min_times)],
                  [max_t - m for m, max_t in zip(mean_times, max_times)]],
            marker='o',
            capsize=5,
            label=func_name
        )
        
        plt.xlabel("Data Size (number of points)")
        plt.ylabel("Execution Time (seconds)")
        plt.title(f"Performance of {func_name}")
        plt.grid(True)
        plt.xscale("log")
        plt.yscale("log")
        
        # Add best fit line for scaling analysis
        log_sizes = np.log(data_sizes)
        log_times = np.log(mean_times)
        coeffs = np.polyfit(log_sizes, log_times, 1)
        poly = np.poly1d(coeffs)
        plt.plot(
            data_sizes,
            np.exp(poly(log_sizes)),
            'r--',
            label=f"Fit: O(n^{coeffs[0]:.2f})"
        )
        
        plt.legend()
        
        # Save the plot if output directory is provided
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{func_name}.png"), dpi=300)
        
        plt.close()


def print_benchmark_results(
    results: Dict[str, List[BenchmarkResult]]
) -> None:
    """
    Print benchmark results in a table format.
    
    Args:
        results: Dictionary mapping function names to lists of benchmark results
    """
    assert isinstance(results, dict), "results must be a dictionary"
    
    # Prepare table data
    table_data = []
    
    for func_name, func_results in results.items():
        for result in func_results:
            table_data.append([
                func_name,
                result.data_size,
                f"{result.mean_time:.6f}",
                f"{result.std_time:.6f}",
                f"{result.min_time:.6f}",
                f"{result.max_time:.6f}"
            ])
    
    # Sort table by function name and data size
    table_data.sort(key=lambda row: (row[0], row[1]))
    
    # Print table
    headers = ["Function", "Data Size", "Mean Time (s)", "Std Dev (s)", "Min Time (s)", "Max Time (s)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark MDR functions")
    
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 1000, 10000, 100000],
        help="Data sizes to benchmark"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs for each benchmark"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run benchmarks."""
    # Parse command-line arguments
    args = parse_args()
    
    # Run benchmarks
    results = run_benchmarks(
        data_sizes=args.sizes,
        num_runs=args.runs,
        output_dir=args.output_dir
    )
    
    # Print results
    print_benchmark_results(results)
    
    # Print location of output files
    if args.output_dir:
        print(f"\nBenchmark results saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()