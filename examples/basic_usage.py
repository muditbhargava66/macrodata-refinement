#!/usr/bin/env python
"""
Basic usage example for Macrodata Refinement (MDR).

This script demonstrates the core functionality of MDR through
a simple end-to-end workflow.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from mdr.core.refinement import RefinementConfig, apply_refinement_pipeline
from mdr.core.validation import validate_data
from mdr.core.transformation import transform_data
from mdr.io.readers import read_csv
from mdr.io.writers import write_csv
from mdr.visualization.plots import (
    plot_time_series,
    plot_refinement_comparison,
    plot_validation_results,
    plot_correlation_matrix,
    PlotConfig,
    save_plot
)
from mdr.utils.logging import setup_logger, get_logger, LogLevel


def generate_sample_data(
    num_points: int = 100,
    outlier_rate: float = 0.05,
    missing_rate: float = 0.05,
    noise_level: float = 0.2,
    output_dir: str = "./data"
) -> str:
    """
    Generate sample data for demonstration purposes.
    
    Args:
        num_points: Number of data points to generate
        outlier_rate: Fraction of data points that will be outliers
        missing_rate: Fraction of data points that will be missing
        noise_level: Standard deviation of the added noise
        output_dir: Directory to save the data
        
    Returns:
        Path to the generated CSV file
    """
    assert isinstance(num_points, int), "num_points must be an integer"
    assert num_points > 0, "num_points must be positive"
    assert isinstance(outlier_rate, float), "outlier_rate must be a floating-point number"
    assert 0.0 <= outlier_rate <= 1.0, "outlier_rate must be between 0 and 1"
    assert isinstance(missing_rate, float), "missing_rate must be a floating-point number"
    assert 0.0 <= missing_rate <= 1.0, "missing_rate must be between 0 and 1"
    assert isinstance(noise_level, float), "noise_level must be a floating-point number"
    assert noise_level >= 0.0, "noise_level must be non-negative"
    assert isinstance(output_dir, str), "output_dir must be a string"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate time points
    time = np.linspace(0, 10, num_points)
    
    # Generate clean sinusoidal data
    temperature = 20.0 + 5.0 * np.sin(time) + noise_level * np.random.randn(num_points)
    pressure = 101.3 + 0.5 * np.cos(time) + noise_level * 0.1 * np.random.randn(num_points)
    humidity = 50.0 + 10.0 * np.sin(time + 1.0) + noise_level * 3.0 * np.random.randn(num_points)
    
    # Add outliers
    num_outliers = int(num_points * outlier_rate)
    if num_outliers > 0:
        outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
        
        # Add outliers to temperature (high values)
        temperature[outlier_indices[:num_outliers//3]] += 20.0
        
        # Add outliers to pressure (low values)
        pressure[outlier_indices[num_outliers//3:2*num_outliers//3]] -= 10.0
        
        # Add outliers to humidity (high values)
        humidity[outlier_indices[2*num_outliers//3:]] += 30.0
    
    # Add missing values
    num_missing = int(num_points * missing_rate)
    if num_missing > 0:
        missing_indices = np.random.choice(num_points, num_missing * 3, replace=False)
        
        # Add missing values to temperature
        temperature[missing_indices[:num_missing]] = np.nan
        
        # Add missing values to pressure
        pressure[missing_indices[num_missing:2*num_missing]] = np.nan
        
        # Add missing values to humidity
        humidity[missing_indices[2*num_missing:]] = np.nan
    
    # Create a DataFrame
    df = pd.DataFrame({
        "time": time,
        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity
    })
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "sample_data.csv")
    df.to_csv(csv_path, index=False)
    
    return csv_path


def main() -> None:
    """
    Run the example workflow.
    """
    # Set up logging
    setup_logger(level=LogLevel.INFO)
    logger = get_logger()
    
    # Create output directories
    output_dir = "mdr_example_output"
    data_dir = os.path.join(output_dir, "data")
    plot_dir = os.path.join(output_dir, "plots")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Step 1: Generate sample data
    logger.info("Step 1: Generating sample data")
    data_path = generate_sample_data(
        num_points=100,
        outlier_rate=0.05,
        missing_rate=0.05,
        noise_level=0.2,
        output_dir=data_dir
    )
    logger.info(f"Sample data saved to: {data_path}")
    
    # Step 2: Read the data
    logger.info("Step 2: Reading data")
    data_dict = read_csv(data_path)
    
    # Extract time array
    time = data_dict.pop("time")
    
    # Step 3: Visualize the raw data
    logger.info("Step 3: Visualizing raw data")
    fig, ax = plot_time_series(data_dict, time)
    save_plot(fig, os.path.join(plot_dir, "raw_data.png"))
    plt.close(fig)
    
    # Step 4: Validate the data
    logger.info("Step 4: Validating data")
    validation_results = validate_data(
        data_dict,
        checks=["range", "missing", "outliers"],
        params={
            "range": {"min_value": -10.0, "max_value": 200.0},
            "missing": {"threshold": 0.1},
            "outliers": {"threshold": 2.5, "method": "zscore"}
        }
    )
    
    # Print validation results
    for var_name, result in validation_results.items():
        if result.is_valid:
            logger.info(f"Variable '{var_name}' passed validation")
        else:
            logger.warning(f"Variable '{var_name}' failed validation:")
            for msg in result.error_messages:
                logger.warning(f"  - {msg}")
    
    # Visualize validation results
    logger.info("Visualizing validation results")
    
    # Convert validation results to a format suitable for plotting
    plot_results = {}
    for var_name, result in validation_results.items():
        plot_results[var_name] = {
            "is_valid": result.is_valid,
            "error_messages": result.error_messages,
            "statistics": result.statistics or {}
        }
    
    fig, axes = plot_validation_results(plot_results)
    save_plot(fig, os.path.join(plot_dir, "validation_results.png"))
    plt.close(fig)
    
    # Step 5: Refine the data
    logger.info("Step 5: Refining data")
    
    # Create a refinement configuration
    config = RefinementConfig(
        smoothing_factor=0.2,
        outlier_threshold=2.5,
        imputation_method="linear",
        normalization_type="minmax"
    )
    
    # Apply refinement
    refined_data = apply_refinement_pipeline(data_dict, config)
    
    # Visualize the refinement for each variable
    logger.info("Visualizing refinement results")
    for var_name, values in data_dict.items():
        fig, axes = plot_refinement_comparison(
            values,
            refined_data[var_name],
            time
        )
        save_plot(fig, os.path.join(plot_dir, f"refinement_{var_name}.png"))
        plt.close(fig)
    
    # Step 6: Apply transformations
    logger.info("Step 6: Applying transformations")
    transformations = [
        {"type": "normalize", "method": "minmax"},
        {"type": "scale", "factor": 2.0, "offset": 1.0}
    ]
    
    # Apply transformations to each variable
    transformed_data = {}
    for var_name, values in refined_data.items():
        transformed_data[var_name] = transform_data(values, transformations)
    
    # Visualize the transformed data
    logger.info("Visualizing transformed data")
    fig, ax = plot_time_series(transformed_data, time)
    save_plot(fig, os.path.join(plot_dir, "transformed_data.png"))
    plt.close(fig)
    
    # Step 7: Save the results
    logger.info("Step 7: Saving results")
    
    # Add time back to the dictionaries
    data_dict["time"] = time
    refined_data["time"] = time
    transformed_data["time"] = time
    
    # Save raw data
    write_csv(data_dict, os.path.join(data_dir, "raw_data.csv"))
    
    # Save refined data
    write_csv(refined_data, os.path.join(data_dir, "refined_data.csv"))
    
    # Save transformed data
    write_csv(transformed_data, os.path.join(data_dir, "transformed_data.csv"))
    
    # Step 8: Demonstrate using pandas DataFrame with visualization
    logger.info("Step 8: Creating correlation matrix with pandas DataFrame")
    
    # Create a pandas DataFrame from the refined data (without time)
    refined_df = pd.DataFrame({
        k: v for k, v in refined_data.items() if k != "time"
    })
    
    # Create a correlation matrix plot
    fig, ax = plot_correlation_matrix(
        refined_df,
        method="pearson",
        cmap="coolwarm",
        config=PlotConfig(title="Variable Correlations")
    )
    save_plot(fig, os.path.join(plot_dir, "correlation_matrix.png"))
    plt.close(fig)
    
    logger.info(f"All output saved to: {os.path.abspath(output_dir)}")
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()