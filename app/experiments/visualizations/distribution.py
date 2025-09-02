"""
Distribution visualizations for experiments.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


def create_text_length_comparison(results: Dict[str, Any]) -> plt.Figure:
    """
    Create a comparison plot of text length distributions across strategies.
    
    Args:
        results: Dictionary with strategy results
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    strategies = []
    means = []
    stds = []
    mins = []
    maxs = []
    
    for strategy, data in results.items():
        if "metrics" in data:
            metrics = data["metrics"]
            strategies.append(strategy)
            means.append(metrics.get("text_length_mean", 0))
            stds.append(metrics.get("text_length_std", 0))
            mins.append(metrics.get("text_length_min", 0))
            maxs.append(metrics.get("text_length_max", 0))
    
    if not strategies:
        return fig
    
    x = np.arange(len(strategies))
    
    # Plot 1: Mean text length with error bars
    ax1 = axes[0]
    ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Text Length (characters)')
    ax1.set_title('Average Text Length by Strategy (with std deviation)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax1.text(i, mean + std + 50, f'{mean:.0f}', ha='center', va='bottom')
    
    # Plot 2: Min-Max range
    ax2 = axes[1]
    width = 0.35
    ax2.bar(x - width/2, mins, width, label='Min', alpha=0.7, color='lightcoral')
    ax2.bar(x + width/2, maxs, width, label='Max', alpha=0.7, color='darkseagreen')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Text Length (characters)')
    ax2.set_title('Text Length Range (Min/Max) by Strategy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_performance_comparison(results: Dict[str, Any]) -> plt.Figure:
    """
    Create a performance comparison plot across strategies.
    
    Args:
        results: Dictionary with strategy results
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    strategies = []
    total_times = []
    per_product_times = []
    
    for strategy, data in results.items():
        if "metrics" in data:
            metrics = data["metrics"]
            strategies.append(strategy)
            total_times.append(metrics.get("total_time", 0))
            per_product_times.append(metrics.get("avg_time_per_product", 0) * 1000)  # Convert to ms
    
    if not strategies:
        return fig
    
    x = np.arange(len(strategies))
    
    # Plot 1: Total processing time
    ax1 = axes[0]
    bars1 = ax1.bar(x, total_times, alpha=0.7, color='coral')
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Total Processing Time by Strategy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time in zip(bars1, total_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{time:.1f}s', ha='center', va='bottom')
    
    # Plot 2: Per-product processing time
    ax2 = axes[1]
    bars2 = ax2.bar(x, per_product_times, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Time (milliseconds)')
    ax2.set_title('Average Time per Product')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time in zip(bars2, per_product_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time:.1f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig