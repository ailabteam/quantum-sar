# File: examples/generate_paper_assets.py
"""
Master script to generate all figures and data tables for the QuantumSAR paper.

This script performs the final, statistically robust experiments and saves all
necessary assets to the `results/paper_assets` directory. It is designed to be
the single source of truth for reproducing the paper's key findings.

Warning: This script is computationally intensive and may take a significant
amount of time to complete (30 minutes to several hours).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from skimage.restoration import unwrap_phase
from neal import SimulatedAnnealingSampler
import seaborn as sns
import tqdm
from typing import Dict, Tuple

# Import the core library functions
from quantum_sar.qubo_builder import (
    build_qubo_matrix,
    build_qubo_matrix_multibit,
    build_qubo_matrix_robust
)

# --- 1. CONFIGURATION ---
OUTPUT_DIR = "results/paper_assets"
DPI = 600  # High resolution for publication
plt.style.use('seaborn-v0_8-whitegrid')


# --- 2. HELPER FUNCTIONS ---

def evaluate_result(ground_truth: np.ndarray, unwrapped: np.ndarray) -> float:
    """Calculates the Mean Squared Error (MSE) after correcting for a global offset."""
    offset = np.mean(ground_truth - unwrapped)
    mse = np.mean((ground_truth - (unwrapped + offset))**2)
    return mse

def run_qubo_solver(wrapped_phase: np.ndarray, builder_func, builder_args: Dict) -> np.ndarray:
    """
    Constructs and solves a QUBO model for phase unwrapping.

    Args:
        wrapped_phase (np.ndarray): The 2D wrapped phase image.
        builder_func (callable): The QUBO builder function to use 
                                 (e.g., build_qubo_matrix_robust).
        builder_args (Dict): A dictionary of arguments for the builder function.

    Returns:
        np.ndarray: The 2D unwrapped phase image.
    """
    qubo = builder_func(wrapped_phase, **builder_args)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo, num_reads=10)
    solution = sampleset.first.sample
    
    height, width = wrapped_phase.shape
    num_bits = builder_args.get('num_bits', 1)
    offset = builder_args.get('offset', 0)
    
    # Reconstruct the integer `k` values from the binary solution
    k_prime_values = np.zeros(height * width)
    for i in range(height * width):
        val = 0
        for p in range(num_bits):
            bit_val = solution.get(i * num_bits + p, 0)
            val += (2**p) * bit_val
        k_prime_values[i] = val
        
    k_values = k_prime_values - offset
    k_matrix = k_values.reshape((height, width))
    
    # Reconstruct the final unwrapped phase
    unwrapped = wrapped_phase + 2 * np.pi * k_matrix
    return unwrapped


# --- 3. ASSET GENERATION FUNCTIONS ---

def generate_figure_1_problem_illustration():
    """Generates Figure 1, illustrating the phase unwrapping problem."""
    print("Generating Figure 1: The InSAR Phase Unwrapping Problem...")
    size = 100
    x, y = np.ogrid[-5:5:complex(size), -5:5:complex(size)]
    gt_phase = (x**2 - y**2) * 0.8
    wrapped_phase = np.angle(np.exp(1j * gt_phase))
    # For illustration, we show the ideal unwrapped result
    unwrapped_ideal = unwrap_phase(wrapped_phase)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    
    titles = ["(a) Ground Truth Phase", "(b) Wrapped Phase", "(c) Ideal Unwrapped Phase"]
    data_maps = [gt_phase, wrapped_phase, unwrapped_ideal]
    
    for i, ax in enumerate(axes):
        im = ax.imshow(data_maps[i], cmap='viridis')
        ax.set_title(titles[i], fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_1_Problem_Illustration.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def run_noise_sweep_experiment() -> pd.DataFrame:
    """
    Runs the main statistical experiment over a range of noise levels.

    This is the most computationally intensive part of the script. It iterates
    through noise levels, and for each level, runs multiple trials to gather
    statistics on the performance of each unwrapping method.

    Returns:
        pd.DataFrame: A DataFrame containing the raw results of all runs.
    """
    print("\nGenerating Figure 3 & Main Results: Noise Sweep Analysis...")
    config = {
        'image_size': 30,
        'n_bits': 3,
        'offset': 4,
        'noise_levels': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'num_runs': 10  # Number of statistical runs per noise level
    }
    
    all_runs_data = []
    
    # Setup the progress bar
    total_iterations = len(config['noise_levels']) * config['num_runs']
    pbar = tqdm.tqdm(total=total_iterations, desc="Noise Sweep Runs")
    
    for noise in config['noise_levels']:
        for _ in range(config['num_runs']):
            # 1. Create a new dataset instance for each run
            x, y = np.ogrid[-5:5:complex(config['image_size']), -5:5:complex(config['image_size'])]
            gt_phase = (x**2 - y**2) * 0.8
            if noise > 0:
                noise_instance = np.random.normal(0, noise, size=(config['image_size'], config['image_size']))
                wrapped_phase = np.angle(np.exp(1j * (gt_phase + noise_instance)))
            else:
                wrapped_phase = np.angle(np.exp(1j * gt_phase))
            
            # 2. Run Classical Baseline
            unwrapped_cl = unwrap_phase(wrapped_phase)
            mse_cl = evaluate_result(gt_phase, unwrapped_cl)
            all_runs_data.append({'noise_level': noise, 'method': 'Classical', 'mse': mse_cl})

            # 3. Run QUBO L2-norm
            unwrapped_l2 = run_qubo_solver(wrapped_phase, build_qubo_matrix_multibit, {'num_bits': config['n_bits'], 'offset': config['offset']})
            mse_l2 = evaluate_result(gt_phase, unwrapped_l2)
            all_runs_data.append({'noise_level': noise, 'method': 'QUBO L2-norm', 'mse': mse_l2})

            # 4. Run QUBO Robust
            unwrapped_robust = run_qubo_solver(wrapped_phase, build_qubo_matrix_robust, {'num_bits': config['n_bits'], 'offset': config['offset']})
            mse_robust = evaluate_result(gt_phase, unwrapped_robust)
            all_runs_data.append({'noise_level': noise, 'method': 'QUBO Robust', 'mse': mse_robust})
            
            pbar.update(1)
    
    pbar.close()
    
    df = pd.DataFrame(all_runs_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "Figure_3_Noise_Sweep_Raw_Data.csv"), index=False)
    print("Raw data for Figure 3 saved to CSV.")
    return df

def generate_figure_3_plot(df: pd.DataFrame):
    """Generates Figure 3, the main plot of the paper, from the experimental data."""
    print("Plotting Figure 3...")
    plt.figure(figsize=(10, 6))
    
    # Use seaborn for a professional-looking plot with statistical error bands
    sns.lineplot(data=df, x='noise_level', y='mse', hue='method', marker='o', errorbar='sd', err_style="band")
    
    # Use a logarithmic scale for the y-axis to visualize the large performance gap
    plt.yscale('log')
    plt.title(f"Performance under Varying Noise (Avg. of {df.groupby('noise_level').size().iloc[0]} runs)", fontsize=16)
    plt.xlabel("Noise Level (Std. Dev. of Gaussian Noise)", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE, log scale)", fontsize=12)
    plt.legend(title='Method')
    plt.grid(True, which="both", ls="--")
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_Noise_Sweep_Plot.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Figure 3 plot saved.")

def generate_figure_4_visual_comparison():
    """Generates Figure 4, a visual comparison of methods on one noisy example."""
    print("\nGenerating Figure 4: Visual Comparison...")
    size = 40
    noise_level = 0.2 # A representative noise level
    
    # Create the dataset
    x, y = np.ogrid[-5:5:complex(size), -5:5:complex(size)]
    gt_phase = (x**2 - y**2) * 0.8
    noise = np.random.normal(0, noise_level, size=(size, size))
    wrapped_phase = np.angle(np.exp(1j * (gt_phase + noise)))
    
    # Run all methods
    unwrapped_cl = unwrap_phase(wrapped_phase)
    unwrapped_l2 = run_qubo_solver(wrapped_phase, build_qubo_matrix_multibit, {'num_bits': 3, 'offset': 4})
    unwrapped_robust = run_qubo_solver(wrapped_phase, build_qubo_matrix_robust, {'num_bits': 3, 'offset': 4})
    
    results = {
        '(a) Ground Truth': gt_phase,
        '(b) Wrapped Input': wrapped_phase,
        '(c) Classical': unwrapped_cl,
        '(d) QUBO L2-norm': unwrapped_l2,
        '(e) QUBO Robust': unwrapped_robust
    }

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharex=True, sharey=True)
    fig.suptitle(f"Visual Comparison on a Noisy Interferogram (Noise Level={noise_level})", fontsize=16, y=1.03)
    
    vmin, vmax = gt_phase.min(), gt_phase.max()
    
    for i, (name, data) in enumerate(results.items()):
        ax = axes[i]
        im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Calculate MSE for result plots
        if name not in ['(a) Ground Truth', '(b) Wrapped Input']:
            mse = evaluate_result(gt_phase, data)
            title = f"{name}\nMSE: {mse:.2f}"
        else:
            title = name
            
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_4_Visual_Comparison_Noisy.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved.")


def main():
    """Main function to generate all assets for the paper."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    # Generate static assets
    generate_figure_1_problem_illustration()
    
    # Run the main statistical experiment and save the raw data
    noise_sweep_df = run_noise_sweep_experiment()
    
    # Generate the main plot from the data
    generate_figure_3_plot(noise_sweep_df)
    
    # Generate the final visual comparison figure
    generate_figure_4_visual_comparison()
    
    print("\n\nAll assets for the paper have been successfully generated in:")
    print(os.path.abspath(OUTPUT_DIR))


if __name__ == '__main__':
    # Ensure necessary visualization libraries are installed
    try:
        import seaborn
        import tqdm
    except ImportError as e:
        print(f"Error: Missing required library. Please run 'pip install seaborn tqdm'")
        print(f"Original error: {e}")
        exit()
        
    main()
