# File: examples/generate_paper_assets.py

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from skimage.restoration import unwrap_phase
from neal import SimulatedAnnealingSampler
import seaborn as sns
import tqdm # Thư viện để hiển thị thanh tiến trình

# Import các hàm builder từ package của chúng ta
from quantum_sar.qubo_builder import (
    build_qubo_matrix,
    build_qubo_matrix_multibit,
    build_qubo_matrix_robust
)

# ==============================================================================
# SCRIPT TỔNG HỢP TẠO TÀI SẢN CHO BÀI BÁO
# ==============================================================================

# --- 1. CẤU HÌNH CHUNG ---
OUTPUT_DIR = "results/paper_assets"
DPI = 600 # Chất lượng cao cho paper
plt.style.use('seaborn-v0_8-whitegrid')

# Các hàm helper (đã được kiểm chứng)
def evaluate_result(ground_truth, unwrapped):
    offset = np.mean(ground_truth - unwrapped)
    mse = np.mean((ground_truth - (unwrapped + offset))**2)
    return mse

def run_qubo_solver(wrapped_phase, builder_func, builder_args):
    qubo = builder_func(wrapped_phase, **builder_args)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo, num_reads=10) # num_reads trong sampler
    solution = sampleset.first.sample

    height, width = wrapped_phase.shape
    num_bits = builder_args.get('num_bits', 1)
    offset = builder_args.get('offset', 0)

    k_prime_values = np.zeros(height * width)
    for i in range(height * width):
        val = 0
        for p in range(num_bits):
            val += (2**p) * solution.get(i * num_bits + p, 0)
        k_prime_values[i] = val

    k_values = k_prime_values - offset
    k_matrix = k_values.reshape((height, width))
    unwrapped = wrapped_phase + 2 * np.pi * k_matrix
    return unwrapped

# --- 2. HÀM TẠO CÁC FIGURE & TABLE ---

def generate_figure_1():
    print("Generating Figure 1: The InSAR Phase Unwrapping Problem...")
    size = 100
    x, y = np.ogrid[-5:5:complex(size), -5:5:complex(size)]
    gt_phase = (x**2 - y**2) * 0.8
    wrapped_phase = np.angle(np.exp(1j * gt_phase))
    unwrapped_phase = unwrap_phase(wrapped_phase)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    titles = ["(a) Ground Truth Phase", "(b) Wrapped Phase", "(c) Unwrapped Phase"]
    data = [gt_phase, wrapped_phase, unwrapped_phase]
    cmaps = ['viridis', 'viridis', 'viridis']

    for i, ax in enumerate(axes):
        im = ax.imshow(data[i], cmap=cmaps[i])
        ax.set_title(titles[i], fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_1_Problem_Illustration.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def generate_table_2_and_figure_4_data():
    print("\nGenerating Table 2 & Figure 4 Data: Ablation Study...")
    size = 30
    x, y = np.ogrid[-5:5:complex(size), -5:5:complex(size)]
    gt_phase = (x**2 - y**2) * 1.5 # Dữ liệu dốc
    wrapped_phase = np.angle(np.exp(1j * gt_phase))

    results = {}
    table_data = []

    # Classical
    unwrapped_cl = unwrap_phase(wrapped_phase)
    mse_cl = evaluate_result(gt_phase, unwrapped_cl)
    results['Classical'] = unwrapped_cl
    table_data.append({'Method': 'Classical', 'MSE': mse_cl})

    # QUBO 1-bit
    unwrapped_q1 = run_qubo_solver(wrapped_phase, build_qubo_matrix, {})
    mse_q1 = evaluate_result(gt_phase, unwrapped_q1)
    results['QUBO 1-bit'] = unwrapped_q1
    table_data.append({'Method': 'QUBO 1-bit', 'MSE': mse_q1})

    # QUBO 3-bit (no offset)
    unwrapped_q3_no = run_qubo_solver(wrapped_phase, build_qubo_matrix_multibit, {'num_bits': 3, 'offset': 0})
    mse_q3_no = evaluate_result(gt_phase, unwrapped_q3_no)
    results['QUBO 3-bit (no offset)'] = unwrapped_q3_no
    table_data.append({'Method': 'QUBO 3-bit (no offset)', 'MSE': mse_q3_no})

    # QUBO 3-bit (with offset)
    unwrapped_q3_off = run_qubo_solver(wrapped_phase, build_qubo_matrix_multibit, {'num_bits': 3, 'offset': 4})
    mse_q3_off = evaluate_result(gt_phase, unwrapped_q3_off)
    results['QUBO 3-bit (offset=4)'] = unwrapped_q3_off
    table_data.append({'Method': 'QUBO 3-bit (offset=4)', 'MSE': mse_q3_off})

    # Save Table 2
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "Table_2_Ablation_Study.csv"), index=False, float_format='%.4f')
    print("Table 2 saved to CSV.")

    # Return data for Figure 4
    return gt_phase, wrapped_phase, results

def generate_figure_3_and_main_results():
    print("\nGenerating Figure 3 & Main Results: Noise Sweep Analysis...")
    config = {
        'image_size': 30,
        'n_bits': 3,
        'offset': 4,
        'noise_levels': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'num_runs': 10 # Số lần chạy thống kê
    }

    all_runs_data = []

    pbar = tqdm.tqdm(total=len(config['noise_levels']) * config['num_runs'], desc="Noise Sweep Runs")


    for noise in config['noise_levels']:
        for run in range(config['num_runs']):
            x, y = np.ogrid[-5:5:complex(config['image_size']), -5:5:complex(config['image_size'])]
            gt_phase = (x**2 - y**2) * 0.8
            if noise > 0:
                n = np.random.normal(0, noise, size=(config['image_size'], config['image_size']))
                wrapped_phase = np.angle(np.exp(1j * (gt_phase + n)))
            else:
                wrapped_phase = np.angle(np.exp(1j * gt_phase))

            # Classical
            unwrapped_cl = unwrap_phase(wrapped_phase)
            mse_cl = evaluate_result(gt_phase, unwrapped_cl)
            all_runs_data.append({'noise_level': noise, 'method': 'Classical', 'mse': mse_cl})

            # QUBO L2-norm
            unwrapped_l2 = run_qubo_solver(wrapped_phase, build_qubo_matrix_multibit, {'num_bits': config['n_bits'], 'offset': config['offset']})
            mse_l2 = evaluate_result(gt_phase, unwrapped_l2)
            all_runs_data.append({'noise_level': noise, 'method': 'QUBO L2-norm', 'mse': mse_l2})

            # QUBO Robust
            unwrapped_robust = run_qubo_solver(wrapped_phase, build_qubo_matrix_robust, {'num_bits': config['n_bits'], 'offset': config['offset']})
            mse_robust = evaluate_result(gt_phase, unwrapped_robust)
            all_runs_data.append({'noise_level': noise, 'method': 'QUBO Robust', 'mse': mse_robust})

            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(all_runs_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "Figure_3_Noise_Sweep_Raw_Data.csv"), index=False)
    print("Raw data for Figure 3 saved to CSV.")

    # Plot Figure 3
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='noise_level', y='mse', hue='method', marker='o', errorbar='sd', err_style="band")
    plt.yscale('log')
    plt.title(f"Performance under Varying Noise (Avg. of {config['num_runs']} runs)", fontsize=16)
    plt.xlabel("Noise Level (Std. Dev.)", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE, log scale)", fontsize=12)
    plt.legend(title='Method')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_Noise_Sweep_Plot.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Figure 3 plot saved.")

    return df # Return data for visual comparison figure

def generate_visual_comparison_figure(fig_data_noisy, fig_data_steep):
    print("\nGenerating Figure 4: Visual Comparisons...")
    # Figure 4a: Noisy data
    gt_phase_n, wrapped_phase_n, results_n_all = fig_data_noisy
    # Filter to only methods of interest
    results_n = {
        'Ground Truth': gt_phase_n,
        'Wrapped Input': wrapped_phase_n,
        'Classical': results_n_all['Classical'],
        'QUBO L2-norm': results_n_all['QUBO L2-norm'],
        'QUBO Robust': results_n_all['QUBO Robust']
    }

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharex=True, sharey=True)
    fig.suptitle("Visual Comparison on a Noisy Interferogram (Noise Level=0.2)", fontsize=16, y=1.02)

    vmin = gt_phase_n.min(); vmax = gt_phase_n.max()
    for i, (name, data) in enumerate(results_n.items()):
        ax = axes[i]
        im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        mse = evaluate_result(gt_phase_n, data) if name not in ['Ground Truth', 'Wrapped Input'] else None
        title = f"{name}\n" + (f"MSE: {mse:.2f}" if mse is not None else "")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_4_Visual_Comparison_Noisy.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def create_noisy_dataset(size, noise_level):
    """Tạo dữ liệu ground truth và wrapped phase có nhiễu."""
    print(f"\nCreating dataset of size {size}x{size} with noise level {noise_level}...")
    x, y = np.ogrid[-5:5:complex(size), -5:5:complex(size)]
    ground_truth_phase = (x**2 - y**2) * 0.8
    noise = np.random.normal(0, noise_level, size=(size, size))
    noisy_ground_truth_phase = ground_truth_phase + noise
    wrapped_phase = np.angle(np.exp(1j * noisy_ground_truth_phase))
    print("Dataset created.")
    return ground_truth_phase, wrapped_phase


def main():
    """Main function to generate all assets."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Generate static assets
    generate_figure_1()

    # Generate data-dependent assets
    #_, _, table2_fig_data = generate_table_2_and_figure_4_data()

    # Run the main statistical experiment
    noise_sweep_df = generate_figure_3_and_main_results()

    # Generate the visual comparison figure using one sample from the statistical run
    print("\nGenerating data for final visual comparison figure...")
    gt_vis, wrapped_vis = create_noisy_dataset(40, 0.2)
    classical_vis = unwrap_phase(wrapped_vis)
    l2_vis = run_qubo_solver(wrapped_vis, build_qubo_matrix_multibit, {'num_bits': 3, 'offset': 4})
    robust_vis = run_qubo_solver(wrapped_vis, build_qubo_matrix_robust, {'num_bits': 3, 'offset': 4})

    visual_results = {
        'Ground Truth': gt_vis,
        'Wrapped Input': wrapped_vis,
        'Classical': classical_vis,
        'QUBO L2-norm': l2_vis,
        'QUBO Robust': robust_vis
    }
    generate_visual_comparison_figure( (gt_vis, wrapped_vis, visual_results) , None)

    print("\n\nAll assets for the paper have been generated in:")
    print(os.path.abspath(OUTPUT_DIR))


if __name__ == '__main__':
    # Cài đặt các thư viện cần thiết
    try:
        import seaborn
        import tqdm
    except ImportError:
        print("Please install required libraries: pip install seaborn tqdm")
        exit()

    main()
