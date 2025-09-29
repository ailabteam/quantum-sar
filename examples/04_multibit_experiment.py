import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from skimage.restoration import unwrap_phase
from neal import SimulatedAnnealingSampler

from quantum_sar.qubo_builder import build_qubo_matrix, build_qubo_matrix_multibit

# ==============================================================================
# SCRIPT THÍ NGHIỆM VỚI MÔ HÌNH CẢI TIẾN
# Thêm offset để cho phép k nhận giá trị âm.
# ==============================================================================

# --- 1. CẤU HÌNH THÍ NGHIỆM ---
OUTPUT_DIR = "results"
DPI = 300
IMAGE_SIZE = 30
BITS_TO_TEST = [2, 3, 4] # Bắt đầu từ 2 bit mới có ý nghĩa khi dùng offset
OFFSET_CONFIG = {
    2: 1,  # k' in {0,1,2,3} -> k in {-1,0,1,2}
    3: 4,  # k' in {0..7} -> k in {-4..3}
    4: 8,  # k' in {0..15} -> k in {-8..7}
}
NUM_READS = 10

def create_dataset(size, noise_level=0.15): # Thêm tham số noise_level
    """Tạo dữ liệu ground truth và wrapped phase, thêm nhiễu."""
    print(f"\nCreating dataset of size {size}x{size} with noise level {noise_level}...")
    x, y = np.ogrid[-5:5:complex(size), -5:5:complex(size)]
    
    # Ground Truth Phase (độ dốc vừa phải)
    ground_truth_phase = (x**2 - y**2) * 0.8 
    
    # Thêm nhiễu Gauss vào pha gốc (để mô phỏng lỗi trong tín hiệu)
    noise = np.random.normal(0, noise_level, size=(size, size))
    noisy_ground_truth_phase = ground_truth_phase + noise
    
    # Gói pha bị nhiễu
    wrapped_phase = np.angle(np.exp(1j * noisy_ground_truth_phase))
    print("Dataset created.")
    
    # Chúng ta vẫn dùng ground_truth_phase không nhiễu để tính MSE
    return ground_truth_phase, wrapped_phase 


def run_classical_baseline(wrapped_phase):
    """Chạy thuật toán kinh điển."""
    print("\nRunning classical baseline (skimage.unwrap_phase)...")
    start_time = time.time()
    unwrapped = unwrap_phase(wrapped_phase)
    elapsed_time = time.time() - start_time
    print(f"...done in {elapsed_time:.4f} seconds.")
    return unwrapped, elapsed_time

def run_qubo_solver(wrapped_phase, num_bits, offset):
    """Chạy giải pháp QUBO với số bit và offset được chỉ định."""
    height, width = wrapped_phase.shape
    num_pixels = height * width

    print(f"\n--- Running QUBO Solver ({num_bits}-bit, offset={offset}) ---")
    print("Step 1: Building QUBO matrix...")
    qubo = build_qubo_matrix_multibit(wrapped_phase, num_bits=num_bits, offset=offset)
    num_vars = num_pixels * num_bits

    print(f"...QUBO built for {num_vars} variables.")

    print("Step 2: Solving with SimulatedAnnealingSampler...")
    sampler = SimulatedAnnealingSampler()
    start_time = time.time()
    sampleset = sampler.sample_qubo(qubo, num_reads=NUM_READS)
    elapsed_time = time.time() - start_time
    solution = sampleset.first.sample
    print(f"...solved in {elapsed_time:.4f} seconds.")

    print("Step 3: Reconstructing phase...")
    k_prime_values = np.zeros(num_pixels)
    for i in range(num_pixels):
        val = 0
        for p in range(num_bits):
            qubo_idx = i * num_bits + p
            bit = solution.get(qubo_idx, 0)
            val += (2**p) * bit
        k_prime_values[i] = val

    # Áp dụng offset để có giá trị k cuối cùng
    k_values = k_prime_values - offset
    k_matrix = k_values.reshape((height, width))

    unwrapped = wrapped_phase + 2 * np.pi * k_matrix
    print("...reconstruction complete.")

    return unwrapped, elapsed_time

def evaluate_result(ground_truth, unwrapped):
    """Tính toán MSE sau khi hiệu chỉnh offset toàn cục."""
    offset = np.mean(ground_truth - unwrapped)
    mse = np.mean((ground_truth - (unwrapped + offset))**2)
    return mse

def visualize_comparison(results, filename):
    """Lưu hình ảnh so sánh."""
    num_methods = len(results)
    fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 5), sharex=True, sharey=True)

    vmin = results['Ground Truth']['data'].min()
    vmax = results['Ground Truth']['data'].max()

    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        im = ax.imshow(result['data'], cmap='viridis', vmin=vmin, vmax=vmax)
        mse_val = result.get('mse')
        if isinstance(mse_val, float):
            title = f"{name}\nMSE: {mse_val:.2f}"
        else: # Dành cho Ground Truth và Wrapped
             title = name
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"\nComparison visualization saved to {filename}")

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gt_phase, wrapped_phase = create_dataset(IMAGE_SIZE, noise_level=0.15)

    all_results = {
        "Ground Truth": {"data": gt_phase},
        "Wrapped Input": {"data": wrapped_phase}
    }
    summary_data = []

    unwrapped_cl, time_cl = run_classical_baseline(wrapped_phase)
    mse_cl = evaluate_result(gt_phase, unwrapped_cl)
    all_results["Classical"] = {"data": unwrapped_cl, "mse": mse_cl}
    summary_data.append({"Method": "Classical", "MSE": mse_cl, "Time (s)": time_cl, "Num Bits": "N/A", "Offset": "N/A"})

    # Chạy các thí nghiệm QUBO với offset
    for n_bits in BITS_TO_TEST:
        offset_val = OFFSET_CONFIG.get(n_bits, 0)
        unwrapped_qb, time_qb = run_qubo_solver(wrapped_phase, num_bits=n_bits, offset=offset_val)
        mse_qb = evaluate_result(gt_phase, unwrapped_qb)
        method_name = f"QUBO {n_bits}-bit (off={offset_val})"
        all_results[method_name] = {"data": unwrapped_qb, "mse": mse_qb}
        summary_data.append({"Method": method_name, "MSE": mse_qb, "Time (s)": time_qb, "Num Bits": n_bits, "Offset": offset_val})

    summary_df = pd.DataFrame(summary_data)
    print("\n--- IMPROVED EXPERIMENT SUMMARY ---")
    print(summary_df.to_string(index=False))

    fig_filename = os.path.join(OUTPUT_DIR, f"05_offset_comparison_{IMAGE_SIZE}x{IMAGE_SIZE}.png")
    visualize_comparison(all_results, fig_filename)
