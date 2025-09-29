import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from skimage.restoration import unwrap_phase
from neal import SimulatedAnnealingSampler

# Import tất cả các hàm builder của chúng ta
from quantum_sar.qubo_builder import build_qubo_matrix_multibit, build_qubo_matrix_robust

# ==============================================================================
# SCRIPT THÍ NGHIỆM CUỐI CÙNG
# So sánh hiệu quả của phương pháp Robust L2-norm trên dữ liệu nhiễu.
# ==============================================================================

# --- 1. CẤU HÌNH THÍ NGHIỆM ---
OUTPUT_DIR = "results"
DPI = 300
IMAGE_SIZE = 40 # Thử với ảnh lớn hơn một chút
NOISE_LEVEL = 0.2 # Tăng nhiễu lên một chút để thử thách các phương pháp
N_BITS = 3 # Chỉ tập trung vào 3-bit để so sánh công bằng
OFFSET = 4 # Offset tương ứng cho 3-bit
NUM_READS = 10

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

def run_solver(wrapped_phase, method_name, num_bits, offset):
    """Hàm chung để chạy các phương pháp QUBO."""
    height, width = wrapped_phase.shape
    
    print(f"\n--- Running QUBO Solver ({method_name}) ---")
    print("Step 1: Building QUBO matrix...")
    if method_name == 'L2-norm':
        qubo = build_qubo_matrix_multibit(wrapped_phase, num_bits=num_bits, offset=offset)
    elif method_name == 'Robust L2-norm':
        qubo = build_qubo_matrix_robust(wrapped_phase, num_bits=num_bits, offset=offset)
    else:
        raise ValueError("Unknown method name")
    
    print(f"...QUBO built for {height*width*num_bits} variables.")

    print("Step 2: Solving with SimulatedAnnealingSampler...")
    sampler = SimulatedAnnealingSampler()
    start_time = time.time()
    sampleset = sampler.sample_qubo(qubo, num_reads=NUM_READS)
    elapsed_time = time.time() - start_time
    solution = sampleset.first.sample
    print(f"...solved in {elapsed_time:.4f} seconds.")

    print("Step 3: Reconstructing phase...")
    k_prime_values = np.zeros(height*width)
    for i in range(height*width):
        val = 0
        for p in range(num_bits):
            qubo_idx = i * num_bits + p
            bit = solution.get(qubo_idx, 0)
            val += (2**p) * bit
        k_prime_values[i] = val
    
    k_values = k_prime_values - offset
    k_matrix = k_values.reshape((height, width))
    unwrapped = wrapped_phase + 2 * np.pi * k_matrix
    print("...reconstruction complete.")
    
    return unwrapped, elapsed_time

def evaluate_result(ground_truth, unwrapped):
    offset = np.mean(ground_truth - unwrapped)
    mse = np.mean((ground_truth - (unwrapped + offset))**2)
    return mse

def visualize_final_comparison(results, filename):
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
        else:
             title = name
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFinal comparison visualization saved to {filename}")

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gt_phase, wrapped_phase = create_noisy_dataset(IMAGE_SIZE, NOISE_LEVEL)
    
    all_results = {
        "Ground Truth": {"data": gt_phase},
        "Wrapped Input": {"data": wrapped_phase}
    }
    summary_data = []

    # 1. Chạy Baseline kinh điển
    print("\nRunning classical baseline (skimage.unwrap_phase)...")
    start_cl = time.time()
    unwrapped_cl = unwrap_phase(wrapped_phase)
    time_cl = time.time() - start_cl
    mse_cl = evaluate_result(gt_phase, unwrapped_cl)
    all_results["Classical"] = {"data": unwrapped_cl, "mse": mse_cl}
    summary_data.append({"Method": "Classical", "MSE": mse_cl, "Time (s)": time_cl})

    # 2. Chạy phương pháp QUBO L2-norm (cũ)
    unwrapped_l2, time_l2 = run_solver(wrapped_phase, 'L2-norm', N_BITS, OFFSET)
    mse_l2 = evaluate_result(gt_phase, unwrapped_l2)
    all_results["QUBO L2-norm (3-bit)"] = {"data": unwrapped_l2, "mse": mse_l2}
    summary_data.append({"Method": "QUBO L2-norm (3-bit)", "MSE": mse_l2, "Time (s)": time_l2})
    
    # 3. Chạy phương pháp QUBO Robust L2-norm (mới)
    unwrapped_robust, time_robust = run_solver(wrapped_phase, 'Robust L2-norm', N_BITS, OFFSET)
    mse_robust = evaluate_result(gt_phase, unwrapped_robust)
    all_results["QUBO Robust (3-bit)"] = {"data": unwrapped_robust, "mse": mse_robust}
    summary_data.append({"Method": "QUBO Robust (3-bit)", "MSE": mse_robust, "Time (s)": time_robust})

    # 4. In bảng kết quả cuối cùng
    summary_df = pd.DataFrame(summary_data)
    print("\n--- FINAL ROBUSTNESS EXPERIMENT SUMMARY ---")
    print(summary_df.to_string(index=False))
    
    # 5. Lưu hình ảnh
    fig_filename = os.path.join(OUTPUT_DIR, f"06_robust_comparison_{IMAGE_SIZE}x{IMAGE_SIZE}_noise{NOISE_LEVEL}.png")
    visualize_final_comparison(all_results, fig_filename)
