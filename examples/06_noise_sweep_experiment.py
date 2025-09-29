import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from skimage.restoration import unwrap_phase
from neal import SimulatedAnnealingSampler

# Import các hàm builder
from quantum_sar.qubo_builder import build_qubo_matrix_multibit, build_qubo_matrix_robust

# ==============================================================================
# SCRIPT THÍ NGHIỆM KHẢO SÁT ẢNH HƯỞNG CỦA NHIỄU
# Chạy các phương pháp trên nhiều mức độ nhiễu và vẽ biểu đồ so sánh.
# ==============================================================================

# --- 1. CẤU HÌNH THÍ NGHIỆM ---
OUTPUT_DIR = "results"
DPI = 300
IMAGE_SIZE = 30  # Giảm size để chạy nhanh hơn qua nhiều mức nhiễu
N_BITS = 3
OFFSET = 4
NUM_READS = 10
RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "07_noise_sweep_results.csv")

# Các mức nhiễu cần khảo sát
NOISE_LEVELS_TO_TEST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# (Các hàm helper create_dataset, run_solver, evaluate_result giữ nguyên từ script trước)
# ... (Copy-paste chúng vào đây hoặc import nếu bạn đã module hóa chúng)
# Để cho gọn, tôi sẽ copy chúng vào đây.

def create_noisy_dataset(size, noise_level):
    x, y = np.ogrid[-5:5:complex(size), -5:5:complex(size)]
    ground_truth_phase = (x**2 - y**2) * 0.8
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, size=(size, size))
        noisy_ground_truth_phase = ground_truth_phase + noise
    else:
        noisy_ground_truth_phase = ground_truth_phase
    wrapped_phase = np.angle(np.exp(1j * noisy_ground_truth_phase))
    return ground_truth_phase, wrapped_phase

def run_qubo_solver(wrapped_phase, method_name, num_bits, offset):
    height, width = wrapped_phase.shape
    builder_func = build_qubo_matrix_multibit if method_name == 'L2-norm' else build_qubo_matrix_robust
    qubo = builder_func(wrapped_phase, num_bits=num_bits, offset=offset)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo, num_reads=NUM_READS)
    solution = sampleset.first.sample
    k_prime_values = np.zeros(height*width)
    for i in range(height*width):
        val = 0;
        for p in range(num_bits):
            val += (2**p) * solution.get(i * num_bits + p, 0)
        k_prime_values[i] = val
    k_values = k_prime_values - offset
    k_matrix = k_values.reshape((height, width))
    unwrapped = wrapped_phase + 2 * np.pi * k_matrix
    return unwrapped

def evaluate_result(ground_truth, unwrapped):
    offset = np.mean(ground_truth - unwrapped)
    mse = np.mean((ground_truth - (unwrapped + offset))**2)
    return mse

def run_full_experiment():
    """
    Hàm chính để chạy vòng lặp thí nghiệm và lưu kết quả vào file CSV.
    """
    results_list = []
    
    for noise_level in NOISE_LEVELS_TO_TEST:
        print(f"\n========================================================")
        print(f"  RUNNING EXPERIMENT FOR NOISE LEVEL: {noise_level:.2f}")
        print(f"========================================================")

        # Mỗi lần lặp, tạo bộ dữ liệu mới để kết quả khách quan
        gt_phase, wrapped_phase = create_noisy_dataset(IMAGE_SIZE, noise_level)

        # --- 1. Classical Baseline ---
        start_time = time.time()
        unwrapped_cl = unwrap_phase(wrapped_phase)
        time_cl = time.time() - start_time
        mse_cl = evaluate_result(gt_phase, unwrapped_cl)
        results_list.append({'noise_level': noise_level, 'method': 'Classical', 'mse': mse_cl, 'time': time_cl})
        print(f"Classical           - MSE: {mse_cl:.4f}, Time: {time_cl:.4f}s")

        # --- 2. QUBO L2-norm ---
        start_time = time.time()
        unwrapped_l2 = run_qubo_solver(wrapped_phase, 'L2-norm', N_BITS, OFFSET)
        time_l2 = time.time() - start_time
        mse_l2 = evaluate_result(gt_phase, unwrapped_l2)
        results_list.append({'noise_level': noise_level, 'method': 'QUBO L2-norm', 'mse': mse_l2, 'time': time_l2})
        print(f"QUBO L2-norm        - MSE: {mse_l2:.4f}, Time: {time_l2:.4f}s")

        # --- 3. QUBO Robust ---
        start_time = time.time()
        unwrapped_robust = run_qubo_solver(wrapped_phase, 'Robust L2-norm', N_BITS, OFFSET)
        time_robust = time.time() - start_time
        mse_robust = evaluate_result(gt_phase, unwrapped_robust)
        results_list.append({'noise_level': noise_level, 'method': 'QUBO Robust', 'mse': mse_robust, 'time': time_robust})
        print(f"QUBO Robust         - MSE: {mse_robust:.4f}, Time: {time_robust:.4f}s")

    # Lưu kết quả vào file CSV
    df = pd.DataFrame(results_list)
    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nExperiment results saved to {RESULTS_CSV_PATH}")
    return df

def plot_results(df):
    """
    Hàm để đọc file CSV và vẽ biểu đồ kết quả.
    """
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid') # Dùng style cho đẹp
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = df['method'].unique()
    for method in methods:
        method_df = df[df['method'] == method]
        ax.plot(method_df['noise_level'], method_df['mse'], marker='o', linestyle='-', label=method)

    ax.set_xlabel("Noise Level (Standard Deviation of Gaussian Noise)", fontsize=12)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
    ax.set_title(f"Performance Comparison of Unwrapping Methods under Varying Noise\n(Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}, QUBO Model: {N_BITS}-bit)", fontsize=14)
    ax.set_yscale('log') # Dùng thang log cho Y để dễ nhìn sự khác biệt
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--")
    
    plot_filename = os.path.join(OUTPUT_DIR, "07_noise_sweep_plot.png")
    plt.savefig(plot_filename, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {plot_filename}")

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Chạy thí nghiệm và lưu file CSV
    results_df = run_full_experiment()
    
    # Vẽ biểu đồ từ kết quả
    plot_results(results_df)
