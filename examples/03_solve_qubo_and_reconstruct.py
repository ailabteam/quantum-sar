import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ==============================================================================
# PHƯƠNG ÁN B: SỬ DỤNG TRÌNH GIẢI OFFLINE (KHÔNG CẦN API TOKEN)
# Thay thế LeapHybridSampler bằng SimulatedAnnealingSampler từ thư viện 'neal'
# ==============================================================================
from neal import SimulatedAnnealingSampler

# Import hàm build_qubo_matrix từ package 'quantum_sar'
from quantum_sar.qubo_builder import build_qubo_matrix


# --- Configuration ---
OUTPUT_DIR = "results"
DPI = 600

def solve_and_reconstruct(wrapped_phase_image):
    """
    Hàm chính thực hiện:
    1. Xây dựng ma trận QUBO.
    2. Giải QUBO bằng SimulatedAnnealingSampler (chạy offline trên CPU).
    3. Tái tạo lại ảnh đã mở pha từ kết quả.
    """
    height, width = wrapped_phase_image.shape
    print("Step 1: Building the QUBO matrix...")
    qubo_matrix = build_qubo_matrix(wrapped_phase_image)
    print(f"...QUBO matrix built for {height*width} variables.")

    print("\nStep 2: Solving the QUBO problem using SimulatedAnnealingSampler (Offline)...")
    # Khởi tạo sampler. Sampler này chạy hoàn toàn trên máy của bạn.
    sampler = SimulatedAnnealingSampler()
    
    start_time = time.time()
    # Gửi bài toán QUBO đến sampler và nhận kết quả
    sampleset = sampler.sample_qubo(qubo_matrix, num_reads=10) # num_reads: số lần chạy annealing
    end_time = time.time()
    
    # Lấy lời giải tốt nhất (có năng lượng thấp nhất)
    solution = sampleset.first.sample
    energy = sampleset.first.energy
    
    print(f"...QUBO solved in {end_time - start_time:.4f} seconds.")
    print(f"Lowest energy found: {energy:.4f}")

    print("\nStep 3: Reconstructing the unwrapped phase image...")
    # Chuyển kết quả (một dict) thành một mảng các giá trị k
    k_values_flat = np.array([solution[i] for i in range(height * width)])
    
    # Reshape mảng k về kích thước ảnh gốc
    k_matrix = k_values_flat.reshape((height, width))
    
    # Tái tạo lại ảnh đã mở pha: unwrapped = wrapped + 2*pi*k
    unwrapped_phase_quantum = wrapped_phase_image + 2 * np.pi * k_matrix
    print("...Reconstruction complete.")
    
    return k_matrix, unwrapped_phase_quantum

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Tạo dữ liệu giả lập ---
    print("--- Generating Test Data ---")
    # Sử dụng ảnh nhỏ hơn (ví dụ: 50x50) để chạy nhanh và kiểm tra
    image_size = 50
    x, y = np.ogrid[-5:5:complex(image_size), -5:5:complex(image_size)] 
    ground_truth_phase = (x**2 - y**2) * 0.5
    wrapped_phase = np.angle(np.exp(1j * ground_truth_phase))
    print(f"Test data ({image_size}x{image_size}) generated.")

    # --- Giải và tái tạo ---
    k_matrix_result, unwrapped_quantum_result = solve_and_reconstruct(wrapped_phase)

    # --- Trực quan hóa và lưu kết quả ---
    print("\n--- Visualizing and Saving Results ---")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    
    vmin = ground_truth_phase.min()
    vmax = ground_truth_phase.max()

    axes[0].set_title('1. Ground Truth')
    im0 = axes[0].imshow(ground_truth_phase, cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].set_title('2. Wrapped Input')
    im1 = axes[1].imshow(wrapped_phase, cmap='viridis')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].set_title('3. "Quantum" Result (k-values)')
    im2 = axes[2].imshow(k_matrix_result, cmap='gray') # k-values là 0 hoặc 1
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].set_title('4. Reconstructed Phase')
    im3 = axes[3].imshow(unwrapped_quantum_result, cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    # Đặt tên file output rõ ràng hơn
    output_fig_path = os.path.join(OUTPUT_DIR, f"03_simulated_annealing_unwrapping_{image_size}x{image_size}.png")
    plt.savefig(output_fig_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Result visualization saved to {output_fig_path}")

    # --- Đánh giá ---
    # Hiệu chỉnh offset trước khi tính MSE
    offset = np.mean(ground_truth_phase - unwrapped_quantum_result)
    mse = np.mean((ground_truth_phase - (unwrapped_quantum_result + offset))**2)
    print(f"\nMean Squared Error (MSE) of Simulated Annealing method: {mse:.6f}")
