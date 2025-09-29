import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
import os

# --- Configuration ---
# Thư mục lưu kết quả
OUTPUT_DIR = "results"
# Độ phân giải ảnh
DPI = 600

def main():
    """
    Hàm chính thực hiện toàn bộ quy trình:
    1. Tạo dữ liệu giả lập.
    2. Chạy thuật toán mở pha kinh điển.
    3. Lưu hình ảnh và kết quả.
    """
    print("Starting the classical baseline script...")

    # Đảm bảo thư mục output tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved in: {OUTPUT_DIR}")

    # --- 1. Tạo dữ liệu giả lập ---
    # Tạo một bề mặt hình yên ngựa (saddle)
    x, y = np.ogrid[-10:10:100j, -10:10:100j]
    ground_truth_phase = (x**2 - y**2)
    # Chuẩn hóa để giá trị pha không quá lớn
    ground_truth_phase = (ground_truth_phase - ground_truth_phase.min()) / (ground_truth_phase.max() - ground_truth_phase.min()) * 6 * np.pi
    print("Step 1: Ground truth and wrapped phase data created.")

    # "Gói" pha lại để tạo dữ liệu đầu vào
    wrapped_phase = np.angle(np.exp(1j * ground_truth_phase))
    
    # --- 2. Lưu hình ảnh dữ liệu đầu vào ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    im0 = axes[0].imshow(ground_truth_phase, cmap='viridis')
    axes[0].set_title('Ground Truth Phase')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(wrapped_phase, cmap='viridis')
    axes[1].set_title('Wrapped Phase')
    fig.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    input_fig_path = os.path.join(OUTPUT_DIR, "01_input_data.png")
    plt.savefig(input_fig_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig) # Đóng hình để giải phóng bộ nhớ
    print(f"Step 2: Saved input data visualization to {input_fig_path}")

    # --- 3. Sử dụng thuật toán kinh điển để mở pha ---
    print("Step 3: Running classical phase unwrapping algorithm...")
    unwrapped_classical = unwrap_phase(wrapped_phase)
    print("...Classical unwrapping complete.")

    # --- 4. Lưu hình ảnh so sánh kết quả ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    
    # Set common color limits for better comparison
    vmin = ground_truth_phase.min()
    vmax = ground_truth_phase.max()

    im0 = axes[0].imshow(ground_truth_phase, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('1. Ground Truth')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(wrapped_phase, cmap='viridis')
    axes[1].set_title('2. Wrapped Input')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(unwrapped_classical, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title('3. Classical Unwrapped')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_fig_path = os.path.join(OUTPUT_DIR, "02_classical_unwrapping_result.png")
    plt.savefig(output_fig_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Step 4: Saved result comparison to {output_fig_path}")

    # --- 5. Đánh giá và in kết quả ---
    # Loại bỏ sai số do offset toàn cục trước khi tính MSE
    # Vì phase unwrapping có thể có offset 2*pi*k so với ground truth
    offset = np.mean(ground_truth_phase - unwrapped_classical)
    mse = np.mean((ground_truth_phase - (unwrapped_classical + offset))**2)
    print("\n--- Evaluation Metric ---")
    print(f"Mean Squared Error (MSE) after offset correction: {mse:.6f}")
    print("-------------------------\n")
    print("Script finished successfully.")

if __name__ == '__main__':
    main()
