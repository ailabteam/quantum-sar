import numpy as np

def build_qubo_matrix(wrapped_phase_image):
    """
    Xây dựng ma trận Q cho bài toán Phase Unwrapping (phiên bản đơn giản k in {0, 1}).

    Input:
        wrapped_phase_image (np.array): ảnh pha bị gói 2D.

    Output:
        Q (dict): Ma trận QUBO dưới dạng dictionary.
                  Key là tuple (i, j) của các biến, value là hệ số.
    """
    height, width = wrapped_phase_image.shape
    num_variables = height * width
    Q = {} # Sử dụng dict để lưu trữ ma trận thưa (sparse matrix)

    # Hàm tiện ích để chuyển từ tọa độ (r, c) sang chỉ số biến 1D
    def to_idx(r, c):
        return r * width + c

    # Lặp qua tất cả các pixel để xây dựng các phần tử của ma trận Q
    for r in range(height):
        for c in range(width):
            # Lấy chỉ số của biến hiện tại
            k_idx = to_idx(r, c)

            # --- Xét các cặp hàng xóm ---
            # 1. Hàng xóm bên phải
            if c + 1 < width:
                neighbor_idx = to_idx(r, c + 1)

                # Tính hằng số C_ij
                # C = (wrapped(i) - wrapped(j)) / (2*pi)
                delta_phase = wrapped_phase_image[r, c] - wrapped_phase_image[r, c+1]
                C = np.round(delta_phase / (2 * np.pi))

                # Từ (k_i - k_j + C)^2 = k_i^2 - 2k_i*k_j + 2C*k_i + k_j^2 - 2C*k_j + C^2
                # Vì k^2 = k cho biến nhị phân, ta có:
                # = k_i - 2k_i*k_j + 2C*k_i + k_j - 2C*k_j  (bỏ hằng số C^2)
                # = (1 + 2C)k_i + (1 - 2C)k_j - 2k_i*k_j

                # Thêm vào Q
                Q[(k_idx, k_idx)] = Q.get((k_idx, k_idx), 0) + (1 + 2 * C)
                Q[(neighbor_idx, neighbor_idx)] = Q.get((neighbor_idx, neighbor_idx), 0) + (1 - 2 * C)
                Q[(k_idx, neighbor_idx)] = Q.get((k_idx, neighbor_idx), 0) - 2.0

            # 2. Hàng xóm bên dưới
            if r + 1 < height:
                neighbor_idx = to_idx(r + 1, c)

                delta_phase = wrapped_phase_image[r, c] - wrapped_phase_image[r+1, c]
                C = np.round(delta_phase / (2 * np.pi))

                Q[(k_idx, k_idx)] = Q.get((k_idx, k_idx), 0) + (1 + 2 * C)
                Q[(neighbor_idx, neighbor_idx)] = Q.get((neighbor_idx, neighbor_idx), 0) + (1 - 2 * C)
                Q[(k_idx, neighbor_idx)] = Q.get((k_idx, neighbor_idx), 0) - 2.0

    return Q

# Dán hàm này vào file quantum_sar/qubo_builder.py

# Trong file quantum_sar/qubo_builder.py

def build_qubo_matrix_multibit(wrapped_phase_image, num_bits=2, offset=0):
    """
    Xây dựng ma trận Q, sử dụng nhiều bit và có offset để biểu diễn số âm.
    
    k_i = (sum_{p=0}^{num_bits-1} 2^p * b_{i,p}) - offset
    
    Args:
        wrapped_phase_image (np.array): ảnh pha bị gói.
        num_bits (int): Số bit để biểu diễn mỗi biến.
        offset (int): Giá trị offset để dịch chuyển dải giá trị của k.
                      Ví dụ: num_bits=2, offset=1 -> k in {-1, 0, 1, 2}.
    
    Returns:
        Q (dict): Ma trận QUBO.
    """
    height, width = wrapped_phase_image.shape
    num_pixels = height * width
    Q = {}

    def to_qubo_idx(pixel_idx, bit_idx):
        return pixel_idx * num_bits + bit_idx
    def to_pixel_idx(r, c):
        return r * width + c

    for r in range(height):
        for c in range(width):
            # Xét hàng xóm (chỉ cần làm 1 lần cho mỗi cặp)
            neighbors = []
            if c + 1 < width: neighbors.append((r, c + 1))
            if r + 1 < height: neighbors.append((r + 1, c))
            
            for r_n, c_n in neighbors:
                i = to_pixel_idx(r, c)
                j = to_pixel_idx(r_n, c_n)
                
                delta_phase = wrapped_phase_image[r, c] - wrapped_phase_image[r_n, c_n]
                C_ij = np.round(delta_phase / (2 * np.pi))
                
                # Mục tiêu: minimize (k_i - k_j + C_ij)^2
                # k_i = k'_i - offset, k_j = k'_j - offset
                # -> ( (k'_i - offset) - (k'_j - offset) + C_ij )^2
                # -> (k'_i - k'_j + C_ij)^2
                # -> Công thức QUBO cho k' giống hệt như trước!
                # Chỉ có bước tái tạo là thay đổi.
                
                # Khai triển (k'_i - k'_j)^2 = k'_i^2 - 2k'_i*k'_j + k'_j^2
                for p in range(num_bits):
                    for q in range(num_bits):
                        idx_ip, idx_iq = to_qubo_idx(i, p), to_qubo_idx(i, q)
                        idx_jp, idx_jq = to_qubo_idx(j, p), to_qubo_idx(j, q)
                        
                        Q[(idx_ip, idx_iq)] = Q.get((idx_ip, idx_iq), 0) + (2**p * 2**q)
                        Q[(idx_jp, idx_jq)] = Q.get((idx_jp, idx_jq), 0) + (2**p * 2**q)
                        Q[(idx_ip, idx_jq)] = Q.get((idx_ip, idx_jq), 0) - 2 * (2**p * 2**q)

                # Khai triển 2*C_ij*(k'_i - k'_j)
                for p in range(num_bits):
                    idx_ip = to_qubo_idx(i, p)
                    idx_jp = to_qubo_idx(j, p)
                    # Đưa vào đường chéo
                    Q[(idx_ip, idx_ip)] = Q.get((idx_ip, idx_ip), 0) + 2 * C_ij * (2**p)
                    Q[(idx_jp, idx_jp)] = Q.get((idx_jp, idx_jp), 0) - 2 * C_ij * (2**p)
    return Q

if __name__ == '__main__':
    # Tạo một ảnh wrapped phase 3x3 đơn giản để test
    test_image = np.array([
        [ 0.1,  0.2,  3.0], # 3.0 gần -3.14 (bước nhảy)
        [-3.0,  0.3,  0.4], # -3.0 gần 3.14
        [ 0.2,  0.1, -0.1]
    ])

    print("Test wrapped phase image (3x3):")
    print(test_image)

    # Xây dựng ma trận QUBO
    qubo_matrix = build_qubo_matrix(test_image)

    print("\nGenerated QUBO matrix (in dictionary form):")
    for (i, j), val in qubo_matrix.items():
        print(f"  Q[({i}, {j})] = {val}")

    num_vars = test_image.shape[0] * test_image.shape[1]
    print(f"\nQUBO problem has {num_vars} variables.")
