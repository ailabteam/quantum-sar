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
