# File: tests/test_qubo_builder.py

import numpy as np
import pytest # Import pytest framework

# Import các hàm cần test từ thư viện của chúng ta
from quantum_sar.qubo_builder import (
    build_qubo_matrix,
    build_qubo_matrix_multibit,
    build_qubo_matrix_robust,
)

# --- Test Case 1: Simple 1-bit Model ---

def test_build_qubo_matrix_1bit_output_type():
    """Tests if the 1-bit builder returns a dictionary."""
    # Create a tiny 2x1 wrapped phase image
    # phase difference is pi, so C_ij = round(pi / 2pi) = round(0.5) = 1
    test_image = np.array([[0.0], [np.pi]])
    qubo = build_qubo_matrix(test_image)
    assert isinstance(qubo, dict)

# File: tests/test_qubo_builder.py

def test_build_qubo_matrix_1bit_values():
    """Tests the correctness of QUBO values for a simple 1-bit case."""
    # For this test, we choose a delta_phase that makes C non-zero and integer.
    # Let's use delta_phase = -2*pi, which makes C = -1.
    test_image = np.array([[0.0], [2 * np.pi]])
    qubo = build_qubo_matrix(test_image)
    
    # With C = np.round((0 - 2*pi) / (2*pi)) = -1
    # From formula: (1+2C)k_0 + (1-2C)k_1 - 2k_0*k_1
    # With C=-1: (1-2)k_0 + (1+2)k_1 - 2k_0*k_1 = -1*k_0 + 3*k_1 - 2k_0*k_1
    # Expected: Q[0,0] = -1, Q[1,1] = 3, Q[0,1] = -2
    
    assert qubo.get((0, 0), 0) == pytest.approx(-1.0)
    assert qubo.get((1, 1), 0) == pytest.approx(3.0)
    
    # The off-diagonal term can be split between (0,1) and (1,0)
    # So we check their sum.
    off_diagonal_sum = qubo.get((0, 1), 0) + qubo.get((1, 0), 0)
    assert off_diagonal_sum == pytest.approx(-2.0)
    

# --- Test Case 2: Multi-bit Model ---

def test_build_qubo_matrix_multibit_variables():
    """Tests if the multi-bit builder creates the correct number of variables."""
    test_image = np.array([[0.0, 1.0], [2.0, 3.0]]) # 4 pixels
    num_bits = 3
    qubo = build_qubo_matrix_multibit(test_image, num_bits=num_bits, offset=4)

    # Total variables should be num_pixels * num_bits = 4 * 3 = 12
    # We find the highest variable index in the QUBO keys
    max_idx = 0
    for i, j in qubo.keys():
        max_idx = max(max_idx, i, j)

    assert max_idx == (4 * num_bits - 1)


# --- Test Case 3: Robust Model ---

def test_build_qubo_matrix_robust_clipping_effect():
    """Tests if the robust builder correctly clips large C_ij values."""
    # Create a large phase jump where C_ij = round(6.0 / 6.28) = 1.0
    # Then create a huge jump where C_ij = round(12.0 / 6.28) = 2.0

    # C_ij = 1 should be unaffected
    image_c1 = np.array([[0.0], [6.0]])
    qubo_c1 = build_qubo_matrix_robust(image_c1, num_bits=2, offset=1)

    # C_ij = 2 should be clipped to 1
    image_c2 = np.array([[0.0], [12.0]])
    qubo_c2 = build_qubo_matrix_robust(image_c2, num_bits=2, offset=1)

    # The resulting QUBOs should be identical because C=2 was clipped to C=1
    assert qubo_c1 == qubo_c2
