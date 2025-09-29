# file: quantum_sar/qubo_builder.py
"""
Core module for constructing QUBO matrices for InSAR Phase Unwrapping.

This module provides functions to translate the phase unwrapping optimization
problem into a Quadratic Unconstrained Binary Optimization (QUBO) format,
suitable for solving with quantum annealers or classical heuristic solvers like
Simulated Annealing.
"""

import numpy as np
from typing import Dict, Tuple

def build_qubo_matrix(wrapped_phase_image: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Builds a QUBO matrix for phase unwrapping using a simple 1-bit model.

    This formulation assumes the integer ambiguity `k` for each pixel can only
    be 0 or 1. It serves as a foundational proof-of-concept model. The objective
    is to minimize the squared difference of phase values between adjacent pixels.

    Args:
        wrapped_phase_image (np.ndarray): A 2D NumPy array representing the
                                          wrapped phase interferogram, with
                                          values in the range [-pi, pi].

    Returns:
        dict: A dictionary representing the sparse QUBO matrix. Keys are tuples
              of variable indices `(i, j)`, and values are the corresponding
              quadratic coefficients Q_ij.
    """
    height, width = wrapped_phase_image.shape
    Q = {}

    def _to_pixel_idx(r: int, c: int) -> int:
        """Converts 2D image coordinates to a 1D pixel index."""
        return r * width + c

    for r in range(height):
        for c in range(width):
            # Process horizontal neighbors
            if c + 1 < width:
                idx_i = _to_pixel_idx(r, c)
                idx_j = _to_pixel_idx(r, c + 1)
                
                delta_phase = wrapped_phase_image[r, c] - wrapped_phase_image[r, c + 1]
                C = np.round(delta_phase / (2 * np.pi))

                # From (k_i - k_j + C)^2 and using k^2 = k for binary variables,
                # the energy contribution is (1+2C)k_i + (1-2C)k_j - 2k_i*k_j
                Q[(idx_i, idx_i)] = Q.get((idx_i, idx_i), 0) + (1 + 2 * C)
                Q[(idx_j, idx_j)] = Q.get((idx_j, idx_j), 0) + (1 - 2 * C)
                Q[(idx_i, idx_j)] = Q.get((idx_i, idx_j), 0) - 2.0

            # Process vertical neighbors
            if r + 1 < height:
                idx_i = _to_pixel_idx(r, c)
                idx_j = _to_pixel_idx(r + 1, c)

                delta_phase = wrapped_phase_image[r, c] - wrapped_phase_image[r + 1, c]
                C = np.round(delta_phase / (2 * np.pi))
                
                Q[(idx_i, idx_i)] = Q.get((idx_i, idx_i), 0) + (1 + 2 * C)
                Q[(idx_j, idx_j)] = Q.get((idx_j, idx_j), 0) + (1 - 2 * C)
                Q[(idx_i, idx_j)] = Q.get((idx_i, idx_j), 0) - 2.0
    
    return Q


def build_qubo_matrix_multibit(wrapped_phase_image: np.ndarray, num_bits: int = 3, offset: int = 4) -> Dict[Tuple[int, int], float]:
    """Builds a QUBO matrix using a multi-bit binary representation for integers.

    This is the standard L2-norm formulation. It represents each integer ambiguity `k`
    using multiple binary variables and an offset, allowing for a wider range of
    positive and negative phase jumps.
    
    The encoding is: k_i = (sum_{p=0}^{B-1} 2^p * b_{i,p}) - offset

    Args:
        wrapped_phase_image (np.ndarray): A 2D NumPy array of the wrapped phase.
        num_bits (int): The number of binary bits to represent each integer `k`.
        offset (int): An integer offset to shift the range of `k`.
                      E.g., for num_bits=3, offset=4, k can range from -4 to 3.

    Returns:
        dict: The sparse QUBO matrix as a dictionary.
    """
    height, width = wrapped_phase_image.shape
    num_pixels = height * width
    Q = {}

    def _to_qubo_idx(pixel_idx: int, bit_idx: int) -> int:
        """Maps (pixel, bit) to a single QUBO variable index."""
        return pixel_idx * num_bits + bit_idx

    def _to_pixel_idx(r: int, c: int) -> int:
        """Converts 2D image coordinates to a 1D pixel index."""
        return r * width + c

    for r in range(height):
        for c in range(width):
            # Iterate over right and bottom neighbors to avoid double counting
            neighbors = []
            if c + 1 < width: neighbors.append((r, c + 1))
            if r + 1 < height: neighbors.append((r + 1, c))

            for r_n, c_n in neighbors:
                idx_pixel_i = _to_pixel_idx(r, c)
                idx_pixel_j = _to_pixel_idx(r_n, c_n)

                delta_phase = wrapped_phase_image[r, c] - wrapped_phase_image[r_n, c_n]
                C_ij = np.round(delta_phase / (2 * np.pi))

                # The objective is to minimize (k_i - k_j + C_ij)^2.
                # Since k_i = k'_i - offset and k_j = k'_j - offset, the offsets cancel,
                # and the objective becomes (k'_i - k'_j + C_ij)^2, where k' is the
                # positive integer part represented by the bits.

                # Term: (k'_i)^2 + (k'_j)^2 - 2*k'_i*k'_j
                for p in range(num_bits):
                    for q in range(num_bits):
                        idx_ip, idx_iq = _to_qubo_idx(idx_pixel_i, p), _to_qubo_idx(idx_pixel_i, q)
                        idx_jp, idx_jq = _to_qubo_idx(idx_pixel_j, p), _to_qubo_idx(idx_pixel_j, q)
                        
                        weight_pq = (2**p) * (2**q)
                        
                        Q[(idx_ip, idx_iq)] = Q.get((idx_ip, idx_iq), 0) + weight_pq
                        Q[(idx_jp, idx_jq)] = Q.get((idx_jp, idx_jq), 0) + weight_pq
                        Q[(idx_ip, idx_jq)] = Q.get((idx_ip, idx_jq), 0) - 2 * weight_pq

                # Term: 2*C_ij*(k'_i - k'_j)
                for p in range(num_bits):
                    idx_ip = _to_qubo_idx(idx_pixel_i, p)
                    idx_jp = _to_qubo_idx(idx_pixel_j, p)
                    
                    weight_p = 2 * C_ij * (2**p)
                    
                    # Linear terms go on the diagonal of the QUBO matrix
                    Q[(idx_ip, idx_ip)] = Q.get((idx_ip, idx_ip), 0) + weight_p
                    Q[(idx_jp, idx_jp)] = Q.get((idx_jp, idx_jp), 0) - weight_p
    return Q


def build_qubo_matrix_robust(wrapped_phase_image: np.ndarray, num_bits: int = 3, offset: int = 4) -> Dict[Tuple[int, int], float]:
    """Builds a QUBO matrix using a robust L2-norm formulation.

    This formulation is designed to be more resilient to noise. It works by
    clipping the discretized phase jumps (C_ij) to the range [-1, 1] before
    incorporating them into the QUBO model. This prevents large, noise-induced
    phase jumps from dominating the optimization objective.

    Args:
        wrapped_phase_image (np.ndarray): A 2D NumPy array of the wrapped phase.
        num_bits (int): The number of binary bits to represent each integer `k`.
        offset (int): An integer offset to shift the range of `k`.

    Returns:
        dict: The sparse QUBO matrix as a dictionary.
    """
    height, width = wrapped_phase_image.shape
    num_pixels = height * width
    Q = {}

    def _to_qubo_idx(pixel_idx: int, bit_idx: int) -> int:
        return pixel_idx * num_bits + bit_idx

    def _to_pixel_idx(r: int, c: int) -> int:
        return r * width + c

    for r in range(height):
        for c in range(width):
            neighbors = []
            if c + 1 < width: neighbors.append((r, c + 1))
            if r + 1 < height: neighbors.append((r + 1, c))

            for r_n, c_n in neighbors:
                idx_pixel_i = _to_pixel_idx(r, c)
                idx_pixel_j = _to_pixel_idx(r_n, c_n)

                delta_phase = wrapped_phase_image[r, c] - wrapped_phase_image[r_n, c_n]
                C_ij = np.round(delta_phase / (2 * np.pi))

                # --- The single most important improvement ---
                C_ij_robust = np.clip(C_ij, -1, 1)
                # --------------------------------------------

                # The rest of the formulation is identical to the standard multi-bit version,
                # but using C_ij_robust instead of C_ij.
                for p in range(num_bits):
                    for q in range(num_bits):
                        idx_ip, idx_iq = _to_qubo_idx(idx_pixel_i, p), _to_qubo_idx(idx_pixel_i, q)
                        idx_jp, idx_jq = _to_qubo_idx(idx_pixel_j, p), _to_qubo_idx(idx_pixel_j, q)
                        
                        weight_pq = (2**p) * (2**q)
                        
                        Q[(idx_ip, idx_iq)] = Q.get((idx_ip, idx_iq), 0) + weight_pq
                        Q[(idx_jp, idx_jq)] = Q.get((idx_jp, idx_jq), 0) + weight_pq
                        Q[(idx_ip, idx_jq)] = Q.get((idx_ip, idx_jq), 0) - 2 * weight_pq

                for p in range(num_bits):
                    idx_ip = _to_qubo_idx(idx_pixel_i, p)
                    idx_jp = _to_qubo_idx(idx_pixel_j, p)
                    
                    weight_p = 2 * C_ij_robust * (2**p)
                    
                    Q[(idx_ip, idx_ip)] = Q.get((idx_ip, idx_ip), 0) + weight_p
                    Q[(idx_jp, idx_jp)] = Q.get((idx_jp, idx_jp), 0) - weight_p
    return Q
