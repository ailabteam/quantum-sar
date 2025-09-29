# File: quantum_sar/main.py
"""
Main command-line interface (CLI) and entry point for the QuantumSAR toolkit.

This module allows users to run the phase unwrapping process directly from the
terminal, handling file I/O for common geospatial raster formats (like GeoTIFF).
"""

import argparse
import numpy as np
import rasterio
import time

from .qubo_builder import build_qubo_matrix_multibit, build_qubo_matrix_robust
from neal import SimulatedAnnealingSampler

# Helper function to reconstruct phase from solution (we can refactor this later)
def _reconstruct_phase(wrapped_phase, solution, num_bits, offset):
    height, width = wrapped_phase.shape
    k_prime_values = np.zeros(height * width)
    for i in range(height * width):
        val = 0
        for p in range(num_bits):
            bit_val = solution.get(i * num_bits + p, 0)
            val += (2**p) * bit_val
        k_prime_values[i] = val
    
    k_values = k_prime_values - offset
    k_matrix = k_values.reshape((height, width))
    unwrapped = wrapped_phase + 2 * np.pi * k_matrix
    return unwrapped

def unwrap_image(input_path: str, output_path: str, method: str, bits: int):
    """
    Main function to load, unwrap, and save an image.
    """
    print(f"--- QuantumSAR Phase Unwrapping Tool ---")
    
    # 1. Load data using rasterio
    print(f"Reading input file: {input_path}")
    with rasterio.open(input_path) as src:
        # Assume single band interferogram
        wrapped_phase = src.read(1).astype(np.float64)
        profile = src.profile
        print(f"Image loaded. Shape: {wrapped_phase.shape}")

    # 2. Select QUBO builder function
    if method == 'l2':
        builder_func = build_qubo_matrix_multibit
        print("Using standard L2-norm QUBO model.")
    elif method == 'robust':
        builder_func = build_qubo_matrix_robust
        print("Using Robust L2-norm QUBO model.")
    else:
        raise ValueError("Method must be 'l2' or 'robust'")

    # 3. Build and solve QUBO
    offset = 2**(bits - 1) # A sensible default offset, e.g., 4 for 3 bits
    print(f"Building QUBO with {bits} bits (offset={offset})...")
    qubo = builder_func(wrapped_phase, num_bits=bits, offset=offset)
    
    print("Solving with Simulated Annealing Sampler...")
    start_time = time.time()
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo, num_reads=10)
    solution = sampleset.first.sample
    elapsed = time.time() - start_time
    print(f"Solver finished in {elapsed:.2f} seconds.")

    # 4. Reconstruct the result
    unwrapped_phase = _reconstruct_phase(wrapped_phase, solution, bits, offset)

    # 5. Save the output using rasterio
    profile.update(dtype=rasterio.float64, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(unwrapped_phase.astype(rasterio.float64), 1)
    print(f"Unwrapped phase saved to: {output_path}")


def main_cli():
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="QuantumSAR: InSAR Phase Unwrapping using QUBO.")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to the input wrapped phase GeoTIFF file."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path to save the output unwrapped GeoTIFF file."
    )
    parser.add_argument(
        "--method", type=str, choices=['l2', 'robust'], default='robust', help="QUBO formulation to use ('l2' or 'robust')."
    )
    parser.add_argument(
        "--bits", "-b", type=int, default=3, help="Number of bits for integer encoding."
    )

    args = parser.parse_args()
    unwrap_image(args.input, args.output, args.method, args.bits)

if __name__ == '__main__':
    main_cli()
