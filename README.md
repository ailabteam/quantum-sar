# QuantumSAR: A Python Toolkit for QUBO-based InSAR Phase Unwrapping

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pytest](https://github.com/ailabteam/quantum-sar/actions/workflows/pytest.yml/badge.svg)](https://github.com/ailabteam/quantum-sar/actions/workflows/pytest.yml)
[![JOSS submission](https://joss.theoj.org/papers/10.21105/joss.xxxxx/status.svg)](https://joss.theoj.org/papers/10.21105/joss.xxxxx)

**QuantumSAR** is an open-source research toolkit for exploring the application of quantum and quantum-inspired optimization algorithms to Synthetic Aperture Radar (SAR) and Interferometric SAR (InSAR) data processing challenges. This initial version focuses on formulating the classic **InSAR Phase Unwrapping** problem as a **Quadratic Unconstrained Binary Optimization (QUBO)** model, making it solvable by quantum annealers and modern classical heuristic solvers.

This repository contains the Python library and experimental scripts to reproduce the findings of our research. Our primary goal is to provide a foundational bridge between the remote sensing and quantum computing communities, enabling new avenues of research into quantum-assisted geoscience.

## 📖 Statement of Need

Interferometric SAR (InSAR) is a powerful technique for measuring ground deformation, but it relies on solving a challenging NP-hard problem known as phase unwrapping. While classical algorithms are well-established, exploring novel computational paradigms like quantum computing is essential for tackling future large-scale, noise-intensive datasets.

`QuantumSAR` directly addresses a key barrier to this exploration: the lack of domain-specific tools to translate the phase unwrapping problem into the native language of quantum annealers (QUBO). It provides researchers with a high-level API to construct and experiment with QUBO models, lowering the barrier to entry for the remote sensing community to engage with quantum computing.

## 🚀 Key Features

*   **QUBO Builders**: Functions to automatically construct QUBO matrices from wrapped phase data.
    *   `build_qubo_matrix_multibit`: A standard L2-norm formulation using multi-bit encoding to represent a wide range of integer phase jumps.
    *   `build_qubo_matrix_robust`: An enhanced formulation designed to be more resilient to noise by clipping the influence of anomalous phase differences.
*   **Reproducibility**: A master script (`examples/generate_paper_assets.py`) to reproduce all key figures and statistical analyses from our paper.
*   **Testing**: A suite of `pytest` tests to ensure the correctness and reliability of the core library functions.

## 🛠️ Installation

We recommend using `conda` to manage the environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ailabteam/quantum-sar.git
    cd quantum-sar
    ```

2.  **Create and activate the conda environment:**
    ```bash
    conda create --name quantumsar python=3.11 -y
    conda activate quantumsar
    ```

3.  **Install the package and its dependencies:**
    This project is structured as an installable Python package. Use the following command to install it in "editable" mode, which means any changes you make to the source code will be immediately effective.
    ```bash
    pip install -e .
    ```
    This command reads the `setup.py` and `requirements.txt` files to install everything needed.

## ⚡ Usage

### Quick Start: Basic Usage

Here is a minimal example of how to use `QuantumSAR` to build a QUBO matrix for a simple interferogram and solve it using a classical sampler.

```python
import numpy as np
from neal import SimulatedAnnealingSampler
from quantum_sar.qubo_builder import build_qubo_matrix_robust

# 1. Create a sample 10x10 wrapped phase image with some noise
size = 10
x, y = np.ogrid[:size, :size]
ground_truth = (x - size/2)**2 + (y - size/2)**2
noise = np.random.normal(0, 0.2, (size, size))
wrapped_phase = np.angle(np.exp(1j * (ground_truth + noise)))

# 2. Build the robust QUBO model using 3 bits per variable
num_bits = 3
offset = 4  # Allows k to range from -4 to 3
qubo_matrix = build_qubo_matrix_robust(
    wrapped_phase,
    num_bits=num_bits,
    offset=offset
)

# 3. Solve the QUBO problem using a classical sampler
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(qubo_matrix, num_reads=10)
solution = sampleset.first.sample

print("QUBO problem solved. Lowest energy found:", sampleset.first.energy)
# The `solution` dictionary now contains the optimal binary variable assignments.
# From here, one can reconstruct the unwrapped phase.
```

### Command-Line Interface (CLI)

For non-programmatic use, `QuantumSAR` can be run directly from the command line. This is useful for integrating into shell scripts or processing single files quickly.

**Syntax:**
```bash
python -m quantum_sar -i <input_file.tif> -o <output_file.tif> [options]
```

**Example:**
```bash
# Unwrap an image using the robust model with 4 bits
python -m quantum_sar -i sample_wrapped.tif -o my_result.tif --method robust --bits 4
```

**Options:**
*   `-i, --input`: Path to the input wrapped phase GeoTIFF file (required).
*   `-o, --output`: Path to save the output unwrapped GeoTIFF file (required).
*   `--method`: QUBO formulation to use (`l2` or `robust`, default: `robust`).
*   `-b, --bits`: Number of bits for integer encoding (default: 3).
    
### Reproducing Paper Results

To generate all figures and data from our associated paper, run the master experiment script.

**Warning:** This script is computationally intensive and may run for a long time.

```bash
python -m examples.generate_paper_assets
```
All outputs will be saved to the `results/paper_assets/` directory.

## ✅ Running Tests

To verify that the core functions are working correctly, you can run the automated tests using `pytest`.

```bash
pytest
```

## 📂 Repository Structure

```
├── quantum_sar/          # The core Python library
│   └── qubo_builder.py   # Main functions to build QUBOs
├── examples/             #
│   └── generate_paper_assets.py # Master script for paper results
├── tests/                # Automated tests for the library
│   └── test_qubo_builder.py
├── results/              # Output directory (ignored by git)
├── joss_paper/           # Source for the JOSS publication
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to open issues, fork the repository, and submit pull requests.

## 📜 Citation

If you use `QuantumSAR` in your research, please cite our paper published in the Journal of Open Source Software (JOSS).

*(Note: The citation details and DOI will be added here upon publication.)*

```bibtex
@article{Do_QuantumSAR_2025,
    doi = {10.21105/joss.xxxxx},
    url = {https://doi.org/10.21105/joss.xxxxx},
    year = {2025},
    publisher = {The Open Journal},
    volume = {x},
    number = {xx},
    pages = {xxxxx},
    author = {Phuc Hao Do},
    title = {QuantumSAR: A Python Toolkit for QUBO-based InSAR Phase Unwrapping},
    journal = {Journal of Open Source Software}
}
```

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
