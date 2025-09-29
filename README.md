# QuantumSAR: A QUBO Framework for InSAR Phase Unwrapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**QuantumSAR** is an open-source research toolkit for exploring the application of quantum and quantum-inspired optimization algorithms to Synthetic Aperture Radar (SAR) and Interferometric SAR (InSAR) data processing challenges.

This initial version focuses on formulating the classic **InSAR Phase Unwrapping** problem as a **Quadratic Unconstrained Binary Optimization (QUBO)** model, making it solvable by quantum annealers and modern classical heuristic solvers.

This repository contains the Python library and experimental scripts associated with our upcoming preprint paper. Our primary goal is to provide a foundational bridge between the remote sensing and quantum computing communities.

## 📖 The Scientific Story

Interferometric SAR (InSAR) is a powerful remote sensing technique for measuring ground deformation, but it relies on solving a challenging inverse problem known as phase unwrapping. Traditionally, this is tackled with classical algorithms. We asked the question: *Can we formulate this problem in a language that quantum computers can understand?*

This project documents our journey:
1.  **Formulation:** We successfully mapped the phase unwrapping problem, which seeks to minimize phase differences between adjacent pixels, into a QUBO model.
2.  **Enhancement:** We developed and tested multiple encoding schemes, including a multi-bit representation to handle large phase jumps and a "Robust" formulation designed to mitigate the effects of noise.
3.  **Analysis:** Through a series of rigorous, statistically-validated experiments on synthetic data, we analyzed the performance of our QUBO models under various conditions (surface slope, noise levels) and compared them against a standard classical algorithm.

Our key finding is that while formulating phase unwrapping as a QUBO is feasible, achieving the accuracy of highly-optimized classical algorithms remains a significant challenge on classical simulators. This highlights the inherent complexity of the problem and underscores the potential need for real quantum hardware to unlock performance advantages.

**The core contribution of this work is not to beat classical methods today, but to provide the foundational model and open-source tools for the quantum-ready SAR processing of tomorrow.**

## 🚀 Features

*   **QUBO Builders:** Python functions to automatically construct QUBO matrices from wrapped phase data.
    *   `build_qubo_matrix`: A simple 1-bit model.
    *   `build_qubo_matrix_multibit`: An advanced multi-bit model with offset to represent positive and negative integer phase jumps.
    *   `build_qubo_matrix_robust`: A variation of the multi-bit model designed for better noise resilience.
*   **Experiment Scripts:** A suite of modular scripts in the `examples/` directory to reproduce all the results from our study.
*   **Modular Design:** The `quantum_sar` library is designed to be extensible for future research into other SAR/InSAR optimization problems.

## 🛠️ Installation and Setup

This project is managed with `conda`.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ailabteam/quantum-sar.git
    cd quantum-sar
    ```

2.  **Create and activate the conda environment:**
    We recommend Python 3.11.
    ```bash
    conda create --name quantumsar python=3.11 -y
    conda activate quantumsar
    ```

3.  **Install required packages:**
    The core dependencies are managed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. See section below.)*

### Creating the `requirements.txt` file

Run the following command in your activated `quantumsar` environment to generate the file:

```bash
pip freeze > requirements.txt
```
The file will contain libraries such as `numpy`, `matplotlib`, `pandas`, `scikit-image`, `dwave-neal`, `seaborn`, `tqdm`, etc.

## ⚡ Running the Experiments

All experiments are designed to be run as modules from the root directory of the project.

### Main Experiment: Noise Robustness Analysis

This is the main experiment that generates the key results for our paper. It performs a statistical analysis by running multiple trials across a range of noise levels.

**Warning:** This script can take a significant amount of time to run (30 minutes to several hours depending on your machine).

```bash
python -m examples.generate_paper_assets
```

After execution, all figures and data tables will be saved in the `results/paper_assets/` directory. The most important output is:

*   `Figure_3_Noise_Sweep_Plot.png`: The plot showing the performance (MSE) of each method vs. noise level.

### Other Experiments

You can also run earlier, simpler experiments:

*   **Classical Baseline:**
    ```bash
    python -m examples.01_classical_baseline
    ```
*   **Robustness Comparison (Single Run):**
    ```bash
    python -m examples.05_robust_experiment
    ```

## 📂 Repository Structure

```
├── quantum_sar/          # The core Python library
│   ├── __init__.py
│   └── qubo_builder.py   # Functions to build QUBO matrices
├── examples/             # Experimental scripts
│   ├── 01_classical_baseline.py
│   ├── ...
│   └── generate_paper_assets.py # Main script for paper results
├── results/              # Output directory (ignored by git)
│   └── paper_assets/     # Final figures and tables for publication
├── .gitignore
├── LICENSE
└── README.md
```

## 🤝 Contributing

We welcome contributions and collaborations! If you have ideas for new QUBO formulations, applications to other SAR problems, or improvements to the existing framework, please feel free to open an issue or submit a pull request.

## 📜 Citation

If you use this code or the concepts from our research in your work, please cite our upcoming preprint (link will be added here once available).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
