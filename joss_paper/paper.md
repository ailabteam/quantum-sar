---
title: 'QuantumSAR: A Python Toolkit for QUBO-based InSAR Phase Unwrapping'
tags:
  - Python
  - InSAR
  - phase unwrapping
  - quantum computing
  - QUBO
  - remote sensing
authors:
  - name: Phuc Hao Do
    orcid: 0000-0003-0645-0021
    affiliation: 1
#  - name: Second Author # Example for adding another author
#    orcid: 0000-0000-0000-0000
#    affiliation: 2
affiliations:
 - name: Bonch-Bruevich St. Petersburg State University of Telecommunications, Russia
   index: 1
#  - name: Affiliation of Second Author, Country
#    index: 2
date: 29 September 2025 # Please change to the correct submission date
bibliography: references.bib
---

# Summary

Interferometric Synthetic Aperture Radar (InSAR) phase unwrapping is a computationally challenging, NP-hard optimization problem central to many geophysical monitoring applications. `QuantumSAR` is a research-focused Python toolkit providing an open-source framework to formulate this problem as a Quadratic Unconstrained Binary Optimization (QUBO) model. By mapping phase unwrapping into the native language of quantum annealers, `QuantumSAR` serves as a crucial bridge between the remote sensing and quantum computing communities, enabling new avenues of research into quantum-assisted geoscience.

# Statement of Need

While numerous classical algorithms for phase unwrapping exist [@ghiglia1998two], exploring novel computational paradigms like quantum computing is essential for tackling future large-scale, noise-intensive datasets. Quantum annealers are purpose-built to solve QUBO problems, but a significant barrier to their application in new domains is the lack of domain-specific tools to translate real-world problems into the required QUBO format. `QuantumSAR` directly addresses this gap for the InSAR community. It provides researchers, who may not be experts in quantum computing, with a high-level API and command-line interface to construct and experiment with QUBO models for phase unwrapping. The toolkit is designed for extensibility and serves as a foundational platform for investigating the performance of quantum and quantum-inspired algorithms on a critical remote sensing challenge.

# Software Description and Features

`QuantumSAR` is a lightweight, well-tested Python library with a modular design. Its core functionality resides in the `qubo_builder` module, which contains methods to construct QUBO matrices from a 2D wrapped phase array. Key features include:

*   **Command-Line Interface (CLI):** Provides a user-friendly way to unwrap GeoTIFF images directly from the terminal without writing Python code.
*   **Multi-bit Encoding:** Implements a multi-bit binary encoding scheme with an offset, allowing each pixel's integer ambiguity variable to take on a range of positive and negative values, a necessary feature for handling realistic deformation gradients [@chen2001phase].
*   **Robust Formulation:** A key feature is the implementation of a "Robust" QUBO formulation, which is specifically designed to be more resilient to noise by clipping the influence of anomalous phase jumps.
*   **Reproducibility:** The repository includes a comprehensive script (`examples/generate_paper_assets.py`) that allows for the full reproduction of our key experimental findings, including a statistical analysis of model performance.

The software is built on standard scientific Python libraries (NumPy, Rasterio) and integrates seamlessly with QUBO solvers like D-Wave's `neal` Simulated Annealing sampler.

# Acknowledgements

We acknowledge the use of D-Wave's Ocean SDK, particularly the `neal` sampler, for the classical simulation of our QUBO models.

# References
