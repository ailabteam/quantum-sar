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
    affiliation: "1" # (Or "1, 2" if you have multiple affiliations)
  # - name: Second Author # Add other authors here if any
  #   orcid: xxxx-xxxx-xxxx-xxxx
  #   affiliation: "2"
affiliations:
 - name: Bonch-Bruevich St. Petersburg State University of Telecommunications, Russia
   index: 1
 # - name: Affiliation of Second Author
 #   index: 2
date: 29 September 2025 # Change to the correct date
bibliography: references.bib
---

# Summary

Interferometric Synthetic Aperture Radar (InSAR) phase unwrapping is a computationally challenging, NP-hard optimization problem central to many geophysical monitoring applications. `QuantumSAR` is a research-focused Python toolkit that provides the first open-source framework for formulating this problem as a Quadratic Unconstrained Binary Optimization (QUBO) model. By mapping phase unwrapping into the native language of quantum annealers, `QuantumSAR` serves as a crucial bridge between the remote sensing and quantum computing communities, enabling new avenues of research into quantum-assisted geoscience.

# Statement of Need

While numerous classical algorithms for phase unwrapping exist [@ghiglia1998two], exploring novel computational paradigms like quantum computing is essential for tackling future large-scale, noise-intensive datasets. Quantum annealers are purpose-built to solve QUBO problems, but a significant barrier to their application in new domains is the lack of domain-specific tools to translate real-world problems into the required QUBO format. `QuantumSAR` directly addresses this gap for the InSAR community. It provides researchers, who may not be experts in quantum computing, with a high-level API to construct and experiment with QUBO models for phase unwrapping. The toolkit is designed for extensibility and serves as a foundational platform for investigating the performance of quantum and quantum-inspired algorithms on a critical remote sensing challenge.

# Software Description

`QuantumSAR` is a lightweight Python library with a modular and easy-to-use design. Its core functionality resides in the `qubo_builder` module, which contains several methods to construct QUBO matrices from a 2D wrapped phase array.

*   **Multi-bit Encoding:** The toolkit implements a multi-bit binary encoding scheme, allowing each pixel's integer ambiguity variable to take on a range of positive and negative values, a necessary feature for handling realistic deformation gradients [@chen2001phase].
*   **Robust Formulation:** A key feature is the implementation of a "Robust" QUBO formulation. This model is specifically designed to be more resilient to noise by clipping the influence of anomalous phase jumps, which our research has shown to be a significant failure mode for standard L2-norm-based models.
*   **Reproducibility:** The repository includes a comprehensive script (`examples/generate_paper_assets.py`) that allows for the full reproduction of our key experimental findings, including a statistical analysis of model performance under varying noise conditions.

The software is built on standard scientific Python libraries such as NumPy and is designed to integrate seamlessly with QUBO solvers like D-Wave's `neal` Simulated Annealing sampler.

# Acknowledgements

We acknowledge the use of D-Wave's Ocean SDK, particularly the `neal` sampler, for the classical simulation of our QUBO models.

# References
