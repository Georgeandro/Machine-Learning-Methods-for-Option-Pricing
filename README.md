# Machine Learning Methods for Option Pricing

This repository contains the official implementation of my undergraduate thesis **"Machine Learning Methods for Option Pricing"**, focusing on solving partial differential equations (PDEs) that arise in financial models using **neural network methods**. The code is written in **PyTorch** and is optimized to run on **Google Colab** for ease of use and GPU acceleration.

---

## Overview

Classical numerical methods for option pricing, such as finite differences or Monte Carlo, can struggle with high-dimensionality or complex volatility models.  
This thesis explores **deep learning-based approaches** for option pricing, leveraging **neural networks** as function approximators for pricing solutions.

We implement and compare methods applied to:
- **Black-Scholes Model**
- **Heston Stochastic Volatility Model**
- **Lifted Heston Model**

A key feature is the use of **Time-Discrete Deep Gradient Flow (TDGF)**:
- **Time discretization**: the PDE is split into time steps.
- **Gated neural architectures**: networks inspired by DGM cells are used to handle the dynamics of the PDE.
- **Whole-domain training**: the neural network is trained across the full space-time domain instead of along trajectories only.

---

## Features

- **Colab-Ready Notebooks** – Easily run on Google Colab with GPU acceleration.
- **PyTorch Implementation** – Modular and extendable deep learning framework.
- **Multiple Financial Models** – Black-Scholes, Heston, and Lifted Heston models.
- **Advanced Training Loop** – Adaptive optimization, energy functional minimization, and loss decomposition.
- **Visualization** – Generate plots of option prices over moneyness and maturities, comparing neural network solutions with analytical or COS-based solutions.

---

## Repository Structure

├── notebooks/
│ ├── black_scholes_tdgf.ipynb # Black-Scholes model with TDGF
│ ├── heston_tdgf.ipynb # Heston model with TDGF
│ └── lifted_heston_tdgf.ipynb # Lifted Heston model with TDGF
│
├── models/
│ ├── dgm_cell.py # Gated DGM-inspired neural network cell
│ └── tdgf_net.py # Full TDGF network architecture
│
├── utils/
│ ├── samplers.py # Data sampling utilities
│ └── plotting.py # Plotting and visualization functions
│
└── README.md

yaml
Copy
Edit

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- NumPy, Matplotlib
- Google Colab (or local GPU environment)

---

## Running on Google Colab

1. Open the desired notebook in Google Colab.
2. Enable GPU acceleration in **Runtime > Change runtime type > GPU**.
3. Run all cells to train and visualize the option pricing solutions.

---

## Results

The trained neural networks approximate option prices with:
- Low mean squared error (≈ 10⁻³ in experiments).

Plots of option price surfaces, implied volatilities, and convergence logs are provided within each notebook.

---

## Motivation

This thesis demonstrates that neural networks, when trained to minimize PDE-based energy functionals, can:
- Scale efficiently to complex volatility models.
- Provide whole-domain solutions without traditional grid-based discretization.
- Achieve high accuracy using relatively simple network architectures.

---

## License

This repository is released under the MIT License.
