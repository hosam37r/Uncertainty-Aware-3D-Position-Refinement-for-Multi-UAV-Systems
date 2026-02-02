# Uncertainty-Aware 3D Position Refinement for Multi-UAV Systems

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the simulation codebase for the paper **"Uncertainty-Aware 3D Position Refinement for Multi-UAV Systems"** by **Hosam Alamleh** and **Damir Pulatov** (University of North Carolina Wilmington).

## 📖 Overview

Reliable 3D positioning is critical for Unmanned Aerial Vehicles (UAVs) performing time-sensitive tasks like mapping and emergency response. However, GNSS performance often suffers from multipath interference, vertical drift, and intentional spoofing.

This project implements a **decentralized, lightweight refinement layer** that enhances 3D positioning by fusing a UAV's local estimates with shared data from nearby neighbors.

### Key Features
* **Uncertainty-Aware Fusion**: Prioritizes high-confidence neighbor estimates and relative geometric constraints to refine local position.
* **Cooperative Bootstrapping**: Facilitates "cold start" initialization and recovery during total localization loss (GNSS-denied environments).
* **Trust Mechanism**: A range-consistency trust mechanism detects and excludes "unhonest" nodes (malicious actors) to prevent spoofing attacks.

---

## 📂 Repository Structure

The codebase consists of four primary simulation scripts, each targeting a specific experimental scenario:

| File | Description |
| :--- | :--- |
| `sim.py` | **Single Run Simulation**: Performs one simulation run. Outputs a qualitative 3D projection (XY, XZ, YZ) of the swarm's final positions and a plot of mean error vs. epoch. |
| `100.py` | **Ensemble Analysis**: Executes 100 simulation runs to generate statistical bounds. Outputs a plot showing the Mean Error + 10th–90th percentile bands. |
| `cold.py` | **Cold Start Recovery**: Simulates a scenario where a cohort of UAVs starts with missing local fixes. Tracks the time-to-recovery using neighbor assistance. |
| `honesty.py` | **Robustness (Scenario 4)**: Validates the trust mechanism by varying the fraction of malicious nodes (0% to 50%) and comparing error rates with and without trust enabled. |

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    The simulations require `numpy`, `matplotlib`, and `Pillow`.
    ```bash
    pip install numpy matplotlib Pillow
    ```

---

## 🚀 Usage

### 1. Run a Single Visualization
To see a single random run with 3D projection snapshots:
```bash
python sim.py
