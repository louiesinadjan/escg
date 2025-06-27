# GPU Acceleration for Evolutionary Spatial Cyclic Games

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains a GPU-accelerated simulation framework for Evolutionary Spatial Cyclic Games (ESCGs), developed as part of a BSc Computer Science dissertation at the University of Bristol. It includes:

- A validated single-threaded C++ baseline
- GPU implementations using Apple Metal (Apple Silicon) and NVIDIA CUDA
- GPU implementations of the Mersenne Twister algorithm using Apple Metal (Apple Silicon) and NVIDIA CUDA

## Highlights

- Up to **28× speedup** (CUDA) over CPU baseline
- Enables simulations with lattice sizes up to **3200×3200**
- Replicates and extends published ESCG studies
- Validated on Apple M1 Pro and NVIDIA RTX A2000

## Requirements

### General

- C++17 compatible compiler (e.g. `g++`, `clang++`)
- Python 3 with:
  - `matplotlib`
  - `seaborn`
  - `pandas`

Install Python dependencies:
```bash
pip install matplotlib seaborn pandas
```

---

### C++ Baseline (`single-threaded-escg/`)

- C++ compiler with C++17 support

To compile:
```bash
make clean && make
```

To run:
```bash
./escg [flags]
```

You can customise the simulation via the following command-line flags:

| CLI Flag             | Description                                      | Default Value |
|----------------------|--------------------------------------------------|---------------|
| `--length`           | Length of the lattice                            | `200`         |
| `--height`           | Height of the lattice                            | `200`         |
| `--mcs`              | Monte Carlo Step Limit                           | `100000`      |
| `--neighbourhood`    | Neighbourhood type (4 or 8-way)                  | `4`           |
| `--printFrequency`   | MCS interval to print density counts             | `200`         |
| `--mobility`         | Mobility of an individual                        | `3e-5`        |
| `--species`          | Number of species in the simulation              | `3`           |
| `--flux`             | Wrap boundary condition                          | `true`        |
| `--empty`            | Initial empty cell probability                   | `0.0`         |
| `--save`             | Export snapshots to `.png`                       | `false`       |
| `--dominance`        | Import dominance matrix from `dominance.csv`              | `false`       |

Example:
```bash
./escg --length 400 --height 400 --species 5 --save true --dominance true
```

---
### CUDA Implementation (`cuda/`)

- NVIDIA GPU with Compute Capability ≥ 5.0
- CUDA Toolkit ≥ 11.0
- `nvcc` (NVIDIA CUDA compiler)

Install on Ubuntu:
```bash
sudo apt install nvidia-cuda-toolkit
```

Alternatively, use the official CUDA installer:  
https://developer.nvidia.com/cuda-downloads

To compile:
```bash
make clean && make
```

To run:
```bash
./cuda-escg [flags]
```

Additional CLI flags available:

| CLI Flag         | Description                                            | Default Value   |
|------------------|--------------------------------------------------------|-----------------|
| `--resume`       | Resume simulation from a given state `grid.csv`                   | `false`         |
| `--numRandoms`   | Number of random numbers to store and generate         | `100000000`     |
| `--maxStep`      | Execute multiple MCS per kernel invocation             | `false`         |

---

### Metal Implementation (`metal-escg/`)

- Apple Silicon with Metal support
- macOS (Monterey or newer recommended)
- Xcode Command Line Tools (for `clang++` and `xcrun`)
- Metal Shading Language support

Install:
```bash
xcode-select --install
```

To compile:
```bash
make clean && make
```

To run:
```bash
./build/metal [flags]
```

Additional CLI flags available:

| CLI Flag         | Description                                            | Default Value   |
|------------------|--------------------------------------------------------|-----------------|
| `--resume`       | Resume simulation from a given state `grid.csv`                   | `false`         |
| `--numRandoms`   | Number of random numbers to store and generate         | `100000000`     |
| `--maxStep`      | Execute multiple MCS per kernel invocation             | `false`         |
