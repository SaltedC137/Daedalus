# Daedalus - Neural Network Framework

A high-performance C++23 neural network framework with MNIST training and parallel processing capabilities.

## Features

- **Core Components**:
  - Matrix operations with SIMD support
  - Activation functions (ReLU, Sigmoid, Tanh, Softmax)
  - Feed-forward neural network architecture
  - Adam optimizer implementation

- **Training**:
  - MNIST dataset support (load, shuffle, batch processing)
  - Multi-epoch training with configurable batch sizes
  - OpenMP-based parallel loss computation
  - Loss visualization and performance tracking

- **Performance**:
  - C++23 with compiler optimizations (O3)
  - Parallel processing for matrix operations
  - Memory-efficient gradient aggregation

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Usage

### Training

```cpp
// Load MNIST data and train neural network
// See src/train.cpp for implementation details
```

Supported platforms: Linux, macOS, Windows (MSVC)

## Project Structure

- `include/` - Header files (activation, matrix, network, dataset)
- `src/` - Implementation files (training, matrix ops, network)
- `assets/data/` - MNIST dataset files

## Dependencies

- C++23 compiler (GCC, Clang, or MSVC)
- CMake 3.15+
- OpenMP (for parallel processing)
- Threads library