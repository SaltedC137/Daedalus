# Architecture Overview

## Module Design

### Matrix Operations (mat.hpp)
- 2D matrix data structure with row/column access
- Element-wise operations and dot product
- SIMD-optimized operations for performance

### Activation Functions (act.hpp)
Implemented activation functions:
- ReLU: `f(x) = max(0, x)`
- Sigmoid: `f(x) = 1 / (1 + e^-x)`
- Tanh: `f(x) = (e^2x - 1) / (e^2x + 1)`
- Softmax: Normalized exponential for multi-class output

### Neural Network (net.hpp)
- Feedforward architecture with configurable layers
- Forward pass computation
- Backpropagation with gradient calculation
- Loss computation (cross-entropy)

### Dataset Handling (set.hpp)
- MNIST image/label loading
- Byte-order swapping for data compatibility
- Batch sampling and shuffling

## Training Pipeline

1. **Data Loading**: Load MNIST images and labels
2. **Initialization**: Xavier weight initialization
3. **Training Loop**:
   - Forward pass through network
   - Loss computation (with parallel OpenMP)
   - Backward pass (gradient calculation)
   - Weight updates via Adam optimizer
4. **Validation**: Track loss across epochs

## Parallel Processing

OpenMP parallelization used in:
- Loss calculation (distributed across threads)
- Gradient aggregation (thread-safe reduction)

## Memory Management

- Pre-allocated buffers for gradients
- Column-major matrix storage for cache efficiency
- Reusable batch buffers to reduce allocations
