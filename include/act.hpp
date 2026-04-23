#pragma once
#include <cmath>

enum class Act { SIGMOID, RELU, TANH, SIN, SOFTMAX };

namespace nn {

inline float actf(float x, Act act) {
  switch (act) {
  case Act::SIGMOID:
    return 1.0f / (1.0f + std::exp(-x));
  case Act::RELU:
    return x > 0 ? x : x * 0.01f;
  case Act::SIN:
    return std::sin(x);
  case Act::TANH: {
    float ex = std::exp(x), enx = std::exp(-x);
    return (ex - enx) / (ex + enx);
  }
  case Act::SOFTMAX: {
    return  x;
  }
  }
  return 0.0f;
}

inline float dactf(float y, Act act) {
  switch (act) {
  case Act::SIGMOID:
    return y * (1.0f - y);
  case Act::RELU:
    return y > 0 ? 1.0f : 0.01f;
  case Act::TANH:
    return 1.0f - y * y;
  case Act::SIN:
    return std::cos(std::asin(y));
  case Act::SOFTMAX: {
    return 1.0f;
  }
  }
  return 0.0f;
}

} // namespace nn
