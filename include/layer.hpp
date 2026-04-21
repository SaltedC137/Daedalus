#pragma once

#include "act.hpp"
#include <cstddef>

namespace nn {

struct Layer {
  size_t input_size;
  size_t output_size;
  Act activation;
  Layer(size_t in, size_t out, Act act)
      : input_size(in), output_size(out), activation(act) {}
};

} // namespace nn