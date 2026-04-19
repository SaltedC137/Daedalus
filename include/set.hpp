#pragma once
#include "mat.hpp"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>

// (20260419-031315)

inline uint32_t swap_endian(uint32_t val) {
  return ((val << 24) & 0xff000000) | ((val << 8) & 0x00ff0000) |
         ((val >> 8) & 0x0000ff00) | ((val >> 24) & 0x000000ff);
}

namespace nn {
inline std::vector<Mat> load_mnist_img(const char *filename) {
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open())
    return {};

  uint32_t magic, count, rows, cols;
  file.read((char *)&magic, 4);
  if (swap_endian(magic) != 2051) return {};

  file.read((char *)&count, 4);
  file.read((char *)&rows, 4);
  file.read((char *)&cols, 4);

  count = swap_endian(count);
  rows = swap_endian(rows);
  cols = swap_endian(cols);

  size_t pixels_per_image = rows * cols;
  std::vector<Mat> images;
  images.reserve(count);

  std::vector<uint8_t> buffer(pixels_per_image);

  for (size_t i = 0; i < count; i++) {
    file.read((char *)buffer.data(), pixels_per_image);

    Mat img(1, pixels_per_image);
    for (size_t j = 0; j < pixels_per_image; j++) {
      img(0, j) = static_cast<float>(buffer[j]) / 255.0f;
    }

    images.push_back(std::move(img));
  }

  return images;
}

inline std::vector<int> load_mnist_lab(const char *filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    return {};

  uint32_t magic, count;

  file.read((char *)&magic, 4);
  if (swap_endian(magic) != 2049) return {}; 

  file.read((char *)&count, 4);

  count = swap_endian(count);

  std::vector<int> labels;
  labels.reserve(count);

  for (size_t i = 0; i < count; i++) {
    uint8_t label;
    file.read((char *)&label, 1);
    labels.push_back(static_cast<int>(label));
  }

  return labels;
}

} // namespace nn
