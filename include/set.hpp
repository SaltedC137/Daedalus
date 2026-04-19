#include "mat.hpp"
#include <cstddef>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

// (20260419-031315)
namespace nn {
inline Mat load_image(const char *filename) {
  int width, height, channels;
  unsigned char *data = stbi_load(filename, &width, &height, &channels, 0);

  if (data == nullptr) {
    return Mat();
  }

  Mat img_mat(1, width * height * channels);

  for (size_t i = 0; i < static_cast<size_t>(width * height * channels); i++) {
    img_mat(0, i) = static_cast<float>(data[i] / 255.0f);
  }

  stbi_image_free(data);

  return img_mat;
}

} // namespace nn
