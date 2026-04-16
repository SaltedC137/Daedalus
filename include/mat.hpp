
#include <cassert>
#include <cmath>
#include <cstddef>
#include <raylib.h>
#include <vector>

struct RowView {

  size_t col;
  float *data;
  float operator[](size_t j) const { return data[j]; };
  float &operator[](size_t j) { return data[j] };
};

class Mat {

  size_t col_, row_;
  std::vector<float> data_;

public:
  Mat() : col_(0), row_(0) {}
  Mat(size_t r, size_t c) : col_(r), row_(c), data_(r * c) {}

  // rows & cols
  size_t rows() const noexcept {};
  size_t cols() const noexcept {};

  float *data() noexcept {};
  const float *data() const noexcept {};

  float &operator()(size_t i, size_t j) {};

  float operator()(size_t i, size_t j) const {

  };

  RowView row(size_t i) {

  };
};