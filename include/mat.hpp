
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

struct RowView {

  size_t cols;
  float *data;
  float operator[](size_t j) const { return data[j]; };
  float &operator[](size_t j) { return data[j]; };
};

namespace nn {

class Matrix_ {

  size_t rows_, cols_;
  std::vector<float> data_;

public:
  Matrix_() : cols_(0), rows_(0) {}
  Matrix_(size_t r, size_t c) : cols_(r), rows_(c), data_(r * c) {}

  // rows & cols
  size_t rows() const noexcept { return rows_; };
  size_t cols() const noexcept { return cols_; };

  float *data() noexcept { return data_.data(); };
  const float *data() const noexcept { return data_.data(); };

  float &operator()(size_t i, size_t j) {
    assert(i < rows_ && j < cols_);
    return data_[i * cols_ + j];
  };

  float operator()(size_t i, size_t j) const {
    assert(i < rows_ && j < cols_);
    return data_[i * cols_ + j];
  };

  RowView row(size_t i);

  void fill(float v);

  void rand_fill(float low, float high, std::mt19937 &rng);

  void copy_from(const Matrix_ &src);

  void add_inplace(const Matrix_ &other);

  void apply(const std::function<float(float)> &fn);

  static void matmul(Matrix_ &dst, const Matrix_ &a, const Matrix_ &b);

  void shuffle_rows(std::mt19937 &rng);

  void print(std::string_view name = {}, int precision = 6) const;
};

} // namespace nn