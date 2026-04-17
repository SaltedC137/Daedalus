
#include <cassert>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>
#include "act.hpp"

struct RowView {

  size_t cols;
  float *data;
  float operator[](size_t j) const { return data[j]; };
  float &operator[](size_t j) { return data[j]; };
};

namespace nn {

class Mat {

  size_t rows_, cols_;
  std::vector<float> data_;

public:
  Mat() : rows_(0), cols_(0) {}
  Mat(size_t r, size_t c) : rows_(r), cols_(c), data_(r * c) {}

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

  void copy_from(const Mat &src);

  void add_inplace(const Mat &other);

  void apply(const std::function<float(float)> &fn);

  void shuffle_rows(std::mt19937 &rng);

  static void matmul(Mat &dst, const Mat &a, const Mat &b);

  static void dot(Mat &dst, const Mat &a, const Mat &b);

  static void sum(Mat &dst, const Mat &src);

  static void act(Mat &dst,Act activation);
  
  void print(std::string_view name = {}, int precision = 6) const;
};

} // namespace nn