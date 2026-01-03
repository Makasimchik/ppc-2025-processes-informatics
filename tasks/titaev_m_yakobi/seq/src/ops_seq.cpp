#include "titaev_m_yakobi/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "titaev_m_yakobi/common/include/common.hpp"

namespace titaev_m_yakobi {

TitaevMYakobiSEQ::TitaevMYakobiSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.n, 0.0);
}

bool TitaevMYakobiSEQ::ValidationImpl() {
  const auto &in = GetInput();
  if (in.n <= 0) {
    return false;
  }
  if (static_cast<std::size_t>(in.n * in.n) != in.A.size()) {
    return false;
  }
  if (static_cast<std::size_t>(in.n) != in.b.size()) {
    return false;
  }
  if (!in.x0.empty() && static_cast<std::size_t>(in.n) != in.x0.size()) {
    return false;
  }
  if (in.eps <= 0.0 || in.max_iter <= 0) {
    return false;
  }
  return true;
}

bool TitaevMYakobiSEQ::PreProcessingImpl() {
  auto &in = GetInput();
  auto &out = GetOutput();
  if (in.x0.empty()) {
    in.x0.assign(in.n, 0.0);
  }
  out = in.x0;
  return true;
}

void TitaevMYakobiSEQ::Iterate(const std::vector<ValueType> &x_old, std::vector<ValueType> &x_new) {
  const auto &in = GetInput();
  const int n = in.n;

  for (int i = 0; i < n; ++i) {
    const ValueType diag = in.A[(i * n) + i];
    if (std::fabs(diag) < 1e-15) {
      continue;
    }

    ValueType sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (j != i) {
        sum += in.A[(i * n) + j] * x_old[j];
      }
    }
    x_new[i] = (in.b[i] - sum) / diag;
  }
}

bool TitaevMYakobiSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  const int n = in.n;

  std::vector<ValueType> x_old = out;
  std::vector<ValueType> x_new(n, 0.0);

  for (int iter = 0; iter < in.max_iter; ++iter) {
    Iterate(x_old, x_new);

    ValueType max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
      const ValueType diff = std::fabs(x_new[i] - x_old[i]);
      max_diff = std::max(diff, max_diff);
    }

    x_old.swap(x_new);
    out = x_old;

    if (max_diff < in.eps) {
      break;
    }
  }

  return true;
}

bool TitaevMYakobiSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_yakobi
