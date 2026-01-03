#include "titaev_m_yakobi/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "titaev_m_yakobi/common/include/common.hpp"

namespace titaev_m_yakobi {

TitaevMYakobiSEQ::TitaevMYakobiSEQ(const InType &in_) : in(in_) {}

bool TitaevMYakobiSEQ::ValidationImpl() {
  if (in.n <= 0) {
    return false;
  }
  if (in.b.size() != static_cast<size_t>(in.n)) {
    return false;
  }
  if (!in.x0.empty() && in.x0.size() != static_cast<size_t>(in.n)) {
    return false;
  }
  return true;
}

void TitaevMYakobiSEQ::Iterate(const std::vector<ValueType> &x_old, std::vector<ValueType> &x_new) const {
  for (int i = 0; i < in.n; ++i) {
    ValueType sum = in.b[i];
    const ValueType diag = in.A[i * in.n + i];

    if (std::fabs(diag) < 1e-15) {
      continue;
    }

    for (int j = 0; j < in.n; ++j) {
      if (j != i) {
        sum -= in.A[i * in.n + j] * x_old[j];
      }
    }
    x_new[i] = sum / diag;
  }
}

bool TitaevMYakobiSEQ::RunImpl() {
  const int n = in.n;

  std::vector<ValueType> x_old = in.x0.empty() ? std::vector<ValueType>(n, 0.0) : in.x0;
  std::vector<ValueType> x_new(n);

  for (int iter = 0; iter < in.max_iter; ++iter) {
    Iterate(x_old, x_new);

    ValueType max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
      max_diff = std::max(max_diff, std::fabs(x_new[i] - x_old[i]));
    }

    x_old = x_new;
    if (max_diff < in.eps) {
      break;
    }
  }

  out = x_old;
  return true;
}

}  // namespace titaev_m_yakobi
