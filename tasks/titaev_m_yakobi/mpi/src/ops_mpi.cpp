#include "titaev_m_yakobi/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "titaev_m_yakobi/common/include/common.hpp"

namespace titaev_m_yakobi {

TitaevMYakobiMPI::TitaevMYakobiMPI(const InType &in_) : in(in_) {}

bool TitaevMYakobiMPI::ValidationImpl() {
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

void TitaevMYakobiMPI::ComputeLocal(const std::vector<ValueType> &x_old, std::vector<ValueType> &x_new, int start_row,
                                    int my_rows) const {
  for (int local_i = 0; local_i < my_rows; ++local_i) {
    const int i = start_row + local_i;
    ValueType sum = in.b[i];
    ValueType diag = in.A[i * in.n + i];

    if (std::fabs(diag) < 1e-15) {
      continue;
    }

    for (int j = 0; j < in.n; ++j) {
      if (j != i) {
        sum -= in.A[i * in.n + j] * x_old[j];
      }
    }
    x_new[local_i] = sum / diag;
  }
}

bool TitaevMYakobiMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = in.n;
  const int rows_per_proc = n / size;
  const int remainder = n % size;
  const int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);

  int start_row = 0;
  for (int p = 0; p < rank; ++p) {
    start_row += rows_per_proc + (p < remainder ? 1 : 0);
  }

  std::vector<ValueType> x_old = in.x0.empty() ? std::vector<ValueType>(n, 0.0) : in.x0;
  std::vector<ValueType> x_local(my_rows);
  std::vector<ValueType> x_new(n);

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  int offset = 0;
  for (int p = 0; p < size; ++p) {
    recvcounts[p] = rows_per_proc + (p < remainder ? 1 : 0);
    displs[p] = offset;
    offset += recvcounts[p];
  }

  for (int iter = 0; iter < in.max_iter; ++iter) {
    ComputeLocal(x_old, x_local, start_row, my_rows);

    MPI_Gatherv(x_local.data(), my_rows, MPI_DOUBLE, x_new.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    int converged = 0;
    if (rank == 0) {
      ValueType max_diff = 0.0;
      for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::fabs(x_new[i] - x_old[i]));
      }
      converged = (max_diff < in.eps) ? 1 : 0;
      x_old = x_new;
    }

    MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(x_old.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (converged != 0) {
      break;
    }
  }

  out = x_old;
  return true;
}

}  // namespace titaev_m_yakobi
