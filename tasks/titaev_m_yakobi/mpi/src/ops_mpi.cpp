#include "titaev_m_yakobi/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "titaev_m_yakobi/common/include/common.hpp"

namespace titaev_m_yakobi {

TitaevMYakobiMPI::TitaevMYakobiMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.n, 0.0);
}

bool TitaevMYakobiMPI::ValidationImpl() {
  const auto &in = GetInput();
  if (in.n <= 0) {
    return false;
  }
  const auto n_size = static_cast<std::size_t>(in.n);
  if (n_size * n_size != in.A.size()) {
    return false;
  }
  if (n_size != in.b.size()) {
    return false;
  }
  if (!in.x0.empty() && n_size != in.x0.size()) {
    return false;
  }
  if (in.eps <= 0.0 || in.max_iter <= 0) {
    return false;
  }
  return true;
}

bool TitaevMYakobiMPI::PreProcessingImpl() {
  auto &in = GetInput();
  auto &out = GetOutput();
  if (in.x0.empty()) {
    in.x0.assign(in.n, 0.0);
  }
  out = in.x0;
  return true;
}

void TitaevMYakobiMPI::ComputeLocal(const std::vector<ValueType> &x_old, std::vector<ValueType> &x_new_local,
                                    int start_row, int my_rows) {
  const auto &in = GetInput();
  const int n = in.n;

  for (int local_i = 0; local_i < my_rows; ++local_i) {
    const int i = start_row + local_i;
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
    x_new_local[local_i] = (in.b[i] - sum) / diag;
  }
}

bool TitaevMYakobiMPI::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  const int n = in.n;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<ValueType> x_old = out;

  const int rows_per_proc = n / size;
  const int remainder = n % size;
  const int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);
  const int start_row = (rank * rows_per_proc) + std::min(rank, remainder);

  std::vector<ValueType> x_new_local(my_rows, 0.0);

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  for (int proc = 0; proc < size; ++proc) {
    const int rows_proc = rows_per_proc + (proc < remainder ? 1 : 0);
    recvcounts[proc] = rows_proc;
    displs[proc] = (proc * rows_per_proc) + std::min(proc, remainder);
  }

  for (int iter = 0; iter < in.max_iter; ++iter) {
    ComputeLocal(x_old, x_new_local, start_row, my_rows);

    std::vector<ValueType> x_new_global;
    if (rank == 0) {
      x_new_global.resize(n);
    }

    MPI_Gatherv(x_new_local.data(), my_rows, MPI_DOUBLE, x_new_global.data(), recvcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int converged = 0;
    if (rank == 0) {
      ValueType max_diff = 0.0;
      for (int i = 0; i < n; ++i) {
        const ValueType diff = std::fabs(x_new_global[i] - x_old[i]);
        max_diff = std::max(diff, max_diff);
      }
      x_old = std::move(x_new_global);
      converged = (max_diff < in.eps) ? 1 : 0;
    }

    MPI_Bcast(x_old.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);

    out = x_old;

    if (converged != 0) {
      break;
    }
  }

  return true;
}

bool TitaevMYakobiMPI::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_yakobi
