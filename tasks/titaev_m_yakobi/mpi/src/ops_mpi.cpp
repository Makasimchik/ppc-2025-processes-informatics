#include "titaev_m_yakobi/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>  // для size_t
#include <cstdint>  // для целочисленных сравнений
#include <vector>

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
  if (static_cast<int>(in.A.size()) != in.n * in.n) {
    return false;
  }
  if (static_cast<int>(in.b.size()) != in.n) {
    return false;
  }
  if (!in.x0.empty() && static_cast<int>(in.x0.size()) != in.n) {
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

bool TitaevMYakobiMPI::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  int n = in.n;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<ValueType> x_old = out;

  int rows_per_proc = n / size;
  int remainder = n % size;
  int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);
  int start_row = (rank * rows_per_proc) + std::min(rank, remainder);  // Добавлены скобки

  std::vector<ValueType> x_new_local(my_rows, 0.0);

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);
  for (int proc_idx = 0; proc_idx < size; ++proc_idx) {        // Изменено имя переменной
    int rows_r = (n / size) + (proc_idx < remainder ? 1 : 0);  // Добавлены скобки
    recvcounts[proc_idx] = rows_r;
    displs[proc_idx] = (proc_idx * (n / size)) + std::min(proc_idx, remainder);  // Добавлены скобки
  }

  for (int iter = 0; iter < in.max_iter; ++iter) {
    for (int local_i = 0; local_i < my_rows; ++local_i) {
      int i = start_row + local_i;
      ValueType diag = in.A[(i * n) + i];  // Добавлены скобки
      if (std::fabs(diag) < 1e-15) {
        return false;
      }

      ValueType sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          sum += in.A[(i * n) + j] * x_old[j];  // Добавлены скобки
        }
      }

      x_new_local[local_i] = (in.b[i] - sum) / diag;
    }

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
        max_diff = std::max(max_diff, std::fabs(x_new_global[i] - x_old[i]));
      }

      x_old = x_new_global;
      converged = (max_diff < in.eps) ? 1 : 0;
    }

    MPI_Bcast(x_old.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ✅ КЛЮЧЕВАЯ СТРОКА: обновление результата на ВСЕХ процессах
    out = x_old;

    if (converged != 0) {  // Явное сравнение с 0
      break;
    }
  }

  return true;
}

bool TitaevMYakobiMPI::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_yakobi
