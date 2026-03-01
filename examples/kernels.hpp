#pragma once

#include <Kokkos_Core.hpp>

inline void
kernel_axpy(Kokkos::View<double*>& x, Kokkos::View<double*>& y, double alpha)
{
  const int n = static_cast<int>(x.extent(0));
  auto* x_ptr = x.data();
  auto* y_ptr = y.data();
  Kokkos::parallel_for(
      "kernel_axpy", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(int i) { y_ptr[i] = alpha * x_ptr[i] + y_ptr[i]; });
  Kokkos::fence();
}

inline double
kernel_reduce(const Kokkos::View<double*>& y)
{
  const int n = static_cast<int>(y.extent(0));
  auto* y_ptr = y.data();
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "kernel_reduce", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(int i, double& lsum) { lsum += y_ptr[i] * 0.5; }, sum);
  Kokkos::fence();
  return sum;
}

inline void
kernel_update(Kokkos::View<double*>& x, int step)
{
  const int n = static_cast<int>(x.extent(0));
  auto* x_ptr = x.data();
  const double scale = 1.0 + 1.0e-6 * static_cast<double>(step);
  Kokkos::parallel_for(
      "kernel_update", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(int i) { x_ptr[i] = x_ptr[i] * scale + double(i & 7); });
  Kokkos::fence();
}
