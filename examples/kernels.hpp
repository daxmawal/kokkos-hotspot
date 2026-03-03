#pragma once

#include <Kokkos_Core.hpp>

inline auto
kernel_axpy(Kokkos::View<double*>& x_view, double alpha,
            Kokkos::View<double*>& y_view) -> void
{
  const int num_elements = static_cast<int>(x_view.extent(0));
  auto* x_data = x_view.data();
  auto* y_data = y_view.data();
  Kokkos::parallel_for(
      "kernel_axpy", Kokkos::RangePolicy<>(0, num_elements),
      KOKKOS_LAMBDA(int index) {
        y_data[index] = alpha * x_data[index] + y_data[index];
      });
  Kokkos::fence();
}

inline auto
kernel_reduce(const Kokkos::View<double*>& y_view) -> double
{
  constexpr double kReductionScale = 0.5;
  const int num_elements = static_cast<int>(y_view.extent(0));
  auto* y_data = y_view.data();
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "kernel_reduce", Kokkos::RangePolicy<>(0, num_elements),
      KOKKOS_LAMBDA(int index, double& local_sum) {
        local_sum += y_data[index] * kReductionScale;
      },
      sum);
  Kokkos::fence();
  return sum;
}

inline auto
kernel_update(Kokkos::View<double*>& x_view, int step) -> void
{
  constexpr double kStepScale = 1.0e-6;
  constexpr int kIndexMask = 7;
  const int num_elements = static_cast<int>(x_view.extent(0));
  auto* x_data = x_view.data();
  const double scale = 1.0 + kStepScale * step;
  Kokkos::parallel_for(
      "kernel_update", Kokkos::RangePolicy<>(0, num_elements),
      KOKKOS_LAMBDA(int index) {
        x_data[index] = x_data[index] * scale + (index & kIndexMask);
      });
  Kokkos::fence();
}
