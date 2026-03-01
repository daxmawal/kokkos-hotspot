#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "kernels.hpp"

static int
get_env_rounds()
{
  const char* env = std::getenv("KOKKOS_ROUNDS");
  if (!env || !env[0]) {
    return 100000;
  }
  char* endptr = nullptr;
  long value = std::strtol(env, &endptr, 10);
  if (endptr == env || value <= 0) {
    return 100000;
  }
  if (value > INT_MAX) {
    return INT_MAX;
  }
  return static_cast<int>(value);
}

int
main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  {
    const int n = 1 << 20;
    Kokkos::View<double*> x("x", n);
    Kokkos::View<double*> y("y", n);

    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i) {
          x(i) = 0.5 * static_cast<double>(i);
          y(i) = 1.0;
        });
    Kokkos::fence();

    const int rounds = get_env_rounds();
    const auto start = std::chrono::steady_clock::now();
    double checksum = 0.0;

    for (int round = 0; round < rounds; ++round) {
      kernel_axpy(x, y, 0.75);
      checksum += kernel_reduce(y);
      if ((round & 0x7) == 0) {
        kernel_update(x, round);
      }
    }

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    const auto host_y =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
    std::cout << "y0=" << host_y(0) << " ylast=" << host_y(n - 1)
              << " checksum=" << checksum << " rounds=" << rounds
              << " time_ms=" << elapsed.count() << "\n";

    const char* summary_csv = std::getenv("KOKKOS_APP_SUMMARY");
    if (summary_csv && summary_csv[0]) {
      std::ofstream out(summary_csv);
      if (out.is_open()) {
        out << "rounds," << rounds << "\n";
        out << "checksum," << checksum << "\n";
        out << "elapsed_ms," << elapsed.count() << "\n";
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
