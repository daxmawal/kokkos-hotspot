#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "kernels.hpp"

namespace {

constexpr int kDefaultRounds = 100000;
constexpr int kDecimalBase = 10;
constexpr int kNumElements = 1 << 20;
constexpr double kInitScale = 0.5;
constexpr double kInitOffset = 1.0;
constexpr double kAxpyAlpha = 0.75;
constexpr int kUpdateMask = 0x7;

}  // namespace

static auto
get_env_rounds() -> int
{
  const char* env_value = std::getenv("KOKKOS_ROUNDS");
  if ((env_value == nullptr) || (*env_value == '\0')) {
    return kDefaultRounds;
  }
  char* end_ptr = nullptr;
  long parsed_value = std::strtol(env_value, &end_ptr, kDecimalBase);
  if ((end_ptr == env_value) || (parsed_value <= 0)) {
    return kDefaultRounds;
  }
  if (parsed_value > INT_MAX) {
    return INT_MAX;
  }
  return static_cast<int>(parsed_value);
}

auto
main(int argc, char** argv) -> int
{
  Kokkos::initialize(argc, argv);
  {
    const int num_elements = kNumElements;
    Kokkos::View<double*> x_view("x", num_elements);
    Kokkos::View<double*> y_view("y", num_elements);

    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy<>(0, num_elements),
        KOKKOS_LAMBDA(int index) {
          x_view(index) = kInitScale * index;
          y_view(index) = kInitOffset;
        });
    Kokkos::fence();

    const int rounds = get_env_rounds();
    const auto start = std::chrono::steady_clock::now();
    double checksum = 0.0;

    for (int round = 0; round < rounds; ++round) {
      kernel_axpy(x_view, kAxpyAlpha, y_view);
      checksum += kernel_reduce(y_view);
      if ((round & kUpdateMask) == 0) {
        kernel_update(x_view, round);
      }
    }

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    const auto host_y_view =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_view);
    std::cout << "y0=" << host_y_view(0)
              << " ylast=" << host_y_view(num_elements - 1)
              << " checksum=" << checksum << " rounds=" << rounds
              << " time_ms=" << elapsed.count() << "\n";

    const char* summary_csv_path = std::getenv("KOKKOS_APP_SUMMARY");
    if ((summary_csv_path != nullptr) && (*summary_csv_path != '\0')) {
      std::ofstream out(summary_csv_path);
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
