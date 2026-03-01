cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(kokkos_hotspot_examples CXX)
  find_package(Kokkos REQUIRED)
endif()

add_executable(kokkos_app app.cpp)
target_link_libraries(kokkos_app PRIVATE Kokkos::kokkos)
set_target_properties(
  kokkos_app
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
