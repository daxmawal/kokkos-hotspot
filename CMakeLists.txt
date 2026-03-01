cmake_minimum_required(VERSION 3.16)
project(kokkos_hotspot_tool CXX)

find_package(Kokkos REQUIRED)

option(KOKKOS_BUILD_EXAMPLES
       "Build the sample Kokkos app used to capture kernel timings" ON)

add_library(kokkos_profiling_tool SHARED src/tool_profiling.cpp
                                         src/tool_common.cpp)
target_link_libraries(kokkos_profiling_tool PRIVATE Kokkos::kokkos)
target_include_directories(kokkos_profiling_tool
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
set_target_properties(
  kokkos_profiling_tool
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON)

if(KOKKOS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
