cmake_minimum_required(VERSION 3.16)
project(kokkos_hotspot_tool CXX)

option(KOKKOS_USE_SUBMODULE "Use bundled external/kokkos submodule by default"
       ON)
option(KOKKOS_AUTO_BACKENDS
       "Auto-detect Kokkos backends (CUDA/HIP/OpenMP) when using the submodule"
       ON)

if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(KOKKOS_USE_SUBMODULE)
  set(KOKKOS_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos")
  if(NOT EXISTS "${KOKKOS_SUBMODULE_DIR}/CMakeLists.txt")
    message(
      FATAL_ERROR
        "KOKKOS_USE_SUBMODULE=ON but external/kokkos is missing. "
        "Run 'git submodule update --init --recursive' or set "
        "-DKOKKOS_USE_SUBMODULE=OFF and provide Kokkos_DIR.")
  endif()

  if(NOT DEFINED Kokkos_ENABLE_PROFILING)
    set(Kokkos_ENABLE_PROFILING
        ON
        CACHE BOOL "Enable Kokkos profiling hooks")
  endif()

  if(KOKKOS_AUTO_BACKENDS)
    include(CheckLanguage)

    set(_kokkos_auto_device_backend "none")
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      set(Kokkos_ENABLE_CUDA
          ON
          CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
      set(Kokkos_ENABLE_HIP
          OFF
          CACHE BOOL "Enable Kokkos HIP backend" FORCE)
      set(_kokkos_auto_device_backend "CUDA")
    else()
      find_program(HIPCC_EXECUTABLE hipcc)
      if(HIPCC_EXECUTABLE)
        set(Kokkos_ENABLE_HIP
            ON
            CACHE BOOL "Enable Kokkos HIP backend" FORCE)
        set(Kokkos_ENABLE_CUDA
            OFF
            CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
        set(_kokkos_auto_device_backend "HIP")
      else()
        set(Kokkos_ENABLE_CUDA
            OFF
            CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
        set(Kokkos_ENABLE_HIP
            OFF
            CACHE BOOL "Enable Kokkos HIP backend" FORCE)
      endif()
      unset(HIPCC_EXECUTABLE CACHE)
    endif()
    message(
      STATUS "Kokkos auto backend (device): ${_kokkos_auto_device_backend}")
    unset(_kokkos_auto_device_backend)

    find_package(
      OpenMP
      COMPONENTS CXX
      QUIET)
    if(OpenMP_CXX_FOUND)
      set(Kokkos_ENABLE_OPENMP
          ON
          CACHE BOOL "Enable Kokkos OpenMP backend" FORCE)
    else()
      set(Kokkos_ENABLE_OPENMP
          OFF
          CACHE BOOL "Enable Kokkos OpenMP backend" FORCE)
    endif()

    set(Kokkos_ENABLE_SERIAL
        ON
        CACHE BOOL "Enable Kokkos SERIAL backend" FORCE)

    message(
      STATUS
        "Kokkos backend config: CUDA=${Kokkos_ENABLE_CUDA} HIP=${Kokkos_ENABLE_HIP} OPENMP=${Kokkos_ENABLE_OPENMP} SERIAL=${Kokkos_ENABLE_SERIAL}"
    )
  endif()

  unset(Kokkos_ENABLE_CUDA_LAMBDA CACHE)

  add_subdirectory("${KOKKOS_SUBMODULE_DIR}" external/kokkos)
else()
  if(KOKKOS_AUTO_BACKENDS)
    message(
      WARNING
        "KOKKOS_AUTO_BACKENDS=ON is ignored when KOKKOS_USE_SUBMODULE=OFF. "
        "Configure backends from your external Kokkos build or disable "
        "KOKKOS_AUTO_BACKENDS.")
  endif()
  find_package(Kokkos REQUIRED)
endif()

option(KOKKOS_BUILD_EXAMPLES
       "Build the sample Kokkos app used to capture kernel timings" ON)

add_library(kokkos_profiling_tool SHARED src/tool_profiling.cpp
                                         src/tool_common.cpp)
target_link_libraries(kokkos_profiling_tool PRIVATE Kokkos::kokkos)
target_include_directories(kokkos_profiling_tool
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_features(kokkos_profiling_tool PRIVATE cxx_std_20)

if(KOKKOS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
