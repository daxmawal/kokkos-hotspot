FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG KOKKOS_REF=5.0.0
ARG KOKKOS_INSTALL_PREFIX=/opt/kokkos

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  clang \
  cmake \
  git \
  python3 \
  python3-matplotlib \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --branch ${KOKKOS_REF} --depth 1 https://github.com/kokkos/kokkos.git
RUN cmake -S /tmp/kokkos -B /tmp/kokkos/build \
  -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_PREFIX} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DKokkos_ENABLE_PROFILING=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_TESTS=OFF \
  -DKokkos_ENABLE_EXAMPLES=OFF \
  && cmake --build /tmp/kokkos/build -j"$(nproc)" \
  && cmake --install /tmp/kokkos/build \
  && rm -rf /tmp/kokkos

WORKDIR /work
COPY . /work
RUN chmod +x /work/scripts/analyze_hotspots.sh
RUN cmake -S /work -B /work/build \
  -DKokkos_DIR=${KOKKOS_INSTALL_PREFIX}/lib/cmake/Kokkos \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DKOKKOS_BUILD_EXAMPLES=ON \
  && cmake --build /work/build -j"$(nproc)"

CMD ["bash"]
