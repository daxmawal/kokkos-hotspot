# Docker Usage

## CPU Image

```bash
docker build -t kokkos-hotspot .
```

## CUDA Image

```bash
docker build --network host -f Dockerfile.cuda \
  --build-arg KOKKOS_ARCH=ADA89 \
  -t kokkos-hotspot-cuda .
```

`KOKKOS_ARCH` must match your target GPU architecture.
