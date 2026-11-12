# Keep the original Rocky 8 glibc baseline and CUDA 12.8 toolkit version.
FROM nvidia/cuda:12.8.0-devel-rockylinux8

RUN dnf -y install \
      gcc-toolset-11-gcc gcc-toolset-11-gcc-c++ \
      cmake make git patch which openssl-devel pkgconf-pkg-config \
      tar gzip ca-certificates \
    && dnf clean all

ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-11/root/usr/lib64:/opt/rh/gcc-toolset-11/root/usr/lib:${LD_LIBRARY_PATH}
