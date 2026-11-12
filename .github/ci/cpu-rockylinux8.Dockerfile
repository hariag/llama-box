# Rebuild the public development environment without changing the Rocky 8
# glibc baseline of the CPU release artifacts.
FROM rockylinux:8.9

RUN dnf -y install \
      gcc-toolset-11-gcc gcc-toolset-11-gcc-c++ \
      cmake make git patch which openssl-devel openblas-devel \
      pkgconf-pkg-config tar gzip ca-certificates \
    && dnf clean all

ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-11/root/usr/lib64:/opt/rh/gcc-toolset-11/root/usr/lib
