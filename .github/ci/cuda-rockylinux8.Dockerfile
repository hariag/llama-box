# Keep the original Rocky 8 glibc baseline and requested CUDA toolkit version.
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-rockylinux8

RUN dnf -y install \
      gcc-toolset-11-gcc gcc-toolset-11-gcc-c++ \
      cmake make git patch which curl perl-core pkgconf-pkg-config \
      tar gzip ca-certificates \
    && dnf clean all

ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-11/root/usr/lib64:/opt/rh/gcc-toolset-11/root/usr/lib:${LD_LIBRARY_PATH}

COPY install-openssl3.sh /tmp/install-openssl3.sh
RUN /bin/sh /tmp/install-openssl3.sh && rm /tmp/install-openssl3.sh
ENV OPENSSL_ROOT_DIR=/opt/openssl3
