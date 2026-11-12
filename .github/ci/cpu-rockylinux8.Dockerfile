# Rebuild the public development environment without changing the Rocky 8
# glibc baseline of the CPU release artifacts.
FROM rockylinux:8.9

RUN dnf -y --enablerepo=powertools install \
      gcc-toolset-11-gcc gcc-toolset-11-gcc-c++ \
      cmake make git patch which openblas-devel openblas-static curl perl-core \
      pkgconf-pkg-config tar gzip ca-certificates \
    && dnf clean all

ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-11/root/usr/lib64:/opt/rh/gcc-toolset-11/root/usr/lib

COPY install-openssl3.sh /tmp/install-openssl3.sh
RUN /bin/sh /tmp/install-openssl3.sh && rm /tmp/install-openssl3.sh
ENV OPENSSL_ROOT_DIR=/opt/openssl3
