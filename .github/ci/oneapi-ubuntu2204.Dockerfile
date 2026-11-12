# Keep the oneAPI 2025.0.0 SDK and Ubuntu 22.04 ABI used by the release.
FROM intel/oneapi-basekit:2025.0.0-0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git patch pkg-config curl perl ca-certificates tar gzip \
    && rm -rf /var/lib/apt/lists/*

COPY install-openssl3.sh /tmp/install-openssl3.sh
RUN /bin/sh /tmp/install-openssl3.sh && rm /tmp/install-openssl3.sh
ENV OPENSSL_ROOT_DIR=/opt/openssl3

# CI replaces the image entrypoint, so the build script must source setvars.sh.
RUN test -f /opt/intel/oneapi/setvars.sh \
    && /bin/bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null && command -v icx && command -v icpx'
