# Official SourceFind/Hygon image, pinned to the verified DTK 25.04 manifest.
# The development tools are checked explicitly rather than inferred from its tag.
FROM image.sourcefind.cn:5000/dcu/admin/base/pytorch@sha256:07cc2367f67c16aa6e80d1a2a63dce933b705413391e3c880099b3ce63c8621b

RUN test -x /opt/dtk/llvm/bin/clang \
    && test -x /opt/dtk/llvm/bin/clang++ \
    && test -x /opt/dtk/hip/bin/hipconfig \
    && test -f /opt/dtk/env.sh \
    && test -d /opt/dtk/.hyhal \
    && { test -f /opt/dtk/include/hip/hip_runtime.h || test -f /opt/dtk/hip/include/hip/hip_runtime.h; } \
    && test -f /opt/dtk/hip/lib/cmake/hip/hip-config-version.cmake \
    || { echo 'DTK 25.04 image lacks the compiler, headers, environment or .hyhal required by the build' >&2; exit 1; }

RUN printf '%s\n' \
      'deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse' \
      'deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse' \
      'deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse' \
      > /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git patch pkg-config curl perl ca-certificates tar gzip \
    && rm -rf /var/lib/apt/lists/*

COPY install-openssl3.sh /tmp/install-openssl3.sh
RUN /bin/sh /tmp/install-openssl3.sh && rm /tmp/install-openssl3.sh
ENV OPENSSL_ROOT_DIR=/opt/openssl3
