# Preserve the Ubuntu 18.04 glibc baseline for the arm64 CPU artifact.
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg make git patch pkg-config libopenblas-dev perl tar gzip \
 && curl -fL --retry 3 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC8EC952E2A0E1FBDC5090F6A2C277A0A352154E5' -o /tmp/toolchain-key.asc \
 && fingerprint="$(gpg --with-colons --import-options show-only --dry-run --import /tmp/toolchain-key.asc | awk -F: '$1 == "fpr" { print $10; exit }')" \
 && test "$fingerprint" = C8EC952E2A0E1FBDC5090F6A2C277A0A352154E5 \
 && gpg --dearmor < /tmp/toolchain-key.asc > /usr/share/keyrings/ubuntu-toolchain-r.gpg \
 && printf '%s\n' 'deb [signed-by=/usr/share/keyrings/ubuntu-toolchain-r.gpg] https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu bionic main' > /etc/apt/sources.list.d/ubuntu-toolchain-r.list \
 && apt-get update && apt-get install -y --no-install-recommends gcc-11 g++-11 \
 && curl -fL --retry 3 https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-aarch64.tar.gz -o /tmp/cmake.tar.gz \
 && echo '7909cc2128ce9442c63ce674a0bfb0e4f4ce04cef667d887e15ad5670d594ba7  /tmp/cmake.tar.gz' | sha256sum -c - \
 && tar -xzf /tmp/cmake.tar.gz -C /opt \
 && ln -s /opt/cmake-3.28.6-linux-aarch64/bin/cmake /usr/local/bin/cmake \
 && rm -rf /var/lib/apt/lists/* /tmp/cmake.tar.gz /tmp/toolchain-key.asc

ENV CC=gcc-11 CXX=g++-11
COPY install-openssl3.sh /tmp/install-openssl3.sh
RUN /bin/sh /tmp/install-openssl3.sh && rm /tmp/install-openssl3.sh
ENV OPENSSL_ROOT_DIR=/opt/openssl3
