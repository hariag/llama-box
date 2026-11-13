# CANN 8.1/8.2 RC1 on the original Ubuntu 20.04/glibc 2.31 ABI.
# Official packages and verification key are downloaded independently for
# each architecture and chip variant. GPUStack historically resolved the
# 8.1.rc1.beta1 label to the vendor 8.1.RC1 download URL; the present file
# cannot be proven identical to the package downloaded historically.
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CANN_ARCH
ARG CANN_CHIP
ARG CANN_VERSION
RUN case "$CANN_ARCH" in x86_64|aarch64) ;; *) echo 'unsupported CANN_ARCH' >&2; exit 1;; esac \
    && case "$CANN_CHIP" in 910b|310p) ;; *) echo 'unsupported CANN_CHIP' >&2; exit 1;; esac \
    && case "$CANN_VERSION" in 8.1.RC1|8.2.RC1) ;; *) echo 'unsupported CANN_VERSION' >&2; exit 1;; esac \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      curl ca-certificates gnupg \
    && curl -fL --retry 3 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC8EC952E2A0E1FBDC5090F6A2C277A0A352154E5' -o /tmp/toolchain-key.asc \
    && install -d -m 0700 /root/.gnupg \
    && fingerprint="$(gpg --with-colons --import-options show-only --dry-run --import /tmp/toolchain-key.asc | awk -F: '$1 == "fpr" { print $10; exit }')" \
    && test "$fingerprint" = C8EC952E2A0E1FBDC5090F6A2C277A0A352154E5 \
    && gpg --dearmor < /tmp/toolchain-key.asc > /usr/share/keyrings/ubuntu-toolchain-r.gpg \
    && printf '%s\n' 'deb [signed-by=/usr/share/keyrings/ubuntu-toolchain-r.gpg] https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu focal main' > /etc/apt/sources.list.d/ubuntu-toolchain-r.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      gcc-11 g++-11 build-essential cmake git patch pkg-config wget perl \
      tar gzip gnupg python3 python3-pip python3-dev unzip pciutils net-tools libblas-dev gfortran patchelf \
    && rm -rf /var/lib/apt/lists/* /tmp/toolchain-key.asc
ENV CC=gcc-11 CXX=g++-11

# CANN toolkit runtime dependencies; numpy/protobuf/scipy are pinned to
# versions available for Ubuntu 20.04's Python 3.8.
RUN python3 -m pip install --no-cache-dir \
      attrs cython numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 \
      psutil protobuf==3.20.3 scipy==1.10.1 requests absl-py

# Huawei's published OpenPGP fingerprint is B1000AC38C41525A19BDC08799AD81DF27A74824.
# Verify the fingerprint and require its VALIDSIG status for each runfile.
RUN set -eu; \
    export GNUPGHOME=/tmp/cann-gnupg; mkdir -m 700 "$GNUPGHOME"; \
    curl -fsSL --retry 3 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xB1000AC38C41525A19BDC08799AD81DF27A74824' -o /tmp/huawei-key.asc; \
    gpg --batch --import /tmp/huawei-key.asc; \
    gpg --batch --with-colons --fingerprint B1000AC38C41525A19BDC08799AD81DF27A74824 | grep -q '^fpr:::::::::B1000AC38C41525A19BDC08799AD81DF27A74824:'; \
    base="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}"; \
    for part in toolkit "kernels-${CANN_CHIP}"; do \
      name="Ascend-cann-${part}_${CANN_VERSION}_linux-${CANN_ARCH}.run"; \
      curl -fL --retry 3 -H 'Referer: https://www.hiascend.com/' "${base}/${name}" -o "/tmp/${name}"; \
      curl -fL --retry 3 -H 'Referer: https://www.hiascend.com/' "${base}/${name}.asc" -o "/tmp/${name}.asc"; \
      gpg --batch --status-fd 1 --verify "/tmp/${name}.asc" "/tmp/${name}" | grep -q '^\[GNUPG:\] VALIDSIG B1000AC38C41525A19BDC08799AD81DF27A74824 '; \
      chmod +x "/tmp/${name}"; \
      "/tmp/${name}" --quiet --install --install-for-all; \
      rm "/tmp/${name}" "/tmp/${name}.asc"; \
    done; \
    rm -rf "$GNUPGHOME" /tmp/huawei-key.asc

RUN test -f /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && test -f /usr/local/Ascend/ascend-toolkit/latest/include/acl/acl.h \
    || { echo 'CANN toolkit environment or ACL headers are missing' >&2; exit 1; }

COPY install-openssl3.sh /tmp/install-openssl3.sh
RUN /bin/sh /tmp/install-openssl3.sh && rm /tmp/install-openssl3.sh
ENV OPENSSL_ROOT_DIR=/opt/openssl3
