# Preserve the released MUSA rc4.2 toolkit and Ubuntu 22.04 ABI.
FROM mthreads/musa:rc4.2.0-devel-ubuntu22.04-amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git patch pkg-config libssl-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# The rc4.2 development image omits the CUB and Thrust headers. Use a
# matching upstream muAlg/muThrust pair: the older 1.12 CUB does not provide
# DeviceSegmentedSort, which the current backend needs.
RUN set -eu; \
    curl -fL --retry 3 'https://codeload.github.com/MooreThreads/muAlg/tar.gz/e189e81f545694cfc79db7d216a049637235d176' -o /tmp/mualg.tar.gz; \
    echo '750e8ad1acbe7e8fffe0998f4d484a6edb9f3abee5f2db2248cd72423ca2bc0c  /tmp/mualg.tar.gz' | sha256sum -c -; \
    curl -fL --retry 3 'https://codeload.github.com/MooreThreads/muThrust/tar.gz/c2e8bbfc8f07b7e755e87ef360505f63fbe9e59e' -o /tmp/muthrust.tar.gz; \
    echo 'a39e0b31ad734d4a0b4f74a1594c2809765b25222d845cb1b746da9f07e9782a  /tmp/muthrust.tar.gz' | sha256sum -c -; \
    mkdir -p /opt/musa-cub /opt/musa-thrust; \
    tar xzf /tmp/mualg.tar.gz -C /opt/musa-cub --strip-components=1; \
    tar xzf /tmp/muthrust.tar.gz -C /opt/musa-thrust --strip-components=1; \
    test -f /opt/musa-cub/cub/cub.cuh; \
    test -f /opt/musa-thrust/thrust/version.h; \
    rm /tmp/mualg.tar.gz /tmp/muthrust.tar.gz

ENV CPLUS_INCLUDE_PATH=/opt/musa-cub:/opt/musa-thrust
