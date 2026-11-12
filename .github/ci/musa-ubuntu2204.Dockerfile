# Preserve the released MUSA rc4.2 toolkit and Ubuntu 22.04 ABI.
FROM mthreads/musa:rc4.2.0-devel-ubuntu22.04-amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git patch pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
