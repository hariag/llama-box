# Preserve the Vulkan 1.4.313 SDK and Ubuntu 22.04 ABI used by the release.
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git patch pkg-config curl perl ca-certificates tar gzip xz-utils \
      libx11-6 libxcb1 libxrandr2 libwayland-client0 \
    && rm -rf /var/lib/apt/lists/*

COPY install-openssl3.sh /tmp/install-openssl3.sh
RUN /bin/sh /tmp/install-openssl3.sh && rm /tmp/install-openssl3.sh
ENV OPENSSL_ROOT_DIR=/opt/openssl3

RUN curl -fsSL --retry 3 https://sdk.lunarg.com/sdk/download/1.4.313.0/linux/vulkansdk-linux-x86_64-1.4.313.0.tar.xz -o /tmp/vulkansdk.tar.xz \
    && echo '4e957b66ade85eeaee95932aa7e3b45aea64db373c58a5eaefc8228cc71445c2  /tmp/vulkansdk.tar.xz' | sha256sum -c - \
    && mkdir -p /opt/vulkan \
    && tar -xJf /tmp/vulkansdk.tar.xz -C /opt/vulkan --strip-components=1 \
    && rm /tmp/vulkansdk.tar.xz
ENV VULKAN_SDK=/opt/vulkan/x86_64
ENV PATH=${VULKAN_SDK}/bin:${PATH}
ENV LD_LIBRARY_PATH=${VULKAN_SDK}/lib:${LD_LIBRARY_PATH}
RUN glslc --version && test -f "${VULKAN_SDK}/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake"
