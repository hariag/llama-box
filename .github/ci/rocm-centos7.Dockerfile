# Preserve ROCm 6.2.4 and the CentOS 7 (glibc 2.17) release ABI.
FROM rocm/dev-centos-7:6.2.4-complete

# CentOS 7 reached end of life. Use the archived repositories with the
# official CentOS 7 and SCLo signing keys; verify the pinned key digests.
RUN curl -fsSL https://www.centos.org/keys/RPM-GPG-KEY-CentOS-7 -o /tmp/centos-7.key \
    && echo '8b48b04b336bd725b9e611c441c65456a4168083c4febc28e88828d8ec14827f  /tmp/centos-7.key' | sha256sum -c - \
    && curl -fsSL https://www.centos.org/keys/RPM-GPG-KEY-CentOS-SIG-SCLo -o /tmp/centos-sclo.key \
    && echo '1402b1ea263cd0e51e7fe77c89ab98d64dd08fdc54e7a4afe27b2782ddfc351c  /tmp/centos-sclo.key' | sha256sum -c - \
    && rpm --import /tmp/centos-7.key /tmp/centos-sclo.key \
    && printf '%s\n' \
      '[vault-base]' 'name=CentOS 7 base' \
      'baseurl=https://vault.centos.org/7.9.2009/os/$basearch/' 'enabled=1' 'gpgcheck=1' 'gpgkey=file:///tmp/centos-7.key' \
      '[vault-updates]' 'name=CentOS 7 updates' \
      'baseurl=https://vault.centos.org/7.9.2009/updates/$basearch/' 'enabled=1' 'gpgcheck=1' 'gpgkey=file:///tmp/centos-7.key' \
      '[vault-sclo-rh]' 'name=CentOS 7 software collections' \
      'baseurl=https://vault.centos.org/7.9.2009/sclo/$basearch/rh/' 'enabled=1' 'gpgcheck=1' 'gpgkey=file:///tmp/centos-sclo.key' \
      > /etc/yum.repos.d/llama-box-vault.repo \
    && yum -y --disablerepo='*' --enablerepo='vault-*' install \
      devtoolset-11-gcc devtoolset-11-gcc-c++ make git patch which \
      curl ca-certificates tar gzip perl-core pkgconfig \
    && yum clean all

ENV PATH=/opt/rh/devtoolset-11/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-11/root/usr/lib64:/opt/rh/devtoolset-11/root/usr/lib:${LD_LIBRARY_PATH}

# HIP language support needs a newer CMake than the CentOS 7 package.
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz -o /tmp/cmake.tar.gz \
    && echo '931e3c0d546ee03ca72bb147ccd9b49e3b6252f765f66bf21b9d165519940458  /tmp/cmake.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/cmake.tar.gz -C /usr/local --strip-components=1 \
    && rm /tmp/cmake.tar.gz

# Current httplib requires OpenSSL 3; keep it static and retain the libc ABI.
COPY install-openssl3.sh /tmp/install-openssl3.sh
RUN /bin/sh /tmp/install-openssl3.sh && rm /tmp/install-openssl3.sh
ENV OPENSSL_ROOT_DIR=/opt/openssl3
