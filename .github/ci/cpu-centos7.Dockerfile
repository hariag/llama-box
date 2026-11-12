# CentOS 7's original repositories are EOL; use the signed official archives.
FROM centos:7

RUN printf '%s\n' \
    '[ci-base]' 'name=CentOS 7 base archive' 'baseurl=https://vault.centos.org/7.9.2009/os/$basearch/' 'enabled=1' 'gpgcheck=1' 'gpgkey=https://www.centos.org/keys/RPM-GPG-KEY-CentOS-7' \
    '[ci-updates]' 'name=CentOS 7 updates archive' 'baseurl=https://vault.centos.org/7.9.2009/updates/$basearch/' 'enabled=1' 'gpgcheck=1' 'gpgkey=https://www.centos.org/keys/RPM-GPG-KEY-CentOS-7' \
    '[ci-sclo]' 'name=CentOS 7 SCLo archive' 'baseurl=https://vault.centos.org/7.9.2009/sclo/$basearch/rh/' 'enabled=1' 'gpgcheck=1' 'gpgkey=https://www.centos.org/keys/RPM-GPG-KEY-CentOS-SIG-SCLo' \
    '[ci-epel]' 'name=EPEL 7 archive' 'baseurl=https://archives.fedoraproject.org/pub/archive/epel/7/$basearch/' 'enabled=1' 'gpgcheck=1' 'gpgkey=https://archives.fedoraproject.org/pub/epel/RPM-GPG-KEY-EPEL-7' \
    > /etc/yum.repos.d/llama-box-ci.repo \
 && yum --disablerepo='*' --enablerepo=ci-base,ci-updates,ci-sclo,ci-epel -y install \
      devtoolset-11-gcc devtoolset-11-gcc-c++ cmake3 make git patch which \
      openblas-devel openblas-static pkgconfig curl perl-core tar gzip ca-certificates \
 && ln -s /usr/bin/cmake3 /usr/local/bin/cmake \
 && yum clean all

ENV PATH=/opt/rh/devtoolset-11/root/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-11/root/usr/lib64:/opt/rh/devtoolset-11/root/usr/lib
COPY install-openssl3.sh /tmp/install-openssl3.sh
RUN /bin/sh /tmp/install-openssl3.sh && rm /tmp/install-openssl3.sh
ENV OPENSSL_ROOT_DIR=/opt/openssl3
