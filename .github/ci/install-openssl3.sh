#!/bin/sh
# Build one pinned OpenSSL release against the container's libc baseline.
# Only static libraries are installed so shipped binaries do not depend on an
# extra libssl.so. The hash is from the official OpenSSL release assets.
set -eu
version=3.5.9
sha256=603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a
archive="openssl-${version}.tar.gz"
prefix="${OPENSSL_INSTALL_PREFIX:-/opt/openssl3}"
cd /tmp
curl -fL --retry 3 "https://github.com/openssl/openssl/releases/download/openssl-${version}/${archive}" -o "$archive"
printf '%s  %s\n' "$sha256" "$archive" | sha256sum -c -
tar -xzf "$archive"
cd "openssl-${version}"
./config --prefix="$prefix" --libdir=lib --openssldir=/etc/ssl no-shared no-tests
make -j "${OPENSSL_BUILD_JOBS:-$(nproc)}"
make install_sw
cd /
rm -rf "/tmp/openssl-${version}" "/tmp/${archive}"
