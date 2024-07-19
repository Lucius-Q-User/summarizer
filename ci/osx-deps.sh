#!/bin/bash
set -euo pipefail
export MACOSX_DEPLOYMENT_TARGET=11.0

function build_arch_pkg {
    arch=$1
    shift
    mkdir build-$arch
    cd build-$arch
    arch -$arch ../configure "$@"
    arch -$arch make -j$(sysctl -n hw.ncpu)
    mkdir ../dist-$arch
    arch -$arch make DESTDIR="$(realpath ../dist-$arch)" install
    cd ..
}

function build_arch {
    arch=$1
    cd opus-1.4/
    build_arch_pkg $arch --enable-shared=no
    cd ../FFmpeg-n7.0/
    export PKG_CONFIG_PATH="$(realpath ../opus-1.4/dist-$arch/usr/local/lib/pkgconfig/)"
    build_arch_pkg $arch --enable-shared --enable-pthreads --disable-ffplay --enable-libopus --disable-lzma --pkg-config-flags='--define-prefix' --disable-xlib --disable-avdevice
    cd ..
}

curl -L https://github.com/xiph/opus/releases/download/v1.4/opus-1.4.tar.gz -O
tar xf opus-1.4.tar.gz
curl -L https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n7.0.tar.gz -O
tar xf n7.0.tar.gz
build_arch arm64
build_arch x86_64

cd FFmpeg-n7.0/dist-arm64
find . -type f -name '*.dylib' -exec lipo -create {} ../dist-x86_64/{} -output {} \;
find . -name '*.a' -exec lipo -create {} ../dist-x86_64/{} -output {} \;
find usr/local/bin -type f -exec lipo -create {} ../dist-x86_64/{} -output {} \;
tar czf ../../ffmpeg.tar.gz .
