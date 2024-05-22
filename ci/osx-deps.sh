#!/bin/bash
export MACOSX_DEPLOYMENT_TARGET=11.0
curl -L https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n7.0.tar.gz -O
tar xf n7.0.tar.gz
cd FFmpeg-n7.0/
./configure --enable-shared --enable-pthreads
make -j$(sysctl -n hw.ncpu)
mkdir dist-arm64
make DESTDIR=dist-arm64 install
make clean
arch -x86_64 ./configure --enable-shared --enable-pthreads
arch -x86_64 make -j$(sysctl -n hw.ncpu)
mkdir dist-x86_64
arch -x86_64 make DESTDIR=dist-x86_64 install
cd dist-arm64
find . -type f -name '*.dylib' -exec lipo -create {} ../dist-x86_64/{} -output {} \;
find . -name '*.a' -exec lipo -create {} ../dist-x86_64/{} -output {} \;
find usr/local/bin -type f -exec lipo -create {} ../dist-x86_64/{} -output {} \;
tar czf ../../ffmpeg.tar.gz .
