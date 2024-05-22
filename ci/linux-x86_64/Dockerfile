FROM quay.io/pypa/manylinux2014_x86_64
RUN curl -L https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n7.0.tar.gz -O && \
    tar xf n7.0.tar.gz && \
    cd FFmpeg-n7.0/ && \
    ./configure --prefix=/usr --libdir=/usr/lib64 --pkgconfigdir=/usr/share/pkgconfig/ --enable-shared --enable-pthreads && \
    make -j$(nproc) && \
    mkdir dist && \
    make DESTDIR=dist install
FROM quay.io/pypa/manylinux2014_x86_64
COPY --from=0 /FFmpeg-n7.0/dist/ /
