FROM quay.io/pypa/manylinux2014_x86_64
RUN curl -L https://github.com/xiph/opus/releases/download/v1.4/opus-1.4.tar.gz -O && \
    tar xf opus-1.4.tar.gz && \
    cd opus-1.4 && \
    ./configure --prefix=/usr --libdir=/usr/lib64 --enable-shared && \
    make -j$(nproc) && \
    mkdir dist && \
    make DESTDIR=/opus-1.4/dist install && \
    make install && \
    cd .. && \
    curl -L https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n7.0.tar.gz -O && \
    tar xf n7.0.tar.gz && \
    cd FFmpeg-n7.0/ && \
    ./configure --prefix=/usr --libdir=/usr/lib64 --pkgconfigdir=/usr/share/pkgconfig/ --enable-shared --enable-pthreads --enable-libopus && \
    make -j$(nproc) && \
    mkdir dist && \
    make DESTDIR=dist install
FROM quay.io/pypa/manylinux2014_x86_64
COPY --from=0 /opus-1.4/dist/ /
COPY --from=0 /FFmpeg-n7.0/dist/ /
