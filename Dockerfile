ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

#Second apt install line is used for building x265/ffmpeg from source
RUN apt update && \
    apt install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libsm6 libxext6 wget build-essential libssl-dev libopenjp2-7 libopenjp2-tools && \
    apt install -y libnuma-dev nasm autoconf automake libass-dev libfreetype6-dev libgnutls28-dev libmp3lame-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev meson pkg-config texinfo yasm zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Build cmake from source
RUN cd / && \
	wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0.tar.gz && \
	tar -zxvf cmake-3.21.0.tar.gz && \
	cd /cmake-3.21.0/ && \
	./bootstrap && \
	make -j 8 && \
	make install && \
	rm /cmake-3.21.0.tar.gz

# Build x265 and ffmpeg from source
RUN mkdir /ffmpeg /ffmpeg/build /ffmpeg/bin && \
    cd /ffmpeg && \
    git clone --depth 1 --branch 3.5 https://bitbucket.org/multicoreware/x265_git.git && \
    cd x265_git/build && \
    PATH="/ffmpeg/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/ffmpeg/build" -DENABLE_SHARED=off ../source && \
    PATH="/ffmpeg/bin:$PATH" make -j 8 && \
    make install && \
    cd /ffmpeg && \
    wget -O ffmpeg-6.1.tar.bz2 https://ffmpeg.org/releases/ffmpeg-6.1.tar.bz2 && \
    tar xjvf ffmpeg-6.1.tar.bz2 && \
    cd /ffmpeg/ffmpeg-6.1 && \
    PATH="/ffmpeg/bin:$PATH" PKG_CONFIG_PATH="/ffmpeg/build/lib/pkgconfig" ./configure \
        --prefix="/ffmpeg/build" \
        --pkg-config-flags="--static" \
        --extra-cflags="-I/ffmpeg/build/include" \
        --extra-ldflags="-L/ffmpeg/build/lib" \
        --extra-libs="-lpthread -lm" \
        --ld="g++" \
        --bindir="/ffmpeg/bin" \
        --enable-gpl \
        --enable-libx265 && \
    PATH="/ffmpeg/bin:$PATH" make -j 8 && \
    make install && \
    hash -r

# Build VVenC from source
RUN cd / && \
	git clone --depth 1 --branch v1.4.0 https://github.com/fraunhoferhhi/vvenc.git && \
	cd /vvenc/ && \
	make install-release

# Build VVdeC from source
RUN cd / && \
	git clone --depth 1 --branch v1.4.0 https://github.com/fraunhoferhhi/vvdec.git && \
	cd /vvdec/ && \
	mkdir build/ && \
	cd build/ && \
	cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j 8

# Build JM from source
RUN cd / && \
	git clone --depth 1 --branch JM-19.0 https://vcgit.hhi.fraunhofer.de/jvet/JM.git && \
	cd /JM/ && \
	chmod +x unixprep.sh && \
	./unixprep.sh && \
	make -j 8

#python version is 3.8.10 ; pip version is 21.2.4
ADD requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user