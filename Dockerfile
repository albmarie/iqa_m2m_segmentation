ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

RUN apt update && apt install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg libsm6 libxext6 wget build-essential libssl-dev libopenjp2-7 libopenjp2-tools && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

#python version is 3.8.10 ; pip version is 21.2.4
ADD requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user