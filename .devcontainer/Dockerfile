FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG USER_NAME
ARG GROUP_NAME
ARG UID
ARG GID

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libopencv-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd -g $GID $GROUP_NAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USER_NAME

USER $USER_NAME

WORKDIR /workspace

RUN echo "export PATH='/home/$USER_NAME/.local/bin:$PATH'" >> ~/.bashrc && . ~/.bashrc

COPY --chown=$USER_NAME:$USER_NAME requirements.txt requirements.txt

RUN pip install -r requirements.txt
