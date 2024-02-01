FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG USERNAME=yoshi
ARG GROUPNAME=yoshi

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libopencv-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m $USERNAME && \
echo "$USERNAME:$USERNAME" | chpasswd

USER $USERNAME

WORKDIR /workspace/deep_reinforcement_learning

COPY --chown=$USERNAME:$USERNAME requirements.txt requirements.txt

RUN pip install -r requirements.txt
