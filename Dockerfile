FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

COPY ./requirements.txt /tmp/

USER root

ENV TZ=Asia/Tokyo

RUN apt-get update
RUN apt-get install -y fonts-dejavu
RUN apt-get install -y python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libgl1-mesa-dev \
    libglib2.0-0

RUN apt-get install -y vim fish git gcc libmariadb-dev tmux htop

RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
