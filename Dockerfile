FROM nvidia/cuda:10.1-runtime-ubuntu18.04

WORKDIR /code
COPY requirements.txt /code/
