FROM nvidia/cuda:10.1-runtime-ubuntu18.04

RUN apt-get update -y && \
	apt-get install -y libjpeg8-dev zlib1g-dev git python3.6 python3-pip && \
	python3.6 -m pip install -U pip setuptools 

WORKDIR /code
COPY requirements.txt /code/
RUN python3.6 -m pip install -r requirements.txt 
COPY . .
WORKDIR /code/src
