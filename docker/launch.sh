#!/bin/bash
IMAGE=nvcr.io/nvidia/pytorch
TAG=23.12-py3

docker run \
	--cap-add=SYS_ADMIN \
	--cap-add=SYS_NICE \
	--ipc=host \
	--gpus all \
	-it -v $(dirname `pwd`):/workspace \
	--name dekim-diamond \
	$IMAGE:$TAG /bin/bash

