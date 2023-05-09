#!/bin/bash
compile=0

while getopts "c:" flag; do
case "${flag}" in
       c) compile=${OPTARG};;
       ?) echo "Usage: $0 [-c compile]"
          exit 1;;
esac
done

# Compile docker image if specified
set -e
if [[ ${compile} -eq 1 ]]; then
       echo "Compiling docker image ..."
       docker build --rm -t onnx_fp -f docker/Dockerfile-torch2.0.0 .
fi

# Run docker image
docker run -it --rm \
        -v ~:/home/$(id -un) \
        -v /etc/group:/etc/group:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -u $(id -u):$(id -g) \
        -w $(pwd) \
        -e HISTFILE=$(pwd)/.docker_history \
        --gpus all \
        --privileged --net=host --ipc=host \
        onnx_fp
