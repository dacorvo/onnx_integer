#!/bin/sh

set -e

if [ ! -f 'mnist_cnn.pts' ]; then
    # Train MNIST model with pytorch
    python train_mnist_model.py --save-model 'mnist_cnn.pts'
fi
if [ ! -f 'mnist_cnn.onnx' ]; then
    # Convert MNIST model to ONNX format
    python torch_to_onnx.py --model 'mnist_cnn.pts' --save_model 'mnist_cnn.onnx'
fi

# Quantize model
python onnx_mnist_quantize.py --per_axis --model 'mnist_cnn.onnx' --save_model 'mnist_cnn_quantized.onnx'

# Check quantized model accuracy
python onnx_mnist.py --model mnist_cnn_quantized.onnx