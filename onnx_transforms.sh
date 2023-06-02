#!/bin/sh

set -e

model=${1:-'mnist_cnn_quantized.onnx'}

echo "Adding rescaling"
python onnx_transforms.py --model ${model} \
                          --save_model mnist_cnn_rescaled.onnx \
                          --add_rescaling --scale 0.0127 --zero-point 33
python onnx_mnist.py --model mnist_cnn_rescaled.onnx


echo "Embedding missing activations"
python onnx_transforms.py --model mnist_cnn_rescaled.onnx \
                          --save_model mnist_cnn_activations.onnx \
                          --embed_act
python onnx_mnist.py --model mnist_cnn_activations.onnx

echo "Pruning QDQs"
python onnx_transforms.py --model mnist_cnn_activations.onnx \
                          --save_model mnist_cnn_pruned.onnx \
                          --prune_qdq
python onnx_mnist.py --model mnist_cnn_pruned.onnx

echo "Folding scales"
python onnx_transforms.py --model mnist_cnn_pruned.onnx \
                          --save_model mnist_cnn_folded.onnx \
                          --fold_op_scales
python onnx_mnist.py --model mnist_cnn_folded.onnx

echo "Splitting bias additions"
python onnx_transforms.py --model mnist_cnn_folded.onnx \
                          --save_model mnist_cnn_split.onnx \
                          --split_bias_add
python onnx_mnist.py --model mnist_cnn_split.onnx

echo "Replacing float operations"
python onnx_transforms.py --model mnist_cnn_split.onnx \
                          --save_model mnist_cnn_integer.onnx \
                          --integer_ops
python onnx_mnist.py --model mnist_cnn_integer.onnx

echo "Replacing output quantizers by scale-out"
python onnx_transforms.py --model mnist_cnn_integer.onnx \
                          --save_model mnist_cnn_scale_out.onnx \
                          --scale_out
python onnx_mnist.py --model mnist_cnn_scale_out.onnx
