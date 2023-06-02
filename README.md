# ONNX integer graph proof-of-concept
A proof-of-concept for the conversion of a quantized ONN model to an integer graph of operations

## Initial setup

```
python3 -m venv onnx_venv
source ./onnx_venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Training a basic MNIST model

The first step is to instantiate and train a basic MNIST model for a few epochs.

```
python train_mnist_model.py --epochs 5 --save_model
```

This should produce a model that reaches `99 %` accuracy, and save it as `mnist_cnn.pts`.

Note: the model is saved in TorchScript to prepare the conversion to ONNX.

## Converting the trained MNIST model to ONNX

```
python torch_to_onnx.py --model mnist_cnn.pts \
                        --save_model mnist_cnn.onnx
```

Once converted, the model can be used for inference on the MNIST test set.

```
python onnx_mnist.py --model mnist_cnn.onnx \
                     --test-batch-size 1000
```

## Quantize a trained ONNX MNIST model

The next step is to statically quantize the float MNIST model to 8-bit.

```
python onnx_mnist_quantize.py --model mnist_cnn.onnx \
                              --per-axis \
                              --save_model mnist_cnn_quantized.onnx
```

This will quantize the model to 8-bit and calibrate it, resulting in a model
with an equivalent accuracy (99 %).

## Modify the ONNX quantized model to use integer-only operation

### Add preprocessing to the graph

Since we only want integer operations, we need to integrate the uint8 inputs preprocessing in the graph.

The MNIST preprocessing applies the following Normalization:

```
float_inputs = (inputs / 255 - mean) / std

With:
 mean = 0.1307
 std = 0.3081
```

We can replace it with a `DequantizeLinear` operation:

```
float_inputs = scale * (inputs - zero_point)

With:
  scale = 1 / (255 * std) = 0.0127
  zero_point = round(255 * mean) = 33
```

This transformation is available through the new `add_rescaling` transform:

```
python onnx_transforms.py --model mnist_cnn_sanitized.onnx \
                          --save_model mnist_cnn_rescaled.onnx \
                          --add_rescaling --scale 0.0127 --zero-point 33
```

### Add explicit ReLU activations

ReLU operations are not strictly necessary, as the uint8 quantization operation after an operation clips the outputs,
implicitly applying a ReLU with an upper maximum value.

The ONNX quantizaiton algorithm therefore omits the ReLU in the graph when it is redundant.

This makes writing transformations easier however to have them explicitly, so this transformation adds them back:

```
python onnx_transforms.py --model mnist_cnn_rescaled.onnx \
                          --save_model mnist_cnn_activations.onnx \
                          --embed_act
```


### Remove useless Quantize/Dequantize sequences

When a Node corresponds to an operation that accepts either float or integer
input, there is no need to quantize and dequantize the outputs of the previous
operation.

Example:

```
Linear -> Quantize -> Cast -> Dequantize -> ReLU   <=>   Linear -> ReLU
```

This transformation is available using the new `prune_qdq` transform:

```
python onnx_transforms.py --model mnist_cnn_activations.onnx \
                          --save_model mnist_cnn_pruned.onnx \
                          --prune_qdq
```

### Fold inputs and weights scale into the layer output scale

Each operation has the following pattern:

Dequantize(inputs,           Dequantize(weights,
           i_scale,                     w_scale,
           i_zeropoint)                 w_zeropoint)
                 \              /
                     operation
                         |
                      Quantize(outputs, o_scale, o_zeropoint)

The goal of this transformation is to fold the inputs and weights scales into
the output scale:

o_scale = o_scale / (i_scale * w_scale)
i_scale = 1.0
w_scale = 1.0

Note: bias scale wich is equal to i_scale * w_scale before folding is also set to 1.0

```
python onnx_transforms.py --model mnist_cnn_pruned.onnx \
                          --save_model mnist_cnn_folded.onnx \
                          --fold_op_scales
```

### Perform bias addition as a separate operation

ConvInteger and MatMulInteger operations do not support biases.

The graph is therefore modified to remove the biases from the Conv and Linear operations and
add them as an explicit operation instead.

```
python onnx_transforms.py --model mnist_cnn_folded.onnx \
                          --save_model mnist_cnn_split.onnx \
                          --split_bias_add
```

### Replace float operations by integer operations

This transformation replaces Conv by ConvInteger and Gemm by MatMulInteger.

It also removes the useless `DequantizeLinear` operations.

Note that the MatMulInteger weights have to be transposed because unlike Gemm,
the operation does not include the transposition of its inputs.

```
python onnx_transforms.py --model mnist_cnn_split.onnx \
                          --save_model mnist_cnn_integer.onnx \
                          --integer_ops
```

### Replace output quantizers by a scale-out sequence of operations

This transformation replaces the output quantizers by two operations
using a (mantissa, frac_bits) fixed-point representation of the output scale:

- a multiplication by the mantissa,
- a division by 2 ^ -frac_bits.

The second operation should be a right-shift, but it cannot be performed on signed integer.

```
python onnx_transforms.py --model mnist_cnn_integer.onnx \
                          --save_model mnist_cnn_scale_out.onnx \
                          --scale_out
```