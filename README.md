# ONNX FixedPoint proof-of-concept
A proof-of-concept for the conversion of a quantized pytorch model to an ONNX FixedPOint graph of operations

## Docker setup

### Build docker image

```
./docker.sh -c 1
```

### Run docker image

```
./docker.sh
```

## Venv setup

### Create venv

```
python -m venv onnx_tf_venv
```

### Activate venv

```
source ./onnx_tf_venv/bin/activate
```

## Training of a basic MNIST model

The first step is to instantiate and train a basic MNIST model for a few epochs.

```
python train_mnist_model.py --epochs 5
```

This should produce a model that reaches `99 %` accuracy, and save it as `mnist_cnn.pt`.

Note: the model is saved as a state_dict, so it cannot be instantiated without
the original model definition.

## Quantize a trained MNIST model

The next step is to statically quantize the float MNIST model to 8-bit.

```
python quantize_model.py --model mnist_cnn.pt \
                         --save_model mnist_cnn_quantized.pts
```

This will quantize the model to 8-bit and calibrate it, resulting in a model
with an equivalent accuracy (99 %).

Note that the model is calibrated with random inputs: calibrating it with inputs
from the dataset will likely improve the accuracy.

Note also that the quantization script does not use the default quantization
backend configuration, as it includes a fused operator that is not supported
by ONNX (the default config is therefore edited to remove the transformation
producing the fused operator).

The quantized model is saved as a `mnist_cnn_quantized.pts` TorchScript model
that can be used for inference or conversion to ONNX.

## Convert the quantized model to ONNX

The conversion to ONNX is rather straightforward. The only caveat is that the
batch size must be specified at conversion, as it is not dynamic.

```
python torch_to_onnx.py --model mnist_cnn_quantized.pts \
                        --save_model mnist_cnn_quantized.onnx \
                        --test-batch-size 1000
```

Once converted, the model can be sued for inference on the MNIST test set.

```
python onnx_mnist.py --model mnist_cnn_quantized.onnx \
                     --test-batch-size 1000
```
