import argparse
import numpy as np

from torchvision import datasets, transforms
from onnxruntime import InferenceSession
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class MnistDataReader(CalibrationDataReader):

    def __init__(self, inputs_name, num_samples):
        self.inputs_name = inputs_name
        self.num_samples = num_samples
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.dataset = datasets.MNIST('../data', train=False, transform=transform)
        self.index = 0

    def get_next(self):
        if self.index == self.num_samples:
            return None
        image, _ = self.dataset[self.index]
        self.index += 1
        return {self.inputs_name : np.expand_dims(to_numpy(image), axis=0)}

    def rewind(self):
        self.index = 0


def main():
    parser = argparse.ArgumentParser(description='ONNX MNIST training script')
    parser.add_argument('--model', type=str, required=True,
                        help='the ONNX model to quantize')
    parser.add_argument('--save_model', type=str, required=True,
                        help='the quantized ONNX model')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='number of calibration samples (default: 1000)')
    parser.add_argument('--per_axis', action='store_true',
                        help='quantize weights per-axis')
    args = parser.parse_args()

    # Instantiate an InferenceSession to read model inputs name
    ort_session = InferenceSession(args.model, providers=['OpenVINOExecutionProvider'])

    # Prepare calibration data reader
    dr = MnistDataReader(ort_session.get_inputs()[0].name, args.num_samples)

    # We force all layer outputs except those returning signed inputs to be quantized.
    # This will allow us to defer the decision of where to perform the actual downscale operation
    # when we will apply the integer graph transformations.
    # Note that this works only for models where there are no intermediate signed outputs.
    extra_options = {
        'OpTypesToExcludeOutputQuantization' : ['Conv', 'MatMul', 'Gemm'],
        'ForceQuantizeNoInputCheck': True
    }
    # Static 8-bit quantization
    quantize_static(args.model,
                    args.save_model,
                    dr,
                    quant_format=QuantFormat.QDQ,
                    per_channel=args.per_axis,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8,
                    optimize_model=False,
                    extra_options=extra_options)


if __name__ == '__main__':
    main()
