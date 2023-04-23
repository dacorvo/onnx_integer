import argparse
import copy
import torch
import platform

from torchvision import datasets, transforms
from torch.quantization import quantize_fx

from train_model import test
from mnist_model import MNISTModel


def machine_backend():
    # Pytorch uses FBGEMM on x86 and QNNPACK on ARM
    return 'x86' if platform.machine() in ("i386", "AMD64", "x86_64") else 'qnnpack'


def main():
    parser = argparse.ArgumentParser(description='PyTorch to ONNX conversion script')
    parser.add_argument('--model', type=str, required=True,
                        help='the quantized pytorch model to convert (in TorchScript)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--save_model', type=str, required=True,
                        help='the path to save the quantized ONNX model')
    args = parser.parse_args()
    torch.backends.quantized.engine = machine_backend()
    # Load the quantized model
    model = torch.jit.load(args.model)

    # Check inference
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_kwargs = {'batch_size': args.test_batch_size}
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
    # Evaluate the base model on CPU (!! Hangs on GPU)
    device = torch.device("cpu")
    test(model, device, test_loader)
    # Export to onnx (it will be saved to disk in the process)
    dummy_inputs = torch.rand(args.test_batch_size,1,28,28)
    input_names = ["inputs"]
    output_names = ["outputs"]
    torch.onnx.export(model,
                      dummy_inputs,
                      args.save_model,
                      opset_version=15,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)


if __name__ == '__main__':
    main()
