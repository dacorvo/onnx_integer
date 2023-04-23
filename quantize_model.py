import argparse
import copy
import torch
import platform

from torchvision import datasets, transforms
from torch.quantization import quantize_fx
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.backend_config.x86 import get_x86_backend_config
from torch.ao.quantization.backend_config.qnnpack import get_qnnpack_backend_config

from train_model import test
from mnist_model import MNISTModel


def machine_backend():
    # Pytorch uses FBGEMM on x86 and QNNPACK on ARM
    return 'x86' if platform.machine() in ("i386", "AMD64", "x86_64") else 'qnnpack'


def get_onnx_backend_config(backend):
    # The proper way to create a dedicated config would be to add only the relevant
    # (pattern, BackendPatternConfig) pairs.
    # Here we just start from the base config and pop unsupported fused operators.
    base_config = get_qnnpack_backend_config() if backend == 'qnnpack' else get_x86_backend_config()
    unsupported_onnx_ops = [
        torch.ao.nn.intrinsic.modules.fused.LinearReLU
    ]
    onnx_config = BackendConfig()
    for config in base_config.configs:
        clone_config = copy.deepcopy(config)
        pattern = clone_config.pattern
        if clone_config.fused_module in unsupported_onnx_ops:
            print(f"Removing {pattern} pattern associated to an unsupported fused operator.")
        else:
            onnx_config.set_backend_pattern_config(clone_config)
    return onnx_config


def quantize_model(model, samples=None, backend=None):
    if backend is None:
        backend = machine_backend()
    torch.backends.quantized.engine = backend
    # Use default config for activations and weights:
    # - HistogramObserver for activations
    # - symmetric int8 PerAxisMinMaxQuantizer for weights
    qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(backend)
    # Copy the model
    model_q = copy.deepcopy(model)
    model_q.eval()
    # Prepare model for quantization by inserting observers and passing samples
    print(f"Creating intermediate model to prepare quantization.")
    sample = torch.rand(1,1,28,28)
    model_q = quantize_fx.prepare_fx(model_q,
                                     qconfig_mapping,
                                     sample,
                                     backend_config=get_onnx_backend_config(backend))
    # Calibrate
    if samples is None:
        samples = torch.rand(10, 1, 28, 28)

    n_samples = samples.shape[0]
    print(f"Calibrating intermediate model wih {n_samples} samples.")
    with torch.inference_mode():
        for i in range(samples.shape[0]):
            model_q(samples[i:i+1, :, :, :])
    # quantize
    print(f"Quantizing model.")
    model_q = quantize_fx.convert_fx(model_q,
                                     qconfig_mapping=qconfig_mapping,
                                     backend_config=get_onnx_backend_config(backend))
    return model_q


def main():
    parser = argparse.ArgumentParser(description='PyTorch Quantization script')
    parser.add_argument('--model', type=str, required=True,
                        help='the MNIST pytorch model to quantize')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--save_model', type=str, required=True,
                        help='the path to save the quantized model (in TorchScript)')
    args = parser.parse_args()
    model = MNISTModel()

    # Get dataset
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_kwargs = {'batch_size': args.test_batch_size}
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    # Extract calibration samples
    samples, _ = next(iter(test_loader))

    # Load and quantize the model
    model.load_state_dict(torch.load(args.model))
    model_q = quantize_model(model, samples)

    # Evaluate the base model
    device = torch.device("cpu")
    print("Evaluating source model")
    test(model, device, test_loader)
    # Evaluate the quantized model (!!hangs on GPU)
    print("Evaluating quantized model")
    test(model_q, device, test_loader)
    # Save the quantized model in TorchScript
    model_qs = torch.jit.script(model_q)
    model_qs.save(args.save_model)


if __name__ == '__main__':
    main()
