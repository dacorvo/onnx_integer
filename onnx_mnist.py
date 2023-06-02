import argparse
import torch
import onnxruntime

from torchvision import datasets, transforms
from timeit import default_timer as timer


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def onnx_eval(ort_session, test_loader): 
    correct = 0
    start = timer()
    for data, target in test_loader:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
        ort_outputs = ort_session.run(None, ort_inputs)
        pred = ort_outputs[0].argmax(axis=1)
        correct += (pred == to_numpy(target)).sum()
    elapsed = timer() - start
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(f'Inference took {elapsed:.4f} seconds.')

def main():
    parser = argparse.ArgumentParser(description='ONNX MNIST inference script')
    parser.add_argument('--model', type=str, required=True,
                        help='the ONNX model to evaluate')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    args = parser.parse_args()
    # Load the model into an onnx runtime session
    ort_session = onnxruntime.InferenceSession(args.model, providers=['OpenVINOExecutionProvider'])

    # Prepare dataloader
    if ort_session.get_inputs()[0].type == 'tensor(float)':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    else:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.uint8)
            ])
    dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_kwargs = {'batch_size': args.test_batch_size}
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    # Check inference
    onnx_eval(ort_session, test_loader)


if __name__ == '__main__':
    main()
