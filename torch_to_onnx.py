import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='PyTorch to ONNX conversion script')
    parser.add_argument('--model', type=str, required=True,
                        help='the quantized pytorch model to convert (in TorchScript)')
    parser.add_argument('--save_model', type=str, required=True,
                        help='the path to save the ONNX model')
    args = parser.parse_args()
    # Load the quantized model
    model = torch.jit.load(args.model)

    # Load the base model on CPU (!! Hangs on GPU)
    device = torch.device("cpu")
    model.to(device)
    # Export to onnx (it will be saved to disk in the process)
    dummy_inputs = torch.rand(1,1,28,28)
    input_names = ["inputs"]
    output_names = ["outputs"]
    dynamic_axes = {'inputs' : {0: 'batch_size'},
                    'outputs' : {0 : 'batch_size'}}
    torch.onnx.export(model,
                      dummy_inputs,
                      args.save_model,
                      opset_version=15,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    main()
