import argparse
import onnx
import numpy as np

import onnx_graphsurgeon as gs
from onnx_graphsurgeon import Node


def skip_QDQ(node):
    # A skippable QDQ sequence starts with a Quantizer with a single output
    if node.op != 'QuantizeLinear':
        return node
    # Get the next node (using o() shortcut instead of dereferencing outputs)
    next_node = node.o()
    if next_node.op == 'Cast':
        # Skip Cast from uint8 to int8
        next_node = next_node.o()
    if next_node.op != 'DequantizeLinear':
        return node
    return next_node.o()


def add_rescaling(graph, scale, zero_point):
    # Extract input
    if len(graph.inputs) > 1 or len(graph.inputs[0].outputs) > 1:
        raise ValueError("Cannot add rescaling on a graph with multiple inputs.")
    # Get input node
    input_node = graph.inputs[0]
    # Get first node
    first_node = input_node.outputs[0]
    # Skip useless Quantize-Dequantize sequence (if any)
    first_node = skip_QDQ(first_node)
    # Create a Dequantizer node with the specified scale and offset
    rescaling_inputs = [input_node, np.array(scale, dtype=np.float32), np.array(zero_point, dtype=np.uint8)]
    input_node.outputs.clear()
    input_node.dtype = np.uint8
    rescaling_outputs = [first_node.inputs[0]]
    first_node.inputs[0].inputs.clear()
    rescaling = graph.layer(op='DequantizeLinear', inputs=rescaling_inputs, outputs=[first_node.inputs[0]])
    return graph


def sanitize(graph):
    # Call the graph surgeon helper:
    # - fold 'Shape' nodes,
    # - partitioning = None, meaning that a single failure invalidates all foldings,
    # - size_threshold = None, all constants are folded regardless of their size.
    graph.fold_constants(fold_shapes=True,
                         partitioning=None,
                         size_threshold=None)
    return graph


def apply_transforms(onnx_model, transforms):
    # Extract graph
    graph = gs.import_onnx(onnx_model)

    # Apply transformations sequentially
    for transform in transforms:
        graph = transform[0](graph, *transform[1])

    # Remove unused nodes/tensors, and topologically sort the graph
    # ONNX requires nodes to be topologically sorted to be considered valid.
    # Therefore, you should only need to sort the graph when you have added new nodes out-of-order.
    # To be on the safer side, we always sort.
    graph.cleanup().toposort()

    # Return a new ONNX model
    return gs.export_onnx(graph)


def main():
    parser = argparse.ArgumentParser(description='ONNX to akida conversion script')
    parser.add_argument('--model', type=str, required=True,
                        help='the ONNX model to modify')
    parser.add_argument('--save_model', type=str, required=True,
                        help='the path to save the modified model')
    parser.add_argument('--add_rescaling', action='store_true',
                        help='include rescaling in the graph (requires scale and zero-point)')
    parser.add_argument("--scale",
                        type=float,
                        default=1.0,
                        help="The scale factor applied on uint8 inputs.")
    parser.add_argument("--zero-point",
                        type=int,
                        default=0,
                        help="The zero-point subtracted from uint8 inputs.")
    args = parser.parse_args()
    # Load the model
    onnx_model = onnx.load(args.model)

    # Extract graph
    graph = gs.import_onnx(onnx_model)

    # Gather transformations as (transform, args) tuples
    transforms = []
    # The first transformation is always to sanitize the model
    transforms.append([sanitize, []])
    if args.add_rescaling:
        transforms.append([add_rescaling, [args.scale, args.zero_point]])

    # Apply transformations to obtain an new ONNX model
    new_model = apply_transforms(onnx_model, transforms)

    # Save the model
    onnx.save(new_model, args.save_model)


if __name__ == '__main__':
    main()
