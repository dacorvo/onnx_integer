import argparse
import onnx
import numpy as np

import onnx_graphsurgeon as gs


def truncate_model(onnx_model, n_ops):
    # Extract graph
    graph = gs.import_onnx(onnx_model)

    if len(graph.nodes) <= n_ops:
        return graph

    # Start from the graph input
    node = graph.inputs[0].outputs[0]
    assert isinstance(node, gs.Node)
    for i in range(n_ops - 1):
        node = node.o()
    assert isinstance(node, gs.Node)
    graph.outputs.clear()
    graph.outputs.append(node.outputs[0])
    
    graph.cleanup().toposort()

    # Return a new ONNX model
    return gs.export_onnx(graph)


def main():
    parser = argparse.ArgumentParser(description='Limit an ONNX model to a specified number of operations')
    parser.add_argument('--model', type=str, required=True,
                        help='the ONNX model to modify')
    parser.add_argument('--save_model', type=str, required=True,
                        help='the path to save the modified model')
    parser.add_argument('--ops', type=int, required=True,
                        help='The maximum number of operations in the graph')
    args = parser.parse_args()
    # Load the model
    onnx_model = onnx.load(args.model)

    # Apply transformations to obtain an new ONNX model
    new_model = truncate_model(onnx_model, args.ops)

    # Save the model
    onnx.save(new_model, args.save_model)


if __name__ == '__main__':
    main()
