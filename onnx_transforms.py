import argparse
import onnx
import numpy as np

import onnx_graphsurgeon as gs
from onnx_graphsurgeon import Node


def _skip_next_ops(node, condition):
    """Skip the next operations matching the specified condition

    Args:
        node (`Node`): the source node
        condition (function): a function taking an operation string as parameter
        and returning a boolean.

    Returns:
        The first node not matching the conditioni after the current node

    """
    if len(node.outputs) == 0:
        # The node does not have any following nodes
        return None
    elif len(node.outputs) > 1:
        # We can only skip within a single sequence of nodes
        return node
    if isinstance(node, gs.Variable):
        # The next node is immediately following this node
        next_node = node.outputs[0]
    else:
        # Get the next node (using o() shortcut instead of dereferencing the tensor)
        next_node = node.o()
    if condition(next_node.op):
        # Skip the node containing that operation and repeat
        return _skip_next_ops(next_node, condition)
    return next_node


def skip_next_ops_in(node, skip_ops):
    """Skip the next operations in the specified list

    Args:
        node (`Node`): the source node
        skip_ops (list): a list of skippable operations

    Returns:
        The first node after the current node that has an operation not in the list

    """
    return _skip_next_ops(node, lambda n_op: n_op in skip_ops)


def skip_next_ops_until(node, target_ops):
    """Skip the next operations not in the specified list

    Args:
        node (`Node`): the source node
        target_ops (list): a list of operations

    Returns:
        The first node that is in the target list

    """
    return _skip_next_ops(node, lambda n_op: n_op not in target_ops)


def skip_QDQ(node):
    if len(node.outputs) > 1:
        return node
    if isinstance(node, gs.Variable):
        # This is likely an input node
        next_node = node.outputs[0]
    else:
        # Get the next node (using o() shortcut instead of dereferencing outputs)
        next_node = node.o()
    # A skippable QDQ sequence starts with a Quantizer with a single output
    if next_node.op != 'QuantizeLinear':
        return next_node
    # Get the next node (using o() shortcut instead of dereferencing outputs)
    next_node = next_node.o()
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
    # Get first node, ignoring useless Quantize-Dequantize sequence (if any)
    first_node = skip_QDQ(input_node)
    # Insert a Dequantizer node with the specified scale and offset
    rescaling_inputs = [input_node, np.array(scale, dtype=np.float32), np.array(zero_point, dtype=np.uint8)]
    input_node.outputs.clear()
    input_node.dtype = np.uint8
    rescaling_outputs = [first_node.inputs[0]]
    first_node.inputs[0].inputs.clear()
    rescaling = graph.layer(op='DequantizeLinear', inputs=rescaling_inputs, outputs=[first_node.inputs[0]])
    # We need now to update the scale of the biases of the first layer with the
    # correct bias scale: input_scale * weight_scale
    w = first_node.i(1)
    w_scale = w.inputs[1].values
    bias = first_node.i(2)
    # Set correct scale
    bias.inputs[1].values = (scale * w_scale).astype(np.float32)
    return graph


def prune_QDQs(graph):
    """Remove useless Quantize-Dequantize sequences of operations

    when a Node corresponds to an operation that:
    - accepts either float or integer,
    - does not modify the quantization axis.
    Then there is no need to quantize and dequantize the outputs of the previous
    operation.
    """
    # These operations require and return dequantized inputs
    dq_ops = ['Conv', 'Linear', 'Gemm', 'Flatten']
    for qnode in graph.nodes:
        # The sequence must start with a quantizer node
        if qnode.op != 'QuantizeLinear':
            continue
        # Identify the next node after a Quantize-Dequantize sequence (if any)
        consumer_node = skip_next_ops_in(qnode, ['Cast', 'DequantizeLinear'])
        if consumer_node != qnode.o():
            # Check if we can skip the QDQ
            parent_node = qnode.i(0)
            if parent_node.op in dq_ops or consumer_node.op not in dq_ops:
                # unplug useless QDQ sequence
                qdq_last_node = consumer_node.i()
                qnode.inputs.clear()
                parent_node.outputs = qdq_last_node.outputs
                qdq_last_node.outputs.clear()
    return graph


def fold_op_scales(graph):
    """Fold inputs and weights scales into the operation quantizer scale

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

    """
    # Identify operation nodes preceded by a dequantizer
    target_patterns = []
    # Identify input dequantizers
    for idq in graph.nodes:
        if idq.op == 'DequantizeLinear':
            # Inputs dequantizer is always the first input
            if idq.o(0).i(0) == idq:
                # Look for a processing node
                pnode = skip_next_ops_until(idq, ['Conv', 'Linear', 'Gemm'])
                if pnode:
                    # Get weights dequantizer node
                    wdq = pnode.i(1)
                    assert wdq.op == 'DequantizeLinear'
                    # Get bias dequantizer node if any
                    bdq = pnode.i(2) if len(pnode.inputs) == 3 else None
                    # Get outputs quantizer node
                    oq = skip_next_ops_in(pnode, ['Relu', 'MaxPool'])
                    if oq.op == 'QuantizeLinear':
                        target_patterns.append([idq, wdq, bdq, oq])
    for pattern in target_patterns:
        idq, wdq, bdq, oq = pattern
        # Evaluate scales
        i_scale = idq.inputs[1].values
        w_scale = wdq.inputs[1].values
        b_scale = bdq.inputs[1].values if bdq else None
        o_scale = oq.inputs[1].values
        # Sanity check
        if b_scale is not None:
           np.testing.assert_allclose(b_scale, i_scale * w_scale)
        # Update scales
        oq.inputs[1].values = o_scale / (i_scale * w_scale)
        idq.inputs[1].values = np.ones_like(i_scale)
        wdq.inputs[1].values = np.ones_like(w_scale)
        if b_scale is not None:
            bdq.inputs[1].values = np.ones_like(b_scale)
        if not "axis" in oq.attrs:
            if "axis" in idq.attrs or "axis" in wdq.attrs:
                # Output quantizer was previously per-tensor: set it per-axis
                oq.attrs['axis'] = 1
                # Reshape zeropoint
                zp = oq.inputs[2].values
                filters = oq.inputs[1].shape[0]
                oq.inputs[2].values = np.full((filters,), zp, dtype=np.uint8)
    return graph


def split_bias_add(graph):
    """Perform bias addition as a separate operation

    Since ConvInteger and MatMulInteger do not support biases, we need to
    perform the bias addition as a separate operation before converting to integer
    operations.
    """
    # Operations using biases
    biased_ops = ['Conv', 'Linear', 'Gemm']
    for node in graph.nodes:
        # If the operation uses a bias, it has a third input
        if node.op not in biased_ops or len(node.inputs) != 3:
            continue
        # Get bias node
        bias = node.i(2)
        assert bias.op == 'DequantizeLinear'
        # Reshape the bias values to make them broadcastable on the node outputs
        bias_values = bias.inputs[0].values
        filters = bias_values.shape[0]
        bias_shape = [1, filters, 1, 1] if node.op == 'Conv' else [1, filters]
        bias.inputs[0].values = bias_values.reshape(bias_shape)
        # Modify the bias dequantization axis accordingly
        bias.attrs['axis'] = 1
        # Create a new Add node
        add_name = f"{node.name}/bias_add"
        add_inputs = [node.outputs[0], bias.outputs[0]]
        add_outputs = [gs.Variable(f"{add_name}/out")]
        bias_add = gs.Node(op='Add',
                           name=add_name,
                           inputs=add_inputs,
                           outputs=add_outputs)
        # Add it to the graph
        graph.nodes.append(bias_add)
        # Remove the bias from the node inputs
        node.inputs = node.inputs[0:2]
        # Update the next node input to use the bias_add outputs
        assert len(node.outputs) == 1
        next_node = node.o(0)
        next_node.inputs[0] = add_outputs[0]
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
    parser.add_argument('--prune_qdq', action='store_true',
                        help='remove useless Quantize/Dequantize sequences.')
    parser.add_argument('--fold_op_scales', action='store_true',
                        help='fold all operations inputs and weights scales)')
    parser.add_argument('--split_bias_add', action='store_true',
                        help='add biases as a separate operation)')
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
    if args.prune_qdq:
        transforms.append([prune_QDQs, []])
    if args.fold_op_scales:
        transforms.append([fold_op_scales, []])
    if args.split_bias_add:
        transforms.append([split_bias_add, []])

    # Apply transformations to obtain an new ONNX model
    new_model = apply_transforms(onnx_model, transforms)

    # Save the model
    onnx.save(new_model, args.save_model)


if __name__ == '__main__':
    main()
