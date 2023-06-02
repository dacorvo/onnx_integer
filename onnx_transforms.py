import argparse
import onnx
import numpy as np

import onnx_graphsurgeon as gs

from fixed_point import to_fixed_point


def _find_next_op(node, condition):
    """Skip operations inside a sequence until the one matching the specified condition

    Args:
        node (`Node`): the source node
        condition (function): a function taking an operation string as parameter
           and returning a boolean.

    Returns:
        The first node matching the condition or None

    """
    if condition(node.op):
        return node
    if len(node.outputs) == 0:
        # The node does not have any following nodes
        return None
    elif len(node.outputs) > 1:
        # We can only skip within a single sequence of nodes
        return None
    elif _is_leaf(node):
        # This is this end of a branch
        return None
    # Evaluate the next node
    if isinstance(node, gs.Variable):
        # The next node is immediately following this node
        next_node = node.outputs[0]
    else:
        # Get the next node (using o() shortcut instead of dereferencing the tensor)
        next_node = node.o()
    return _find_next_op(next_node, condition)


def skip_next_ops_in(node, skip_ops):
    """Skip the next operations in the specified list

    Args:
        node (`Node`): the first node to check
        skip_ops (list): a list of skippable operations

    Returns:
        The first node that has an operation not in the list

    """
    return _find_next_op(node, lambda n_op: n_op not in skip_ops)


def skip_next_ops_until(node, target_ops):
    """Skip the next operations not in the specified list

    Args:
        node (`Node`): the source node
        target_ops (list): a list of operations

    Returns:
        The first node that is in the target list

    """
    return _find_next_op(node, lambda n_op: n_op in target_ops)


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
    # Get input node and set its type to uint8
    input_node = graph.inputs[0]
    input_node.dtype = np.uint8
    # Get first node, ignoring useless Quantize-Dequantize sequence (if any)
    first_node = skip_QDQ(input_node)
    # Insert a single rescaling dequantizer
    rescaling_name = f"{input_node.name}/rescaling"
    rescaling_inputs = [input_node,
                        gs.Constant(f"{rescaling_name}/scale", np.array(scale, dtype=np.float32)),
                        gs.Constant(f"{rescaling_name}/zeropoint", np.array(zero_point, dtype=np.uint8))]
    rescaling = gs.Node('DequantizeLinear',
                        name=rescaling_name,
                        inputs=rescaling_inputs,
                        outputs=[gs.Variable(name=f"{rescaling_name}/outputs")])
    graph.nodes.append(rescaling)
    first_node.inputs[0] = rescaling.outputs[0]
    # We need now to update the scale of the biases of the first layer with the
    # correct bias scale: input_scale * weight_scale
    w = first_node.i(1)
    w_scale = w.inputs[1].values
    bias = first_node.i(2)
    # Set correct scale
    bias.inputs[1].values = (scale * w_scale).astype(np.float32)
    return graph


def _is_leaf(node):
    for output in node.outputs:
        if len(output.outputs) == 0:
            return True
    return False


def prune_QDQs(graph):
    """Remove useless Quantize-Dequantize sequences of operations

    when a Node corresponds to an operation that:
    - accepts either float or integer,
    - does not modify the quantization axis.
    Then there is no need to quantize and dequantize the outputs of the previous
    operation.
    """
    # These operations produce outputs that are not rescaled
    scaled_output_ops = ['Conv', 'Linear', 'Gemm', 'Add', 'Relu', 'GlobalAveragePool']
    # These operation receive inputs that must be rescaled
    rescaled_input_ops = ['Conv', 'Linear', 'Gemm', 'Add', 'MaxPool', 'Flatten', 'Reshape']
    for qnode in graph.nodes:
        # The sequence must start with a quantizer node.
        if qnode.op != 'QuantizeLinear' or len(qnode.inputs[0].inputs) == 0:
            continue
        # Identify the next node after a Quantize-Dequantize sequence (if any)
        consumer_node = skip_next_ops_in(qnode.o(), ['Cast', 'DequantizeLinear'])
        if consumer_node != qnode.o() and not _is_leaf(consumer_node):
            # If the consumer node does not require rescaled inputs OR if the parent
            # does not produce scaled outputs, then we can skip the QDQ
            parent_node = qnode.i(0)
            if parent_node.op not in scaled_output_ops or consumer_node.op not in rescaled_input_ops:
                # unplug useless QDQ sequence
                qdq_last_node = consumer_node.i()
                qnode.inputs.clear()
                parent_node.outputs = qdq_last_node.outputs
                qdq_last_node.outputs.clear()
    return graph


def _get_quantized_op_nodes(graph):
    """Get the group of nodes corresponding to an operation

    Starting from each DequantizeLinear node in the graph, identify all nodes
    involved in the downstream operation.

    Each operation must have the following pattern:

    Dequantize(a,        Dequantize(b,        Dequantize(c,
               a_scale,             b_scale,             c_scale,
               a_zp)                b_zp)                c_zp)
                       \          |            /
                             operation
                                 |
                    Quantize(outputs, o_scale, o_zeropoint)

    Note:
    - some nodes not modifying the quantization can be inserted
    between the nodes corresponding to the operation.
    - c input is optional

    """
    # Identify operation nodes preceded by a dequantizer
    op_blocks = []
    for adq in graph.nodes:
        # Identify input dequantizers
        if adq.op == 'DequantizeLinear' and not _is_leaf(adq):
            # To avoid duplicates, we only look for the first inputs
            if adq.o(0).i(0) == adq:
                # Look for a processing node
                op_node = skip_next_ops_until(adq, ['Conv', 'Linear', 'Gemm'])
                if op_node:
                    # Get second input dequantizer node
                    bdq = op_node.i(1)
                    assert bdq.op == 'DequantizeLinear'
                    # Get third dequantizer node if any
                    cdq = op_node.i(2) if len(op_node.inputs) == 3 else None
                    # Get outputs quantizer node
                    oq = skip_next_ops_until(op_node, ['QuantizeLinear'])
                    op_blocks.append([op_node, adq, bdq, cdq, oq])
    return op_blocks


def fold_op_scales(graph):
    """Fold inputs and weights scales into the operation quantizer scale

    Each operation has the following pattern:

    Dequantize(inputs,    Dequantize(weights,  Dequantize(bias
               i_scale,              w_scale,             b_scale,
               i_zp)                 w_zp)                b_zp)
                     \              |                 /
                                operation
                                    |
                    Quantize(outputs, o_scale, o_zeropoint)

    The goal of this transformation is to fold the inputs and weights scales into
    the output scale:

    o_scale = o_scale / (i_scale * w_scale)
    i_scale = 1.0
    w_scale = 1.0

    """
    # Identify operation nodes with their dequantized inputs
    op_nodes = _get_quantized_op_nodes(graph)
    for nodes in op_nodes:
        # Isolate input, weight, bias and output dequantizers
        _, idq, wdq, bdq, oq = nodes
        if oq is None:
            # No output quantizer: cannot fold scales
            continue
        # Evaluate scales
        i_scale = idq.inputs[1].values
        w_scale = wdq.inputs[1].values
        b_scale = bdq.inputs[1].values if bdq else None
        o_scale = oq.inputs[1].values
        # Sanity check
        if b_scale is not None:
            np.testing.assert_allclose(b_scale, i_scale * w_scale)
        # Create new tensors to store updated scales
        oq.inputs[1] = gs.Constant(name=f"{oq.inputs[1].name}/rescaled",
                                   values=np.array(o_scale / (i_scale * w_scale)))
        idq.inputs[1] = gs.Constant(name=f"{idq.inputs[1].name}/folded",
                                    values=np.ones_like(i_scale))
        wdq.inputs[1] = gs.Constant(name=f"{wdq.inputs[1].name}/folded",
                                    values=np.ones_like(w_scale))
        if b_scale is not None:
            bdq.inputs[1] = gs.Constant(name=f"{bdq.inputs[1].name}/folded",
                                        values=np.ones_like(b_scale))
        # Check zeropoint shape
        zp = oq.inputs[2].values
        filters = oq.inputs[1].shape[0]
        if zp.shape != (filters,):
            # Output quantizer was previously per-tensor: set it per-axis
            oq.attrs['axis'] = 1
            # Replace zeropoint by a vector
            oq.inputs[2] = gs.Constant(name=f"{oq.inputs[2].name}/reshaped",
                                       values=np.full((filters,), zp, dtype=np.uint8))
    return graph


def split_bias_add(graph):
    """Perform bias addition as a separate operation

    Since ConvInteger and MatMulInteger do not support biases, we need to
    perform the bias addition as a separate operation before converting to integer
    operations.
    """
    # Operations using biases
    biased_ops = ['Conv', 'Linear', 'Gemm']
    # Identify operation nodes with their dequantized inputs
    op_nodes = _get_quantized_op_nodes(graph)
    for nodes in op_nodes:
        # Isolate op, input, weight, bias and output dequantizers
        op_node, idq, wdq, bdq, _ = nodes
        if bdq is None:
            # No bias
            continue
        # Reshape the bias values to make them broadcastable on the node outputs
        bias_values = bdq.inputs[0].values
        filters = bias_values.shape[0]
        bias_shape = [1, filters, 1, 1] if op_node.op == 'Conv' else [1, filters]
        bdq.inputs[0].values = bias_values.reshape(bias_shape)
        # Modify the bias dequantization axis accordingly
        bdq.attrs['axis'] = 1
        # Modify expected output shape
        bdq.outputs[0].shape = bias_shape
        # Create a new Add node
        add_name = f"{op_node.name}/bias_add"
        add_inputs = [op_node.outputs[0], bdq.outputs[0]]
        add_outputs = [gs.Variable(f"{add_name}/out",
                                   shape=op_node.outputs[0].shape,
                                   dtype=op_node.outputs[0].dtype)]
        bias_add = gs.Node(op='Add',
                           name=add_name,
                           inputs=add_inputs,
                           outputs=add_outputs)
        # Add it to the graph
        graph.nodes.append(bias_add)
        # Remove the bias from the node inputs
        op_node.inputs = op_node.inputs[0:2]
        # Update the next node input to use the bias_add outputs
        assert len(op_node.outputs) == 1
        next_node = op_node.o(0)
        next_node.inputs[0] = add_outputs[0]
    return graph


def _remove_dequantizer(node):
    # Update output node inputs with the dequantizer first input
    output_node = node.o()
    for i, input in enumerate(output_node.inputs):
        if input == node.outputs[0]:
            output_node.inputs[i] = node.inputs[0]
    node.inputs.clear()
    node.outputs.clear()
    return node


def _convert_integer_op(op_node):
    if op_node.op == 'Conv':
        int_op_node = gs.Node('ConvInteger')
        int_op_node.attrs = op_node.attrs.copy()
    elif op_node.op in ('Gemm', 'Linear'):
        int_op_node = gs.Node('MatMulInteger')
    int_op_node.name = f"{op_node.name}/int"
    return int_op_node


def _set_integer_output_tensors(node, dtype=np.int32):
    # Check if this node modifies its output type
    if node.op == 'DequantizeLinear':
        # Float outputs: stop here
        return
    if node.op == 'QuantizeLinear':
        dtype = np.uint8
    elif node.op in ['ConvInteger', 'MatMulInteger']:
        dtype = np.int32
    if len(node.outputs) > 0:
        output = node.outputs[0]
        if isinstance(output, gs.Variable):
            output.dtype = dtype
        for next_node in output.outputs:
            if isinstance(next_node, gs.Node):
                _set_integer_output_tensors(next_node, dtype)


def integer_ops(graph):
    """Convert float operations to their integer equivalent

    """
    # Identify operation nodes with their dequantized inputs
    op_nodes = _get_quantized_op_nodes(graph)
    pruned_dqs = []
    for nodes in op_nodes:
        # Isolate op, input, weight, bias and output dequantizers
        op_node, idq, wdq, bdq, oq = nodes
        if bdq is not None:
            print(f"{op_node.name} cannot be converted because it uses a bias."
                  "Transform the graph first to split the bias addition.")
            continue

        # Extract dequantizers scales and zero-points
        scales = []
        zero_points = []
        for dq in (idq, wdq):
            scales.append(dq.inputs[1])
            zero_points.append(dq.inputs[2])

        if oq is not None:
            if (np.any(scales[0].values.astype(np.float32) != 1.) or
                    np.any(scales[1].values.astype(np.float32) != 1.)):
                print(f"{op_node.name} cannot be converted because it receives"
                      " a dequantized input with a non-identity scale."
                      "Transform the graph first to fold the input and weight scales.")
                continue

        # Instantiate a new node with an integer operation
        new_op_node = _convert_integer_op(op_node)
        # Replace the operation node
        new_op_node.inputs = op_node.inputs
        new_op_node.outputs = op_node.outputs
        op_node.inputs.clear()
        op_node.outputs.clear()
        # Add inputs and weights zero-point to the new node
        new_op_node.inputs.append(zero_points[0])
        new_op_node.inputs.append(zero_points[1])
        graph.nodes.append(new_op_node)
        # Now, remove inputs and weights dequantizers
        pruned_dqs.append(_remove_dequantizer(idq))
        pruned_dqs.append(_remove_dequantizer(wdq))
        # Check if we need to transpose weights
        if "transB" in op_node.attrs:
            w = new_op_node.inputs[1]
            w.values = np.transpose(w.values)
        # Check if there is a bias addition
        bias = None
        if not _is_leaf(new_op_node) and new_op_node.o().op == 'Add':
            bias = new_op_node.o()
        if bias:
            # Remove the bias addition Dequantizer
            # Note that we don't know if it is the first or second input
            dq = bias.i()
            if dq == new_op_node:
                dq = bias.i(1)
            pruned_dqs.append(_remove_dequantizer(dq))
        if oq is None:
            # Get the last node corresponding to the operation
            insertion_node = new_op_node if bias is None else bias
            if not _is_leaf(new_op_node):
                next_node = new_op_node.o()
                if next_node.op == 'Add':
                    insertion_node = next_node
            # Insert a final Dequantizer
            dq_name = f"{new_op_node.name}/dequantizer"
            output_scale = scales[0].values * scales[1].values
            dq_inputs = [insertion_node.outputs[0],
                         gs.Constant(name=f"{dq_name}/output_scale", values=output_scale)]
            dq_outputs = [gs.Variable(name=f"{dq_name}/output",
                                      dtype=np.float32)]
            dq = gs.Node('DequantizeLinear',
                         name=dq_name,
                         inputs=dq_inputs,
                         outputs=dq_outputs)
            graph.nodes.append(dq)
            next_node = insertion_node.o()
            next_node.inputs[0] = dq.outputs[0]
    # Remove detached dequantizers from the graph
    for dq in pruned_dqs:
        graph.nodes.remove(dq)
    # Update graph tensor types
    first_node = graph.inputs[0].outputs[0]
    _set_integer_output_tensors(first_node)
    return graph


def _create_scale_out_sequence(inputs, graph, mantissa, frac_bits):
    def create_array(inputs, t):
        t = np.array(t, dtype=np.int32)
        if len(t.shape) != 0:
            # Expand tensor dim to make it broadcastable
            n_output_dims = len(inputs.shape)
            axis = (0,) + tuple(range(2, n_output_dims))
            t = np.expand_dims(t, axis=axis)
        return t
    # Create a new Multiply node to multiply by the mantissa
    mul_node_name = f"{inputs.name}/scale_out"
    mantissa = create_array(inputs, mantissa)
    mul_node_inputs = [inputs,
                       gs.Constant(name=f"{mul_node_name}/scale", values=mantissa)]
    mul_node = gs.Node('Mul',
                       name=mul_node_name,
                       inputs=mul_node_inputs,
                       outputs=[gs.Variable(name=f"{mul_node_name}/outputs", dtype=np.int32)])
    graph.nodes.append(mul_node)
    # Create a new div to apply shift (cannot use bitshift because inputs are signed)
    divisor = create_array(inputs, 2**frac_bits)
    div_node_name = f"{inputs.name}/shift_out"
    div_node_inputs = [mul_node.outputs[0],
                       gs.Constant(name=f"{div_node_name}/shift", values=divisor)]
    div_node = gs.Node('Div',
                       name=div_node_name,
                       inputs=div_node_inputs,
                       outputs=[gs.Variable(name=f"{div_node_name}/outputs", dtype=np.int32)])
    graph.nodes.append(div_node)
    # Create a final cast node
    cast_node = gs.Node('Cast',
                        attrs={'to': np.uint8},
                        inputs=div_node.outputs,
                        outputs=[gs.Variable(name=f"{inputs.name}/cast", dtype=np.uint8)])
    graph.nodes.append(cast_node)
    return cast_node


def scale_out(graph, bitwidth):
    """Replace the layer output QuantizeLinear by integer operations
    """
    pruned_oqs = []
    for node in graph.nodes:
        if node.op in ['ConvInteger', 'MatMulInteger']:
            # Get outputs quantizer node
            oq = skip_next_ops_until(node, ['QuantizeLinear'])
            if oq is None:
                continue
            # Evaluate scale and zero-point
            scale = oq.inputs[1]
            zeropoint = oq.inputs[2]
            if np.any(zeropoint.values != 0):
                # Not supported for now.
                # If the previous operation is a bias addition, the zeropoint could
                # be rescaled, quantized and then folded into the bias
                continue
            # Get a FixedPoint representation of the reciprocal of the scale
            mantissa, frac_bits = to_fixed_point(1 / scale.values, bitwidth=8, signed=False)
            # Create a new scale_out sequence starting after the output quantizer first inputs
            last_node = _create_scale_out_sequence(oq.inputs[0], graph, mantissa, frac_bits)
            # Link the output quantizer output node
            oq.o().inputs[0] = last_node.outputs[0]
            # Unplug output quantizer
            oq.inputs.clear()
            oq.outputs.clear()
            pruned_oqs.append(oq)
    # Remove detached dequantizers from the graph
    for oq in pruned_oqs:
        graph.nodes.remove(oq)
    return graph


def embed_explicit_activations(graph):
    """Add a Relu op between a processing node and QuantizeLinear if the last
    contains the activation implicitly. This can be inferred from two rules

    1. QuantizeLinear zero point is 0
    2. QuantizeLinear output dtype is uint8

    Args:
        graph (gs.graph): Graph to modify

    Returns:
        gs.graph : graph with new activation nodes.
    """
    for op_node in graph.nodes:
        # Look for a processing node
        if op_node.op not in ['Conv', 'Linear', 'Gemm']:
            continue

        # Look for an output quantizer.
        # If relu already exists, there is no need to add a new one.
        qnode = skip_next_ops_until(op_node, ['Relu, QuantizeLinear'])
        if qnode is None or qnode.op in ['Relu']:
            continue

        # Check if output quantizer embeded the activation
        if np.any(qnode.inputs[-1].values) or qnode.outputs[0].dtype != np.dtype("uint8"):
            continue

        # Embed explicit activation BEFORE qnode
        act_node = gs.Node("Relu", inputs=[qnode.inputs[0]],
                           outputs=[gs.Variable(f"{op_node.name}/act",
                                                dtype=op_node.outputs[0].dtype,
                                                shape=op_node.outputs[0].shape)])
        qnode.inputs[0] = act_node.outputs[0]
        graph.nodes.append(act_node)
    return graph


def _fold_cast(graph):
    """Fold cast op into previous one in the following cases:

        1. Input has an output with the same type than value to cast
        2. Input is a QuantizeLinear and zero point match with value to cast

    Args:
        graph (gs.graph): Graph to modify
    """
    # Mapper from enum to type (skipping 'UNDEFINED' type)
    _supported_types = {k: v for v, k in onnx.TensorProto.DataType.items() if v != 'UNDEFINED'}
    nodes_to_prune = []
    for node in graph.nodes:
        if node.op != 'Cast':
            continue
        # We can skip a cast op in two cases
        cast_to = _supported_types.get(node.attrs['to'], None)
        if cast_to is not None and len(node.inputs[0].inputs) == 1:
            input_node = node.i()
            # 1. When input node's output has the same type to cast
            cast_to = np.dtype(cast_to.lower())
            skip_node = input_node.outputs[0].dtype == cast_to
            # 2. When input node is QuantizeLinear and zero point has same type to cast
            if input_node.op == "QuantizeLinear" and input_node.inputs[-1].dtype == cast_to:
                skip_node = True
            if skip_node:
                # Unplug useless cast node
                input_node.outputs = node.outputs
                node.inputs.clear()
                node.outputs.clear()
                # Set output dtype
                input_node.outputs[0].dtype = cast_to
                nodes_to_prune.append(node)
    # Remove nodes of graph
    for ncast in nodes_to_prune:
        graph.nodes.remove(ncast)


def sanitize(graph):
    # Call the graph surgeon helper:
    # - fold 'Shape' nodes,
    # - partitioning = None, meaning that a single failure invalidates all foldings,
    # - size_threshold = None, all constants are folded regardless of their size.
    graph.fold_constants(fold_shapes=True,
                         partitioning=None,
                         size_threshold=None)
    # - fold 'Cast' nodes when output is a Variable (constants were folded in previous step)
    _fold_cast(graph)
    return graph


def apply_transforms(onnx_model, transforms):
    # Some transformations will need to know input-output shapes and dtype
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model, check_type=True)

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

    # Return a new ONNX model, preserving Intermediate Representation version
    return gs.export_onnx(graph, ir_version=onnx_model.ir_version)


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
    parser.add_argument('--embed_act', action='store_true',
                        help='embed explicit activations after processing nodes.')
    parser.add_argument('--prune_qdq', action='store_true',
                        help='remove useless Quantize/Dequantize sequences.')
    parser.add_argument('--fold_op_scales', action='store_true',
                        help='fold all operations inputs and weights scales)')
    parser.add_argument('--split_bias_add', action='store_true',
                        help='add biases as a separate operation)')
    parser.add_argument('--integer_ops', action='store_true',
                        help='use integer operations)')
    parser.add_argument('--scale_out', action='store_true',
                        help='convert output quantizers to integer operations)')
    args = parser.parse_args()
    # Load the model
    onnx_model = onnx.load(args.model)

    # Gather transformations as (transform, args) tuples
    transforms = []
    # The first transformation is always to sanitize the model
    transforms.append([sanitize, []])
    if args.add_rescaling:
        transforms.append([add_rescaling, [args.scale, args.zero_point]])
    if args.embed_act:
        transforms.append([embed_explicit_activations, []])
    if args.prune_qdq:
        transforms.append([prune_QDQs, []])
    if args.fold_op_scales:
        transforms.append([fold_op_scales, []])
    if args.split_bias_add:
        transforms.append([split_bias_add, []])
    if args.integer_ops:
        transforms.append([integer_ops, []])
    if args.scale_out:
        transforms.append([scale_out, [8]])

    # Apply transformations to obtain an new ONNX model
    new_model = apply_transforms(onnx_model, transforms)

    # Save the model
    onnx.save(new_model, args.save_model)


if __name__ == '__main__':
    main()
