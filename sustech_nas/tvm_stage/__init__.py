# pylint: disable=too-many-arguments, E1101
""" Module for tvm stage. """
import torch

import tvm
from tvm import relay
from tvm.relay.frontend.pytorch import get_graph_input_names
from tvm.contrib import graph_runtime

__all__ = ['convert_model', 'compile_model', 'eval_model']


def convert_model(model, input_shape):
    """
    Convert a pytorch model to tvm model.
    """
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    input_name = get_graph_input_names(scripted_model)[0]  # only one input
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_dict)
    return mod, params, input_name


def compile_model(mod, params, target='llvm', target_host='llvm', opt_level=3):
    """
    Compile the tvm model.
    """
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build(mod,
                                         target=target,
                                         target_host=target_host,
                                         params=params)
    return graph, lib, params


def eval_model(graph, lib, params, ctx, input_name, input_data):
    """
    Evaluate the model with input data.
    """
    mod = graph_runtime.create(graph, lib, ctx)
    # Set inputs
    mod.set_input(input_name, tvm.nd.array(input_data.astype('float32')))
    mod.set_input(**params)
    # Execute
    mod.run()
    # Get outputs
    tvm_output = mod.get_output(0)
    return tvm_output
