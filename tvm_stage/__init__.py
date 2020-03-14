import tvm
from tvm import relay
from tvm.relay.frontend.pytorch import get_graph_input_names
from tvm.contrib import graph_runtime

import torch
import torchvision

__all__ = ['convert', 'compile', 'eval']


def convert(model, input_shape):
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    input_name = get_graph_input_names(scripted_model)[0]  # only one input
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_dict)
    return mod, params, input_name


def compile(mod, params, target='llvm', target_host='llvm', opt_level=3):
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build(mod,
                                         target=target,
                                         target_host=target_host,
                                         params=params)
    return graph, lib, params


def eval(graph, lib, params, ctx, input_name, input_data):
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    # Set inputs
    m.set_input(input_name, tvm.nd.array(input_data.astype(dtype)))
    m.set_input(**params)
    # Execute
    m.run()
    # Get outputs
    tvm_output = m.get_output(0)
    return tvm_output


if __name__ == "__main__":
    import numpy as np
    model = torchvision.models.resnet18(pretrained=True)
    model = model.eval()
    input_shape = [1, 3, 224, 224]

    mod, params, input_name = convert(model, input_shape)
    graph, lib, params = compile(mod, params)

    input_data = np.random.normal(size=input_shape)
    dtype = 'float32'
    ctx = tvm.cpu(0)

    m = graph_runtime.create(graph, lib, ctx)
    # Set inputs
    m.set_input(input_name, tvm.nd.array(input_data.astype(dtype)))
    m.set_input(**params)
    # Execute
    m.run()
    # Get outputs
    tvm_output = m.get_output(0)
