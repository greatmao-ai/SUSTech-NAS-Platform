import numpy as np
import tvm
import torchvision

from sustech_nas import tvm_stage as ts


def test_whole_stage():
    model = torchvision.models.resnet18(pretrained=True)
    model = model.eval()
    input_shape = [1, 3, 224, 224]

    mod, params, input_name = ts.convert_model(model, input_shape)
    graph, lib, params = ts.compile_model(mod, params)

    input_data = np.random.normal(size=input_shape).astype('float32')
    ctx = tvm.cpu(0)

    ts.eval_model(graph, lib, params, ctx, input_name, input_data)

if __name__ == "__main__":
    test_whole_stage()
