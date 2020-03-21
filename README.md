# SUSTech-NAS-Platform

## Requirements

- pytorch
- tvm (install from)
- nni

## Structure

Code for nas in `sustech_nas/search`. Code about tvm in `sustech_nas/tvm_stage`. Searched models are put in `sustech_nas/models`.

## Roadmap

1. Implement NAS algorithm(s).

   Reference: https://github.com/microsoft/nni/tree/master/examples/nas/darts, https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch

2. Mapping models: pytorch model (torch scripted) -> tvm.relay model -> specific device.
