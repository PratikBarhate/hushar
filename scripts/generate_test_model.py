# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

"""Exports the single-input f32 sigmoid fixture, for a given feature count.

    conda activate local && python scripts/generate_test_model.py 3
    conda activate local && python scripts/generate_test_model.py 3 --fixed-batch 2

`--fixed-batch N` pins the leading dimension instead of leaving it dynamic, which is
what a model needs to exercise the service's `fixed_batch_size` path. It is a separate
file rather than a replacement, because the dynamic one is what every other test uses.

The 3-feature model is written to both crates' `test-data`, because both test it:
hushar loads it, and `onnxrt-rs` runs it on every execution provider. Each crate's
tests resolve fixtures under its own manifest directory, so `onnxrt-rs` stays
runnable on its own rather than reaching into a sibling crate. The second copy is
written from this one export, so the two cannot drift.
"""

import pathlib
import shutil
import sys

import torch
import torch.nn as nn

HUSHAR = pathlib.Path("hushar/test-data")
ONNXRT = pathlib.Path("onnxrt-rs/test-data")

# Which crates test which size. Sizes not listed go to hushar alone.
DESTINATIONS = {3: [HUSHAR, ONNXRT]}


class SimpleModel(nn.Module):
    def __init__(self, feature_len = 3):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(feature_len, 2)
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":

    feature_len = int(sys.argv[1])
    fixed_batch = (
        int(sys.argv[sys.argv.index("--fixed-batch") + 1])
        if "--fixed-batch" in sys.argv
        else None
    )
    model = SimpleModel(feature_len)

    # Initialize weights and biases with fixed values for deterministic test results
    with torch.no_grad():
        model.linear.weight.fill_(0.5)  # All weights set to 0.5
        model.linear.bias.fill_(0.1)    # All biases set to 0.1

    # A pinned fixture is hushar's alone: onnxrt-rs tests the engine, and a fixed
    # leading dimension is a service-level concern.
    destinations = [HUSHAR] if fixed_batch else DESTINATIONS.get(feature_len, [HUSHAR])
    for out_dir in destinations:
        out_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"sigmoid_model_{feature_len}_batch{fixed_batch}.onnx"
        if fixed_batch
        else f"sigmoid_model_{feature_len}.onnx"
    )
    first, *rest = [out_dir / name for out_dir in destinations]

    dummy_input = torch.randn(fixed_batch or 1, feature_len)
    torch.onnx.export(
        model,
        dummy_input,
        first,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        # Omitting dynamic_axes is what pins the leading dimension to the dummy
        # input's row count, which is the whole point of the pinned fixture.
        dynamic_axes=(
            {}
            if fixed_batch
            else {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        ),
        # The TorchScript exporter, which stopped being the default in torch 2.9.
        # The torch.export one needs `onnxscript` installed and emits a different
        # graph for the same module; this model is three nodes and does not need it.
        dynamo=False,
    )
    for path in rest:
        shutil.copyfile(first, path)

    for path in [first, *rest]:
        print(f"wrote {path} ({path.stat().st_size} bytes)")
