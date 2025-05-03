# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

import sys
import torch
import torch.nn as nn

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
    model = SimpleModel(feature_len)

    # Initialize weights and biases with fixed values for deterministic test results
    with torch.no_grad():
        model.linear.weight.fill_(0.5)  # All weights set to 0.5
        model.linear.bias.fill_(0.1)    # All biases set to 0.1

    dummy_input = torch.randn(1, feature_len) 
    torch.onnx.export(
        model,
        dummy_input,
        f"hushar/test-data/sigmoid_model_{feature_len}.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
