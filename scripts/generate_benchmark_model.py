import torch
import torch.nn as nn
import torch.onnx

class BenchmarkModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(BenchmarkModel, self).__init__()    
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.2))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

model = BenchmarkModel(
    input_size=1000, 
    output_size=5, 
    hidden_sizes=[4096, 4096, 4096, 2048, 2048, 1024, 4096, 512]
)

total_params = sum(p.numel() for p in model.parameters())
total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
print(f"Total parameters: {total_params:,}")
print(f"Model size: {total_size_mb:.2f} MB")

dummy_input = torch.randn(1, 1000)

output_path = "large_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model successfully exported to {output_path}")

import onnx
import onnxruntime

onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("ONNX model verification successful")

ort_session = onnxruntime.InferenceSession(output_path)

random_input = torch.randn(1, 1000).numpy()
ort_inputs = {ort_session.get_inputs()[0].name: random_input}
ort_outputs = ort_session.run(None, ort_inputs)

print(f"ONNX inference successful. Output shape: {ort_outputs[0].shape}")
print(f"Sample output: {ort_outputs[0][0][:5]}")
print("\nInstructions for using the model:")
print("1. The ONNX model is saved as 'large_model.onnx'")
print("2. Input shape: [batch_size, 1000]")
print("3. Output shape: [batch_size, 5]")
print("4. The model uses random weights as requested")
print("5. Use onnxruntime.InferenceSession() to load and run inference")