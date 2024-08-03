import torch
import torch.nn as nn

class TracerPixel(nn.Module):
    def __init__(self):
        super(TracerPixel, self).__init__()
        self.trace = []

    def forward(self, x):
        self.trace.append(x[0, 0, 0, 0].item())  # Track the value of the first pixel
        return x

class DebuggedNetwork(nn.Module):
    def __init__(self, original_model):
        super(DebuggedNetwork, self).__init__()
        self.layers = []
        self.tracer_pixels = []
        
        for name, module in original_model.named_children():
            self.layers.append(module)
            tracer = TracerPixel()
            self.tracer_pixels.append(tracer)
            self.layers.append(tracer)
        
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_trace(self):
        return [tp.trace for tp in self.tracer_pixels]

# Usage example
original_model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 3, 3, padding=1)
)

debugged_model = DebuggedNetwork(original_model)

# Run a forward pass
input_tensor = torch.randn(1, 3, 224, 224)
output = debugged_model(input_tensor)

# Get the trace
trace = debugged_model.get_trace()

# Analyze the trace
for i, layer_trace in enumerate(trace):
    print(f"Layer {i} trace: {layer_trace}")