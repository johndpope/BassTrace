import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union, Tuple
import seaborn as sns
import networkx as nx

class AdvancedTracer(nn.Module):
    def __init__(self, name: str, track_pixels: bool = True, num_pixels: int = 5, 
                 track_stats: bool = True, track_gradients: bool = True):
        super(AdvancedTracer, self).__init__()
        self.name = name
        self.track_pixels = track_pixels
        self.num_pixels = num_pixels
        self.track_stats = track_stats
        self.track_gradients = track_gradients
        self.reset()

    def reset(self):
        self.pixel_trace = []
        self.stats_trace = []
        self.gradient_trace = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.track_pixels:
            pixel_values = x[0, :, :self.num_pixels, :self.num_pixels].flatten().tolist()
            self.pixel_trace.append(pixel_values)

        if self.track_stats:
            stats = {
                'mean': x.mean().item(),
                'std': x.std().item(),
                'min': x.min().item(),
                'max': x.max().item(),
                'num_nan': torch.isnan(x).sum().item(),
                'num_inf': torch.isinf(x).sum().item(),
                'l1_norm': x.abs().mean().item(),
                'l2_norm': x.pow(2).mean().sqrt().item()
            }
            self.stats_trace.append(stats)

        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.gradient_trace.append(grad.detach().cpu()))

        return x

class DebuggedModule(nn.Module):
    def __init__(self, module: nn.Module, name: str, tracer: AdvancedTracer):
        super(DebuggedModule, self).__init__()
        self.module = module
        self.name = name
        self.tracer = tracer

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(x, tuple):
            x = tuple(self.tracer(xi) for xi in x)
            return self.module(*x)
        else:
            x = self.tracer(x)
            return self.module(x)

class AdvancedDebuggedNetwork(nn.Module):
    def __init__(self, original_model: nn.Module, track_pixels: bool = True, 
                 num_pixels: int = 5, track_stats: bool = True, 
                 track_gradients: bool = True):
        super(AdvancedDebuggedNetwork, self).__init__()
        self.debugged_modules = nn.ModuleList()
        self.graph = nx.DiGraph()
        self._wrap_model(original_model, track_pixels, num_pixels, track_stats, track_gradients)

    def _wrap_model(self, module: nn.Module, track_pixels: bool, num_pixels: int, 
                    track_stats: bool, track_gradients: bool, parent_name: str = ''):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            tracer = AdvancedTracer(full_name, track_pixels, num_pixels, track_stats, track_gradients)
            debugged_module = DebuggedModule(child, full_name, tracer)
            self.debugged_modules.append(debugged_module)
            self.graph.add_node(full_name)
            
            if parent_name:
                self.graph.add_edge(parent_name, full_name)
            
            if list(child.children()):  # If the child has children, recurse
                self._wrap_model(child, track_pixels, num_pixels, track_stats, track_gradients, full_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_outputs = {}
        for module in self.debugged_modules:
            if module.name in self.graph.predecessors(module.name):
                predecessors = list(self.graph.predecessors(module.name))
                if len(predecessors) == 1:
                    x = module_outputs[predecessors[0]]
                else:
                    x = tuple(module_outputs[pred] for pred in predecessors)
            
            output = module(x)
            module_outputs[module.name] = output
            
            if isinstance(output, tuple):
                x = output[0]  # Assume the first output is the main one
            else:
                x = output
        
        return x

    def reset_traces(self):
        for module in self.debugged_modules:
            module.tracer.reset()

    def get_traces(self) -> Dict[str, Dict[str, List[Any]]]:
        traces = {}
        for module in self.debugged_modules:
            traces[module.name] = {
                'pixels': module.tracer.pixel_trace,
                'stats': module.tracer.stats_trace,
                'gradients': module.tracer.gradient_trace
            }
        return traces

class NetworkDebugger:
    def __init__(self, model: nn.Module, track_pixels: bool = True, 
                 num_pixels: int = 5, track_stats: bool = True, 
                 track_gradients: bool = True):
        self.debugged_model = AdvancedDebuggedNetwork(model, track_pixels, num_pixels, 
                                                      track_stats, track_gradients)
        self.original_model = model

    def run_and_debug(self, input_tensor: torch.Tensor, target: torch.Tensor = None, 
                      loss_fn: nn.Module = None, optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
        self.debugged_model.reset_traces()
        
        # Forward pass
        output = self.debugged_model(input_tensor)
        
        # Backward pass (if applicable)
        if target is not None and loss_fn is not None:
            loss = loss_fn(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss.backward()

        return self.debugged_model.get_traces()

    def visualize_traces(self, traces: Dict[str, Dict[str, List[Any]]], 
                         plot_type: str = 'all', save_path: str = None):
        if plot_type in ['all', 'pixels'] and traces[list(traces.keys())[0]]['pixels']:
            self._plot_pixel_traces(traces, save_path)
        
        if plot_type in ['all', 'stats'] and traces[list(traces.keys())[0]]['stats']:
            self._plot_stats_traces(traces, save_path)
        
        if plot_type in ['all', 'gradients'] and traces[list(traces.keys())[0]]['gradients']:
            self._plot_gradient_traces(traces, save_path)

        if plot_type in ['all', 'architecture']:
            self._plot_architecture(save_path)

    def _plot_pixel_traces(self, traces: Dict[str, Dict[str, List[Any]]], save_path: str = None):
        # Implementation remains the same as before
        pass

    def _plot_stats_traces(self, traces: Dict[str, Dict[str, List[Any]]], save_path: str = None):
        # Implementation remains the same as before
        pass

    def _plot_gradient_traces(self, traces: Dict[str, Dict[str, List[Any]]], save_path: str = None):
        # Implementation remains the same as before
        pass

    def _plot_architecture(self, save_path: str = None):
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(self.debugged_model.graph)
        nx.draw(self.debugged_model.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=8, arrows=True)
        nx.draw_networkx_labels(self.debugged_model.graph, pos, font_size=6)
        plt.title("Network Architecture")
        if save_path:
            plt.savefig(f"{save_path}_architecture.png")
        plt.show()

# Example usage with a ResNet-like model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
 
class NetworkDebugger:
    def __init__(self, model: nn.Module, track_pixels: bool = True, 
                 num_pixels: int = 5, track_stats: bool = True, 
                 track_gradients: bool = True):
        self.debugged_model = AdvancedDebuggedNetwork(model, track_pixels, num_pixels, 
                                                      track_stats, track_gradients)
        self.original_model = model
        self.traces = {}

    def run_and_debug(self, input_tensor: torch.Tensor, target: torch.Tensor = None, 
                      loss_fn: nn.Module = None, optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
        self.debugged_model.reset_traces()
        
        # Forward pass
        output = self.debugged_model(input_tensor)
        
        # Backward pass (if applicable)
        if target is not None and loss_fn is not None:
            loss = loss_fn(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss.backward()

        self.traces = self.debugged_model.get_traces()
        return self.traces
    
# Create and debug the ResNet model
model = SimpleResNet()
debugger = NetworkDebugger(model)

# Run a forward and backward pass
input_tensor = torch.randn(1, 3, 32, 32)
target = torch.randint(0, 10, (1,))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

traces = debugger.run_and_debug(input_tensor, target, loss_fn, optimizer)

# Visualize the results
debugger.visualize_traces(traces, plot_type='all', save_path='resnet_debug_output')