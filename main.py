import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union, Tuple
import seaborn as sns
import networkx as nx
import pandas as pd
from sklearn.manifold import TSNE
from collections import defaultdict
import wandb
from torch.utils.hooks import RemovableHandle
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import torchvision.models as models
from network_debugger import AdvancedDebuggedNetwork


class NetworkDebugger:
    def __init__(self, model: nn.Module, track_pixels: bool = True, 
                 num_pixels: int = 5, track_stats: bool = True, 
                 track_gradients: bool = True, probe_layers: Dict[str, int] = None):
        self.debugged_model = AdvancedDebuggedNetwork(model, track_pixels, num_pixels, 
                                                      track_stats, track_gradients, probe_layers)
        self.original_model = model
        self.optimizer = None
        self.loss_fn = None
        self.current_input = None
        self.current_target = None
        self.current_output = None
        self.current_loss = None
        self.iteration = 0
        self.history = defaultdict(list)
        self.hooks = {}
        
        wandb.init(project="network_debugger", config={
            "model": model.__class__.__name__,
            "track_pixels": track_pixels,
            "num_pixels": num_pixels,
            "track_stats": track_stats,
            "track_gradients": track_gradients
        })
        
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.debugged_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                self.hooks[f"{name}_forward"] = module.register_forward_hook(self._forward_hook(name))
                self.hooks[f"{name}_backward"] = module.register_backward_hook(self._backward_hook(name))

    def _forward_hook(self, name):
        def hook(module, input, output):
            self.history[f"{name}_activation"].append(output.detach().cpu().numpy())
        return hook

    def _backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.history[f"{name}_gradient"].append(grad_output[0].detach().cpu().numpy())
        return hook

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        wandb.config.update({"optimizer": optimizer.__class__.__name__})

    def set_loss_function(self, loss_fn: nn.Module):
        self.loss_fn = loss_fn
        wandb.config.update({"loss_function": loss_fn.__class__.__name__})

    def set_input(self, input_tensor: torch.Tensor, target: torch.Tensor):
        self.current_input = input_tensor
        self.current_target = target
        self.iteration = 0

    def forward_step(self):
        if self.current_input is None:
            raise ValueError("Input not set. Call set_input() first.")
        
        self.debugged_model.reset_traces()
        self.current_output = self.debugged_model(self.current_input)
        self.iteration += 1
        
        return self.debugged_model.get_traces()

    def backward_step(self):
        if self.current_output is None:
            raise ValueError("Forward step not performed. Call forward_step() first.")
        if self.loss_fn is None:
            raise ValueError("Loss function not set. Call set_loss_function() first.")
        
        self.current_loss = self.loss_fn(self.current_output, self.current_target)
        self.current_loss.backward()
        
        return self.debugged_model.get_traces()

    def optimization_step(self):
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Call set_optimizer() first.")
        
        self.optimizer.step()
        self.optimizer.zero_grad()

    def run_single_iteration(self):
        forward_traces = self.forward_step()
        backward_traces = self.backward_step()
        self.optimization_step()
        self.update_history()
        
        return forward_traces, backward_traces

    def update_history(self):
        self.history['loss'].append(self.current_loss.item())
        accuracy = (self.current_output.argmax(dim=1) == self.current_target).float().mean().item()
        self.history['accuracy'].append(accuracy)
        
        wandb_log = {
            'loss': self.current_loss.item(),
            'accuracy': accuracy
        }
        
        for name, module in self.debugged_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                weight_mean = module.weight.data.mean().item()
                weight_std = module.weight.data.std().item()
                self.history[f'{name}_weight_mean'].append(weight_mean)
                self.history[f'{name}_weight_std'].append(weight_std)
                wandb_log[f'{name}/weight_mean'] = weight_mean
                wandb_log[f'{name}/weight_std'] = weight_std
                
                if module.weight.grad is not None:
                    grad_norm = module.weight.grad.norm().item()
                    self.history[f'{name}_grad_norm'].append(grad_norm)
                    wandb_log[f'{name}/grad_norm'] = grad_norm
        
        wandb.log(wandb_log, step=self.iteration)

    def visualize_current_state(self):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Loss and Accuracy", "Layer Weight Statistics", "Gradient Norms", "Current Output"))
        
        # Loss and Accuracy
        fig.add_trace(go.Scatter(y=self.history['loss'], name="Loss"), row=1, col=1)
        fig.add_trace(go.Scatter(y=self.history['accuracy'], name="Accuracy"), row=1, col=1)
        
        # Layer Weight Statistics
        for name in self.history.keys():
            if name.endswith('_weight_mean'):
                fig.add_trace(go.Scatter(y=self.history[name], name=f"{name} (mean)"), row=1, col=2)
                fig.add_trace(go.Scatter(y=self.history[name.replace('_mean', '_std')], name=f"{name.replace('_mean', '_std')} (std)"), row=1, col=2)
        
        # Gradient Norms
        for name in self.history.keys():
            if name.endswith('_grad_norm'):
                fig.add_trace(go.Scatter(y=self.history[name], name=name), row=2, col=1)
        
        # Current Output
        fig.add_trace(go.Heatmap(z=self.current_output.detach().cpu().numpy()), row=2, col=2)
        
        fig.update_layout(height=800, width=1000, title_text=f"Network State (Iteration {self.iteration})")
        wandb.log({"network_state": fig}, step=self.iteration)

    def visualize_feature_maps(self, layer_name: str):
        activations = self.history.get(f"{layer_name}_activation")
        if activations is None or len(activations) == 0:
            print(f"No activations found for layer {layer_name}")
            return
        
        latest_activation = activations[-1]
        fig = make_subplots(rows=4, cols=4, subplot_titles=[f"Channel {i+1}" for i in range(16)])
        
        for i in range(min(16, latest_activation.shape[1])):
            fig.add_trace(go.Heatmap(z=latest_activation[0, i]), row=i//4+1, col=i%4+1)
        
        fig.update_layout(height=800, width=800, title_text=f"Feature Maps for {layer_name}")
        wandb.log({f"feature_maps_{layer_name}": fig}, step=self.iteration)

    def visualize_gradients(self, layer_name: str):
        gradients = self.history.get(f"{layer_name}_gradient")
        if gradients is None or len(gradients) == 0:
            print(f"No gradients found for layer {layer_name}")
            return
        
        latest_gradient = gradients[-1]
        fig = go.Figure(data=go.Heatmap(z=latest_gradient.reshape(-1, latest_gradient.shape[-1])))
        fig.update_layout(height=600, width=800, title_text=f"Gradient Heatmap for {layer_name}")
        wandb.log({f"gradient_heatmap_{layer_name}": fig}, step=self.iteration)

    def visualize_weight_distribution(self, layer_name: str):
        module = dict(self.debugged_model.named_modules())[layer_name]
        if not hasattr(module, 'weight'):
            print(f"No weights found for layer {layer_name}")
            return
        
        weights = module.weight.data.cpu().numpy().flatten()
        fig = go.Figure(data=[go.Histogram(x=weights, nbinsx=50)])
        fig.update_layout(height=400, width=600, title_text=f"Weight Distribution for {layer_name}")
        wandb.log({f"weight_distribution_{layer_name}": fig}, step=self.iteration)

    def export_traces(self, filename: str):
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)
        print(f"Traces exported to {filename}")
        wandb.save(filename)

    def analyze_sensitivity(self, layer_name: str, num_samples: int = 100):
        module = dict(self.debugged_model.named_modules())[layer_name]
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"Sensitivity analysis is only supported for Conv2d and Linear layers. {layer_name} is {type(module)}")
            return
        
        original_weights = module.weight.data.clone()
        sensitivities = []
        
        for _ in range(num_samples):
            perturbed_weights = original_weights + torch.randn_like(original_weights) * 0.01
            module.weight.data = perturbed_weights
            
            with torch.no_grad():
                output = self.debugged_model(self.current_input)
            
            loss = self.loss_fn(output, self.current_target)
            sensitivities.append(loss.item())
        
        module.weight.data = original_weights
        
        fig = go.Figure(data=[go.Box(y=sensitivities, name="Loss Sensitivity")])
        fig.update_layout(height=400, width=600, title_text=f"Loss Sensitivity Analysis for {layer_name}")
        wandb.log({f"sensitivity_analysis_{layer_name}": fig}, step=self.iteration)

def interactive_debugging_loop(debugger: NetworkDebugger):
    commands = {
        'f': ('Forward step', debugger.forward_step),
        'b': ('Backward step', debugger.backward_step),
        'o': ('Optimization step', debugger.optimization_step),
        'i': ('Run single iteration', debugger.run_single_iteration),
        'v': ('Visualize current state', debugger.visualize_current_state),
        'fm': ('Visualize feature maps', lambda: debugger.visualize_feature_maps(input("Enter layer name: "))),
        'g': ('Visualize gradients', lambda: debugger.visualize_gradients(input("Enter layer name: "))),
        'w': ('Visualize weight distribution', lambda: debugger.visualize_weight_distribution(input("Enter layer name: "))),
        's': ('Analyze layer sensitivity', lambda: debugger.analyze_sensitivity(input("Enter layer name: "))),
        'ex': ('Export traces', lambda: debugger.export_traces(input("Enter filename: "))),
        'q': ('Quit', None)
    }

    while True:
        print("\nAvailable commands:")
        for cmd, (description, _) in commands.items():
            print(f"{cmd}: {description}")
        
        action = input("Enter command: ").lower()
        
        if action in commands:
            if action == 'q':
                break
            commands[action][1]()
            print(f"{commands[action][0]} completed.")
        else:
            print("Invalid command. Please try again.")

    wandb.finish()


# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# If you want to finetune the model, you can modify the last layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 is the number of classes in your task

debugger = NetworkDebugger(model)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
input_tensor = torch.randn(1, 3, 224, 224)  # ResNet expects 224x224 images
target = torch.randint(0, 10, (1,))

debugger.set_optimizer(optimizer)
debugger.set_loss_function(loss_fn)
debugger.set_input(input_tensor, target)

interactive_debugging_loop(debugger)

