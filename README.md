
# 🎛️ BassTrace: Neural Network Debugger 

## 🎵 Dropping the Beat on Neural Network Debugging 

BassTrace is a powerful, interactive debugging tool for neural networks, designed to help you visualize, analyze, and optimize your models with the precision of a drum machine and the flexibility of a synthesizer.

### 🚀 Features

- 🔬 Real-time visualization of network states
- 🧠 Layer-wise analysis of activations and gradients
- 📊 Interactive plots for weights, biases, and gradients
- 🔍 Sensitivity analysis for individual layers
- 📈 Comprehensive logging with Weights & Biases integration
- 🎛️ Interactive debugging loop for step-by-step analysis

### 🎹 How It Works

1. 🎼 Initialize your model and wrap it with BassTrace
2. 🎚️ Set up your optimizer and loss function
3. 🎯 Provide input data
4. 🎛️ Start the interactive debugging loop
5. 🔊 Analyze your model's behavior in real-time

### 🎧 Commands

- `f`: Forward step
- `b`: Backward step
- `o`: Optimization step
- `i`: Run single 
iteration
- `v`: Visualize current state
- `fm`: Visualize feature maps
- `g`: Visualize gradients
- `w`: Visualize weight distribution
- `s`: Analyze layer sensitivity
- `ex`: Export traces

### 🎶 Example Usage




```shell
pip install wandb plotly
```





```python
import torch
import torch.nn as nn
import torchvision.models as models

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

```

### 🎸 Why "BassTrace"?
Just as the bass line provides the foundation for a track, BassTrace lays down the groundwork for understanding your neural networks. It helps you trace the flow of data and gradients, letting you fine-tune your model like a producer tweaking a mix.



### 🎼 Contributing
We welcome contributions! Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated. Please check out our contributing guidelines for more information.

### 🎵 License
BassTrace is released under the MIT License. See the LICENSE file for more details.
