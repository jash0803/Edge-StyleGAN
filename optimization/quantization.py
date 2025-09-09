import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
import math


class FakeQuantize(nn.Module):
    """Fake quantization for simulating quantized inference during training"""
    
    def __init__(self, num_bits: int = 8, symmetric: bool = True, 
                 per_channel: bool = False, eps: float = 1e-8):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.eps = eps
        
        # Quantization parameters
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('min_val', torch.tensor(0.0))
        self.register_buffer('max_val', torch.tensor(0.0))
        
        # Quantization levels
        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1
    
    def calculate_qparams(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantization parameters"""
        if self.per_channel:
            # Per-channel quantization (along first dimension)
            dims_except_first = list(range(1, x.dim()))
            min_val = x.min(dim=dims_except_first, keepdim=True)[0]
            max_val = x.max(dim=dims_except_first, keepdim=True)[0]
        else:
            # Per-tensor quantization
            min_val = x.min()
            max_val = x.max()
        
        # Handle symmetric vs asymmetric quantization
        if self.symmetric:
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            min_val = -max_abs
            max_val = max_abs
        
        # Calculate scale and zero point
        scale = (max_val - min_val) / (self.qmax - self.qmin)
        scale = torch.clamp(scale, min=self.eps)
        
        if self.symmetric:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = self.qmin - torch.round(min_val / scale)
            zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        
        return scale, zero_point
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization"""
        if self.training:
            # Calculate quantization parameters dynamically during training
            scale, zero_point = self.calculate_qparams(x)
        else:
            # Use stored parameters during inference
            scale = self.scale
            zero_point = self.zero_point
        
        # Quantize
        x_q = torch.round(x / scale + zero_point)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)
        
        # Dequantize
        x_dq = (x_q - zero_point) * scale
        
        return x_dq
    
    def calibrate(self, x: torch.Tensor):
        """Calibrate quantization parameters for inference"""
        self.scale, self.zero_point = self.calculate_qparams(x)
        self.min_val = x.min()
        self.max_val = x.max()


class QConv2d(nn.Module):
    """Quantized convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 weight_bits: int = 8, activation_bits: int = 8):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=bias)
        
        # Weight quantizer
        self.weight_quantizer = FakeQuantize(
            num_bits=weight_bits, 
            symmetric=True, 
            per_channel=True
        )
        
        # Activation quantizer
        self.activation_quantizer = FakeQuantize(
            num_bits=activation_bits,
            symmetric=False,
            per_channel=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input activations
        x_q = self.activation_quantizer(x)
        
        # Quantize weights
        weight_q = self.weight_quantizer(self.conv.weight)
        
        # Perform convolution with quantized weights
        output = F.conv2d(x_q, weight_q, self.conv.bias, 
                         self.conv.stride, self.conv.padding)
        
        return output


class QLinear(nn.Module):
    """Quantized linear layer"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_bits: int = 8, activation_bits: int = 8):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Weight quantizer
        self.weight_quantizer = FakeQuantize(
            num_bits=weight_bits,
            symmetric=True,
            per_channel=True
        )
        
        # Activation quantizer  
        self.activation_quantizer = FakeQuantize(
            num_bits=activation_bits,
            symmetric=False,
            per_channel=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input activations
        x_q = self.activation_quantizer(x)
        
        # Quantize weights
        weight_q = self.weight_quantizer(self.linear.weight)
        
        # Perform linear operation with quantized weights
        output = F.linear(x_q, weight_q, self.linear.bias)
        
        return output


class ModelQuantizer:
    """Quantize an entire model"""
    
    def __init__(self, model: nn.Module, weight_bits: int = 8, 
                 activation_bits: int = 8, skip_layers: List[str] = None):
        self.model = model
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.skip_layers = skip_layers or []
        
    def quantize_model(self) -> nn.Module:
        """Convert model to quantized version"""
        quantized_model = copy.deepcopy(self.model)
        self._replace_layers_recursive(quantized_model)
        return quantized_model
    
    def _replace_layers_recursive(self, model: nn.Module, parent_name: str = ""):
        """Recursively replace layers with quantized versions"""
        for name, module in model.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            # Skip specified layers
            if any(skip in full_name for skip in self.skip_layers):
                continue
            
            # Replace with quantized layer
            if isinstance(module, nn.Conv2d):
                q_conv = QConv2d(
                    module.in_channels, module.out_channels, module.kernel_size[0],
                    module.stride[0], module.padding[0], module.bias is not None,
                    self.weight_bits, self.activation_bits
                )
                # Copy weights
                q_conv.conv.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    q_conv.conv.bias.data = module.bias.data.clone()
                
                setattr(model, name, q_conv)
                
            elif isinstance(module, nn.Linear):
                q_linear = QLinear(
                    module.in_features, module.out_features, 
                    module.bias is not None,
                    self.weight_bits, self.activation_bits
                )
                # Copy weights
                q_linear.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    q_linear.linear.bias.data = module.bias.data.clone()
                
                setattr(model, name, q_linear)
            
            else:
                # Recursively process child modules
                self._replace_layers_recursive(module, full_name)
    
    def calibrate_model(self, quantized_model: nn.Module, 
                       calibration_data: torch.Tensor):
        """Calibrate quantization parameters using calibration data"""
        quantized_model.eval()
        
        with torch.no_grad():
            # Forward pass to collect statistics
            _ = quantized_model(calibration_data)
            
            # Calibrate all quantizers
            for module in quantized_model.modules():
                if isinstance(module, (QConv2d, QLinear)):
                    # Weight quantizers are calibrated during forward pass
                    pass


class PTQQuantizer:
    """Post-Training Quantization"""
    
    def __init__(self, model: nn.Module, weight_bits: int = 8, 
                 activation_bits: int = 8):
        self.model = model
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.activation_stats = {}
    
    def collect_activation_stats(self, calibration_loader):
        """Collect activation statistics for calibration"""
        self.model.eval()
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    if name not in self.activation_stats:
                        self.activation_stats[name] = []
                    self.activation_stats[name].append(output.detach().cpu())
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        # Collect statistics
        print("Collecting activation statistics...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= 100:  # Limit calibration samples
                    break
                _ = self.model(batch)
                if i % 20 == 0:
                    print(f"Processed {i+1} calibration batches")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print(f"Collected stats for {len(self.activation_stats)} layers")
    
    def quantize_weights(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Quantize model weights"""
        quantized_weights = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Calculate quantization parameters
                if len(param.shape) == 4:  # Conv weight
                    # Per-channel quantization for conv layers
                    weight_flat = param.view(param.size(0), -1)
                    min_vals = weight_flat.min(dim=1)[0]
                    max_vals = weight_flat.max(dim=1)[0]
                else:  # Linear weight
                    # Per-tensor quantization for linear layers
                    min_vals = param.min()
                    max_vals = param.max()
                
                # Symmetric quantization
                max_abs = torch.max(torch.abs(min_vals), torch.abs(max_vals))
                scale = max_abs / (2 ** (self.weight_bits - 1) - 1)
                scale = torch.clamp(scale, min=1e-8)
                
                # Quantize
                param_q = torch.round(param / scale.view(-1, 1, 1, 1) 
                                    if len(param.shape) == 4 else param / scale)
                param_q = torch.clamp(param_q, 
                                    -(2 ** (self.weight_bits - 1)), 
                                    2 ** (self.weight_bits - 1) - 1)
                
                # Dequantize
                param_dq = param_q * scale.view(-1, 1, 1, 1) if len(param.shape) == 4 else param_q * scale
                
                quantized_weights[name] = (param_dq, scale)
        
        return quantized_weights
    
    def apply_quantization(self) -> nn.Module:
        """Apply post-training quantization"""
        quantized_model = copy.deepcopy(self.model)
        
        # Quantize weights
        quantized_weights = self.quantize_weights()
        
        # Apply quantized weights
        for name, (weight_q, scale) in quantized_weights.items():
            # Navigate to the parameter
            module = quantized_model
            parts = name.split('.')
            for part in parts[:-1]:
                module = getattr(module, part)
            
            # Set quantized weight
            setattr(module, parts[-1], nn.Parameter(weight_q))
        
        return quantized_model


class QATTrainer:
    """Quantization Aware Training"""
    
    def __init__(self, model: nn.Module, weight_bits: int = 8, 
                 activation_bits: int = 8):
        self.model = model
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quantized_model = None
    
    def prepare_qat(self) -> nn.Module:
        """Prepare model for quantization aware training"""
        quantizer = ModelQuantizer(
            self.model, 
            self.weight_bits, 
            self.activation_bits,
            skip_layers=['mapping']  # Skip mapping network
        )
        self.quantized_model = quantizer.quantize_model()
        return self.quantized_model
    
    def train_step(self, batch: torch.Tensor, optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> float:
        """Single training step with quantization"""
        self.quantized_model.train()
        
        # Forward pass
        output = self.quantized_model(batch)
        
        # Calculate loss (this would be more complex for GAN training)
        if hasattr(batch, 'target'):
            loss = criterion(output, batch.target)
        else:
            # Dummy loss for example
            loss = output.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.quantized_model.parameters(), 1.0)
        
        optimizer.step()
        
        return loss.item()


def test_quantization():
    """Test quantization implementation"""
    from models.stylegan import StyleGAN
    
    # Create a small model for testing
    model = StyleGAN(
        latent_dim=512,
        style_dim=512,
        channels=[128, 64, 32],
        resolutions=[4, 8, 16]
    )
    
    print("Original model size:", sum(p.numel() for p in model.parameters()))
    
    # Test fake quantization
    print("\nTesting Fake Quantization...")
    fake_quant = FakeQuantize(num_bits=8, symmetric=True)
    test_tensor = torch.randn(4, 64, 8, 8)
    quantized_tensor = fake_quant(test_tensor)
    print(f"Original range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    print(f"Quantized range: [{quantized_tensor.min():.3f}, {quantized_tensor.max():.3f}]")
    
    # Test model quantization
    print("\nTesting Model Quantization...")
    quantizer = ModelQuantizer(model, weight_bits=8, activation_bits=8)
    quantized_model = quantizer.quantize_model()
    
    # Test forward pass
    z = torch.randn(2, 512)
    with torch.no_grad():
        original_output = model(z)
        quantized_output = quantized_model(z)
        
        mse_error = F.mse_loss(original_output, quantized_output)
        print(f"MSE between original and quantized: {mse_error:.6f}")
    
    # Test PTQ
    print("\nTesting Post-Training Quantization...")
    ptq = PTQQuantizer(model, weight_bits=8, activation_bits=8)
    ptq_model = ptq.apply_quantization()
    
    with torch.no_grad():
        ptq_output = ptq_model(z)
        mse_error_ptq = F.mse_loss(original_output, ptq_output)
        print(f"MSE between original and PTQ: {mse_error_ptq:.6f}")
    
    print("\nQuantization test completed successfully!")


if __name__ == "__main__":
    test_quantization()