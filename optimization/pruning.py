import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
from collections import OrderedDict


class ChannelPruner:
    """Structured pruning for StyleGAN using channel-wise importance scoring"""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.importance_scores = {}
        self.pruning_masks = {}
        self.original_channels = {}
        
    def calculate_channel_importance(self, method: str = 'l1_norm') -> Dict[str, torch.Tensor]:
        """Calculate importance scores for each channel in convolutional layers"""
        importance_scores = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if method == 'l1_norm':
                    # L1 norm of filter weights
                    weights = module.weight.data
                    scores = torch.sum(torch.abs(weights), dim=(1, 2, 3))
                elif method == 'l2_norm':
                    # L2 norm of filter weights
                    weights = module.weight.data
                    scores = torch.sqrt(torch.sum(weights ** 2, dim=(1, 2, 3)))
                elif method == 'gradient':
                    # Gradient-based importance (requires gradients)
                    if module.weight.grad is not None:
                        scores = torch.sum(torch.abs(module.weight.grad), dim=(1, 2, 3))
                    else:
                        # Fallback to L1 norm if no gradients
                        weights = module.weight.data
                        scores = torch.sum(torch.abs(weights), dim=(1, 2, 3))
                else:
                    raise ValueError(f"Unknown importance method: {method}")
                
                importance_scores[name] = scores
                self.original_channels[name] = module.out_channels
        
        self.importance_scores = importance_scores
        return importance_scores
    
    def generate_pruning_masks(self) -> Dict[str, torch.Tensor]:
        """Generate binary masks for channel pruning"""
        pruning_masks = {}
        
        for name, scores in self.importance_scores.items():
            num_channels = len(scores)
            num_prune = int(num_channels * self.pruning_ratio)
            
            if num_prune >= num_channels:
                num_prune = num_channels - 1  # Keep at least one channel
            
            # Sort channels by importance (ascending order)
            _, sorted_indices = torch.sort(scores)
            
            # Create mask (1 to keep, 0 to prune)
            mask = torch.ones(num_channels, dtype=torch.bool, device=scores.device)
            mask[sorted_indices[:num_prune]] = False
            
            pruning_masks[name] = mask
        
        self.pruning_masks = pruning_masks
        return pruning_masks
    
    def apply_structured_pruning(self) -> nn.Module:
        """Apply structured pruning to create a smaller model"""
        pruned_model = copy.deepcopy(self.model)
        
        # First pass: prune channels and update layer dimensions
        layer_mapping = {}  # Track channel dimension changes
        
        for name, module in pruned_model.named_modules():
            if name in self.pruning_masks:
                mask = self.pruning_masks[name]
                new_out_channels = mask.sum().item()
                layer_mapping[name] = {
                    'mask': mask,
                    'new_out_channels': new_out_channels,
                    'original_out_channels': self.original_channels[name]
                }
        
        # Second pass: create new modules with reduced channels
        def prune_conv_layer(module, out_mask, in_mask=None):
            """Create a pruned version of a convolutional layer"""
            if isinstance(module, nn.Conv2d):
                new_out_channels = out_mask.sum().item()
                new_in_channels = module.in_channels if in_mask is None else in_mask.sum().item()
                
                new_conv = nn.Conv2d(
                    new_in_channels, new_out_channels,
                    module.kernel_size, module.stride, module.padding,
                    module.dilation, module.groups, module.bias is not None
                )
                
                # Copy pruned weights
                if in_mask is not None:
                    new_conv.weight.data = module.weight.data[out_mask][:, in_mask]
                else:
                    new_conv.weight.data = module.weight.data[out_mask]
                
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data[out_mask]
                
                return new_conv
            
            return module
        
        # Replace layers in the model
        self._replace_layers_recursive(pruned_model, layer_mapping, prune_conv_layer)
        
        return pruned_model
    
    def _replace_layers_recursive(self, model, layer_mapping, prune_fn):
        """Recursively replace layers in the model"""
        for name, module in model.named_children():
            full_name = name
            if hasattr(model, '_get_name'):
                parent_name = model._get_name()
                if parent_name:
                    full_name = f"{parent_name}.{name}"
            
            if full_name in layer_mapping:
                mask_info = layer_mapping[full_name]
                out_mask = mask_info['mask']
                
                # Find corresponding input mask from previous layer
                in_mask = self._find_input_mask(full_name, layer_mapping)
                
                new_module = prune_fn(module, out_mask, in_mask)
                setattr(model, name, new_module)
            else:
                # Recursively process child modules
                self._replace_layers_recursive(module, layer_mapping, prune_fn)
    
    def _find_input_mask(self, layer_name, layer_mapping):
        """Find the input mask for a layer based on previous layer's output mask"""
        # This is a simplified version - in practice, you'd need more sophisticated
        # layer dependency tracking
        layer_parts = layer_name.split('.')
        if len(layer_parts) > 1:
            # Look for previous layer in the same module
            parent = '.'.join(layer_parts[:-1])
            layer_idx = layer_parts[-1]
            
            # Try to find previous conv layer
            for name, info in layer_mapping.items():
                if name.startswith(parent) and name != layer_name:
                    return info['mask']
        
        return None
    
    def prune_model(self, importance_method: str = 'l1_norm', 
                   fine_tune_epochs: int = 0) -> nn.Module:
        """Complete pruning pipeline"""
        print(f"Starting structured pruning with ratio: {self.pruning_ratio}")
        
        # Calculate importance scores
        print("Calculating channel importance scores...")
        self.calculate_channel_importance(importance_method)
        
        # Generate pruning masks
        print("Generating pruning masks...")
        self.generate_pruning_masks()
        
        # Apply pruning
        print("Applying structured pruning...")
        pruned_model = self.apply_structured_pruning()
        
        # Print pruning statistics
        self._print_pruning_stats(pruned_model)
        
        if fine_tune_epochs > 0:
            print(f"Fine-tuning for {fine_tune_epochs} epochs...")
            # Fine-tuning would be implemented here
            pass
        
        return pruned_model
    
    def _print_pruning_stats(self, pruned_model):
        """Print pruning statistics"""
        original_params = sum(p.numel() for p in self.model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        reduction = (original_params - pruned_params) / original_params * 100
        
        print(f"\nPruning Statistics:")
        print(f"Original parameters: {original_params:,}")
        print(f"Pruned parameters: {pruned_params:,}")
        print(f"Parameter reduction: {reduction:.2f}%")
        
        # Layer-wise statistics
        print(f"\nLayer-wise pruning:")
        for name, info in self.original_channels.items():
            if name in self.pruning_masks:
                mask = self.pruning_masks[name]
                kept = mask.sum().item()
                total = len(mask)
                pruned = total - kept
                print(f"  {name}: {kept}/{total} channels kept ({pruned} pruned)")


class GlobalChannelPruner(ChannelPruner):
    """Global channel pruning across all layers"""
    
    def generate_pruning_masks(self) -> Dict[str, torch.Tensor]:
        """Generate masks using global importance ranking"""
        # Collect all importance scores
        all_scores = []
        layer_names = []
        
        for name, scores in self.importance_scores.items():
            all_scores.extend(scores.cpu().numpy())
            layer_names.extend([name] * len(scores))
        
        # Global ranking
        all_scores = np.array(all_scores)
        layer_names = np.array(layer_names)
        
        # Calculate global pruning threshold
        total_channels = len(all_scores)
        num_prune_global = int(total_channels * self.pruning_ratio)
        
        sorted_indices = np.argsort(all_scores)
        prune_indices = sorted_indices[:num_prune_global]
        
        # Create layer-wise masks
        pruning_masks = {}
        for name in self.importance_scores.keys():
            layer_mask = layer_names != name
            layer_prune_mask = np.isin(np.arange(len(all_scores)), prune_indices) & ~layer_mask
            
            # Convert to layer-specific indices
            layer_indices = np.where(layer_names == name)[0]
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(layer_indices)}
            
            # Create boolean mask for this layer
            num_channels = len(self.importance_scores[name])
            mask = torch.ones(num_channels, dtype=torch.bool)
            
            for global_idx in prune_indices:
                if global_idx in global_to_local:
                    local_idx = global_to_local[global_idx]
                    mask[local_idx] = False
            
            # Ensure at least one channel remains
            if mask.sum() == 0:
                mask[0] = True
            
            pruning_masks[name] = mask
        
        self.pruning_masks = pruning_masks
        return pruning_masks


class GradualPruner:
    """Gradual pruning during training"""
    
    def __init__(self, model: nn.Module, initial_ratio: float = 0.0, 
                 final_ratio: float = 0.5, pruning_frequency: int = 100):
        self.model = model
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.pruning_frequency = pruning_frequency
        self.current_step = 0
        self.current_ratio = initial_ratio
        self.channel_pruner = ChannelPruner(model, initial_ratio)
    
    def step(self):
        """Called during training to potentially apply gradual pruning"""
        self.current_step += 1
        
        if self.current_step % self.pruning_frequency == 0:
            # Update pruning ratio
            progress = min(1.0, self.current_step / (self.pruning_frequency * 10))  # 10 pruning steps total
            self.current_ratio = self.initial_ratio + progress * (self.final_ratio - self.initial_ratio)
            
            # Apply pruning
            self.channel_pruner.pruning_ratio = self.current_ratio
            self.channel_pruner.calculate_channel_importance('gradient')
            self.channel_pruner.generate_pruning_masks()
            
            # Apply soft pruning (mask weights instead of removing them)
            self._apply_soft_pruning()
    
    def _apply_soft_pruning(self):
        """Apply soft pruning by masking weights"""
        for name, module in self.model.named_modules():
            if name in self.channel_pruner.pruning_masks:
                mask = self.channel_pruner.pruning_masks[name]
                if hasattr(module, 'weight'):
                    # Zero out pruned channels
                    module.weight.data[~mask] = 0
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data[~mask] = 0


def test_pruning():
    """Test the pruning implementation"""
    from models.stylegan import StyleGAN
    
    # Create a small StyleGAN for testing
    model = StyleGAN(
        latent_dim=512,
        style_dim=512,
        channels=[256, 128, 64, 32],  # Small for testing
        resolutions=[4, 8, 16, 32]
    )
    
    print("Original model parameters:", sum(p.numel() for p in model.parameters()))
    
    # Test channel pruning
    pruner = ChannelPruner(model, pruning_ratio=0.3)
    pruned_model = pruner.prune_model(importance_method='l1_norm')
    
    # Test the pruned model
    z = torch.randn(2, 512)
    with torch.no_grad():
        original_output = model(z)
        pruned_output = pruned_model(z)
        print(f"Original output shape: {original_output.shape}")
        print(f"Pruned output shape: {pruned_output.shape}")
    
    # Test global pruning
    print("\n" + "="*50)
    print("Testing Global Channel Pruning")
    global_pruner = GlobalChannelPruner(model, pruning_ratio=0.3)
    globally_pruned_model = global_pruner.prune_model()
    
    print("Pruning test completed successfully!")


if __name__ == "__main__":
    test_pruning()