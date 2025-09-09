import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import copy
from collections import OrderedDict


class FeatureDistillationLoss(nn.Module):
    """Feature-level distillation loss for intermediate representations"""
    
    def __init__(self, student_channels: int, teacher_channels: int, 
                 loss_type: str = 'mse', temperature: float = 4.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        
        # Adaptation layer to match dimensions
        if student_channels != teacher_channels:
            self.adaptation = nn.Conv2d(student_channels, teacher_channels, 1)
        else:
            self.adaptation = nn.Identity()
    
    def forward(self, student_features: torch.Tensor, 
                teacher_features: torch.Tensor) -> torch.Tensor:
        # Adapt student features to match teacher dimensions
        student_adapted = self.adaptation(student_features)
        
        # Ensure spatial dimensions match
        if student_adapted.shape != teacher_features.shape:
            student_adapted = F.interpolate(
                student_adapted, 
                size=teacher_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        if self.loss_type == 'mse':
            return F.mse_loss(student_adapted, teacher_features)
        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            student_norm = F.normalize(student_adapted.flatten(1), p=2, dim=1)
            teacher_norm = F.normalize(teacher_features.flatten(1), p=2, dim=1)
            cosine_sim = (student_norm * teacher_norm).sum(dim=1).mean()
            return 1 - cosine_sim
        elif self.loss_type == 'kl':
            # KL divergence with temperature
            student_soft = F.log_softmax(student_adapted.flatten(1) / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_features.flatten(1) / self.temperature, dim=1)
            return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class AttentionDistillationLoss(nn.Module):
    """Attention-based distillation loss"""
    
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_features: torch.Tensor, 
                teacher_features: torch.Tensor) -> torch.Tensor:
        # Calculate attention maps
        student_attention = self.calculate_attention(student_features)
        teacher_attention = self.calculate_attention(teacher_features)
        
        # Apply temperature and calculate KL divergence
        student_soft = F.log_softmax(student_attention / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_attention / self.temperature, dim=-1)
        
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
    
    def calculate_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Calculate spatial attention maps"""
        batch, channel, height, width = features.shape
        
        # Sum across channels to get spatial attention
        attention = features.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        attention = attention.view(batch, -1)  # [B, H*W]
        
        return attention


class StyleDistillationLoss(nn.Module):
    """Style-based distillation for StyleGAN-specific knowledge transfer"""
    
    def __init__(self, style_dim: int = 512):
        super().__init__()
        self.style_dim = style_dim
    
    def forward(self, student_styles: torch.Tensor, 
                teacher_styles: torch.Tensor) -> torch.Tensor:
        # Direct style matching
        style_loss = F.mse_loss(student_styles, teacher_styles)
        
        # Style correlation matching
        student_corr = self.compute_style_correlation(student_styles)
        teacher_corr = self.compute_style_correlation(teacher_styles)
        correlation_loss = F.mse_loss(student_corr, teacher_corr)
        
        return style_loss + 0.1 * correlation_loss
    
    def compute_style_correlation(self, styles: torch.Tensor) -> torch.Tensor:
        """Compute style correlation matrix"""
        batch_size = styles.shape[0]
        styles_normalized = F.normalize(styles, p=2, dim=1)
        correlation = torch.mm(styles_normalized, styles_normalized.t())
        return correlation


class GANKnowledgeDistiller:
    """Knowledge distillation framework for StyleGAN"""
    
    def __init__(self, teacher_generator: nn.Module, student_generator: nn.Module,
                 teacher_discriminator: nn.Module = None, student_discriminator: nn.Module = None,
                 distillation_layers: List[str] = None, feature_loss_weight: float = 1.0,
                 style_loss_weight: float = 0.5, attention_loss_weight: float = 0.1):
        
        self.teacher_generator = teacher_generator
        self.student_generator = student_generator
        self.teacher_discriminator = teacher_discriminator
        self.student_discriminator = student_discriminator
        
        # Loss weights
        self.feature_loss_weight = feature_loss_weight
        self.style_loss_weight = style_loss_weight
        self.attention_loss_weight = attention_loss_weight
        
        # Distillation layers
        self.distillation_layers = distillation_layers or self._auto_select_layers()
        
        # Initialize loss functions
        self.feature_losses = nn.ModuleDict()
        self.attention_loss = AttentionDistillationLoss()
        self.style_loss = StyleDistillationLoss()
        
        self._setup_feature_losses()
        
        # Feature hooks
        self.teacher_features = {}
        self.student_features = {}
        self._register_hooks()
    
    def _auto_select_layers(self) -> List[str]:
        """Automatically select layers for distillation"""
        layers = []
        for name, module in self.student_generator.named_modules():
            if isinstance(module, nn.Conv2d) and 'synthesis' in name:
                layers.append(name)
        return layers
    
    def _setup_feature_losses(self):
        """Setup feature distillation losses for each layer"""
        for layer_name in self.distillation_layers:
            # Get channel dimensions
            teacher_module = dict(self.teacher_generator.named_modules())[layer_name]
            student_module = dict(self.student_generator.named_modules())[layer_name]
            
            teacher_channels = teacher_module.out_channels
            student_channels = student_module.out_channels
            
            # Create feature distillation loss
            feature_loss = FeatureDistillationLoss(
                student_channels, teacher_channels, loss_type='mse'
            )
            self.feature_losses[layer_name] = feature_loss
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features"""
        def make_hook(features_dict, layer_name):
            def hook(module, input, output):
                features_dict[layer_name] = output
            return hook
        
        # Register hooks for teacher
        for layer_name in self.distillation_layers:
            teacher_module = dict(self.teacher_generator.named_modules())[layer_name]
            teacher_module.register_forward_hook(
                make_hook(self.teacher_features, layer_name)
            )
        
        # Register hooks for student
        for layer_name in self.distillation_layers:
            student_module = dict(self.student_generator.named_modules())[layer_name]
            student_module.register_forward_hook(
                make_hook(self.student_features, layer_name)
            )
    
    def compute_distillation_loss(self, z: torch.Tensor, 
                                styles: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute knowledge distillation loss"""
        losses = {}
        
        # Generate images and capture features
        with torch.no_grad():
            self.teacher_generator.eval()
            teacher_output = self.teacher_generator(z, styles=styles)
            teacher_styles = self.teacher_generator.mapping(z) if styles is None else styles
        
        student_output = self.student_generator(z, styles=styles)
        student_styles = self.student_generator.mapping(z) if styles is None else styles
        
        # Feature-level distillation
        feature_loss = 0.0
        for layer_name in self.distillation_layers:
            if layer_name in self.teacher_features and layer_name in self.student_features:
                layer_loss = self.feature_losses[layer_name](
                    self.student_features[layer_name],
                    self.teacher_features[layer_name]
                )
                feature_loss += layer_loss
        
        losses['feature_loss'] = feature_loss / len(self.distillation_layers)
        
        # Style distillation
        losses['style_loss'] = self.style_loss(student_styles, teacher_styles)
        
        # Output-level distillation
        losses['output_loss'] = F.mse_loss(student_output, teacher_output)
        
        # Attention distillation (using final layer features)
        if self.distillation_layers:
            final_layer = self.distillation_layers[-1]
            if final_layer in self.teacher_features and final_layer in self.student_features:
                losses['attention_loss'] = self.attention_loss(
                    self.student_features[final_layer],
                    self.teacher_features[final_layer]
                )
            else:
                losses['attention_loss'] = torch.tensor(0.0, device=z.device)
        
        # Total distillation loss
        total_loss = (
            self.feature_loss_weight * losses['feature_loss'] +
            self.style_loss_weight * losses['style_loss'] +
            losses['output_loss'] +
            self.attention_loss_weight * losses['attention_loss']
        )
        
        losses['total_distillation_loss'] = total_loss
        
        return losses
    
    def train_step(self, z: torch.Tensor, optimizer: torch.optim.Optimizer,
                   adversarial_loss_fn: Callable = None, 
                   adversarial_weight: float = 1.0) -> Dict[str, float]:
        """Single training step with knowledge distillation"""
        
        # Compute distillation losses
        distillation_losses = self.compute_distillation_loss(z)
        
        total_loss = distillation_losses['total_distillation_loss']
        
        # Add adversarial loss if discriminator is available
        if adversarial_loss_fn is not None and self.student_discriminator is not None:
            student_output = self.student_generator(z)
            fake_logits = self.student_discriminator(student_output)
            adversarial_loss = adversarial_loss_fn(fake_logits)
            total_loss += adversarial_weight * adversarial_loss
            distillation_losses['adversarial_loss'] = adversarial_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student_generator.parameters(), 1.0)
        
        optimizer.step()
        
        # Clear cached features
        self.teacher_features.clear()
        self.student_features.clear()
        
        # Return scalar losses
        return {k: v.item() if torch.is_tensor(v) else v 
                for k, v in distillation_losses.items()}


class ProgressiveKnowledgeDistiller:
    """Progressive knowledge distillation for StyleGAN training"""
    
    def __init__(self, teacher_generator: nn.Module, student_generator: nn.Module,
                 num_progressive_stages: int = 4, base_resolution: int = 4):
        self.teacher_generator = teacher_generator
        self.student_generator = student_generator
        self.num_progressive_stages = num_progressive_stages
        self.base_resolution = base_resolution
        
        self.current_stage = 0
        self.distillers = {}
        
        self._setup_progressive_distillers()
    
    def _setup_progressive_distillers(self):
        """Setup distillers for each progressive stage"""
        for stage in range(self.num_progressive_stages):
            resolution = self.base_resolution * (2 ** stage)
            
            # Select layers appropriate for this resolution
            distillation_layers = self._get_layers_for_resolution(resolution)
            
            distiller = GANKnowledgeDistiller(
                self.teacher_generator,
                self.student_generator,
                distillation_layers=distillation_layers
            )
            
            self.distillers[stage] = distiller
    
    def _get_layers_for_resolution(self, target_resolution: int) -> List[str]:
        """Get distillation layers for specific resolution"""
        layers = []
        current_res = 4
        
        for name, module in self.student_generator.named_modules():
            if isinstance(module, nn.Conv2d) and 'synthesis' in name:
                if current_res <= target_resolution:
                    layers.append(name)
                # Update resolution tracking logic here
                if 'upsample' in name:
                    current_res *= 2
        
        return layers
    
    def set_stage(self, stage: int):
        """Set current progressive training stage"""
        self.current_stage = min(stage, self.num_progressive_stages - 1)
    
    def train_step(self, z: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Progressive training step"""
        return self.distillers[self.current_stage].train_step(z, optimizer)


class OnlineKnowledgeDistiller:
    """Online knowledge distillation between multiple student models"""
    
    def __init__(self, student_models: List[nn.Module], 
                 ensemble_weight: float = 0.5, diversity_weight: float = 0.1):
        self.student_models = student_models
        self.ensemble_weight = ensemble_weight
        self.diversity_weight = diversity_weight
        self.num_students = len(student_models)
        
        # Feature extraction hooks
        self.student_features = [{} for _ in range(self.num_students)]
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for all student models"""
        def make_hook(features_dict, layer_name):
            def hook(module, input, output):
                features_dict[layer_name] = output
            return hook
        
        for i, model in enumerate(self.student_models):
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and 'synthesis' in name:
                    module.register_forward_hook(
                        make_hook(self.student_features[i], name)
                    )
    
    def compute_ensemble_loss(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute online distillation loss using ensemble of students"""
        losses = {}
        
        # Generate outputs from all students
        outputs = []
        for model in self.student_models:
            output = model(z)
            outputs.append(output)
        
        # Compute ensemble target (average of all student outputs)
        ensemble_target = torch.stack(outputs).mean(dim=0)
        
        # Individual student losses against ensemble
        ensemble_loss = 0.0
        for i, output in enumerate(outputs):
            loss = F.mse_loss(output, ensemble_target.detach())
            ensemble_loss += loss
            losses[f'student_{i}_ensemble_loss'] = loss
        
        losses['ensemble_loss'] = ensemble_loss / self.num_students
        
        # Diversity loss to encourage different solutions
        diversity_loss = 0.0
        for i in range(self.num_students):
            for j in range(i + 1, self.num_students):
                # Negative correlation to encourage diversity
                correlation = F.cosine_similarity(
                    outputs[i].flatten(1), outputs[j].flatten(1)
                ).mean()
                diversity_loss -= correlation
        
        num_pairs = self.num_students * (self.num_students - 1) / 2
        losses['diversity_loss'] = diversity_loss / num_pairs
        
        # Total loss
        total_loss = (
            self.ensemble_weight * losses['ensemble_loss'] +
            self.diversity_weight * losses['diversity_loss']
        )
        
        losses['total_online_distillation_loss'] = total_loss
        
        return losses
    
    def train_step(self, z: torch.Tensor, 
                   optimizers: List[torch.optim.Optimizer]) -> Dict[str, float]:
        """Training step for online distillation"""
        
        # Compute ensemble losses
        losses = self.compute_ensemble_loss(z)
        
        # Update each student
        for i, optimizer in enumerate(optimizers):
            optimizer.zero_grad()
            
            # Individual loss for this student
            student_loss = losses[f'student_{i}_ensemble_loss']
            total_loss = (
                student_loss + 
                self.diversity_weight * losses['diversity_loss']
            )
            
            total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.student_models[i].parameters(), 1.0)
            optimizer.step()
        
        # Clear features
        for features_dict in self.student_features:
            features_dict.clear()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


class AdaptiveKnowledgeDistiller:
    """Adaptive knowledge distillation with dynamic loss weighting"""
    
    def __init__(self, teacher_generator: nn.Module, student_generator: nn.Module,
                 initial_temperature: float = 4.0, min_temperature: float = 1.0,
                 max_temperature: float = 8.0):
        self.teacher_generator = teacher_generator
        self.student_generator = student_generator
        
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        
        self.distiller = GANKnowledgeDistiller(teacher_generator, student_generator)
        
        # Adaptive parameters
        self.loss_history = []
        self.temperature_history = []
        self.adaptation_frequency = 100
        self.step_count = 0
    
    def adapt_temperature(self, current_loss: float):
        """Adapt temperature based on training progress"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) >= self.adaptation_frequency:
            # Calculate loss trend
            recent_losses = self.loss_history[-self.adaptation_frequency:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            # Adapt temperature based on trend
            if loss_trend > 0:  # Loss increasing, increase temperature
                self.temperature = min(self.temperature * 1.1, self.max_temperature)
            else:  # Loss decreasing, decrease temperature
                self.temperature = max(self.temperature * 0.95, self.min_temperature)
            
            self.temperature_history.append(self.temperature)
            
            # Clear old history
            self.loss_history = self.loss_history[-self.adaptation_frequency//2:]
    
    def train_step(self, z: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Adaptive training step"""
        # Update temperature in distillation losses
        for loss_fn in self.distiller.feature_losses.values():
            if hasattr(loss_fn, 'temperature'):
                loss_fn.temperature = self.temperature
        
        # Regular training step
        losses = self.distiller.train_step(z, optimizer)
        
        # Adapt temperature
        self.adapt_temperature(losses['total_distillation_loss'])
        self.step_count += 1
        
        losses['temperature'] = self.temperature
        
        return losses


def test_knowledge_distillation():
    """Test knowledge distillation implementation"""
    from models.stylegan import StyleGAN
    
    # Create teacher and student models
    teacher = StyleGAN(
        latent_dim=512,
        style_dim=512,
        channels=[256, 128, 64, 32],
        resolutions=[4, 8, 16, 32]
    )
    
    student = StyleGAN(
        latent_dim=512,
        style_dim=512,
        channels=[128, 64, 32, 16],  # Smaller student
        resolutions=[4, 8, 16, 32]
    )
    
    print("Teacher parameters:", sum(p.numel() for p in teacher.parameters()))
    print("Student parameters:", sum(p.numel() for p in student.parameters()))
    
    # Test knowledge distillation
    distiller = GANKnowledgeDistiller(teacher, student)
    
    # Test forward pass
    z = torch.randn(2, 512)
    losses = distiller.compute_distillation_loss(z)
    
    print("\nDistillation Losses:")
    for name, loss in losses.items():
        if torch.is_tensor(loss):
            print(f"  {name}: {loss.item():.6f}")
        else:
            print(f"  {name}: {loss:.6f}")
    
    # Test progressive distillation
    print("\nTesting Progressive Distillation...")
    progressive_distiller = ProgressiveKnowledgeDistiller(teacher, student)
    progressive_losses = progressive_distiller.train_step(z, torch.optim.Adam(student.parameters()))
    
    print("Progressive distillation losses:")
    for name, loss in progressive_losses.items():
        print(f"  {name}: {loss:.6f}")
    
    # Test online distillation
    print("\nTesting Online Distillation...")
    student2 = StyleGAN(
        latent_dim=512,
        style_dim=512,
        channels=[128, 64, 32, 16],
        resolutions=[4, 8, 16, 32]
    )
    
    online_distiller = OnlineKnowledgeDistiller([student, student2])
    online_losses = online_distiller.compute_ensemble_loss(z)
    
    print("Online distillation losses:")
    for name, loss in online_losses.items():
        if torch.is_tensor(loss):
            print(f"  {name}: {loss.item():.6f}")
        else:
            print(f"  {name}: {loss:.6f}")
    
    print("\nKnowledge distillation test completed successfully!")


if __name__ == "__main__":
    test_knowledge_distillation()