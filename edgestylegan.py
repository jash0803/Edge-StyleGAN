import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import os
import json
import time
from pathlib import Path

from models.stylegan import StyleGAN, Discriminator
from optimization.pruning import ChannelPruner, GlobalChannelPruner
from optimization.quantization import ModelQuantizer, PTQQuantizer, QATTrainer
from optimization.knowledge_distillation import GANKnowledgeDistiller, ProgressiveKnowledgeDistiller


class EdgeStyleGAN:
    """Main EdgeStyleGAN class that integrates all optimization techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize models
        self.teacher_generator = None
        self.student_generator = None
        self.discriminator = None
        
        # Optimization modules
        self.pruner = None
        self.quantizer = None
        self.distiller = None
        
        # Training state
        self.current_epoch = 0
        self.training_stats = {
            'losses': [],
            'optimization_stats': {},
            'inference_times': []
        }
        
        self._build_models()
        self._setup_optimization()
    
    def _build_models(self):
        """Build teacher and student models"""
        model_config = self.config['model']
        
        # Teacher model (full StyleGAN)
        if self.config.get('teacher_model_path'):
            self.teacher_generator = self._load_pretrained_teacher()
        else:
            self.teacher_generator = StyleGAN(
                latent_dim=model_config['latent_dim'],
                style_dim=model_config['style_dim'],
                channels=model_config['teacher_channels'],
                resolutions=model_config['resolutions']
            ).to(self.device)
        
        # Student model (lightweight)
        self.student_generator = StyleGAN(
            latent_dim=model_config['latent_dim'],
            style_dim=model_config['style_dim'],
            channels=model_config['student_channels'],
            resolutions=model_config['resolutions']
        ).to(self.device)
        
        # Discriminator for adversarial training
        if model_config.get('use_discriminator', True):
            self.discriminator = Discriminator(
                channels=model_config['discriminator_channels'],
                resolutions=model_config['resolutions'][::-1]  # Reverse for discriminator
            ).to(self.device)
        
        print(f"Teacher parameters: {sum(p.numel() for p in self.teacher_generator.parameters()):,}")
        print(f"Student parameters: {sum(p.numel() for p in self.student_generator.parameters()):,}")
        if self.discriminator:
            print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _load_pretrained_teacher(self) -> StyleGAN:
        """Load pretrained teacher model"""
        teacher_path = self.config['teacher_model_path']
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(f"Teacher model not found: {teacher_path}")
        
        checkpoint = torch.load(teacher_path, map_location=self.device)
        
        # Build teacher model
        teacher_config = checkpoint.get('model_config', self.config['model'])
        teacher = StyleGAN(
            latent_dim=teacher_config['latent_dim'],
            style_dim=teacher_config['style_dim'],
            channels=teacher_config.get('teacher_channels', teacher_config['channels']),
            resolutions=teacher_config['resolutions']
        ).to(self.device)
        
        teacher.load_state_dict(checkpoint['generator_state_dict'])
        teacher.eval()
        
        print(f"Loaded pretrained teacher from: {teacher_path}")
        return teacher
    
    def _setup_optimization(self):
        """Setup optimization techniques"""
        opt_config = self.config['optimization']
        
        # Setup pruning
        if opt_config.get('enable_pruning', False):
            pruning_config = opt_config['pruning']
            if pruning_config.get('global_pruning', False):
                self.pruner = GlobalChannelPruner(
                    self.student_generator,
                    pruning_ratio=pruning_config['pruning_ratio']
                )
            else:
                self.pruner = ChannelPruner(
                    self.student_generator,
                    pruning_ratio=pruning_config['pruning_ratio']
                )
        
        # Setup quantization
        if opt_config.get('enable_quantization', False):
            quant_config = opt_config['quantization']
            if quant_config.get('quantization_aware_training', False):
                self.quantizer = QATTrainer(
                    self.student_generator,
                    weight_bits=quant_config['weight_bits'],
                    activation_bits=quant_config['activation_bits']
                )
            else:
                self.quantizer = ModelQuantizer(
                    self.student_generator,
                    weight_bits=quant_config['weight_bits'],
                    activation_bits=quant_config['activation_bits']
                )
        
        # Setup knowledge distillation
        if opt_config.get('enable_distillation', False) and self.teacher_generator:
            distill_config = opt_config['distillation']
            if distill_config.get('progressive_distillation', False):
                self.distiller = ProgressiveKnowledgeDistiller(
                    self.teacher_generator,
                    self.student_generator,
                    num_progressive_stages=distill_config.get('num_stages', 4)
                )
            else:
                self.distiller = GANKnowledgeDistiller(
                    self.teacher_generator,
                    self.student_generator,
                    feature_loss_weight=distill_config.get('feature_weight', 1.0),
                    style_loss_weight=distill_config.get('style_weight', 0.5)
                )
    
    def optimize_model(self) -> nn.Module:
        """Apply all optimization techniques to create optimized model"""
        optimized_model = self.student_generator
        optimization_stats = {}
        
        print("Starting model optimization...")
        
        # Step 1: Knowledge Distillation (if enabled)
        if self.distiller and self.config['optimization'].get('enable_distillation'):
            print("Applying knowledge distillation...")
            start_time = time.time()
            
            # Train with distillation
            self._train_with_distillation()
            
            distillation_time = time.time() - start_time
            optimization_stats['distillation_time'] = distillation_time
            print(f"Knowledge distillation completed in {distillation_time:.2f}s")
        
        # Step 2: Structured Pruning (if enabled)
        if self.pruner and self.config['optimization'].get('enable_pruning'):
            print("Applying structured pruning...")
            start_time = time.time()
            
            original_params = sum(p.numel() for p in optimized_model.parameters())
            optimized_model = self.pruner.prune_model(
                importance_method='l1_norm',
                fine_tune_epochs=self.config['optimization']['pruning'].get('fine_tune_epochs', 10)
            )
            pruned_params = sum(p.numel() for p in optimized_model.parameters())
            
            pruning_time = time.time() - start_time
            optimization_stats['pruning'] = {
                'original_params': original_params,
                'pruned_params': pruned_params,
                'compression_ratio': original_params / pruned_params,
                'pruning_time': pruning_time
            }
            print(f"Pruning completed in {pruning_time:.2f}s")
            print(f"Parameter reduction: {(1 - pruned_params/original_params)*100:.2f}%")
        
        # Step 3: Quantization (if enabled)
        if self.quantizer and self.config['optimization'].get('enable_quantization'):
            print("Applying quantization...")
            start_time = time.time()
            
            if isinstance(self.quantizer, QATTrainer):
                # Quantization-aware training already applied during distillation
                optimized_model = self.quantizer.quantized_model
            else:
                # Post-training quantization
                if isinstance(self.quantizer, PTQQuantizer):
                    optimized_model = self.quantizer.apply_quantization()
                else:
                    optimized_model = self.quantizer.quantize_model()
            
            quantization_time = time.time() - start_time
            optimization_stats['quantization_time'] = quantization_time
            print(f"Quantization completed in {quantization_time:.2f}s")
        
        self.training_stats['optimization_stats'] = optimization_stats
        return optimized_model
    
    def _train_with_distillation(self, num_epochs: int = None):
        """Train student model with knowledge distillation"""
        if num_epochs is None:
            num_epochs = self.config['training'].get('distillation_epochs', 100)
        
        # Setup optimizers
        optimizer_g = torch.optim.Adam(
            self.student_generator.parameters(),
            lr=self.config['training'].get('learning_rate', 0.0001),
            betas=(0.5, 0.999)
        )
        
        optimizer_d = None
        if self.discriminator:
            optimizer_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.config['training'].get('learning_rate', 0.0001),
                betas=(0.5, 0.999)
            )
        
        # Training loop
        self.student_generator.train()
        if self.discriminator:
            self.discriminator.train()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Generate training batch
            batch_size = self.config['training'].get('batch_size', 16)
            z = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device)
            
            # Train discriminator (if available)
            if self.discriminator and optimizer_d:
                self._train_discriminator_step(z, optimizer_d)
            
            # Train generator with distillation
            if isinstance(self.distiller, ProgressiveKnowledgeDistiller):
                # Update progressive stage
                stage = min(epoch // (num_epochs // 4), 3)
                self.distiller.set_stage(stage)
            
            losses = self.distiller.train_step(z, optimizer_g)
            epoch_losses.append(losses['total_distillation_loss'])
            
            # Log progress
            if epoch % 10 == 0:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_loss:.6f}")
                
                # Log all losses
                for loss_name, loss_value in losses.items():
                    if loss_name != 'total_distillation_loss':
                        print(f"  {loss_name}: {loss_value:.6f}")
            
            self.training_stats['losses'].append(losses)
    
    def _train_discriminator_step(self, z: torch.Tensor, optimizer_d: torch.optim.Optimizer):
        """Single discriminator training step"""
        batch_size = z.shape[0]
        
        # Real images (from teacher)
        with torch.no_grad():
            real_images = self.teacher_generator(z)
        
        # Fake images (from student)
        fake_images = self.student_generator(z).detach()
        
        # Discriminator predictions
        real_pred = self.discriminator(real_images)
        fake_pred = self.discriminator(fake_images)
        
        # Discriminator loss (WGAN-GP style)
        d_loss = fake_pred.mean() - real_pred.mean()
        
        # Gradient penalty
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        interpolated_pred = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=interpolated_pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_pred),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        d_loss += 10 * gradient_penalty
        
        # Update discriminator
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()
    
    def benchmark_inference(self, num_samples: int = 100, 
                          warmup_samples: int = 10) -> Dict[str, float]:
        """Benchmark inference speed"""
        print(f"Benchmarking inference speed with {num_samples} samples...")
        
        self.student_generator.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_samples):
                z = torch.randn(1, self.config['model']['latent_dim'], device=self.device)
                _ = self.student_generator(z)
        
        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_samples):
                z = torch.randn(1, self.config['model']['latent_dim'], device=self.device)
                _ = self.student_generator(z)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_samples
        fps = 1.0 / avg_time
        
        benchmark_results = {
            'total_time': total_time,
            'average_inference_time': avg_time,
            'fps': fps,
            'samples_per_second': fps
        }
        
        print(f"Inference benchmark results:")
        print(f"  Average inference time: {avg_time*1000:.2f}ms")
        print(f"  FPS: {fps:.2f}")
        
        return benchmark_results
    
    def evaluate_quality(self, num_samples: int = 1000) -> Dict[str, float]:
        """Evaluate generated image quality"""
        print(f"Evaluating image quality with {num_samples} samples...")
        
        self.student_generator.eval()
        generated_images = []
        
        with torch.no_grad():
            for i in range(0, num_samples, 32):  # Batch processing
                batch_size = min(32, num_samples - i)
                z = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device)
                images = self.student_generator(z)
                generated_images.append(images.cpu())
        
        generated_images = torch.cat(generated_images, dim=0)
        
        # Calculate basic statistics
        quality_metrics = {
            'mean_pixel_value': generated_images.mean().item(),
            'std_pixel_value': generated_images.std().item(),
            'min_pixel_value': generated_images.min().item(),
            'max_pixel_value': generated_images.max().item()
        }
        
        print(f"Quality evaluation results:")
        for metric, value in quality_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return quality_metrics
    
    def save_model(self, save_path: str, include_teacher: bool = False):
        """Save the optimized model"""
        save_dict = {
            'student_generator_state_dict': self.student_generator.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
            'model_info': {
                'student_parameters': sum(p.numel() for p in self.student_generator.parameters()),
                'optimization_applied': {
                    'pruning': self.pruner is not None,
                    'quantization': self.quantizer is not None,
                    'distillation': self.distiller is not None
                }
            }
        }
        
        if include_teacher and self.teacher_generator:
            save_dict['teacher_generator_state_dict'] = self.teacher_generator.state_dict()
        
        if self.discriminator:
            save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
        
        torch.save(save_dict, save_path)
        print(f"Model saved to: {save_path}")
    
    def load_model(self, load_path: str):
        """Load a saved model"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.student_generator.load_state_dict(checkpoint['student_generator_state_dict'])
        
        if 'teacher_generator_state_dict' in checkpoint and self.teacher_generator:
            self.teacher_generator.load_state_dict(checkpoint['teacher_generator_state_dict'])
        
        if 'discriminator_state_dict' in checkpoint and self.discriminator:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        print(f"Model loaded from: {load_path}")
    
    @classmethod
    def from_pretrained(cls, model_name: str, cache_dir: str = None):
        """Load a pretrained EdgeStyleGAN model"""
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/edgestylegan")
        
        model_configs = {
            'edgestylegan-ffhq-256': {
                'model': {
                    'latent_dim': 512,
                    'style_dim': 512,
                    'teacher_channels': [512, 512, 512, 256, 128, 64, 32],
                    'student_channels': [256, 256, 128, 64, 32, 16, 8],
                    'discriminator_channels': [32, 64, 128, 256, 512, 512, 512],
                    'resolutions': [4, 8, 16, 32, 64, 128, 256]
                },
                'optimization': {
                    'enable_pruning': True,
                    'enable_quantization': True,
                    'enable_distillation': True,
                    'pruning': {'pruning_ratio': 0.5},
                    'quantization': {'weight_bits': 8, 'activation_bits': 8},
                    'distillation': {'feature_weight': 1.0, 'style_weight': 0.5}
                },
                'training': {
                    'batch_size': 16,
                    'learning_rate': 0.0001,
                    'distillation_epochs': 100
                }
            },
            'edgestylegan-ffhq-512': {
                'model': {
                    'latent_dim': 512,
                    'style_dim': 512,
                    'teacher_channels': [512, 512, 512, 512, 256, 128, 64, 32, 16],
                    'student_channels': [256, 256, 256, 128, 64, 32, 16, 8, 4],
                    'discriminator_channels': [16, 32, 64, 128, 256, 512, 512, 512, 512],
                    'resolutions': [4, 8, 16, 32, 64, 128, 256, 512]
                },
                'optimization': {
                    'enable_pruning': True,
                    'enable_quantization': True,
                    'enable_distillation': True,
                    'pruning': {'pruning_ratio': 0.6},
                    'quantization': {'weight_bits': 8, 'activation_bits': 8},
                    'distillation': {'feature_weight': 1.0, 'style_weight': 0.5}
                },
                'training': {
                    'batch_size': 8,
                    'learning_rate': 0.0001,
                    'distillation_epochs': 150
                }
            }
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_configs.keys())}")
        
        config = model_configs[model_name]
        
        # Create instance
        edge_stylegan = cls(config)
        
        # Try to load pretrained weights
        model_path = os.path.join(cache_dir, f"{model_name}.pth")
        if os.path.exists(model_path):
            edge_stylegan.load_model(model_path)
        else:
            print(f"Pretrained weights not found at {model_path}")
            print("Using randomly initialized weights. Train the model or download pretrained weights.")
        
        return edge_stylegan
    
    def generate(self, z: torch.Tensor = None, num_samples: int = 1, 
                truncation: float = 1.0, seed: int = None) -> torch.Tensor:
        """Generate images using the optimized model"""
        if seed is not None:
            torch.manual_seed(seed)
        
        if z is None:
            z = torch.randn(num_samples, self.config['model']['latent_dim'], device=self.device)
        
        self.student_generator.eval()
        with torch.no_grad():
            if hasattr(self.student_generator, 'generate'):
                images = self.student_generator.generate(z, truncation=truncation)
            else:
                images = self.student_generator(z)
        
        return images
    
    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, 
                   num_steps: int = 10) -> torch.Tensor:
        """Generate interpolation between two latent codes"""
        self.student_generator.eval()
        
        interpolated_images = []
        with torch.no_grad():
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                image = self.student_generator(z_interp)
                interpolated_images.append(image)
        
        return torch.cat(interpolated_images, dim=0)
    
    def style_mixing(self, z1: torch.Tensor, z2: torch.Tensor, 
                    mixing_layer: int = 4) -> torch.Tensor:
        """Generate images with style mixing"""
        self.student_generator.eval()
        
        with torch.no_grad():
            # Generate styles
            style1 = self.student_generator.mapping(z1)
            style2 = self.student_generator.mapping(z2)
            
            # Mix styles
            mixed_style = style1.clone()
            mixed_style[:, mixing_layer:] = style2[:, mixing_layer:]
            
            # Generate with mixed styles
            image = self.student_generator(z1, styles=mixed_style)
        
        return image
    
    def export_onnx(self, output_path: str, input_size: Tuple[int, ...] = (1, 512)):
        """Export model to ONNX format for deployment"""
        self.student_generator.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_size, device=self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.student_generator,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['latent_code'],
            output_names=['generated_image'],
            dynamic_axes={
                'latent_code': {0: 'batch_size'},
                'generated_image': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to ONNX format: {output_path}")
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        summary = {
            'model_info': {
                'latent_dim': self.config['model']['latent_dim'],
                'style_dim': self.config['model']['style_dim'],
                'output_resolution': self.config['model']['resolutions'][-1],
                'student_parameters': sum(p.numel() for p in self.student_generator.parameters()),
            },
            'optimization_applied': {
                'pruning': self.pruner is not None,
                'quantization': self.quantizer is not None,
                'distillation': self.distiller is not None
            },
            'training_stats': self.training_stats.get('optimization_stats', {}),
            'device': str(self.device)
        }
        
        if self.teacher_generator:
            summary['model_info']['teacher_parameters'] = sum(p.numel() for p in self.teacher_generator.parameters())
            summary['model_info']['compression_ratio'] = (
                summary['model_info']['teacher_parameters'] / 
                summary['model_info']['student_parameters']
            )
        
        if self.discriminator:
            summary['model_info']['discriminator_parameters'] = sum(p.numel() for p in self.discriminator.parameters())
        
        return summary


def create_default_config() -> Dict:
    """Create default configuration for EdgeStyleGAN"""
    return {
        'model': {
            'latent_dim': 512,
            'style_dim': 512,
            'teacher_channels': [512, 512, 256, 128, 64, 32],
            'student_channels': [256, 128, 64, 32, 16, 8],
            'discriminator_channels': [32, 64, 128, 256, 512, 512],
            'resolutions': [4, 8, 16, 32, 64, 128],
            'use_discriminator': True
        },
        'optimization': {
            'enable_pruning': True,
            'enable_quantization': True,
            'enable_distillation': True,
            'pruning': {
                'pruning_ratio': 0.5,
                'global_pruning': False,
                'fine_tune_epochs': 10
            },
            'quantization': {
                'weight_bits': 8,
                'activation_bits': 8,
                'quantization_aware_training': False
            },
            'distillation': {
                'feature_weight': 1.0,
                'style_weight': 0.5,
                'progressive_distillation': False,
                'num_stages': 4
            }
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 0.0001,
            'distillation_epochs': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    }


def demo_edgestylegan():
    """Demo EdgeStyleGAN functionality"""
    print("EdgeStyleGAN Demo")
    print("=" * 50)
    
    # Create configuration
    config = create_default_config()
    config['model']['resolutions'] = [4, 8, 16, 32]  # Smaller for demo
    config['training']['distillation_epochs'] = 10   # Fewer epochs for demo
    
    # Initialize EdgeStyleGAN
    edge_stylegan = EdgeStyleGAN(config)
    
    # Show model summary
    summary = edge_stylegan.get_model_summary()
    print("\nModel Summary:")
    for key, value in summary['model_info'].items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    # Optimize model
    print("\nOptimizing model...")
    optimized_model = edge_stylegan.optimize_model()
    
    # Benchmark inference
    print("\nBenchmarking inference speed...")
    benchmark_results = edge_stylegan.benchmark_inference(num_samples=50)
    
    # Generate sample images
    print("\nGenerating sample images...")
    sample_images = edge_stylegan.generate(num_samples=4, seed=42)
    print(f"Generated images shape: {sample_images.shape}")
    
    # Test interpolation
    print("\nTesting interpolation...")
    z1 = torch.randn(1, config['model']['latent_dim'])
    z2 = torch.randn(1, config['model']['latent_dim'])
    interpolated = edge_stylegan.interpolate(z1, z2, num_steps=5)
    print(f"Interpolated images shape: {interpolated.shape}")
    
    # Evaluate quality
    print("\nEvaluating image quality...")
    quality_metrics = edge_stylegan.evaluate_quality(num_samples=100)
    
    # Save model
    save_path = "edgestylegan_demo.pth"
    edge_stylegan.save_model(save_path)
    
    print(f"\nDemo completed! Model saved to {save_path}")
    print(f"Final model parameters: {sum(p.numel() for p in optimized_model.parameters()):,}")


if __name__ == "__main__":
    demo_edgestylegan()