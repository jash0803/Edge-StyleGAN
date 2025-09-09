#!/usr/bin/env python3
"""
Training script for EdgeStyleGAN
Supports full training pipeline with pruning, quantization, and knowledge distillation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import argparse
import yaml
import os
import json
import time
import numpy as np
from PIL import Image
from pathlib import Path
import wandb
from typing import Dict, List, Optional

from edgestylegan import EdgeStyleGAN, create_default_config
from models.stylegan import StyleGAN, Discriminator
from optimization.pruning import ChannelPruner, GlobalChannelPruner
from optimization.quantization import QATTrainer, ModelQuantizer
from optimization.knowledge_distillation import GANKnowledgeDistiller


class ImageDataset(Dataset):
    """Dataset for loading images from directory"""
    
    def __init__(self, image_dir: str, image_size: int = 256, 
                 transform: Optional[transforms.Compose] = None):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # Supported image extensions
        self.extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Find all image files
        self.image_paths = []
        for ext in self.extensions:
            self.image_paths.extend(list(self.image_dir.glob(f'**/*{ext}')))
            self.image_paths.extend(list(self.image_dir.glob(f'**/*{ext.upper()}')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class EdgeStyleGANTrainer:
    """Comprehensive trainer for EdgeStyleGAN"""
    
    def __init__(self, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.device = torch.device(config['training']['device'])
        
        # Initialize EdgeStyleGAN
        self.edge_stylegan = EdgeStyleGAN(config)
        
        # Training components
        self.optimizers = {}
        self.schedulers = {}
        self.data_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_fid = float('inf')
        
        # Logging
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            wandb.init(project="edgestylegan", config=config)
        
        self._setup_data_loader()
        self._setup_optimizers()
        self._setup_directories()
    
    def _setup_data_loader(self):
        """Setup data loader for training"""
        if self.args.dataset_path:
            dataset = ImageDataset(
                self.args.dataset_path,
                image_size=self.config['model']['resolutions'][-1]
            )
            
            self.data_loader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )
            print(f"Data loader created with {len(dataset)} images")
    
    def _setup_optimizers(self):
        """Setup optimizers and schedulers"""
        lr = self.config['training']['learning_rate']
        
        # Generator optimizer
        self.optimizers['generator'] = torch.optim.Adam(
            self.edge_stylegan.student_generator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )
        
        # Discriminator optimizer
        if self.edge_stylegan.discriminator:
            self.optimizers['discriminator'] = torch.optim.Adam(
                self.edge_stylegan.discriminator.parameters(),
                lr=lr,
                betas=(0.5, 0.999)
            )
        
        # Learning rate schedulers
        for name, optimizer in self.optimizers.items():
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training'].get('epochs', 1000),
                eta_min=lr * 0.01
            )
            self.schedulers[name] = scheduler
    
    def _setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def train_epoch_distillation(self) -> Dict[str, float]:
        """Train one epoch with knowledge distillation"""
        epoch_losses = {}
        num_batches = len(self.data_loader) if self.data_loader else 100
        
        self.edge_stylegan.student_generator.train()
        if self.edge_stylegan.discriminator:
            self.edge_stylegan.discriminator.train()
        
        for batch_idx in range(num_batches):
            # Generate latent codes
            batch_size = self.config['training']['batch_size']
            z = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device)
            
            # Train discriminator
            if self.edge_stylegan.discriminator:
                d_loss = self._train_discriminator_step(z)
                if 'discriminator_loss' not in epoch_losses:
                    epoch_losses['discriminator_loss'] = []
                epoch_losses['discriminator_loss'].append(d_loss)
            
            # Train generator with distillation
            if self.edge_stylegan.distiller:
                g_losses = self.edge_stylegan.distiller.train_step(
                    z, self.optimizers['generator']
                )
                
                # Accumulate losses
                for loss_name, loss_value in g_losses.items():
                    if loss_name not in epoch_losses:
                        epoch_losses[loss_name] = []
                    epoch_losses[loss_name].append(loss_value)
            
            self.global_step += 1
            
            # Log batch losses
            if batch_idx % 10 == 0:
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}")
                if self.edge_stylegan.distiller:
                    for loss_name, loss_value in g_losses.items():
                        print(f"  {loss_name}: {loss_value:.6f}")
        
        # Average losses
        avg_losses = {}
        for loss_name, loss_values in epoch_losses.items():
            avg_losses[loss_name] = sum(loss_values) / len(loss_values)
        
        return avg_losses
    
    def _train_discriminator_step(self, z: torch.Tensor) -> float:
        """Train discriminator for one step"""
        batch_size = z.shape[0]
        
        # Real images (from teacher or dataset)
        if self.data_loader:
            try:
                real_images = next(iter(self.data_loader))[:batch_size].to(self.device)
            except:
                # Fallback to teacher generator
                with torch.no_grad():
                    real_images = self.edge_stylegan.teacher_generator(z)
        else:
            # Use teacher generator
            with torch.no_grad():
                real_images = self.edge_stylegan.teacher_generator(z)
        
        # Fake images from student
        fake_images = self.edge_stylegan.student_generator(z).detach()
        
        # Discriminator predictions
        real_pred = self.edge_stylegan.discriminator(real_images)
        fake_pred = self.edge_stylegan.discriminator(fake_images)
        
        # WGAN-GP loss
        d_loss = fake_pred.mean() - real_pred.mean()
        
        # Gradient penalty
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        interpolated_pred = self.edge_stylegan.discriminator(interpolated)
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
        self.optimizers['discriminator'].zero_grad()
        d_loss.backward()
        self.optimizers['discriminator'].step()
        
        return d_loss.item()
    
    def generate_samples(self, num_samples: int = 64, save_path: str = None) -> torch.Tensor:
        """Generate sample images"""
        self.edge_stylegan.student_generator.eval()
        
        with torch.no_grad():
            samples = self.edge_stylegan.generate(num_samples=num_samples, seed=42)
        
        if save_path:
            grid = make_grid(samples, nrow=8, normalize=True, range=(-1, 1))
            save_image(grid, save_path)
        
        return samples
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate model quality and performance"""
        print("Evaluating model...")
        
        # Benchmark inference speed
        benchmark_results = self.edge_stylegan.benchmark_inference(num_samples=100)
        
        # Evaluate image quality
        quality_metrics = self.edge_stylegan.evaluate_quality(num_samples=500)
        
        # Generate sample images
        samples_path = self.output_dir / 'samples' / f'epoch_{self.current_epoch:04d}.png'
        self.generate_samples(num_samples=64, save_path=str(samples_path))
        
        # Combine all metrics
        eval_results = {**benchmark_results, **quality_metrics}
        
        return eval_results
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint_data = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'student_generator_state_dict': self.edge_stylegan.student_generator.state_dict(),
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'schedulers': {name: sch.state_dict() for name, sch in self.schedulers.items()},
            'config': self.config,
            'best_fid': self.best_fid
        }
        
        if self.edge_stylegan.teacher_generator:
            checkpoint_data['teacher_generator_state_dict'] = self.edge_stylegan.teacher_generator.state_dict()
        
        if self.edge_stylegan.discriminator:
            checkpoint_data['discriminator_state_dict'] = self.edge_stylegan.discriminator.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / 'checkpoints' / f'checkpoint_epoch_{self.current_epoch:04d}.pth'
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoints' / 'latest.pth'
        torch.save(checkpoint_data, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best.pth'
            torch.save(checkpoint_data, best_path)
            print(f"New best model saved! FID: {self.best_fid:.3f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_fid = checkpoint.get('best_fid', float('inf'))
        
        self.edge_stylegan.student_generator.load_state_dict(checkpoint['student_generator_state_dict'])
        
        if 'teacher_generator_state_dict' in checkpoint and self.edge_stylegan.teacher_generator:
            self.edge_stylegan.teacher_generator.load_state_dict(checkpoint['teacher_generator_state_dict'])
        
        if 'discriminator_state_dict' in checkpoint and self.edge_stylegan.discriminator:
            self.edge_stylegan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load optimizers and schedulers
        for name, state_dict in checkpoint['optimizers'].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state_dict)
        
        for name, state_dict in checkpoint['schedulers'].items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(state_dict)
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Resumed from epoch {self.current_epoch}, global step {self.global_step}")
    
    def train(self):
        """Main training loop"""
        print("Starting EdgeStyleGAN training...")
        print(f"Training configuration:")
        print(f"  Epochs: {self.config['training'].get('epochs', 1000)}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        print(f"  Device: {self.device}")
        
        # Load checkpoint if resuming
        if self.args.resume:
            self.load_checkpoint(self.args.resume)
        
        epochs = self.config['training'].get('epochs', 1000)
        
        # Phase 1: Knowledge Distillation Training
        print("\nPhase 1: Knowledge Distillation Training")
        distillation_epochs = self.config['training'].get('distillation_epochs', epochs // 2)
        
        for epoch in range(self.current_epoch, distillation_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train epoch
            epoch_losses = self.train_epoch_distillation()
            epoch_time = time.time() - start_time
            
            # Update learning rates
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Log losses
            print(f"\nEpoch {epoch}/{distillation_epochs} completed in {epoch_time:.2f}s")
            for loss_name, loss_value in epoch_losses.items():
                print(f"  {loss_name}: {loss_value:.6f}")
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'epoch_time': epoch_time,
                    **epoch_losses
                })
            
            # Evaluation and checkpointing
            if epoch % self.args.eval_freq == 0 or epoch == distillation_epochs - 1:
                eval_results = self.evaluate_model()
                
                # Check if best model
                current_fid = eval_results.get('mean_pixel_value', float('inf'))  # Placeholder
                is_best = current_fid < self.best_fid
                if is_best:
                    self.best_fid = current_fid
                
                self.save_checkpoint(is_best=is_best)
                
                if self.use_wandb:
                    wandb.log({
                        'evaluation': eval_results,
                        'is_best': is_best
                    })
        
        # Phase 2: Apply Optimization Techniques
        print("\nPhase 2: Applying Optimization Techniques")
        optimized_model = self.edge_stylegan.optimize_model()
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_eval = self.evaluate_model()
        
        # Save final model
        final_model_path = self.output_dir / 'edgestylegan_final.pth'
        self.edge_stylegan.save_model(str(final_model_path))
        
        # Export to ONNX
        if self.args.export_onnx:
            onnx_path = self.output_dir / 'edgestylegan_final.onnx'
            self.edge_stylegan.export_onnx(str(onnx_path))
        
        print(f"\nTraining completed!")
        print(f"Final model saved to: {final_model_path}")
        print("Final evaluation results:")
        for metric, value in final_eval.items():
            print(f"  {metric}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Train EdgeStyleGAN')
    
    # Model configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='Path to pre-trained teacher StyleGAN model')
    parser.add_argument('--architecture', type=str, default='mobilenet_v3',
                        choices=['mobilenet_v3', 'efficientnet_b0', 'ghost_net'],
                        help='Student generator architecture')
    
    # Training data
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to training dataset directory')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Training image resolution')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0002,
                        help='Learning rate for optimizers')
    parser.add_argument('--distillation-epochs', type=int, default=None,
                        help='Number of epochs for distillation phase (default: half of total)')
    
    # Optimization settings
    parser.add_argument('--enable-pruning', action='store_true',
                        help='Enable channel pruning')
    parser.add_argument('--pruning-ratio', type=float, default=0.3,
                        help='Channel pruning ratio')
    parser.add_argument('--enable-quantization', action='store_true',
                        help='Enable quantization-aware training')
    parser.add_argument('--quantization-bits', type=int, default=8,
                        help='Quantization bits (8 or 16)')
    
    # Knowledge distillation
    parser.add_argument('--distillation-weight', type=float, default=1.0,
                        help='Weight for distillation loss')
    parser.add_argument('--feature-weight', type=float, default=0.1,
                        help='Weight for feature matching loss')
    parser.add_argument('--adversarial-weight', type=float, default=0.01,
                        help='Weight for adversarial loss')
    
    # Output and logging
    parser.add_argument('--output-dir', type=str, default='./outputs/edgestylegan',
                        help='Output directory for models and logs')
    parser.add_argument('--eval-freq', type=int, default=10,
                        help='Evaluation frequency (epochs)')
    parser.add_argument('--save-freq', type=int, default=50,
                        help='Checkpoint saving frequency (epochs)')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='edgestylegan',
                        help='Weights & Biases project name')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Export options
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export final model to ONNX format')
    parser.add_argument('--export-tflite', action='store_true',
                        help='Export final model to TensorFlow Lite format')
    
    # Device settings
    parser.add_argument('--device', type=str, default='auto',
                        help='Training device (cuda, cpu, or auto)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Debugging and development
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (smaller model, fewer epochs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = create_default_config()
        print("Using default configuration")
    
    # Override config with command line arguments
    config['training']['device'] = device
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.learning_rate
    
    if args.distillation_epochs:
        config['training']['distillation_epochs'] = args.distillation_epochs
    
    # Model configuration
    if args.architecture:
        config['model']['student_architecture'] = args.architecture
    
    config['model']['resolutions'] = [args.image_size]
    
    # Optimization settings
    config['optimization']['pruning']['enabled'] = args.enable_pruning
    config['optimization']['pruning']['ratio'] = args.pruning_ratio
    config['optimization']['quantization']['enabled'] = args.enable_quantization
    config['optimization']['quantization']['bits'] = args.quantization_bits
    
    # Knowledge distillation weights
    if 'knowledge_distillation' not in config:
        config['knowledge_distillation'] = {}
    
    config['knowledge_distillation']['distillation_weight'] = args.distillation_weight
    config['knowledge_distillation']['feature_weight'] = args.feature_weight
    config['knowledge_distillation']['adversarial_weight'] = args.adversarial_weight
    
    # Teacher model path
    if args.teacher_model:
        config['model']['teacher_model_path'] = args.teacher_model
    
    # Debug mode adjustments
    if args.debug:
        print("Debug mode enabled - using smaller model and fewer epochs")
        config['training']['epochs'] = min(config['training']['epochs'], 10)
        config['training']['batch_size'] = min(config['training']['batch_size'], 4)
        config['model']['base_channels'] = 32  # Smaller model
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    try:
        trainer = EdgeStyleGANTrainer(config, args)
        
        # Start training
        trainer.train()
        
        # Clean up
        if args.use_wandb:
            wandb.finish()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if args.use_wandb:
            wandb.finish()
            
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        if args.use_wandb:
            wandb.finish()
        raise


if __name__ == '__main__':
    main()