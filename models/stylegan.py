import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bias_init=0, lr_mul=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_mul = lr_mul
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.full((out_features,), bias_init))
        else:
            self.register_parameter('bias', None)
        
        self.weight_gain = lr_mul / math.sqrt(in_features)

    def forward(self, x):
        weight = self.weight * self.weight_gain
        bias = self.bias
        
        if bias is not None:
            bias = bias * self.lr_mul
            
        return F.linear(x, weight, bias)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, lr_mul=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr_mul = lr_mul
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        fan_in = in_channels * kernel_size ** 2
        self.weight_gain = lr_mul / math.sqrt(fan_in)

    def forward(self, x):
        weight = self.weight * self.weight_gain
        bias = self.bias
        
        if bias is not None:
            bias = bias * self.lr_mul
            
        return F.conv2d(x, weight, bias, self.stride, self.padding)


class AdaIN(nn.Module):
    def __init__(self, num_features, style_dim):
        super().__init__()
        self.num_features = num_features
        self.style_dim = style_dim
        
        self.style_scale = EqualizedLinear(style_dim, num_features, bias_init=1)
        self.style_bias = EqualizedLinear(style_dim, num_features, bias_init=0)

    def forward(self, x, style):
        batch, channel, height, width = x.shape
        
        # Calculate instance statistics
        x_mean = x.view(batch, channel, -1).mean(dim=2, keepdim=True).view(batch, channel, 1, 1)
        x_std = x.view(batch, channel, -1).std(dim=2, keepdim=True).view(batch, channel, 1, 1) + 1e-8
        
        # Normalize
        x_normalized = (x - x_mean) / x_std
        
        # Apply style
        scale = self.style_scale(style).view(batch, channel, 1, 1)
        bias = self.style_bias(style).view(batch, channel, 1, 1)
        
        return x_normalized * scale + bias


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        batch, channel, height, width = x.shape
        
        if noise is None:
            noise = torch.randn(batch, 1, height, width, device=x.device)
        
        return x + self.weight * noise


class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=False, use_noise=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.use_noise = use_noise
        
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        
        self.noise1 = NoiseInjection(out_channels) if use_noise else None
        self.noise2 = NoiseInjection(out_channels) if use_noise else None
        
        self.adain1 = AdaIN(out_channels, style_dim)
        self.adain2 = AdaIN(out_channels, style_dim)
        
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise1=None, noise2=None):
        if self.upsample:
            x = self.upsample_layer(x)
        
        # First convolution
        x = self.conv1(x)
        if self.noise1 is not None:
            x = self.noise1(x, noise1)
        x = self.activation(x)
        x = self.adain1(x, style)
        
        # Second convolution
        x = self.conv2(x)
        if self.noise2 is not None:
            x = self.noise2(x, noise2)
        x = self.activation(x)
        x = self.adain2(x, style)
        
        return x


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, style_dim=512, num_layers=8, lr_mul=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_layers = num_layers
        
        layers = []
        layers.append(PixelNorm())
        
        for i in range(num_layers):
            layers.append(EqualizedLinear(latent_dim if i == 0 else style_dim, style_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2))
        
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


class SynthesisNetwork(nn.Module):
    def __init__(self, style_dim=512, channels=[512, 512, 512, 512, 256, 128, 64, 32], 
                 resolutions=[4, 8, 16, 32, 64, 128, 256, 512]):
        super().__init__()
        self.style_dim = style_dim
        self.channels = channels
        self.resolutions = resolutions
        self.num_layers = len(channels)
        
        # Constant input
        self.constant = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        
        # Initial style block
        self.initial_block = StyleBlock(channels[0], channels[0], style_dim, upsample=False)
        
        # Progressive blocks
        self.blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            block = StyleBlock(channels[i-1], channels[i], style_dim, upsample=True)
            self.blocks.append(block)
        
        # RGB output layers
        self.to_rgb_layers = nn.ModuleList()
        for i, channel in enumerate(channels):
            to_rgb = EqualizedConv2d(channel, 3, 1)
            self.to_rgb_layers.append(to_rgb)

    def forward(self, styles, noise_inputs=None, start_layer=0, end_layer=None, alpha=1.0):
        batch_size = styles.shape[0]
        
        if end_layer is None:
            end_layer = len(self.channels) - 1
        
        # Start with constant input
        x = self.constant.repeat(batch_size, 1, 1, 1)
        
        # Apply initial block
        if start_layer == 0:
            noise1 = noise_inputs[0] if noise_inputs else None
            noise2 = noise_inputs[1] if noise_inputs else None
            x = self.initial_block(x, styles, noise1, noise2)
            
            if end_layer == 0:
                return torch.tanh(self.to_rgb_layers[0](x))
        
        # Apply progressive blocks
        for i, block in enumerate(self.blocks):
            layer_idx = i + 1
            
            if layer_idx < start_layer:
                continue
            
            if layer_idx > end_layer:
                break
            
            noise_idx = layer_idx * 2
            noise1 = noise_inputs[noise_idx] if noise_inputs else None
            noise2 = noise_inputs[noise_idx + 1] if noise_inputs else None
            
            x = block(x, styles, noise1, noise2)
        
        # Convert to RGB
        rgb = torch.tanh(self.to_rgb_layers[end_layer](x))
        
        return rgb


class StyleGAN(nn.Module):
    def __init__(self, latent_dim=512, style_dim=512, num_mapping_layers=8, 
                 channels=[512, 512, 512, 512, 256, 128, 64, 32],
                 resolutions=[4, 8, 16, 32, 64, 128, 256, 512]):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_mapping_layers = num_mapping_layers
        self.channels = channels
        self.resolutions = resolutions
        
        self.mapping = MappingNetwork(latent_dim, style_dim, num_mapping_layers)
        self.synthesis = SynthesisNetwork(style_dim, channels, resolutions)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def generate_noise(self, batch_size, device):
        noise_inputs = []
        resolution = 4
        
        for i in range(len(self.channels)):
            noise = torch.randn(batch_size, 1, resolution, resolution, device=device)
            noise_inputs.append(noise)  # First noise for each layer
            noise_inputs.append(noise)  # Second noise for each layer
            
            if i > 0:  # Don't upscale for the first layer
                resolution *= 2
                
        return noise_inputs
    
    def forward(self, z, noise_inputs=None, styles=None, start_layer=0, end_layer=None, alpha=1.0):
        batch_size = z.shape[0]
        
        if styles is None:
            styles = self.mapping(z)
        
        if noise_inputs is None:
            noise_inputs = self.generate_noise(batch_size, z.device)
        
        rgb = self.synthesis(styles, noise_inputs, start_layer, end_layer, alpha)
        
        return rgb
    
    def generate(self, z, truncation=1.0, truncation_cutoff=None):
        """Generate images with optional truncation trick"""
        with torch.no_grad():
            if truncation < 1.0:
                # Apply truncation trick
                styles = self.mapping(z)
                if truncation_cutoff is None:
                    truncation_cutoff = self.num_mapping_layers
                
                # Calculate average style
                avg_style = styles.mean(dim=0, keepdim=True)
                
                # Apply truncation
                for i in range(min(truncation_cutoff, styles.shape[1])):
                    styles[:, i] = avg_style + truncation * (styles[:, i] - avg_style)
                
                return self.forward(z, styles=styles)
            else:
                return self.forward(z)


class Discriminator(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 512, 512, 512, 512], 
                 resolutions=[512, 256, 128, 64, 32, 16, 8, 4]):
        super().__init__()
        self.channels = channels
        self.resolutions = resolutions
        
        # From RGB layers
        self.from_rgb_layers = nn.ModuleList()
        for channel in channels:
            from_rgb = nn.Sequential(
                EqualizedConv2d(3, channel, 1),
                nn.LeakyReLU(0.2)
            )
            self.from_rgb_layers.append(from_rgb)
        
        # Progressive blocks
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.Sequential(
                EqualizedConv2d(channels[i], channels[i+1], 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(channels[i+1], channels[i+1], 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            )
            self.blocks.append(block)
        
        # Final layers
        self.final_block = nn.Sequential(
            EqualizedConv2d(channels[-1], channels[-1], 3, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(channels[-1], channels[-1], 4),
            nn.LeakyReLU(0.2)
        )
        
        self.classifier = EqualizedLinear(channels[-1], 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, start_layer=0, end_layer=None, alpha=1.0):
        if end_layer is None:
            end_layer = len(self.channels) - 1
        
        # Convert from RGB
        x = self.from_rgb_layers[start_layer](x)
        
        # Apply progressive blocks
        for i in range(start_layer, end_layer):
            x = self.blocks[i](x)
        
        # Final processing
        x = self.final_block(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        
        return x


def test_stylegan():
    """Test the StyleGAN implementation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    generator = StyleGAN(
        latent_dim=512,
        style_dim=512,
        channels=[512, 256, 128, 64],  # Smaller for testing
        resolutions=[4, 8, 16, 32]
    ).to(device)
    
    discriminator = Discriminator(
        channels=[64, 128, 256, 512],
        resolutions=[32, 16, 8, 4]
    ).to(device)
    
    # Test generation
    batch_size = 2
    z = torch.randn(batch_size, 512, device=device)
    
    # Generate images
    fake_images = generator(z)
    print(f"Generated images shape: {fake_images.shape}")
    
    # Test discriminator
    disc_output = discriminator(fake_images.detach())
    print(f"Discriminator output shape: {disc_output.shape}")
    
    # Test with real images
    real_images = torch.randn(batch_size, 3, 32, 32, device=device)
    disc_real = discriminator(real_images)
    print(f"Discriminator real output shape: {disc_real.shape}")
    
    print("StyleGAN test completed successfully!")


if __name__ == "__main__":
    test_stylegan()