import torch
import torch.nn as nn
import torch.nn.functional as F
from wavelet_fusion import WaveletFusionBlock
import math
import copy

class WaveletAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet_block = WaveletFusionBlock(in_channels)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        wavelet_features = self.wavelet_block(x)
        combined_features = torch.cat([x, wavelet_features], dim=1)
        channel_weights = self.channel_attention(combined_features)
        channel_refined = x * channel_weights
        spatial_max, _ = torch.max(channel_refined, dim=1, keepdim=True)
        spatial_avg = torch.mean(channel_refined, dim=1, keepdim=True)
        spatial_features = torch.cat([spatial_max, spatial_avg], dim=1)
        spatial_weights = self.spatial_attention(spatial_features)
        output = channel_refined * spatial_weights
        return output + x

class WaveletEnhancedBlock(nn.Module):
    def __init__(self, in_channels, num_features=256):
        super().__init__()
        self.wavelet_attention = WaveletAttention(in_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        wavelet_enhanced = self.wavelet_attention(x)
        combined = torch.cat([x, wavelet_enhanced], dim=1)
        fused = self.fusion(combined)
        output = self.projection(fused)
        return output

class WaveletEnhancedMoCo(nn.Module):
    def __init__(self, base_encoder, dim=256, K=65536, m=0.999, T=0.07, mlp=False):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)
        if hasattr(base_encoder, 'feature_channels'):
            feature_channels = base_encoder.feature_channels
        else:
            feature_channels = 2048
        self.wavelet_enhanced_q = WaveletEnhancedBlock(feature_channels, dim)
        self.wavelet_enhanced_k = WaveletEnhancedBlock(feature_channels, dim)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.wavelet_enhanced_q.parameters(), self.wavelet_enhanced_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        if mlp:
            self.mlp_q = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            self.mlp_k = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.wavelet_enhanced_q.parameters(), self.wavelet_enhanced_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size-remaining] = keys[remaining:].T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
        
    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = self.wavelet_enhanced_q(q)
        if hasattr(self, 'mlp_q'):
            q = self.mlp_q(q)
        else:
            q = F.adaptive_avg_pool2d(q, 1).flatten(1)
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = self.wavelet_enhanced_k(k)
            if hasattr(self, 'mlp_k'):
                k = self.mlp_k(k)
            else:
                k = F.adaptive_avg_pool2d(k, 1).flatten(1)
            k = F.normalize(k, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels 