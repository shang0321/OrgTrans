import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from copy import deepcopy

class WaveletTransform(nn.Module):
    def __init__(self, wave_type='haar'):
        super().__init__()
        self.wave_type = wave_type
        
    def forward(self, x):
        B, C, H, W = x.shape
        wavelet_coeffs = []
        
        for b in range(B):
            channel_coeffs = []
            for c in range(C):
                img = x[b, c].cpu().numpy()
                coeffs = pywt.dwt2(img, self.wave_type)
                channel_coeffs.append(coeffs)
            wavelet_coeffs.append(channel_coeffs)
            
        LL, (LH, HL, HH) = self.reorganize_coeffs(wavelet_coeffs, B, C)
        
        return LL.to(x.device), LH.to(x.device), HL.to(x.device), HH.to(x.device)
    
    def reorganize_coeffs(self, coeffs, B, C):
        LL = torch.stack([torch.from_numpy(coeffs[b][c][0]) 
                         for b in range(B) for c in range(C)])
        LH = torch.stack([torch.from_numpy(coeffs[b][c][1][0]) 
                         for b in range(B) for c in range(C)])
        HL = torch.stack([torch.from_numpy(coeffs[b][c][1][1]) 
                         for b in range(B) for c in range(C)])
        HH = torch.stack([torch.from_numpy(coeffs[b][c][1][2]) 
                         for b in range(B) for c in range(C)])
        
        LL = LL.view(B, C, *LL.shape[1:])
        LH = LH.view(B, C, *LH.shape[1:])
        HL = HL.view(B, C, *HL.shape[1:])
        HH = HH.view(B, C, *HH.shape[1:])
        
        return LL, (LH, HL, HH)

class MoCoQueue:
    def __init__(self, dim, K):
        self.K = K
        self.dim = dim
        self.queue = torch.randn(K, dim)
        self.queue = F.normalize(self.queue, dim=1)
        self.ptr = 0
        
    @torch.no_grad()
    def update(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.ptr)
        
        if ptr + batch_size > self.K:
            remain = self.K - ptr
            self.queue[ptr:] = keys[:remain]
            self.queue[:batch_size-remain] = keys[remain:]
            ptr = batch_size-remain
        else:
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.K
            
        self.ptr = ptr

class WaveletMoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        
        self.encoder_q = base_encoder
        self.encoder_k = deepcopy(base_encoder)
        
        self.wavelet = WaveletTransform()
        
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False
            
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                  self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size > self.K:
            remain = self.K - ptr
            self.queue[:, ptr:] = keys[:remain].T
            self.queue[:, :batch_size-remain] = keys[remain:].T
            ptr = batch_size-remain
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
            
        self.queue_ptr[0] = ptr
        
    def forward(self, im_q, im_k=None, is_eval=False):
        if is_eval:
            return self.encoder_q(im_q)
            
        q_LL, q_LH, q_HL, q_HH = self.wavelet(im_q)
        k_LL, k_LH, k_HL, k_HH = self.wavelet(im_k)
        
        q = self.encoder_q(q_LL)
        q = F.normalize(q, dim=1)
        q = self.projector(q)
        q = F.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(k_LL)
            k = F.normalize(k, dim=1)
            
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        self._dequeue_and_enqueue(k)
        
        return logits, labels, (q_LH, q_HL, q_HH), (k_LH, k_HL, k_HH)

class WaveletMoCoLoss(nn.Module):
    def __init__(self, lambda_h=0.2, lambda_w=0.2):
        super().__init__()
        self.lambda_h = lambda_h
        self.lambda_w = lambda_w
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits, labels, q_coeffs, k_coeffs):
        contrast_loss = self.criterion(logits, labels)
        
        wavelet_loss = 0
        for q_coeff, k_coeff in zip(q_coeffs, k_coeffs):
            wavelet_loss += F.mse_loss(q_coeff, k_coeff)
            
        total_loss = contrast_loss + self.lambda_w * wavelet_loss
        
        return total_loss 