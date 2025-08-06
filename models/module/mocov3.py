import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.wavelet_utils import wavelet_transform_torch

class MoCoV3(nn.Module):
    def __init__(self, base_encoder, dim=256, K=65536, m=0.999, T=0.2, use_wavelet=False, cfg=None):
        super(MoCoV3, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.use_wavelet = use_wavelet
        self.encoder_q = base_encoder(cfg)
        self.encoder_k = base_encoder(cfg)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.proj_q = nn.Linear(self.encoder_q.out_shape['C5_size'], dim)
        self.proj_k = nn.Linear(self.encoder_k.out_shape['C5_size'], dim)
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            first_len = self.K - ptr
            self.queue[:, ptr:] = keys[:first_len].T
            self.queue[:, :batch_size - first_len] = keys[first_len:].T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        if self.use_wavelet:
            im_q = wavelet_transform_torch(im_q)
            im_k = wavelet_transform_torch(im_k)

        if im_q.shape[0] != im_k.shape[0]:
            min_batch_size = min(im_q.shape[0], im_k.shape[0])
            im_q = im_q[:min_batch_size]
            im_k = im_k[:min_batch_size]

        _, _, q = self.encoder_q(im_q)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, _, k = self.encoder_k(im_k)

        q = self.proj_q(q.mean(dim=[2, 3]))
        k = self.proj_k(k.mean(dim=[2, 3]))

        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        assert q.shape == k.shape, f"q shape: {q.shape}, k shape: {k.shape}"

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k)
        return logits, labels 