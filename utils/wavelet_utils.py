import numpy as np
import torch
try:
    import pywt
except ImportError:
    pywt = None

def wavelet_transform_torch(x, wavelet='haar', level=1):
    if pywt is None:
        raise ImportError('Please install pywt library first: pip install PyWavelets')
    x_np = x.detach().cpu().numpy()
    B, C, H, W = x_np.shape
    out = []
    for b in range(B):
        c_out = []
        for c in range(C):
            coeffs2 = pywt.dwt2(x_np[b, c], wavelet)
            cA, (cH, cV, cD) = coeffs2
            c_out.append(cA)
        c_out = np.stack(c_out, axis=0)
        out.append(c_out)
    out = np.stack(out, axis=0)
    out_tensor = torch.from_numpy(out).to(x.device, dtype=x.dtype)
    return out_tensor 