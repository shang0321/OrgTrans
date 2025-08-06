
from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile

def check_train_batch_size(model, imgsz=640, amp=True):
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)

def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):

    prefix = colorstr('AutoBatch: ')
    LOGGER.info(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = next(model.parameters()).device
    if device.type == 'cpu':
        LOGGER.info(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size

    gb = 1 << 30
    d = str(device).upper()
    properties = torch.cuda.get_device_properties(device)
    t = properties.total_memory / gb
    r = torch.cuda.memory_reserved(device) / gb
    a = torch.cuda.memory_allocated(device) / gb
    f = t - (r + a)
    LOGGER.info(f'{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')

    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')

    y = [x[2] for x in results if x]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)
    b = int((f * fraction - p[1]) / p[0])
    if None in results:
        i = results.index(None)
        if b >= batch_sizes[i]:
            b = batch_sizes[max(i - 1, 0)]
    if b < 1 or b > 1024:
        b = batch_size
        LOGGER.warning(f'{prefix}WARNING: ⚠️ CUDA anomaly detected, recommend restart environment and retry command.')

    fraction = np.polyval(p, b) / t
    LOGGER.info(f'{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅')
    return b
