import torch
import torch.nn as nn

class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.SiLU(),
                
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.SiLU(),
                
                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
            ),
            
            nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.SiLU(),
            ),
            
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.SiLU(),
                nn.Conv2d(256, 80, kernel_size=1),
            )
        ])
        
    def forward(self, x):
        features = []
        for i, m in enumerate(self.model):
            x = m(x)
            if i < 2:
                features.append(x)
        return features

class Model(DetectionModel):
    pass

def attempt_load(weights_path, device='cpu'):
    model = Model()
    try:
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt['model'].float().state_dict()
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded weights from {weights_path}")
    except Exception as e:
        print(f"Warning: Could not load weights: {str(e)}")
        print("Using model without pre-trained weights")
    return model 