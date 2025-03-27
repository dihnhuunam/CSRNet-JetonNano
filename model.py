import torch.nn as nn
import torch
from torchvision import models

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=1):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=dilation, dilation=dilation)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNet(nn.Module):
    def __init__(self, config_type='A', load_weights=False):
        super(CSRNet, self).__init__()
        
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_configs = {
            'A': [512, 512, 512, 256, 128, 64],
            'B': [512, 512, 512, 256, 128, 64],
            'C': [512, 512, 512, 256, 128, 64],
            'D': [512, 512, 512, 256, 128, 64]
        }
        self.dilation_rates = {'A': 1, 'B': 2, 'C': 2, 'D': 4}
        
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_configs[config_type], in_channels=512, dilation=self.dilation_rates[config_type])
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            mod = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self._initialize_weights()
            frontend_state = list(self.frontend.state_dict().items())
            mod_state = list(mod.state_dict().items())
            for i in range(len(frontend_state)):
                frontend_state[i][1].copy_(mod_state[i][1])

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

model = CSRNet(config_type='C')