import torch.nn as nn
from ..backbone.common import Conv, Concat, C2f
from utils.general import make_divisible

class YoloV8Neck(nn.Module):

    def __init__(self, cfg):
        super(YoloV8Neck, self).__init__()
        self.gd = cfg.Model.depth_multiple
        self.gw = cfg.Model.width_multiple

        input_p3, input_p4, input_p5 = cfg.Model.Neck.in_channels
        output_p3, output_p4, output_p5 = cfg.Model.Neck.out_channels

        self.channels = {
            'input_p3': input_p3,
            'input_p4': input_p4,
            'input_p5': input_p5,
            'output_p3': output_p3,
            'output_p4': output_p4,
            'output_p5': output_p5,
        }
        self.re_channels_out()

        self.input_p3 = self.channels['input_p3']
        self.input_p4 = self.channels['input_p4']
        self.input_p5 = self.channels['input_p5']

        self.output_p3 = self.channels['output_p3']
        self.output_p4 = self.channels['output_p4']
        self.output_p5 = self.channels['output_p5']

        if cfg.Model.Neck.activation == 'SiLU':
            CONV_ACT = 'silu'
            C_ACT = 'silu'
        elif cfg.Model.Neck.activation == 'ReLU':
            CONV_ACT = 'relu'
            C_ACT = 'relu'
        else:
            CONV_ACT = 'hard_swish'
            C_ACT = 'hard_swish'

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.C1 = C2f(self.input_p5 + self.input_p4, self.input_p4, self.get_depth(3), False, 1, 0.5, C_ACT)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.C2 = C2f(self.input_p4 + self.input_p3, self.output_p3, self.get_depth(3), False, 1, 0.5, C_ACT)

        self.conv3 = Conv(self.output_p3 , self.output_p3, 3, 2, None, 1, CONV_ACT)
        self.C3 = C2f(self.output_p3 + self.input_p4, self.output_p4, self.get_depth(3), False, 1, 0.5, C_ACT)

        self.conv4 = Conv(self.output_p4, self.output_p4, 3, 2, None, 1, CONV_ACT)
        self.C4 = C2f(self.output_p4 + self.input_p5, self.output_p5, self.get_depth(3), False, 1, 0.5, C_ACT)

        self.concat = Concat()

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels.items():
            self.channels[k] = self.get_width(v)

    def forward(self, inputs):
        P3, P4, P5 = inputs
        x1 = self.upsample1(P5)
        x1 = self.concat([x1, P4])
        x1 = self.C1(x1)

        x2 = self.upsample2(x1)
        x2 = self.concat([x2, P3])
        x2 = self.C2(x2)

        x3 = self.conv3(x2)
        x3 = self.concat([x3, x1])
        x3 = self.C3(x3)

        x4 = self.conv4(x3)
        x4 = self.concat([x4, P5])
        x4 = self.C4(x4)
       
        return [x2, x3, x4]
