import torch.nn as nn

from rlda_conv import RLDAConv


class RLDAResNet18(nn.Module):
    def __init__(self, original_model, lut_base_path):
        super().__init__()
        self.model = original_model

        layer_specs = [
            ('layer1.0.conv1', 64, 64, 3, 1, 1),
            ('layer1.0.conv2', 64, 64, 3, 1, 1),
            ('layer1.1.conv1', 64, 64, 3, 1, 1),
            ('layer1.1.conv2', 64, 64, 3, 1, 1),
            ('layer2.0.conv1', 64, 128, 3, 2, 1),
            ('layer2.0.conv2', 128, 128, 3, 1, 1),
            ('layer2.1.conv1', 128, 128, 3, 1, 1),
            ('layer2.1.conv2', 128, 128, 3, 1, 1),
            ('layer3.0.conv1', 128, 256, 3, 2, 1),
            ('layer3.0.conv2', 256, 256, 3, 1, 1),
            ('layer3.1.conv1', 256, 256, 3, 1, 1),
            ('layer3.1.conv2', 256, 256, 3, 1, 1),
            ('layer4.0.conv1', 256, 512, 3, 2, 1),
            ('layer4.0.conv2', 512, 512, 3, 1, 1),
            ('layer4.1.conv1', 512, 512, 3, 1, 1),
            ('layer4.1.conv2', 512, 512, 3, 1, 1)
        ]

        for layer_name, in_ch, out_ch, kernel_size, stride, padding in layer_specs:
            layer_path = layer_name.split('.')
            target_layer = self.model
            for name in layer_path[:-1]:
                if name.isdigit():
                    target_layer = target_layer[int(name)]
                else:
                    target_layer = getattr(target_layer, name)
            # Remove conv_weight parameter, as our RLDAConv now only expects the LUT path and the convolution parameters.
            rlda_conv = RLDAConv(
                lut_path=f'{lut_base_path}/lut_{layer_name.replace(".", "_")}.pth',
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            setattr(target_layer, layer_path[-1], rlda_conv)

    def forward(self, x):
        return self.model(x)
