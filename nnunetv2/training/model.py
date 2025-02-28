from typing import Union

from torch import nn
from torch.nn import functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_channels, num_classes=3, conv_layer_configs=None, dropout_prob=0.2):
        super(ClassificationHead, self).__init__()

        if conv_layer_configs is None:
            conv_layer_configs = [
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1}
            ]

        convolutional_blocks = []
        in_channels = input_channels

        for config in conv_layer_configs:
            convolutional_blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        config['out_channels'],
                        kernel_size=config['kernel_size'],
                        stride=config.get('stride', 1),
                        padding=config.get('padding', 0)
                    ),
                    nn.BatchNorm3d(config['out_channels']),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout3d(dropout_prob)
                )
            )
            in_channels = config['out_channels']

        self.feature_extractor = nn.Sequential(*convolutional_blocks)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.classifier_head = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        return self.classifier_head(x)


class ResidualEncoderUNetWithClassification(ResidualEncoderUNet):
    def __init__(self, **kwargs):
        super(ResidualEncoderUNetWithClassification, self).__init__(**kwargs)
        features_per_stage = kwargs['features_per_stage']
        self.classification_decoder = ClassificationHead(features_per_stage[-1])
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        skips = self.encoder(x)
        return {"segmentation": self.decoder(skips), "classification": self.classification_decoder(skips[-1])}
