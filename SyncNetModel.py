#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def save(model, filename):
    """Save the model to a file."""
    with open(filename, "wb") as f:
        torch.save(model, f)
        print("%s saved." % filename)


def load(filename):
    """Load the model from a file."""
    net = torch.load(filename)
    return net


class S(nn.Module):
    """
    SyncNet Model class for audio-visual synchronization feature extraction.
    The model extracts features from both audio and lip (visual) inputs and maps them into a shared representation space.
    """
    def __init__(self, num_layers_in_fc_layers=1024):
        super(S, self).__init__()

        # Audio feature extraction layers
        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # FC layers for audio
        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        # Lip (video) feature extraction layers
        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(256, 512, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        # FC layers for lip
        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

    def forward_aud(self, x):
        """
        Forward pass for audio stream.
        x: Audio input tensor of shape (N, 1, frequency_bins, time_frames).
        Returns feature vector of shape (N, 1024).
        """
        mid = self.netcnnaud(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfcaud(mid)
        return out

    def forward_lip(self, x):
        """
        Forward pass for lip (video) stream.
        x: Video input tensor of shape (N, C=3, frames=5, H, W).
        Returns feature vector of shape (N, 1024).
        """
        mid = self.netcnnlip(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfclip(mid)
        return out

    def forward_lipfeat(self, x):
        """
        Extract intermediate lip features without the FC layers.
        Useful if we need lower-level visual features.
        """
        mid = self.netcnnlip(x)
        out = mid.view((mid.size()[0], -1))
        return out