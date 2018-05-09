import torch
import torch.nn as nn
import torch


##########################################################
# Models for predicting next states and rewards
# Input: current states, actions
#
# input size: 3 x (210 x 160)
# after conv layers size: 128 x (11 x 8)
#
# action size: 14
##########################################################

class PredictNet(nn.Module):
    def __init__(self, num_actions):
        super(PredictNet, self).__init__()

        # Convolutional
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=(0, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=6, stride=2, padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, strid=2, padding=(0, 0))

        self.hidden_units = 128 * 11 * 8

        # Fully-connected layers
        self.fc1 = nn.Linear(self.hidden_units, 2048)
        self.encode = nn.Linear(2048, 2048)
        self.embed_action = nn.Linear(num_actions, 2048)
        self.decode = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, self.hidden_units)
        self.fc3 = nn.Linear(self.hidden_units, 1)

        # De-convolutional
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=6, stride=2, padding=(1, 1))
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=8, stride=2, padding=(0, 1))

        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, x, action):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view((-1, self.hidden_units))

        x = self.relu(self.fc1(x))
        x = self.encode(x)
        action = self.embed_action(action)
        x = torch.mul(x, action)
        x = self.decode(x)
        x = self.relu(self.fc2(x))

        r = self.fc3(x)

        x = x.view((-1, 128, 11, 8))
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        return x, r
