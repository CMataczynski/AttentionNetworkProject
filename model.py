from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channel_nums, window_sizes):
        super(ResidualBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_nums[0], channel_nums[1], window_sizes[0], padding=1),
            nn.BatchNorm2d(channel_nums[1]),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_nums[1], channel_nums[2], window_sizes[1], padding=1),
            nn.BatchNorm2d(channel_nums[2]),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(channel_nums[2], channel_nums[3], window_sizes[2], padding=1),
            nn.BatchNorm2d(channel_nums[3]),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x + residual
        out = self.relu(x)
        return out


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        pass

    def forward(self, x):
        pass


class AttentionNetworks(nn.Module):
    def __init__(self,in_channels=3, mode="Attention56"):
        super().__init__()
        self.mode = mode

        self.conv_1 = nn.Conv2d(in_channels, 64, (7, 7), 2)
        self.pool_2 = nn.MaxPool2d((3,3), 2)
        