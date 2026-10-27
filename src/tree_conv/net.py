from torch import nn
from tree_conv.tcnn import BinaryTreeConv, TreeLayerNorm
from tree_conv.tcnn import TreeActivation, DynamicPooling
from tree_conv.util import prepare_trees
from utils.logger import setup_custom_logger

logger = setup_custom_logger("NET")

def left_child(x):
    if len(x) != 3:
        return None
    return x[1]


def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    if len(x) == 3:
        return x[0]
    return x


class ActorNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.__in_channels = in_channels
        self.__cuda = False

        self.tree_conv = nn.Sequential(BinaryTreeConv(self.__in_channels, 256),
                                       TreeLayerNorm(),
                                       TreeActivation(nn.LeakyReLU()),
                                       BinaryTreeConv(256, 128), TreeLayerNorm(),
                                       TreeActivation(nn.LeakyReLU()),
                                       BinaryTreeConv(128, 64),
                                       TreeLayerNorm(), DynamicPooling(),
                                       nn.Linear(64, 32),
                                       nn.LeakyReLU(),
                                       nn.Linear(32, out_channels),
                                       nn.LogSoftmax(dim=1))

    def in_channels(self):
        return self.__in_channels

    def forward(self, x):
        trees = prepare_trees(x,
                              features,
                              left_child,
                              right_child,
                              cuda=self.__cuda)
        return self.tree_conv(trees)

    def cuda(self):
        self.__cuda = True
        return super().cuda()
    
class CriticNet(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.__in_channels = in_channels
        self.__cuda = False

        self.tree_conv = nn.Sequential(BinaryTreeConv(self.__in_channels, 256),
                                       TreeLayerNorm(),
                                       TreeActivation(nn.LeakyReLU()),
                                       BinaryTreeConv(256, 128), TreeLayerNorm(),
                                       TreeActivation(nn.LeakyReLU()),
                                       BinaryTreeConv(128, 64),
                                       TreeLayerNorm(), DynamicPooling(),
                                       nn.Linear(64, 32),
                                       nn.LeakyReLU(),
                                       nn.Linear(32, 1))

    def in_channels(self):
        return self.__in_channels

    def forward(self, x):
        trees = prepare_trees(x,
                              features,
                              left_child,
                              right_child,
                              cuda=self.__cuda)
        return self.tree_conv(trees)

    def cuda(self):
        self.__cuda = True
        return super().cuda()
