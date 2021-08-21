import math
import paddle.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper


class BBBOriginNet(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBOriginNet, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type == 'lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type == 'bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.flatten = FlattenLayer(32*32)
        self.fc1 = BBBLinear(32*32, 1200, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.fc2 = BBBLinear(1200, 1200, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.fc3 = BBBLinear(1200, outputs, bias=True, priors=self.priors)
