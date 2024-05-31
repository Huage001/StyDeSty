import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from collections import OrderedDict


class ProbMLP(nn.Module):
    
    def __init__(self, in_feat, out_feat, mid_feat):
        super().__init__()
        self.log_var = nn.Sequential(nn.Linear(in_feat, mid_feat), nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(in_feat, mid_feat), nn.LeakyReLU())
        self.linear = nn.Linear(mid_feat, out_feat)
        
    @staticmethod
    def reparameterize(mu, log_var, factor=0.2):
        std = log_var.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + factor * std * eps
 
    def forward(self, x):
        log_var = self.log_var(x)
        mu = self.mu(x)
        if self.training:
            embed = self.reparameterize(mu, log_var)
        else:
            embed = mu
        result = self.linear(embed)
        return result, log_var, mu, embed


class AlexNetCaffe(nn.Module):
    def __init__(self, n_classes=1000):
        super(AlexNetCaffe, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout()),
            ("fc8", nn.Linear(4096, n_classes))]))
     
    def forward(self, x):
        x = self.features(x * 57.6)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class AlexNetBackbone(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x * 57.6)


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.is_in_equal_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.is_in_equal_out) and nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False) or None

    def forward(self, x):
        if not self.is_in_equal_out:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))
        else:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if not self.is_in_equal_out:
            return torch.add(self.conv_shortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    """Layer container for blocks."""

    def __init__(self,
                 nb_layers,
                 in_planes,
                 out_planes,
                 block,
                 stride,
                 drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, drop_rate)

    @staticmethod
    def _make_layer(block, in_planes, out_planes, nb_layers, stride,
                    drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes,
                      i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """WideResNet class."""

    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1,
                                   drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2,
                                   drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2,
                                   drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.mlp = ProbMLP(n_channels[3], num_classes, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.pool(out)
        out = self.flatten(out)
        return self.mlp(out)


def get_network(network, n_classes):
    if network == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        backbone = nn.Sequential(
            *list(model.children())[:6],
            nn.InstanceNorm2d(128, affine=True)
        )
        classifier = nn.Sequential(*list(model.children())[6:-1], nn.Flatten(), ProbMLP(512, n_classes, 512))
    elif network == 'alexnet':
        model = AlexNetCaffe()
        model.load_state_dict(torch.load('pretrained/alexnet_caffe.pth.tar'))
        backbone = AlexNetBackbone(nn.Sequential(
            *list(model.features.children())[:4],
            nn.InstanceNorm2d(96, affine=True)
        ))
        classifier = nn.Sequential(
            *list(model.features.children())[4:],
            model.avgpool,
            nn.Flatten(),
            *list(model.classifier.children())[:-1],
            ProbMLP(4096, n_classes, 512)
        )
    elif network == 'vgg11':
        model = torchvision.models.vgg11(pretrained=True)
        backbone = nn.Sequential(
            *list(model.features.children())[:3],
            nn.InstanceNorm2d(64, affine=True)
        )
        classifier = nn.Sequential(
            *list(model.features.children())[3:],
            model.avgpool,
            nn.Flatten(),
            *list(model.classifier.children())[:-2],
            ProbMLP(4096, n_classes, 512)
        )
    elif network == 'lenet':
        model = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(),
            ProbMLP(1024, n_classes, 1024)
        )
        backbone = nn.Sequential(*list(model.children())[:2], nn.InstanceNorm2d(64, affine=True))
        classifier = nn.Sequential(*list(model.children())[2:])
    elif network == 'wide_resnet164':
        model = WideResNet(16, num_classes=n_classes, widen_factor=4, drop_rate=0.0)
        backbone = nn.Sequential(model.conv1, model.block1,
                                 nn.InstanceNorm2d(64, affine=True))
        classifier = nn.Sequential(model.block2, model.block3, 
                                   model.bn1, model.relu, model.pool, model.flatten, model.mlp)
    else:
        raise NotImplementedError
    return backbone, classifier


class AugNet(nn.Module):
    def __init__(self):
        super(AugNet, self).__init__()
        self.shift_var1 = nn.Parameter(torch.empty(3, 216, 216))
        nn.init.normal_(self.shift_var1, 1, 0.1)
        self.shift_mean1 = nn.Parameter(torch.zeros(3, 216, 216))
        nn.init.normal_(self.shift_mean1, 0, 0.1)

        self.shift_var2 = nn.Parameter(torch.empty(3, 212, 212))
        nn.init.normal_(self.shift_var2, 1, 0.1)
        self.shift_mean2 = nn.Parameter(torch.zeros(3, 212, 212))
        nn.init.normal_(self.shift_mean2, 0, 0.1)

        self.shift_var3 = nn.Parameter(torch.empty(3, 208, 208))
        nn.init.normal_(self.shift_var3, 1, 0.1)
        self.shift_mean3 = nn.Parameter(torch.zeros(3, 208, 208))
        nn.init.normal_(self.shift_mean3, 0, 0.1)

        self.shift_var4 = nn.Parameter(torch.empty(3, 220, 220))
        nn.init.normal_(self.shift_var4, 1, 0.1)
        self.shift_mean4 = nn.Parameter(torch.zeros(3, 220, 220))
        nn.init.normal_(self.shift_mean4, 0, 0.1)

        self.shift_var5 = nn.Parameter(torch.empty(64, 1, 1))
        nn.init.normal_(self.shift_var5, 1, 0.1)
        self.shift_mean5 = nn.Parameter(torch.zeros(64, 1, 1))
        nn.init.normal_(self.shift_mean5, 0, 0.1)

        self.norm1 = nn.InstanceNorm2d(3)
        self.norm2 = nn.InstanceNorm2d(64)

        # Fixed Parameters for MI estimation
        self.spatial1 = nn.Conv2d(3, 3, 9)
        self.spatial_up1 = nn.ConvTranspose2d(3, 3, 9)

        self.spatial2 = nn.Conv2d(3, 3, 13)
        self.spatial_up2 = nn.ConvTranspose2d(3, 3, 13)

        self.spatial3 = nn.Conv2d(3, 3, 17)
        self.spatial_up3 = nn.ConvTranspose2d(3, 3, 17)

        self.spatial4 = nn.Conv2d(3, 3, 5)
        self.spatial_up4 = nn.ConvTranspose2d(3, 3, 5)

        self.spatial5 = nn.Conv2d(3, 64, 5)
        self.spatial_up5 = nn.ConvTranspose2d(64, 3, 5)

        self.color = nn.Conv2d(3, 3, 1)

        for param in list(list(self.color.parameters()) +
                          list(self.spatial1.parameters()) + list(self.spatial_up1.parameters()) +
                          list(self.spatial2.parameters()) + list(self.spatial_up2.parameters()) +
                          list(self.spatial3.parameters()) + list(self.spatial_up3.parameters()) +
                          list(self.spatial4.parameters()) + list(self.spatial_up4.parameters()) +
                          list(self.spatial5.parameters()) + list(self.spatial_up5.parameters())):
            param.requires_grad = False

    def forward(self, x, estimation=False):
        device = x.device
        if not estimation:
            spatial1 = nn.Conv2d(3, 3, 9).to(device)
            spatial_up1 = nn.ConvTranspose2d(3, 3, 9).to(device)
            spatial2 = nn.Conv2d(3, 3, 13).to(device)
            spatial_up2 = nn.ConvTranspose2d(3, 3, 13).to(device)
            spatial3 = nn.Conv2d(3, 3, 17).to(device)
            spatial_up3 = nn.ConvTranspose2d(3, 3, 17).to(device)
            spatial4 = nn.Conv2d(3, 3, 5).to(device)
            spatial_up4 = nn.ConvTranspose2d(3, 3, 5).to(device)
            spatial5 = nn.Conv2d(3, 64, 5).to(device)
            spatial_up5 = nn.ConvTranspose2d(64, 3, 5).to(device)
            color = nn.Conv2d(3, 3, 1).to(device)
            weight = torch.randn(6).to(device)
            x_c = torch.tanh(F.dropout(color(x), p=.2))
        else:
            spatial1 = self.spatial1
            spatial_up1 = self.spatial_up1
            spatial2 = self.spatial2
            spatial_up2 = self.spatial_up2
            spatial3 = self.spatial3
            spatial_up3 = self.spatial_up3
            spatial4 = self.spatial4
            spatial_up4 = self.spatial_up4
            spatial5 = self.spatial5
            spatial_up5 = self.spatial_up5
            color = self.color
            weight = torch.ones(6).to(device)
            x_c = torch.tanh(color(x))
        x_s1down = spatial1(x)
        x_s1down = self.shift_var1 * self.norm1(x_s1down) + self.shift_mean1
        x_s = torch.tanh(spatial_up1(x_s1down))
        x_s2down = spatial2(x)
        x_s2down = self.shift_var2 * self.norm1(x_s2down) + self.shift_mean2
        x_s2 = torch.tanh(spatial_up2(x_s2down))
        x_s3down = spatial3(x)
        x_s3down = self.shift_var3 * self.norm1(x_s3down) + self.shift_mean3
        x_s3 = torch.tanh(spatial_up3(x_s3down))
        x_s4down = spatial4(x)
        x_s4down = self.shift_var4 * self.norm1(x_s4down) + self.shift_mean4
        x_s4 = torch.tanh(spatial_up4(x_s4down))
        x_s5down = spatial5(x)
        x_s5down = self.shift_var5 * self.norm2(x_s5down) + self.shift_mean5
        x_s5 = torch.tanh(spatial_up5(x_s5down))
        output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s2 + weight[3] * x_s3 + weight[4] * x_s4 + weight[5] * x_s5) / weight.sum()
        return output


class AugNetSmall(nn.Module):
    def __init__(self):
        super(AugNetSmall, self).__init__()
        self.shift_var = nn.Parameter(torch.empty(3, 30, 30))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, 30, 30))
        nn.init.normal_(self.shift_mean, 0, 0.01)
        self.shift_var_1 = nn.Parameter(torch.empty(3, 28, 28))
        nn.init.normal_(self.shift_var_1, 1, 0.1)
        self.shift_mean_1 = nn.Parameter(torch.zeros(3, 28, 28))
        nn.init.normal_(self.shift_mean_1, 0, 0.01)
        self.norm = nn.InstanceNorm2d(3)
        # Fixed Parameters for MI estimation
        self.spatial = nn.Conv2d(3, 3, 3)
        self.spatial_up = nn.ConvTranspose2d(3, 3, 3)
        self.spatial_1 = nn.Conv2d(3, 3, 5)
        self.spatial_up_1 = nn.ConvTranspose2d(3, 3, 5)

        self.color = nn.Conv2d(3, 3, 1)

        for param in list(list(self.color.parameters()) +
                          list(self.spatial.parameters()) + list(self.spatial_up.parameters()) + 
                          list(self.spatial_1.parameters()) + list(self.spatial_up_1.parameters())):
            param.requires_grad = False

    def forward(self, x, estimation=False):
        device = x.device
        if not estimation:
            spatial = nn.Conv2d(3, 3, 3).to(device)
            spatial_up = nn.ConvTranspose2d(3, 3, 3).to(device)
            spatial_1 = nn.Conv2d(3, 3, 5).to(device)
            spatial_up_1 = nn.ConvTranspose2d(3, 3, 5).to(device)
            color = nn.Conv2d(3, 3, 1).to(device)
            weight = torch.randn(3).to(device)
            x_c = torch.tanh(F.dropout(color(x), p=.5))
        else:
            spatial = self.spatial
            spatial_up = self.spatial_up
            spatial_1 = self.spatial_1
            spatial_up_1 = self.spatial_up_1
            color = self.color
            weight = torch.ones(3).to(device)
            x_c = torch.tanh(color(x))
        x_down = spatial(x)
        x_down = self.shift_var * self.norm(x_down) + self.shift_mean
        x_s = torch.tanh(spatial_up(x_down))
        x_down_1 = spatial_1(x)
        x_down_1 = self.shift_var_1 * self.norm(x_down_1) + self.shift_mean_1
        x_s_1 = torch.tanh(spatial_up_1(x_down_1))
        output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s_1) / weight.sum()
        return output
