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


class ResNet18NAS(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        model = list(torchvision.models.resnet18(pretrained=True).children())
        block_1 = nn.Sequential(*model[:5])
        norm_1 = nn.InstanceNorm2d(64, affine=True)
        block_2 = model[5]
        norm_2 = nn.InstanceNorm2d(128, affine=True)
        block_3 = model[6]
        norm_3 = nn.InstanceNorm2d(256, affine=True)
        block_4 = model[7]
        norm_4 = nn.InstanceNorm2d(512, affine=True)
        self.blocks = nn.ModuleList([block_1, block_2, block_3, block_4])
        self.norms = nn.ModuleList([norm_1, norm_2, norm_3, norm_4])
        self.classifier = nn.Sequential(model[8], nn.Flatten(), ProbMLP(512, n_classes, 512))

    def forward(self, x, norm_choice):
        in_feats = []
        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            x = block(x)
            in_x = norm(x)
            x = norm_choice[i] * in_x + (1. - norm_choice[i]) * x
            in_feats.append(in_x)
        result, log_var, mu, embed = self.classifier(x)
        return result, in_feats, log_var, mu, embed


class VGG11NAS(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        model = torchvision.models.vgg11(pretrained=True)
        blocks = list(model.features.children())
        block_1 = nn.Sequential(*blocks[:3])
        norm_1 = nn.InstanceNorm2d(64, affine=True)
        block_2 = nn.Sequential(*blocks[3:6])
        norm_2 = nn.InstanceNorm2d(128, affine=True)
        block_3 = nn.Sequential(*blocks[6:11])
        norm_3 = nn.InstanceNorm2d(256, affine=True)
        block_4 = nn.Sequential(*blocks[11:16])
        norm_4 = nn.InstanceNorm2d(512, affine=True)
        self.blocks = nn.ModuleList([block_1, block_2, block_3, block_4])
        self.norms = nn.ModuleList([norm_1, norm_2, norm_3, norm_4])
        self.classifier = nn.Sequential(nn.Sequential(*blocks[16:]), model.avgpool, nn.Flatten(), *list(model.classifier.children())[:-2], ProbMLP(4096, n_classes, 512))

    def forward(self, x, norm_choice):
        in_feats = []
        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            x = block(x)
            in_x = norm(x)
            x = norm_choice[i] * in_x + (1. - norm_choice[i]) * x
            in_feats.append(in_x)
        result, log_var, mu, embed = self.classifier(x)
        return result, in_feats, log_var, mu, embed


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


class AlexNetNAS(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        model = AlexNetCaffe()
        model.load_state_dict(torch.load('pretrained/alexnet_caffe.pth.tar'))
        block_0 = nn.Sequential(*model.features[:4])
        norm_0 = nn.InstanceNorm2d(96, affine=True)
        block_1 = nn.Sequential(*model.features[4:8])
        norm_1 = nn.InstanceNorm2d(256, affine=True)
        block_2 = nn.Sequential(*model.features[8:10])
        norm_2 = nn.InstanceNorm2d(384, affine=True)
        block_3 = nn.Sequential(*model.features[10:12])
        norm_3 = nn.InstanceNorm2d(384, affine=True)
        block_4 = nn.Sequential(*model.features[12:14])
        norm_4 = nn.InstanceNorm2d(256, affine=True)
        self.blocks = nn.ModuleList([block_0, block_1, block_2, block_3, block_4])
        self.norms = nn.ModuleList([norm_0, norm_1, norm_2, norm_3, norm_4])
        self.classifier = nn.Sequential(model.avgpool, nn.Flatten(), *list(model.classifier.children())[:-1],
                                        ProbMLP(4096, n_classes, 512))

    def forward(self, x, norm_choice):
        in_feats = []
        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            x = block(x)
            in_x = norm(x)
            x = norm_choice[i] * in_x + (1. - norm_choice[i]) * x
            in_feats.append(in_x)
        result, log_var, mu, embed = self.classifier(x)
        return result, in_feats, log_var, mu, embed


class LeNetNAS(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv2d(3, 64, 5),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(64, 128, 5),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2)])
        self.norms = nn.ModuleList([nn.InstanceNorm2d(64, affine=True),
                                    nn.InstanceNorm2d(64, affine=True),
                                    nn.InstanceNorm2d(64, affine=True),
                                    nn.InstanceNorm2d(128, affine=True),
                                    nn.InstanceNorm2d(128, affine=True),
                                    nn.InstanceNorm2d(128, affine=True)])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(),
            ProbMLP(1024, n_classes, 1024)
        )

    def forward(self, x, norm_choice):
        in_feats = []
        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            x = block(x)
            in_x = norm(x)
            x = norm_choice[i] * in_x + (1. - norm_choice[i]) * x
            in_feats.append(in_x)
        result, log_var, mu, embed = self.classifier(x)
        return result, in_feats, log_var, mu, embed


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


class WideResNet164NAS(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        model = WideResNet(16, num_classes=n_classes, widen_factor=4, drop_rate=0.0)
        block_0 = model.conv1
        norm_0 = nn.InstanceNorm2d(16, affine=True)
        block_1 = model.block1
        norm_1 = nn.InstanceNorm2d(64, affine=True)
        block_2 = model.block2
        norm_2 = nn.InstanceNorm2d(128, affine=True)
        block_3 = model.block3
        norm_3 = nn.InstanceNorm2d(256, affine=True)
        self.blocks = nn.ModuleList([block_0, block_1, block_2, block_3])
        self.norms = nn.ModuleList([norm_0, norm_1, norm_2, norm_3])
        self.classifier = nn.Sequential(model.bn1, model.relu, model.pool, model.flatten, model.mlp)

    def forward(self, x, norm_choice):
        in_feats = []
        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            x = block(x)
            in_x = norm(x)
            x = norm_choice[i] * in_x + (1. - norm_choice[i]) * x
            in_feats.append(in_x)
        result, log_var, mu, embed = self.classifier(x)
        return result, in_feats, log_var, mu, embed


def get_network(network, n_classes):
    if network == 'resnet18':
        return ResNet18NAS(n_classes), torch.tensor([0.] * 4)
    elif network == 'alexnet':
        return AlexNetNAS(n_classes), torch.tensor([0.] * 5)
    elif network == 'lenet':
        return LeNetNAS(n_classes), torch.tensor([0.] * 6)
    elif network == 'wide_resnet164':
        return WideResNet164NAS(n_classes), torch.tensor([0.] * 4)
    elif network == 'vgg11':
        return VGG11NAS(n_classes), torch.tensor([0.] * 4)
    else:
        raise NotImplementedError


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
        output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s2 + weight[3] * x_s3 +
                  weight[4] * x_s4 + weight[5] * x_s5) / weight.sum()
        return output


class AugNetSmall(nn.Module):
    def __init__(self):
        super(AugNetSmall, self).__init__()
        self.shift_var = nn.Parameter(torch.empty(3, 30, 30))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, 30, 30))
        nn.init.normal_(self.shift_mean, 0, 0.01)
        self.norm = nn.InstanceNorm2d(3)
        # Fixed Parameters for MI estimation
        self.spatial = nn.Conv2d(3, 3, 3)
        self.spatial_up = nn.ConvTranspose2d(3, 3, 3)
        self.color = nn.Conv2d(3, 3, 1)

        for param in list(list(self.color.parameters()) +
                          list(self.spatial.parameters()) + list(self.spatial_up.parameters())):
            param.requires_grad = False

    def forward(self, x, estimation=False):
        device = x.device
        if not estimation:
            spatial = nn.Conv2d(3, 3, 3).to(device)
            spatial_up = nn.ConvTranspose2d(3, 3, 3).to(device)
            color = nn.Conv2d(3, 3, 1).to(device)
            weight = torch.randn(2).to(device)
            x_c = torch.tanh(F.dropout(color(x), p=.5))
        else:
            spatial = self.spatial
            spatial_up = self.spatial_up
            color = self.color
            weight = torch.ones(2).to(device)
            x_c = torch.tanh(color(x))
        x_down = spatial(x)
        x_down = self.shift_var * self.norm(x_down) + self.shift_mean
        x_s = torch.tanh(spatial_up(x_down))
        output = (weight[0] * x_c + weight[1] * x_s) / weight.sum()
        return output
