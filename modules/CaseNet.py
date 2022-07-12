import torch.nn as nn
import torch
from torch.autograd import Variable
import math
from collections import OrderedDict
from torchvision.models.resnet import resnet50, resnet101
import sys
from .CASENet_capital import ConcatLayer, SliceLayer, Bottleneck, gen_mapping_layer_name, CropLayer
import numpy as np
import utils.utils as utils
import os


def init_bilinear(arr):
    weight = np.zeros(np.prod(arr.size()), dtype='float32')
    shape = arr.size()
    f = np.ceil(shape[3] / 2.)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(np.prod(shape)):
        x = i % shape[3]
        y = (i / shape[3]) % shape[2]
        weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))

    return torch.from_numpy(weight.reshape(shape))


def set_require_grad_to_false(m):
    for param in m.parameters():
        param.requires_grad = False


class SideOutput(nn.Module):
    def __init__(self, num_output, num_class, kernel_size=None, stride=None):
        super(SideOutput, self).__init__()
        self.conv = nn.Conv2d(num_output,
                              num_class,
                              1,
                              stride=1,
                              padding=0,
                              bias=True)
        if kernel_size is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(num_class,
                                                num_class,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                bias=False)
            self.upsampled.weight.data = init_bilinear(self.upsampled.weight)
            # set_require_grad_to_false(self.upsampled)
        else:
            self.upsample = False

    def forward(self, res):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)

        return side_output


class Res5Output(nn.Module):
    def __init__(self, num_class, num_output=2048, scale_factor=8):
        super(Res5Output, self).__init__()
        self.conv = nn.Conv2d(num_output, num_class, 1, stride=1, padding=0)
        self.upsampled = nn.ConvTranspose2d(num_class,
                                            num_class,
                                            kernel_size=8,
                                            padding=0,
                                            stride=8,
                                            groups=num_class,
                                            bias=False)
        self.upsampled.weight.data = init_bilinear(self.upsampled.weight)
        # set_require_grad_to_false(self.upsampled)

    def forward(self, res):
        res = self.conv(res)
        res = self.upsampled(res)
        return res


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=1,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # set_require_grad_to_false(self.bn1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, special_case=True
        )  # Notice official resnet is 2, but CASENet here is 1.

        # Added by CASENet to get feature map from each branch in different scales.
        self.score_edge_side1 = nn.Conv2d(64, 1, kernel_size=1, bias=True)
        self.score_edge_side2 = nn.Conv2d(256, 1, kernel_size=1, bias=True)
        self.upsample_edge_side2 = nn.ConvTranspose2d(1,
                                                      1,
                                                      kernel_size=4,
                                                      stride=2,
                                                      bias=False)
        # set_require_grad_to_false(self.upsample_edge_side2)

        self.score_edge_side3 = nn.Conv2d(512, 1, kernel_size=1, bias=True)
        self.upsample_edge_side3 = nn.ConvTranspose2d(1,
                                                      1,
                                                      kernel_size=8,
                                                      stride=4,
                                                      bias=False)
        # set_require_grad_to_false(self.upsample_edge_side3)

        self.score_cls_side5 = nn.Conv2d(2048,
                                         num_classes,
                                         kernel_size=1,
                                         bias=True)
        self.upsample_cls_side5 = nn.ConvTranspose2d(num_classes,
                                                     num_classes,
                                                     kernel_size=16,
                                                     stride=8,
                                                     groups=num_classes,
                                                     bias=False)
        # set_require_grad_to_false(self.upsample_cls_side5)

        self.ce_fusion = nn.Conv2d(num_classes * 4,
                                   num_classes,
                                   kernel_size=1,
                                   groups=num_classes,
                                   bias=True)

        # Define crop, concat layer
        self.crop_layer = CropLayer()
        self.slice_layer = SliceLayer()
        self.concat_layer = ConcatLayer()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Initialize ConvTranspose2D with bilinear.
        self.upsample_edge_side2.weight.data = init_bilinear(
            self.upsample_edge_side2.weight)
        self.upsample_edge_side3.weight.data = init_bilinear(
            self.upsample_edge_side3.weight)
        self.upsample_cls_side5.weight.data = init_bilinear(
            self.upsample_cls_side5.weight)

        # Initialize final conv fusion layer with constant=0.25
        self.ce_fusion.weight.data.fill_(0.25)
        self.ce_fusion.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, special_case=False):
        """
        special case only for res5a branch
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            # set_require_grad_to_false(downsample[1])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, for_vis=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # BS X 64 X 352 X 352
        score_feats1 = self.score_edge_side1(x)  # BS X 1 X 352 X 352

        x = self.maxpool(x)  # BS X 64 X 175 X 175

        x = self.layer1(x)  # BS X 256 X 175 X 175
        score_feats2 = self.score_edge_side2(x)
        upsampled_score_feats2 = self.upsample_edge_side2(score_feats2)  # BS X 1 X 352 X 352
        cropped_score_feats2 = upsampled_score_feats2  # Here don't need to crop. (In official caffe, there's a crop)

        x = self.layer2(x)  # BS X 512 X 88 X 88
        score_feats3 = self.score_edge_side3(x)  # BS X 1 X 356 X 356
        upsampled_score_feats3 = self.upsample_edge_side3(score_feats3)
        cropped_score_feats3 = self.crop_layer(upsampled_score_feats3,
                                               offset=2)  # BS X 1 X 352 X 352

        x = self.layer3(x)
        x = self.layer4(x)
        score_feats5 = self.score_cls_side5(x)
        upsampled_score_feats5 = self.upsample_cls_side5(score_feats5)
        cropped_score_feats5 = self.crop_layer(
            upsampled_score_feats5, offset=4
        )  # BS X 20 X 352 X 352. The output of it will be used to get a loss for this branch.
        sliced_list = self.slice_layer(
            cropped_score_feats5)  # Each element is BS X 1 X 352 X 352

        # Add low-level feats to sliced_list
        final_sliced_list = []
        for i in range(len(sliced_list)):
            final_sliced_list.append(sliced_list[i])
            final_sliced_list.append(score_feats1)
            final_sliced_list.append(cropped_score_feats2)
            final_sliced_list.append(cropped_score_feats3)

        concat_feats = self.concat_layer(final_sliced_list,
                                         dim=1)  # BS X 80 X 352 X 352
        fused_feats = self.ce_fusion(
            concat_feats
        )  # BS X 2 X 352 X 352. The output of this will gen loss for this branch. So, totaly 2 loss. (same loss type)

        if for_vis:
            return score_feats1, cropped_score_feats2, cropped_score_feats3, cropped_score_feats5, fused_feats
        else:
            return cropped_score_feats5, fused_feats


# def build_CaseNet(pretrained=False, num_classes=2):
#     """Constructs a modified ResNet-101 model for CASENet.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on MSCOCO
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
#     if pretrained:
#         if os.path.exists("./Datasets/resnet101.pth"):
#             checkpoint = torch.load(
#                 "./Datasets/resnet101.pth"
#             )
#             params = checkpoint
#             model.load_state_dict(params, strict=False)
#         else:
#             print("start downloading resnet101: pytorch/vision:v0.10.0 \n")
#             if not os.path.exists("./Datasets/"):
#                 os.makedirs("./Datasets/")
#             url = 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'
#             import requests
#             down_res = requests.get(url)
#             with open("./Datasets/resnet101.pth", 'wb') as file:
#                 file.write(down_res.content)
#             checkpoint = torch.load(
#                 "./Datasets/resnet101.pth"
#             )
#             params = checkpoint
#             model.load_state_dict(params, strict=False)
#
#     return model
def build_CaseNet(pretrained=False, num_classes=2):
    """Constructs a modified ResNet-50 model for CASENet.
    Args:
        pretrained (bool): If True, returns a model pre-trained on MSCOCO
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    if pretrained:
        if os.path.exists("./Datasets/resnet50.pth"):
            checkpoint = torch.load(
                "./Datasets/resnet50.pth"
            )
            params = checkpoint
            model.load_state_dict(params, strict=False)
        else:
            print("start downloading resnet50: pytorch/vision:v0.10.0 \n")
            if not os.path.exists("./Datasets/"):
                os.makedirs("./Datasets/")
            url = 'https://download.pytorch.org/models/resnext50_32x8d-8ba56ff5.pth'
            import requests
            down_res = requests.get(url)
            with open("./Datasets/resnet50.pth", 'wb') as file:
                file.write(down_res.content)
            checkpoint = torch.load(
                "./Datasets/resnet50.pth"
            )
            params = checkpoint
            model.load_state_dict(params, strict=False)

    return model


# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=2):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self._num_classes = num_classes
#         self.conv1 = nn.Conv2d(3,
#                                64,
#                                kernel_size=7,
#                                stride=1,
#                                padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

#         self.layer1 = self._make_layer(block, 64, layers[0], 2)  # res2
#         self.layer2 = self._make_layer(block, 128, layers[1], 3,
#                                        stride=2)  # res3
#         self.layer3 = self._make_layer(block, 256, layers[2], 4,
#                                        stride=2)  # res4
#         self.layer4 = self._make_layer(block, 512, layers[3], 5, stride=1)
#         # set_require_grad_to_false(self.conv1)
#         # set_require_grad_to_false(self.bn1)
#         # set_require_grad_to_false(self.layer1)
#         # set_require_grad_to_false(self.layer2)
#         # set_require_grad_to_false(self.layer3)
#         # set_require_grad_to_false(self.layer4)

#         self.SideOutput1 = SideOutput(64, num_classes)
#         self.SideOutput2 = SideOutput(256,
#                                       num_classes,
#                                       kernel_size=4,
#                                       stride=2)
#         self.SideOutput3 = SideOutput(512,
#                                       num_classes,
#                                       kernel_size=4,
#                                       stride=4)
#         self.layer5Output = Res5Output(num_classes)

#         self.slice_layer = SliceLayer()
#         self.concat_layer = ConcatLayer()
#         in_channels = int(4 * self._num_classes)
#         out_channels = self._num_classes
#         self._fuse_conv = nn.Conv2d(in_channels,
#                                     out_channels,
#                                     1,
#                                     groups=self._num_classes).cuda()

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#         self._fuse_conv.weight.data.fill_(1 / 4)
#         self._fuse_conv.bias.data.zero_()

#     def forward(self, x, for_vis=False):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)  # BS X 64 X 352 X 352
#         side_1 = self.SideOutput1(x)

#         x = self.maxpool(x)  # BS X 64 X 175 X 175

#         x = self.layer1(x)  # BS X 256 X 175 X 175
#         side_2 = self.SideOutput2(x)

#         x = self.layer2(x)  # BS X 512 X 88 X 88
#         side_3 = self.SideOutput3(x)
#         # cropped_score_feats3 = self.crop_layer(upsampled_score_feats3,
#         #                                        offset=1)  # BS X 1 X 352 X 352

#         x = self.layer3(x)
#         x = self.layer4(x)
#         side_5 = self.layer5Output
#         side_5 = side_5(x)

#         sliced_list = self.slice_layer(side_5)
#         final_sliced_list = []
#         for i in range(len(sliced_list)):
#             final_sliced_list.append(sliced_list[i])
#             final_sliced_list.append(side_1[:, i:i + 1, ...])
#             final_sliced_list.append(side_2[:, i:i + 1, ...])
#             final_sliced_list.append(side_3[:, i:i + 1, ...])

#         # for sliced in final_sliced_list:
#         #     print(sliced.shape)

#         sliced_cat = self.concat_layer(final_sliced_list, dim=1)

#         acts = self._fuse_conv(sliced_cat).cuda()
#         if for_vis:
#             return side_1, side_2, side_3, side_5, acts
#         else:
#             return side_5, acts

#     def _make_layer(self, block, planes, blocks, block_no, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes,
#                           planes * block.expansion,
#                           kernel_size=1,
#                           stride=stride,
#                           bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(
#             block(self.inplanes, planes, block_no, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, block_no))

#         return nn.Sequential(*layers)

# def build_CaseNet(num_classes=2):
#     """Constructs a modified ResNet-101 model for CASENet.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on MSCOCO
#     """

#     model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

#     checkpoint = torch.load(
#         "/media/kerwin/7717c621-eb0d-4de1-bf04-a0f0feb8f7c1/CASENet-master/resnet50.pth"
#     )
#     params = checkpoint
#     # params_ = checkpoint["state_dict"]
#     # params = OrderedDict()
#     # for key, value in params_.items():
#     #     new_key = key.replace("module.", "")
#     #     params[new_key] = value
#     model.load_state_dict(params, strict=False)
#     return model

if __name__ == "__main__":
    model = build_CaseNet(num_classes=2)