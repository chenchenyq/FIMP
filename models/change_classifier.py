import time
import torchvision
from .layers import Up, TemporalInteractionBlock, FourierUnit, Classifier, Concat, Sub, Add
from torch.nn import Module, ModuleList
import torch
from .FusionUNet import FuseModule


class ChangeClassifier(Module):
    def __init__(
        self,
        num_classes,
        num,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="4",
        freeze_backbone=False,
    ):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self.backbone = get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )

        self.tice = ModuleList(
            [
                TemporalInteractionBlock(24, 14),
                TemporalInteractionBlock(32, 28),
                TemporalInteractionBlock(56, 56),
                TemporalInteractionBlock(112, 112),
            ]
        )

        self.fourier = ModuleList(
            [
                FourierUnit(24, 24),
                FourierUnit(32, 32),
                FourierUnit(56, 56),
                FourierUnit(112, 112),
            ]
        )

        self.fuse = FuseModule(14, num)

        self.up = ModuleList(
            [
                Up(112, 56),
                Up(56, 28),
                Up(28, 14),
                Up(14, 7),
            ]
        )

        # Final classification layer:
        self.classify = Classifier(7, num_classes)

    def forward(self, x1, x2):
        # forward backbone resnet
        features_1, features_2 = self.encode(x1, x2)
        # feature filter
        afx1, afx2 = self.frequency(features_1, features_2)
        # feature interaction and fuse
        diff = self.interaction(afx1, afx2)
        d = self.fuse(diff[0], diff[1], diff[2], diff[3])
        latents = self.decoder(d)
        pred = self.classify(latents)
        return pred

    def interaction(self, x1, x2):
        d = []
        for i in range(0, 4):
            x = self.tice[i](x1[i], x2[i])
            d.append(x)
        return d

    def frequency(self, x1, x2):
        afx1 = []
        afx2 = []
        # mp = []
        for i in range(0, 4):
            x_1, x_2= self.fourier[i](x1[i], x2[i])
            afx1.append(x_1)
            afx2.append(x_2)
        return afx1, afx2

    def encode(self, x1, x2):
        x1_downsample = []
        x2_downsample = []
        for num, layer in enumerate(self.backbone):
            x1 = layer(x1)
            x2 = layer(x2)
            if num != 0:
                x1_downsample.append(x1)
                x2_downsample.append(x2)

        return x1_downsample, x2_downsample

    def decoder(self, features):
        x = self.up[0](features[3])
        x = self.up[1](x, features[2])
        x = self.up[2](x, features[1])
        x = self.up[3](x, features[0])
        return x


def get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone):
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(pretrained=pretrained).features

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 256, 256).cuda().float()
    x2 = torch.randn(1, 3, 256, 256).cuda().float()
    net1 = ChangeClassifier(num_classes=2, num=3).cuda()
    s1 = net1(x1, x2)
    print(s1.shape)

    # from fvcore.nn import FlopCountAnalysis
    # flops = FlopCountAnalysis(net1, (x1, x2))
    # total = sum([param.nelement() for param in net1.parameters()])
    # print("Params_Num: %.2fM" % (total/1e6))
    # print(flops.total()/1e9)
    with torch.no_grad():
        for _ in range(10):
            _ = net1(x1, x2)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = net1(x1, x2)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / 100
    print(f"平均推理时间：{avg_inference_time}秒")