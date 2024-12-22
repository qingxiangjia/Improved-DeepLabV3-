import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.backbone.xception import xception
from nets.backbone.mobilenetv2 import mobilenetv2
from nets.backbone.mobilenetv3 import mobilenet_v3
from efficientnet_pytorch import EfficientNet
from nets.backbone.microsoft_swintransformer import SwinTransformer
from torchvision import models
import torch.nn as nn
from nets.Attention.CBAM import CBAMBlock
from nets.Attention.SE import SEAttention
from nets.Attention.CAA import CAA
from nets.Attention.ECA import EfficientChannelAttention
from nets.Attention.CPCA import CPCA
from nets.Attention.TripletAttention import TripletAttention
from nets.Attention.ShuffleAttention import ShuffleAttention
from nets.Attention.EMCAD import EMCAM

class AttentionFactory:
    @staticmethod
    def get_attention(attention_type, in_planes, **kwargs):
        print("*********000001")
        print(attention_type)
        print(in_planes)
        if attention_type == "cbam":
            return CBAMBlock(in_planes, **kwargs)
        elif attention_type == "se":
            return SEAttention(in_planes, **kwargs)
        elif attention_type == "caa":
            return CAA(in_planes, **kwargs)
        elif attention_type == "eca":
            return EfficientChannelAttention(in_planes, **kwargs)
        elif attention_type == "cpca":
            return CPCA(in_planes, **kwargs)
        elif attention_type == "ta":
            return TripletAttention(in_planes, **kwargs)
        elif attention_type == "sa":
            return ShuffleAttention(in_planes, **kwargs)
        elif attention_type == "emcam":
            return EMCAM(in_planes, in_planes,**kwargs)
        else:
            return nn.Identity()  # 默认无注意力机制




class VGG16Backbone(nn.Module):
    def __init__(self, pretrained=True, downsample_factor=16):
        super(VGG16Backbone, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())

        # 根据下采样倍数，选择特征层的截取点
        if downsample_factor == 8:
            self.low_level_features = nn.Sequential(*features[:16])  # 获取前16层作为低特征层
            self.high_level_features = nn.Sequential(*features[16:23])  # 获取接下来几层作为高特征层
        elif downsample_factor == 16:
            self.low_level_features = nn.Sequential(*features[:10])  # 获取前10层作为低特征层
            self.high_level_features = nn.Sequential(*features[10:23])  # 获取接下来几层作为高特征层
        else:
            raise ValueError('Unsupported downsample factor - `{}`, Use 8 or 16.'.format(downsample_factor))

    def forward(self, x):
        low_level_features = self.low_level_features(x)
        x = self.high_level_features(low_level_features)
        return low_level_features, x





class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='efficientnet-b0', pretrained=True, downsample_factor=16):
        super(EfficientNetBackbone, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        self.downsample_factor = downsample_factor

        # 根据下采样倍数选择不同的 reduction 层
        if downsample_factor == 8:
            self.low_level_layer = 'reduction_2'  # 选择 reduction_2 层
            self.high_level_layer = 'reduction_4'  # 选择 reduction_4 层
        elif downsample_factor == 16:
            self.low_level_layer = 'reduction_1'  # 选择 reduction_1 层
            self.high_level_layer = 'reduction_5'  # 选择 reduction_5 层
        else:
            raise ValueError('Unsupported downsample factor - `{}`, Use 8 or 16.'.format(downsample_factor))

    def forward(self, x):
        # 获取EfficientNet的所有特征层
        endpoints = self.model.extract_endpoints(x)
        # 选择合适的层作为 low_level_features 和 x
        low_level_features = endpoints[self.low_level_layer]  # 选取较浅的特征层
        x = endpoints[self.high_level_layer]  # 选取较深的特征层

        return low_level_features, x


class SwinTransformerBackbone(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True, downsample_factor=16):
        super(SwinTransformerBackbone, self).__init__()
        self.model = SwinTransformer(model_name=model_name, pretrained=pretrained)
        self.downsample_factor = downsample_factor

        if downsample_factor == 8:
            self.low_level_layer_idx = 1  # 假设stage1在索引1处
            self.high_level_layer_idx = 3  # 假设stage3在索引3处
        elif downsample_factor == 16:
            self.low_level_layer_idx = 2  # 假设stage2在索引2处
            self.high_level_layer_idx = 4  # 假设stage4在索引4处
        else:
            raise ValueError('Unsupported downsample factor - `{}`, Use 8 or 16.'.format(downsample_factor))

    def forward(self, x):
        features = self.model.forward_features(x)

        # 打印特征的结构和长度
        #print(f"Number of features: {len(features)}")
        #for idx, feature in enumerate(features):
            #print(f"Feature {idx}: {feature.shape}")

        if len(features) > self.low_level_layer_idx:
            low_level_features = features[self.low_level_layer_idx]
        else:
            #print(f"Low level layer index {self.low_level_layer_idx} is out of range.")
            low_level_features = features[0]  # 使用第一个特征层作为默认值

        if len(features) > self.high_level_layer_idx:
            x = features[self.high_level_layer_idx]
        else:
            #print(f"High level layer index {self.high_level_layer_idx} is out of range.")
            x = features[-1]  # 使用最后一个特征层作为默认值

        return low_level_features, x



class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]
        #total_idx是18层
        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:   #i从7到14
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx): #i从14到18
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
    #输入的x是2*3*512*512 输出的low_level_features是2*24*128*128  middle_level_features*96*64*64  高层特征x是2*320*64*64
    def forward(self, x):
        # 将输入 x 传入 features 的前 4 层，提取浅层特征。  浅层特征保留了更多的细节信息，如边缘和纹理信息，非常适合图像分割任务中用于对象的边缘识别和细节保留。
        low_level_features = self.features[:4](x) # 前 4 层提取浅层特征
        middle_level_features = self.features[4:12](low_level_features) ## 从第 4 层到第 8 层提取中层特征
        #将浅层特征传入剩下的层，提取深层特征。 层特征通常捕捉高层次的抽象信息，如整体结构和语义信息。
        x = self.features[12:](middle_level_features) # 继续从第 8 层到最后提取深层特征
        return low_level_features, middle_level_features, x


class MobileNetV3(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV3, self).__init__()
        from functools import partial

        model = mobilenet_v3(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )



    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)


        #mid_level_features = self.features[4:6](low_level_features)

        # = F.max_pool2d(mid_level_features, 2)

        x = self.features[4:](low_level_features)

        return low_level_features,  x






class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()  #输入 x：输入的特征图，其形状为 [b, c, row, col]，分别表示批量大小、通道数、高度、宽度
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        #使用 F.interpolate 对全局特征图进行双线性插值，恢复到输入的空间大小。这使得全局特征与其他分支的特征图大小一致，可以进行拼接。 global_feature维度是2*256*64*64
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。 feature_cat维度是2*1280*64*64  result维度是2*256*64*64
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result  #返回 ASPP 模块融合后的特征图，这些特征包含不同感受野的信息


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv3", pretrained=True, downsample_factor=16,attention_low=None, attention_middle=None, attention_high=None, attention_aspp=None):
        super(DeepLab, self).__init__()

        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif "mobilenetv2"in backbone:
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320 #高层特征通道数
            aspp_channels = 256 #ASPP模块输出通道数
            middle_level_channels = 96 #中层特征通道数
            low_level_channels = 24 #底层特征通道数
        elif "mobilenetv3"in backbone:
            self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 160
            # mid_level_channels = 40
            low_level_channels = 24

        elif 'efficientnet'in backbone:
            self.backbone = EfficientNetBackbone(model_name=backbone, pretrained=pretrained,
                                                 downsample_factor=downsample_factor)
            in_channels = 320 if downsample_factor == 16 else 112  # EfficientNet-B0中 reduction_5 和 reduction_4 的输出通道数
            low_level_channels = 16 if downsample_factor == 16 else 24  # EfficientNet-B0中 reduction_1 和 reduction_2 的输出通道数
        elif "swintransformer" in backbone:
            self.backbone = SwinTransformerBackbone(model_name=backbone, pretrained=pretrained,
                                                    downsample_factor=downsample_factor)
            in_channels = 768  # 根据模型的高特征层输出维度设置
            low_level_channels = 192  # 根据模型的低特征层输出维度设置

        elif  "vgg16"in backbone:
            self.backbone = VGG16Backbone(pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 512  # VGG16高特征层的输出通道数
            low_level_channels = 128  # VGG16低特征层的输出通道数
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #用于低层和高层特征的注意力机制增强，通过 AttentionFactory.get_attention 根据指定的 attention_type 创建对应的注意力模块
        self.attention_low = AttentionFactory.get_attention(attention_low, low_level_channels)
        self.attention_middle = AttentionFactory.get_attention(attention_middle, middle_level_channels)
        self.attention_high = AttentionFactory.get_attention(attention_high, in_channels)
        self.attention_aspp = AttentionFactory.get_attention(attention_aspp, aspp_channels)

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取，用于多尺度特征提取，通过不同的膨胀率来获取不同感受野的特征。
        #   dim_in 是高层特征的输入通道数，dim_out 是输出通道数
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征边  用于对低层特征进行卷积操作，将通道数转换为 48，使得后续拼接特征时更方便
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 高层和中层特征融合
        self.cat_conv0 = nn.Sequential(
            nn.Conv2d(96 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        #特征融合卷积，将低层特征与经过 ASPP 处理后的高层特征拼接后，进行卷积操作，进一步融合特征。
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        #分类卷积，用于生成最终的分割预测，输出通道数等于类别数量 num_classes
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        self.emcad1=EMCAM(in_channels=in_channels,out_channels=in_channels)
        self.emcad2=EMCAM(in_channels=low_level_channels,out_channels=low_level_channels)

    def forward(self, x):
        #保存输入图像的高度和宽度，以便在最后对输出进行恢复  H,W都是512*512  x是batchsize*3*512*512
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 使用主干网络提取低层特征，浅层特征-进行卷积处理 2*24*128*128
        #   x : 主干部分-利用ASPP结构进行加强特征提取 2*320*64*64
        #   middle_level_features 维度是2*96*64*64
        #   low_level_features 是用于细节保留的浅层特征，而 x 是经过深度卷积提取的高层特征，包含更丰富的上下文信息。
        # -----------------------------------------#
        low_level_features, middle_level_features, x = self.backbone(x)
        #对低层特征应用注意力机制，以增强对重要特征的关注
        # low_level_features = self.emcad2(low_level_features)
        #low_level_features是2*24*128*128
        low_level_features = self.attention_low(low_level_features)
        middle_level_features = self.attention_middle(middle_level_features)


        #对高层特征应用注意力机制
        # x = self.attention_high(x)
        #x = self.emcad1(x)


        #通过 ASPP 模块对高层特征进行多尺度特征提取，以获得更多的上下文信息 进入aspp之前的x维度是2*320*64*64  aspp输出的x是2*256*64*64
        x = self.aspp(x)
        # x = self.emcad1(x)
        x = self.attention_aspp(x)  #我将高层特征的注意力移动到了aspp模块之后



        #对低层特征进行卷积操作，减少通道数，使得后续融合更有效 卷积之前low_level_features2*24*128*128 卷积之后2*48*128*128
        low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样 将高层特征上采样到与低层特征相同的尺寸
        #   与浅层特征堆叠后利用卷积进行特征提取  上采样之前x是2*256*64*64   设置目标高度和宽度与 low_level_features 的高度和宽度一致 上采样之后x是2*256*128*128
        #'bilinear' 表示使用 双线性插值 方法，它是一种常见的用于图像上采样的插值方式。 双线性插值的效果通常比最近邻插值更加平滑，适合处理图像特征图的上采样。
        # -----------------------------------------#
        x = F.interpolate(x, size=(middle_level_features.size(2), middle_level_features.size(3)), mode='bilinear',align_corners=True)
        x = self.cat_conv0(torch.cat((x, middle_level_features), dim=1))  #高层和中层特征融合

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',align_corners=True)

        # 特征融合,将高层特征和低层特征在通道维度上拼接，使用卷积层对拼接后的特征进行融合
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))

        #分类卷积,通过一个 1x1 卷积生成最终的分割结果，每个像素点预测属于哪个类别
        x = self.cls_conv(x)
        #将特征上采样到输入图像的原始尺寸，以得到分割后的输出
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x