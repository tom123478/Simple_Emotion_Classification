import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet18_Weights,ResNet34_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16  # 使用预训练的ViT模型


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_vgg19_pretrained(num_classes=1000):
    # 加载预训练的VGG-19模型
    model = models.vgg19(pretrained=True)
    # 替换最后的分类器层以匹配目标类别数
    num_features = model.classifier[6].in_features  # 获取原始分类层的输入特征数
    model.classifier[6] = nn.Linear(num_features, num_classes)  # 替换为新的分类器 
    return model

class ConvNeXtForClassification(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNeXtForClassification, self).__init__()
        # 加载预训练的ConvNeXt模型
        self.convnext = models.convnext_small(pretrained=True)
        
        # 修改最后的分类层
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, num_classes)
    
    def forward(self, x):
        x = self.convnext(x)
        return x

class EfficientNetForEmotionClassification(nn.Module):
    def __init__(self, num_classes=7, version='b0'):
        super(EfficientNetForEmotionClassification, self).__init__()
        # 根据版本加载预训练的EfficientNet模型
        model_versions = {
            'b0': models.efficientnet_b0,
            'b1': models.efficientnet_b1,
            'b2': models.efficientnet_b2,
            'b3': models.efficientnet_b3,
            'b4': models.efficientnet_b4,
            'b5': models.efficientnet_b5,
            'b6': models.efficientnet_b6,
            'b7': models.efficientnet_b7
        }
        self.efficientnet = model_versions[version](pretrained=True)
        # 获取预训练模型的分类器的输入特征数量
        num_ftrs = self.efficientnet.classifier[1].in_features
        # 替换分类器以适应新的类别数
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        x = self.efficientnet(x)
        return x

class vitEmotionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(vitEmotionClassifier, self).__init__()
        # 加载预训练的ViT模型
        self.vit = vit_b_16(pretrained=True)
        # 替换ViT模型的分类器头部（原先针对ImageNet是1000类）
        self.vit.heads = nn.Linear(self.vit.heads.in_features, num_classes)

    def forward(self, x):
        # 使用ViT模型进行特征提取
        x = self.vit(x)
        return x

class resnetEmotionResNetClassifier(nn.Module):
    def __init__(self, num_classes=7, resnet_model='resnet18'):
        super(resnetEmotionResNetClassifier, self).__init__()
        # 选择并加载预训练的ResNet模型
        if resnet_model == 'resnet18':
            self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif resnet_model == 'resnet34':
            self.resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif resnet_model == 'resnet50':
            self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif resnet_model == 'resnet101':
            self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        elif resnet_model == 'resnet152':
            self.resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported ResNet model type. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")

        # 替换ResNet模型的全连接层以适应情绪分类（7类）
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # 使用ResNet模型进行特征提取
        x = self.resnet(x)
        return x
    
class mobilenetEmotionMobileNetClassifier(nn.Module):
    def __init__(self, num_classes=7, mobilenet_version='v2'):
        super(mobilenetEmotionMobileNetClassifier, self).__init__()
        # 根据选择加载预训练的MobileNet模型
        if mobilenet_version == 'v1':
            self.mobilenet = models.mobilenet_v2(pretrained=True)  # torchvision暂无v1版本，使用v2作为替代
        elif mobilenet_version == 'v2':
            self.mobilenet = models.mobilenet_v2(pretrained=True)
        elif mobilenet_version == 'v3_large':
            self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        elif mobilenet_version == 'v3_small':
            self.mobilenet = models.mobilenet_v3_small(pretrained=True)
        else:
            raise ValueError("Unsupported MobileNet version. Choose from 'v1' (using v2 as a substitute), 'v2', 'v3_large', 'v3_small'.")

        # 替换MobileNet的分类器部分以适应情绪分类（7类）
        # MobileNet V2 和 V3 使用不同的分类器结构，因此需要进行相应的修改
        if mobilenet_version in ['v1', 'v2']:
            self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, num_classes)
        else:  # 对于V3的两个版本
            self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_classes)

    def forward(self, x):
        # 使用MobileNet模型进行特征提取
        x = self.mobilenet(x)
        return x
