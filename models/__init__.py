from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
# from .resnet_prelu import ResNet18_prelu
from .resnetv2 import ResNet18, ResNet34
from .resnetv2_3BN import ResNet18 as ResNet18_3BN, ResNet34 as ResNet34_3BN
# from .resnetv2_density import ResNet18_density
from .resnetv2_ordered import ODResNet18
from .wrn_ordered import ODwrn_22_8
from .resnetv3 import resnet50, resnet38, resnet26
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_22_8
from .wrn_3BN import wrn_22_8 as wrn_22_8_3BN
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .custom_vgg import Custom_vgg16_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2

from .customresnetV1 import CustomResNet18, CustomResNet34

from .customresnet_3BN import CustomResNet18 as CustomResNet18_3BN, CustomResNet34 as CustomResNet34_3BN
from .customresnet_trainable_mask import CustomResNet18 as CustomResNet18_TM
# from .customresnet_prelu import CustomResNet18Prelu
# from .customresnet_plot import CustomResNet18_plot
# from .customresnet_density import CustomResNet18_density
# from .custom_wrn import Custom_wrn_22_8
from .custom_wrnV1 import Custom_wrn_22_8


from .custom_wrn_3BN import Custom_wrn_22_8 as Custom_wrn_22_8_3BN
# from .custom_resnetv2_ac import CustomResNet18_AC, CustomResNet34_AC
# from .custom_wrn_ac import Custom_wrn_22_8_AC
# from .custom_vgg_ac import Custom_vgg16_bn_AC
# from .custom_resnetv2_gb_ordered import CustomODResNet18_gb
# from .custom_wrn_gb_ordered import CustomODwrn_22_8_gb

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet18': ResNet18, 
    'ResNet18_3BN': ResNet18_3BN,
    'ODResNet18': ODResNet18,
    'ODwrn': ODwrn_22_8, 
    # 'ResNet18_density': ResNet18_density,
    'ResNet34': ResNet34, 
    'ResNet34_3BN': ResNet34_3BN, 
    'resnet26': resnet26,
    'resnet38': resnet38,
    'resnet50': resnet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_22_8': wrn_22_8,
    'wrn_22_8_3BN': wrn_22_8_3BN,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'CustomResNet18': CustomResNet18,

    # 'CustomResNet18': CustomResNet18,
    'CustomResNet34': CustomResNet34,
    'CustomResNet18_3BN': CustomResNet18_3BN,
    'CustomResNet34_3BN': CustomResNet34_3BN,
    'CustomResNet18_TM': CustomResNet18_TM,
    # 'CustomResNet18Prelu': CustomResNet18Prelu,
    # 'Customvgg16': Custom_vgg16_bn,
    # 'Customvgg16_AC': Custom_vgg16_bn_AC,
    'Custom_wrn_22_8': Custom_wrn_22_8,
    'Custom_wrn_22_8_3BN': Custom_wrn_22_8_3BN,
    # 'CustomResNet18_AC': CustomResNet18_AC,
    # 'CustomResNet34_AC': CustomResNet34_AC,
    # 'Custom_wrn_22_8_AC': Custom_wrn_22_8_AC,
    # 'CustomResNet18_plot': CustomResNet18_plot,
    # 'CustomResNet18_density': CustomResNet18_density,
    # 'CustomODResNet18_gb': CustomODResNet18_gb,
    # 'CustomOD_wrn_22_8_gb': CustomODwrn_22_8_gb
}



model_dict2 = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet18': ResNet18,
    'ResNet18_3BN': ResNet18_3BN,
    'ODResNet18': ODResNet18,
    'ODwrn': ODwrn_22_8,
    # 'ResNet18_density': ResNet18_density,
    'ResNet34': ResNet34,
    'ResNet34_3BN': ResNet34_3BN,
    'resnet26': resnet26,
    'resnet38': resnet38,
    'resnet50': resnet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_22_8': wrn_22_8,
    'wrn_22_8_3BN': wrn_22_8_3BN,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'CustomResNet18': CustomResNet18,
    # 'CustomResNet18': CustomResNet18,
    'CustomResNet34': CustomResNet34,
    'CustomResNet18_3BN': CustomResNet18_3BN,
    'CustomResNet34_3BN': CustomResNet34_3BN,
    'CustomResNet18_TM': CustomResNet18_TM,
    # 'CustomResNet18Prelu': CustomResNet18Prelu,
    # 'Customvgg16': Custom_vgg16_bn,
    # 'Customvgg16_AC': Custom_vgg16_bn_AC,
    'Custom_wrn_22_8': Custom_wrn_22_8,
    'Custom_wrn_22_8_3BN': Custom_wrn_22_8_3BN,
    # 'CustomResNet18_AC': CustomResNet18_AC,
    # 'CustomResNet34_AC': CustomResNet34_AC,
    # 'Custom_wrn_22_8_AC': Custom_wrn_22_8_AC,
    # 'CustomResNet18_plot': CustomResNet18_plot,
    # 'CustomResNet18_density': CustomResNet18_density,
    # 'CustomODResNet18_gb': CustomODResNet18_gb,
    # 'CustomOD_wrn_22_8_gb': CustomODwrn_22_8_gb
}