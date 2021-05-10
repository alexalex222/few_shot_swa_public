from .convnet import convnet4
from .resnet import resnet12
from .wresnet import wrn_28_10


model_pool = [
    'convnet4',
    'resnet12',
    'wrn_28_10',
]

model_dict = {
    'wrn_28_10': wrn_28_10,
    'convnet4': convnet4,
    'resnet12': resnet12,
}