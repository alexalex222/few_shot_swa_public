from .convnet import convnet4
from .resnet_db import resnet12_db
from .resnet_db import seresnet12_db
from .wresnet import wrn_28_10


model_pool = [
    'convnet4',
    'resnet12_db',
    'seresnet12_db',
    'wrn_28_10',
]

model_dict = {
    'wrn_28_10': wrn_28_10,
    'convnet4': convnet4,
    'resnet12_db': resnet12_db,
    'seresnet12_db': seresnet12_db,
}