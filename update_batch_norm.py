import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from torchvision.datasets import ImageFolder
import torch.multiprocessing
from torch.cuda.amp import autocast
from tqdm import tqdm
from models.util import create_model

from dataset.mini_imagenet import ImageNet
from dataset.tiered_imagenet import TieredImageNet
from dataset.cifar import CIFAR100
from dataset.cub import CUB
from dataset.transform_cfg import transforms_options

from util import parse_option_eval


torch.multiprocessing.set_sharing_strategy('file_system')


def update_bn_float16(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for idx, data in tqdm(enumerate(loader), total=len(loader)):
        if isinstance(data, (list, tuple)):
            data = data[0]
        if device is not None:
            data = data.to(device)
        with autocast():
            model(data)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def main():

    opt = parse_option_eval()

    # dataloader
    if opt.use_trainval:
        train_partition = 'trainval'
    elif opt.use_all:
        train_partition = 'all'
    else:
        train_partition = 'train'

    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 80
        elif opt.use_all:
            n_cls = 100
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']

        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    elif opt.dataset == 'CUB':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(CUB(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        n_cls = 100
    elif opt.dataset == 'miniImageNet2CUB':
        train_trans, test_trans = transforms_options['F']
        image_dataset = ImageFolder(root=opt.data_root, transform=train_trans)
        train_loader = DataLoader(image_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    swa_model = AveragedModel(create_model(opt.model, n_cls, opt.dataset))
    if torch.cuda.is_available():
        swa_model = swa_model.cuda()
        cudnn.benchmark = True
    ckpt = torch.load(opt.model_path)
    swa_model.load_state_dict(ckpt['model'])

    # update batch normalization in a forward pass
    update_bn_float16(train_loader, swa_model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # save the last model
    state = {
        'opt': opt,
        'model': swa_model.state_dict(),
    }

    save_file = os.path.join(os.path.dirname(opt.model_path), '{}_swa_last.pth'.format(opt.model))
    torch.save(state, save_file)

    print('Batch norm statistics is updated')


if __name__ == '__main__':
    main()
