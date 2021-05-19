import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np

from models import model_pool
from dataset.transform_cfg import transforms_options, transforms_list


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight,
                                              size_average=size_average,
                                              reduce=reduce,
                                              reduction=reduction,
                                              pos_weight=pos_weight)

    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mixup_data(x: np.ndarray,
               y: np.ndarray,
               alpha=0.2,
               repeat=0):
    if alpha < 0:
        ValueError('alpha must be greater than 0')
    batch_size, feature_size = x.shape
    num_class = y.shape[-1]
    origin_index = np.tile(np.arange(batch_size), repeat)
    random_index = np.tile(np.random.permutation(batch_size), repeat)
    np.random.shuffle(random_index)
    lam = np.random.beta(alpha, alpha, size=(len(random_index), 1))
    lam_x = np.tile(lam, (1, feature_size))
    lam_y = np.tile(lam, (1, num_class))

    mixed_x = lam_x * x[origin_index, :] + (1 - lam_x) * x[random_index, :]
    mixed_y = lam_y * y[origin_index, :] + (1 - lam_y) * y[random_index, :]
    mixed_x = np.concatenate((x, mixed_x), axis=0)
    mixed_y = np.concatenate((y, mixed_y), axis=0)
    return mixed_x, mixed_y


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def preprocess_zca(train_x, test_x, min_divisor=1e-8, zca_bias=0.0001, return_weights=False):
    orig_train_shape = train_x.shape
    orig_test_shape = test_x.shape

    train_x = np.ascontiguousarray(train_x, dtype=np.float32).reshape(train_x.shape[0], -1).astype('float64')
    test_x = np.ascontiguousarray(test_x, dtype=np.float32).reshape(test_x.shape[0], -1).astype('float64')

    n_train = train_x.shape[0]

    # Zero mean every feature
    train_x = train_x - np.mean(train_x, axis=1)[:, np.newaxis]
    test_x = test_x - np.mean(test_x, axis=1)[:, np.newaxis]

    # Normalize
    train_norms = np.linalg.norm(train_x, axis=1)
    test_norms = np.linalg.norm(test_x, axis=1)

    # Make features unit norm
    train_x = train_x/train_norms[:, np.newaxis]
    test_x = test_x/test_norms[:, np.newaxis]

    train_cov_mat = 1.0/n_train * train_x.T.dot(train_x)

    (E, V) = np.linalg.eig(train_cov_mat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_zca = V.dot(inv_sqrt_zca_eigs).dot(V.T)

    train_x = train_x.dot(global_zca)
    test_x = test_x.dot(global_zca)
    if return_weights:
        return (train_x.reshape(orig_train_shape).astype('float64'),
                test_x.reshape(orig_test_shape).astype('float64')), \
               global_zca
    else:
        return train_x.reshape(orig_train_shape).astype('float64'), test_x.reshape(orig_test_shape).astype('float64')


def parse_option_pretrain():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--swa', action='store_true', help='stochastic weight average')
    parser.add_argument('--swa_lr', type=float, default=0.02, help='SWA learning rate')
    parser.add_argument('--swa_start', type=int, default=100, help='the number of epoch to start swa')
    parser.add_argument('--resume', action='store_true', help='resume training from a checkpoint')
    parser.add_argument('--start_epoch', type=int, default=1, help='the start epoch number')
    parser.add_argument('--resume_file', type=str, default='', help='path to saved model to resume')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet',
                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--mix_up', action='store_true', help='use mix up training samples')
    parser.add_argument('--mix_up_alpha', type=float, default=1.0, help='hyper-parameter in beta distribution in mix up')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--update_bn', action='store_true', help='update bn layers')
    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='./pretrained_models', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')
    parser.add_argument('--ckpt_name', type=str, default='ckpt_100.pth', help='checkpoint name')
    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed')
    parser.add_argument('--reg', type=float, default=1.0, help='regularization coefficient')
    parser.add_argument('--temp', type=float, default=1.0, help='temperature scale factor')

    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

    opt = parser.parse_args()

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './pretrained_models'

    time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    opt.tb_path = os.path.join(opt.tb_path, time_string)

    if not opt.data_root:
        opt.data_root = json.load(open('config.json'))[opt.dataset]

    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.transform)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    if opt.swa:
        opt.model_name = '{}_swa'.format(opt.model_name)

    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()

    return opt


def parse_option_eval():

    parser = argparse.ArgumentParser('argument for few-shot evaluation')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--data_root', type=str, default='', metavar='N',
                        help='Root dataset')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--cross', action='store_true', help='cross domain')
    parser.add_argument('--classifier', type=str, default='LR', help='classifier type')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of train batch)')
    parser.add_argument('--use_all', action='store_true', help='use all dataset for cross domain few-shot learning')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--reg', type=float, default=1.0, help='regularization coefficient')
    parser.add_argument('--temp', type=float, default=1.0, help='temperature scale factor')
    parser.add_argument('--swa', action='store_true', help='use swa model')

    opt = parser.parse_args()

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    opt.data_root = json.load(open('config.json'))[opt.dataset]
    opt.data_aug = True

    return opt


def train(epoch, train_loader, model, criterion, optimizer, opt):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    if opt.dataset == 'miniImageNet2CUB':
        for idx, (x, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            x = x.float()
            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()

            # ===================forward=====================
            output = model(x)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            # tensorboard logger
            pass

            # print info
            if idx % opt.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()
    else:
        for idx, (x, target, _) in enumerate(train_loader):
            data_time.update(time.time() - end)

            x = x.float()
            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()

            # ===================forward=====================
            output = model(x)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            # ===================backward=====================
            # use following methods to zero gradient for better performance
            # for param in model.parameters():
            #    param.grad = None
            # Pytorch 1.7+
            # optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            # tensorboard logger
            pass

            # print info
            if idx % opt.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, idx, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_mix_precision(epoch, train_loader, model, criterion, optimizer, scaler, opt):
    """One epoch training using mix precision"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (x, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            x = x.cuda()
            target = target.cuda()

        # ===================forward=====================
        with autocast():
            output = model(x)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_mix_up(epoch, train_loader, model, criterion, optimizer, opt):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total = 0
    correct = 0
    top1 = 0.0

    end = time.time()
    for idx, (x, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x = x.float()

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        lam = np.random.beta(opt.mix_up_alpha, opt.mix_up_alpha)
        x_mix_up = lam * x + (1 - lam) * x[index, :]
        target_a, target_b = target, target[index]
        if torch.cuda.is_available():
            x_mix_up = x_mix_up.cuda()
            target_a = target_a.cuda()
            target_b = target_b.cuda()

        # ===================forward=====================
        output = model(x_mix_up)
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())

        top1 = correct / total * 100.0
        losses.update(loss.item(), x.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1:.3f}'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    print(' * Acc@1 {top1:.3f} '
          .format(top1=top1))

    return top1, losses.avg


def _check_param_device(param, old_param_device):
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


def parameters_to_vector(parameters):
    r"""Convert parameters to one vector
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec, parameters):
    r"""Convert one vector to the parameters
    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
