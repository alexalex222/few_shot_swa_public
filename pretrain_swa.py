import os
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn, SWALR

from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.cub import CUB, MetaCUB
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, parse_option_pretrain, train, train_mix_precision, train_mix_up
from eval.cls_eval import validate
from eval.meta_eval import meta_test


def main():

    opt = parse_option_pretrain()

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
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        # val_loader = DataLoader(TieredImageNet(args=opt, partition='train', transform=test_trans),
        #                        batch_size=opt.batch_size, shuffle=False, drop_last=False,
        #                        num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans,
                                                       fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
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
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
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
        val_loader = DataLoader(CUB(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers)

        meta_valloader = DataLoader(MetaCUB(args=opt, partition='val',
                                            train_transform=train_trans,
                                            test_transform=test_trans,
                                            fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = create_model(opt.model, n_cls, opt.dataset)

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    swa_model = AveragedModel(model)
    if opt.swa_start >= opt.epochs:
        swa_start = int(opt.epochs * 0.8)
    else:
        swa_start = opt.swa_start
    swa_scheduler = SWALR(optimizer, swa_lr=opt.swa_lr)

    # tensorboard
    logger = SummaryWriter(log_dir=opt.tb_folder)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    scaler = GradScaler()

    # routine: supervised pre-training
    best_acc = 0.0
    few_shot_acc = 0.0
    if opt.resume:
        checkpoint_file = os.path.join(opt.save_folder, opt.resume_file)
        assert os.path.isfile(checkpoint_file), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(checkpoint_file)
        opt.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file, checkpoint['epoch']))

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print("==> training...")
        time1 = time.time()
        if opt.dataset == 'CUB':
            train_acc, train_loss = train_mix_precision(epoch, train_loader, model, criterion, optimizer, scaler, opt)
        elif opt.model == 'wrn_28_10':
            train_acc, train_loss = train_mix_precision(epoch, train_loader, model, criterion, optimizer, scaler, opt)
        else:
            if opt.mix_up:
                train_acc, train_loss = train_mix_up(epoch, train_loader, model, criterion, optimizer, opt)
            else:
                train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % 20 == 0:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_{}.pth'.format(epoch))
            torch.save(state, save_file)

        if epoch <= swa_start:
            if opt.cosine:
                scheduler.step()
            else:
                adjust_learning_rate(epoch, opt, optimizer)
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        logger.add_scalar('train_acc', train_acc, epoch)
        logger.add_scalar('train_loss', train_loss, epoch)

        if epoch <= swa_start:
            # test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
            pass
        else:
            update_bn(train_loader, swa_model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # test_acc, test_acc_top5, test_loss = validate(val_loader, swa_model, criterion, opt)
            (few_shot_acc, few_shot_std), _, _, _ = meta_test(swa_model, meta_valloader, use_logit=False)

        logger.add_scalar('few_shot_acc', few_shot_acc, epoch)
        # logger.add_scalar('test_acc_top5', test_acc_top5, epoch)
        # logger.add_scalar('test_loss', test_loss, epoch)

        # regular saving
        if epoch >= swa_start and few_shot_acc > best_acc:
            best_acc = few_shot_acc
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': swa_model.state_dict(),
                'train_acc': train_acc,
                'train_loss': train_loss,
                'few_shot_acc': few_shot_acc,
                'few_shot_std': few_shot_std
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_swa.pth')
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': swa_model.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_swa_last.pth'.format(opt.model))
    torch.save(state, save_file)

    print('best few-shot acc:', best_acc)


if __name__ == '__main__':
    main()
