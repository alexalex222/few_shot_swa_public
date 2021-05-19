import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
import matplotlib
from matplotlib import pyplot as plt

from models.util import create_model

from dataset.mini_imagenet import MetaImageNet
from dataset.tiered_imagenet import MetaTieredImageNet
from dataset.cifar import MetaCIFAR100
from dataset.transform_cfg import transforms_options

from eval.meta_eval import meta_test
from eval.util import plot_calibration_error
from eval.util import calc_calibration_score
from util import parse_option_eval


# plt.style.use('ggplot')
matplotlib.rcParams['figure.dpi'] = 200
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    opt = parse_option_eval()

    print('seed: ', opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.dataset == 'miniImageNet':
        #  29 30 35 36
        train_trans, test_trans = transforms_options['A']
        meta_val_loader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=1, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options['A']
        meta_val_loader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=1, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']
        meta_val_loader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=1, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)

        if opt.dataset == 'CIFAR-FS':
            n_cls = 64
        elif opt.dataset == 'FC100':
            n_cls = 60
        else:
            raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))

    else:
        raise NotImplementedError(opt.dataset)

    # load model
    model = create_model(opt.model, n_cls, opt.dataset)
    model_swa = AveragedModel(model)
    ckpt = torch.load(opt.model_path)

    if torch.cuda.is_available():
        cudnn.benchmark = True
        if opt.swa:
            model_swa = model_swa.cuda()
        else:
            model = model.cuda()

    # reg_pool = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20, 50, 100, 200, 500, 1000, 2000]
    reg_pool = [1.0]
    best_reg = 0.01
    best_acc = 0.0
    for reg_coeff in reg_pool:
        opt.reg = reg_coeff
        if opt.swa:
            model_swa.load_state_dict(ckpt['model'])
            (test_acc, test_std), y_pred, y_prob, y_true, _, _ = meta_test(model_swa, meta_val_loader,
                                                                     classifier=opt.classifier, opt=opt)
        else:
            model.load_state_dict(ckpt['model'])
            (test_acc, test_std), y_pred, y_prob, y_true, _, _ = meta_test(model, meta_val_loader,
                                                                     classifier=opt.classifier, opt=opt)

        print('Reg coeff: {0}, acc: {1}'.format(reg_coeff, test_acc))
        if test_acc >= best_acc:
            best_acc = test_acc
            best_reg = reg_coeff

    print('Best reg coefficient: ', best_reg)

    opt.reg = best_reg
    opt.reg = 1.0
    current_temperature = 0.1
    best_temperature = 0.1
    ece_all = []
    mce_all = []
    bri_all = []
    ece_best = float('inf')
    mce_best = float('inf')
    bri_best = float('inf')
    while current_temperature < 4.0:
        opt.temp = current_temperature
        if opt.swa:
            model_swa.load_state_dict(ckpt['model'])
            (test_acc, test_std), y_pred, y_prob, y_true, _, _ = meta_test(model_swa, meta_val_loader,
                                                                           classifier=opt.classifier, opt=opt)
        else:
            model.load_state_dict(ckpt['model'])
            (test_acc, test_std), y_pred, y_prob, y_true, _, _ = meta_test(model, meta_val_loader,
                                                                           classifier=opt.classifier, opt=opt)

        y_pred = np.concatenate(y_pred, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        y_prob = torch.tensor(y_prob, dtype=torch.float)
        y_true = torch.tensor(y_true, dtype=torch.long)

        ece, mce, brier_score = calc_calibration_score(y_prob, y_true, n_bins=20)
        print('ECE: {0}, MCE: {1}, BRI: {2}'.format(ece, mce, brier_score))
        ece_all.append(ece)
        mce_all.append(mce)
        bri_all.append(brier_score)

        if ece < ece_best:
            ece_best = ece
            best_temperature = current_temperature

        if mce < mce_best:
            mce_best = mce

        if brier_score < bri_best:
            bri_best = brier_score

        current_temperature += 0.1

    print('Best temperature: ', best_temperature)


if __name__ == '__main__':
    main()
