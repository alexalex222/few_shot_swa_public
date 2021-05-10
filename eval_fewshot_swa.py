import os
import time

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
from dataset.cub import MetaCUBImageFolder
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
        train_trans, test_trans = transforms_options['A']
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=1, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options['A']
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=1, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
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

    start = time.time()
    if opt.swa:
        model_swa.load_state_dict(ckpt['model'])
        (test_acc, test_std), y_pred, y_prob, y_true, train_scores, train_y_true = meta_test(model_swa, meta_testloader,
                                                                                             classifier=opt.classifier,
                                                                                             opt=opt)
    else:
        model.load_state_dict(ckpt['model'])
        (test_acc, test_std), y_pred, y_prob, y_true, train_scores, train_y_true = meta_test(model, meta_testloader,
                                                                                             classifier=opt.classifier,
                                                                                             opt=opt)

    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))


    """
     few_shot_eval_results = {
        'test_acc': test_acc,
        'test_std': test_std
    }
    
    save_dir = os.path.dirname(opt.model_path)
    torch.save(few_shot_eval_results, os.path.join(save_dir, 'few_shot_eval_results.pth'))

    y_pred = np.concatenate(y_pred, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    y_prob = torch.tensor(y_prob, dtype=torch.float)
    y_true = torch.tensor(y_true, dtype=torch.long)

    ece, mce, brier_score = calc_calibration_score(y_prob, y_true, n_bins=20)
    print('ECE: {0}, MCE: {1}, BRI: {2}'.format(ece, mce, brier_score))
    """


if __name__ == '__main__':
    main()
