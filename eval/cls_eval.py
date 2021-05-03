import torch
import time

from .util import AverageMeter, accuracy


def validate(val_loader, model, criterion):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (x, target, _) in enumerate(val_loader):

            x = x.float()
            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()

            # compute output
            output = model(x)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.5f}'
              .format(top1=top1, top5=top5, losses=losses))

    return top1.avg, top5.avg, losses.avg
