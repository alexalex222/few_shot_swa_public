import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is normalized probability.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, probs, labels):
        softmaxes = probs
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def plot_calibration_error(probs, targets, ax, n_bins=10):
    targets_one_hot = F.one_hot(targets).type(torch.DoubleTensor)
    brier_score = torch.mean(torch.sum((probs - targets_one_hot)**2, dim=1))
    confidences = probs.max(-1).values.detach().numpy()
    accuracies = probs.argmax(-1).eq(targets).numpy()
    # print(confidences)
    # print(accuracies)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    max_err = 0.0

    plot_acc = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(np.float32).mean()

        if prop_in_bin > 0.0:
            accuracy_in_bin = accuracies[in_bin].astype(np.float32).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if np.abs(avg_confidence_in_bin - accuracy_in_bin) > max_err:
                max_err = np.abs(avg_confidence_in_bin - accuracy_in_bin)

            plot_acc.append(accuracy_in_bin)
        else:
            plot_acc.append(0.0)

    ax.bar(
        bin_lowers, plot_acc, bin_uppers[0], align="edge", linewidth=1, edgecolor="k"
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], c="k", lw=2, linestyle="--")
    text_box_props = dict(facecolor='white', alpha=0.5)
    ax.text(
        0.02,
        0.65,
        "ECE: {:0.4f}\nMCE: {:0.4f}\nBRI: {:0.4f}".format(
            ece, max_err, brier_score
        ),
        fontsize=16,
        fontfamily='monospace',
        bbox=text_box_props
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    #plt.savefig("temp.png", bbox_inches="tight")


def calc_calibration_score(probs, targets, n_bins=10):
    targets_one_hot = F.one_hot(targets).type(torch.DoubleTensor)
    brier_score = torch.mean(torch.sum((probs - targets_one_hot) ** 2, dim=1))
    confidences = probs.max(-1).values.detach().numpy()
    accuracies = probs.argmax(-1).eq(targets).numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    max_err = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(np.float32).mean()

        if prop_in_bin > 0.0:
            accuracy_in_bin = accuracies[in_bin].astype(np.float32).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if np.abs(avg_confidence_in_bin - accuracy_in_bin) > max_err:
                max_err = np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece, max_err, brier_score
