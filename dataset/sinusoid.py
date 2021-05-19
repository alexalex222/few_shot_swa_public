# %%
import torch
import numpy as np
from torch.utils.data import Dataset


# %%
class MetaDataset(object):
    """Base class for a meta-dataset.
    Parameters
    ----------
    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.
    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        `meta_val` and `meta_test`.
    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed
        version. See also `torchvision.transforms`.
    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `transforms.ClassSplitter()`.
    """
    def __init__(self, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, target_transform=None, dataset_transform=None):
        if meta_train + meta_val + meta_test == 0:
            if meta_split is None:
                raise ValueError('The meta-split is undefined. Use either the '
                                 'argument `meta_train=True` (or `meta_val`/`meta_test`), or '
                                 'the argument `meta_split="train"` (or "val"/"test").')
            elif meta_split not in ['train', 'val', 'test']:
                raise ValueError('Unknown meta-split name `{0}`. The meta-split '
                                 'must be in [`train`, `val`, `test`].'.format(meta_split))
            meta_train = (meta_split == 'train')
            meta_val = (meta_split == 'val')
            meta_test = (meta_split == 'test')
        elif meta_train + meta_val + meta_test > 1:
            raise ValueError('Multiple arguments among `meta_train`, `meta_val` '
                             'and `meta_test` are set to `True`. Exactly one must be set to '
                             '`True`.')
        self.meta_train = meta_train
        self.meta_val = meta_val
        self.meta_test = meta_test
        self._meta_split = meta_split
        self.target_transform = target_transform
        self.dataset_transform = dataset_transform
        self.seed()

    @property
    def meta_split(self):
        if self._meta_split is None:
            if self.meta_train:
                self._meta_split = 'train'
            elif self.meta_val:
                self._meta_split = 'val'
            elif self.meta_test:
                self._meta_split = 'test'
            else:
                raise NotImplementedError()
        return self._meta_split

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed=seed)
        # Seed the dataset transform
        if hasattr(self.dataset_transform, 'seed'):
            self.dataset_transform.seed(seed=seed)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def sample_task(self):
        index = self.np_random.randint(len(self))
        return self[index]

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class SinusoidTask(Dataset):
    def __init__(self, index, amplitude, phase, input_range, noise_std,
                 num_samples, transform=None, target_transform=None,
                 np_random=None):
        super(SinusoidTask, self).__init__()  # Regression task
        self.amplitude = amplitude
        self.phase = phase
        self.input_range = input_range
        self.num_samples = num_samples
        self.noise_std = noise_std

        self.transform = transform
        self.target_transform = target_transform

        if np_random is None:
            np_random = np.random.RandomState(None)

        self._inputs = np_random.uniform(input_range[0], input_range[1], size=(num_samples, 1))
        self._targets = amplitude * np.sin(self._inputs - phase)
        if (noise_std is not None) and (noise_std > 0.):
            self._targets += noise_std * np_random.randn(num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x, y = self._inputs[index], self._targets[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


class Sinusoid(MetaDataset):
    """
    Simple regression task, based on sinusoids, as introduced in [1].
    Parameters
    ----------
    num_samples_per_task : int
        Number of examples per task.
    num_tasks : int (default: 1,000,000)
        Overall number of tasks to sample.
    noise_std : float, optional
        Amount of noise to include in the targets for each task. If `None`, then
        nos noise is included, and the target is a sine function of the input.
    transform : callable, optional
        A function/transform that takes a numpy array of size (1,) and returns a
        transformed version of the input.
    target_transform : callable, optional
        A function/transform that takes a numpy array of size (1,) and returns a
        transformed version of the target.
    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.
    Notes
    -----
    The tasks are created randomly as random sinusoid function. The amplitude
    varies within [0.1, 5.0], the phase within [0, pi], and the inputs are
    sampled uniformly in [-5.0, 5.0]. Due to the way PyTorch handles datasets,
    the number of tasks to be sampled needs to be fixed ahead of time (with
    `num_tasks`). This will typically be equal to `meta_batch_size * num_batches`.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, num_samples_per_task, num_tasks=1000000,
                 noise_std=None, transform=None, target_transform=None,
                 dataset_transform=None):
        super(Sinusoid, self).__init__(meta_split='train',
                                       target_transform=target_transform,
                                       dataset_transform=dataset_transform)
        self.num_samples_per_task = num_samples_per_task
        self.num_tasks = num_tasks
        self.noise_std = noise_std
        self.transform = transform

        self._input_range = np.array([-5.0, 5.0])
        self._amplitude_range = np.array([0.1, 5.0])
        self._phase_range = np.array([0, np.pi])

        self._amplitudes = None
        self._phases = None

    @property
    def amplitudes(self):
        if self._amplitudes is None:
            self._amplitudes = self.np_random.uniform(self._amplitude_range[0],
                                                      self._amplitude_range[1],
                                                      size=self.num_tasks)
        return self._amplitudes

    @property
    def phases(self):
        if self._phases is None:
            self._phases = self.np_random.uniform(self._phase_range[0],
                                                  self._phase_range[1],
                                                  size=self.num_tasks)
        return self._phases

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        np_random = np.random.RandomState(None)
        amplitude, phase = self.amplitudes[index], self.phases[index]
        # task = SinusoidTask(index, amplitude, phase, self._input_range,
        #                    self.noise_std, self.num_samples_per_task, self.transform,
        #                    self.target_transform, np_random=self.np_random)

        self._inputs = np.linspace(self._input_range[0], self._input_range[1],
                                   num=self.num_samples_per_task)
        self._targets = amplitude * np.sin(self._inputs - phase)
        if (self.noise_std is not None) and (self.noise_std > 0.):
            self._targets += self.noise_std * np_random.randn(self.num_samples_per_task, 1)

        return self._inputs, self._targets


def generate_sin_data_matrix(num_tasks=1000, num_samples_per_task=200):
    input_range = torch.tensor([-5.0, 5.0])
    amplitude_range = torch.tensor([0.1, 5.0])
    phase_range = torch.tensor([0, np.pi])

    amplitudes = torch.rand(num_tasks) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
    phases = torch.rand(num_tasks) * (phase_range[1] - phase_range[0]) + phase_range[0]
    x = torch.linspace(input_range[0], input_range[1], steps=num_samples_per_task)

    x = x.reshape(1, -1)
    phases = phases.reshape(-1, 1)
    amplitudes = amplitudes.reshape(-1, 1)

    y_true = amplitudes * torch.sin(x - phases)
    y = y_true + 0.1 * torch.randn((x - phases).shape)

    x = x.permute(1, 0)
    y = y.permute(1, 0)
    y_true = y_true.permute(1, 0)

    x_train = x.clone()
    x_test = x.clone()
    y_train = y[:, :500]
    y_test = y[:, 500:]
    y_true_train = y_true[:, :500]
    y_true_test = y_true[:, 500:]

    sin_data = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_true_train': y_true_train,
        'y_true_test': y_true_test
    }

    torch.save(sin_data, './data/regression/sin_data_few_shot.pt')


# %%
if __name__ == '__main__':
    # dataset = Sinusoid(10, num_tasks=1000, noise_std=None)
    # x, y = dataset[0]
    generate_sin_data_matrix()
