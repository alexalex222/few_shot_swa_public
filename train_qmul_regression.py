# %%
import os
import time
import pickle
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.swa_utils import AveragedModel, SWALR
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from tqdm import tqdm
from models.convnet import ConvNetRegression

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% load data
train_file_path = './data/QMUL//qmul_train.pickle'

train_file = open(train_file_path, 'rb')
train_data = pickle.load(train_file)
train_images = train_data['data']
train_labels = train_data['labels']
train_tasks = train_images.shape[0]


# %%
def get_one_batch(train_x, train_y, repeat_per_task=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    num_tasks = train_x.shape[0]
    sample_per_task = train_x.shape[1]
    select_idx = torch.randint(low=0, high=sample_per_task, size=(train_tasks * repeat_per_task,))
    mini_batch_x = torch.empty(len(select_idx), 3, 100, 100)
    mini_batch_y = torch.empty(len(select_idx), )

    for i in range(len(select_idx)):
        task_idx = i % num_tasks
        sample_idx = select_idx[i]
        img = np.asarray(train_x[task_idx, sample_idx]).astype('uint8')
        img = Image.fromarray(img)
        img = transform(img)
        label = train_y[task_idx, sample_idx]
        mini_batch_x[i] = img
        mini_batch_y[i] = torch.tensor(label, dtype=torch.float32)

    return mini_batch_x, mini_batch_y


# %%
def update_bn(model, device=None):
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

    for i in tqdm(range(40*100)):
        batch_train_x, _ = get_one_batch(train_images, train_labels, repeat_per_task=4)
        batch_train_x = batch_train_x.to(device)
        model(batch_train_x)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


# %%
model = ConvNetRegression(num_tasks=train_tasks)
model.train()
model = model.to(device)
swa_model = AveragedModel(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=100000, gamma=0.1)
swa_scheduler = SWALR(optimizer, swa_lr=0.01)
loss_func = nn.MSELoss()
repeat = 4
training_iter = 40*300
swa_iter = 1000

time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
logger = SummaryWriter(log_dir=os.path.join('./pretrained_models', time_string))

log_interval = 40
avg_train_loss = 0
for i in tqdm(range(training_iter + swa_iter)):
    batch_train_x, batch_train_y = get_one_batch(train_images, train_labels, repeat_per_task=repeat)
    batch_train_x, batch_train_y = batch_train_x.to(device), batch_train_y.to(device)
    optimizer.zero_grad()
    output = model(batch_train_x)
    idx1 = torch.arange(len(batch_train_x))
    idx2 = torch.cat(repeat*[torch.arange(train_tasks)])
    select_output = output[idx1, idx2]
    loss = loss_func(select_output, batch_train_y)
    loss.backward()
    optimizer.step()
    if i == training_iter - 1:
        torch.save(model.state_dict(), './pretrained_models/qmul_regression/conv_reg.pt')
    if i < training_iter:
        pass
    else:
        if i % 40 == 0:
            swa_model.update_parameters(model)
            swa_scheduler.step()
    avg_train_loss = avg_train_loss + loss.item()
    if i % log_interval == 0:
        avg_train_loss = avg_train_loss / log_interval
        logger.add_scalar('train_loss', avg_train_loss, i)
        avg_train_loss = 0


update_bn(swa_model, device=device)

torch.save(swa_model.state_dict(), './pretrained_models/qmul_regression/conv_reg_swa.pt')
