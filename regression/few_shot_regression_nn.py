# %%
import sys
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

sys.path.append(os.getcwd() + '/..')
from dataset.sinusoid import Sinusoid
from models.nn_regression import RegressionNN


plt.style.use('ggplot')
matplotlib.rcParams['figure.dpi'] = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
num_tasks = 600
dataset = Sinusoid(100, num_tasks=num_tasks, noise_std=None)
all_rmse = []

for t in range(num_tasks):
    x, y = dataset[t]
    x, y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)


# %%
    n_shots = 10
    idx = torch.randperm(100)
    train_idx, _ = torch.sort(idx[:n_shots])
    test_idx, _ = torch.sort(idx[n_shots:])
    train_x = x[train_idx]
    train_y = y[train_idx]
    test_x = x[test_idx]
    test_y = y[test_idx]

# %%
    model = RegressionNN(input_dim=1, emb_size=1024)
    model.train()
    # Use the adam optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# %%
    training_iter = 20000
    train_x = train_x.reshape(-1, 1)
    train_y = train_y.reshape(-1, 1)
    train_x, train_y = train_x.to(device), train_y.to(device)
    model = model.to(device)
    p_bar = tqdm(total=training_iter)
    p_bar.set_description(f'Task {t + 1}')
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = F.mse_loss(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
        desc = f'task {t + 1} - iter {i + 1} - loss {loss.item():.4f}'
        p_bar.set_description(desc)
        p_bar.update(1)
    p_bar.refresh()
    p_bar.close()


# %%
    model.eval()
    test_x = test_x.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    test_x, test_y = test_x.to(device), test_y.to(device)
    with torch.no_grad():
        pred = model(test_x)

    rmse = mean_squared_error(test_y.cpu().numpy(), pred.cpu().numpy(), squared=True)
    all_rmse.append(rmse)
    print('Task RMSE: ', rmse)

np_all_rmse = np.array(all_rmse)
print('Mean RMSE: ', np_all_rmse.mean())
print('RMSE std: ', np_all_rmse.std())

"""
# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(4, 3))
# Plot training data as black stars
ax.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'kx')
# Plot predictive means as blue line
ax.plot(test_x.cpu().numpy(), pred.detach().cpu().numpy(), '.')
ax.plot(x.numpy(), y.numpy())

#ax.set_ylim([-3, 3])
ax.legend(['Few-shot train data', 'Predicted', 'True function'])
plt.show()
"""
