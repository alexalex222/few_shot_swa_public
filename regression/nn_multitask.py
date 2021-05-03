# %%
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, BayesianRidge

sys.path.append(os.getcwd() + '/..')
from models.nn_regression import RegressionNN

plt.style.use('ggplot')
matplotlib.rcParams['figure.dpi'] = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
if os.name == 'nt':
    sin_data = torch.load('D:\\Temp\\torch_dataset\\regression\\sin_data_few_shot.pt')
else:
    sin_data = torch.load('/media/kuilin/research/temp/torch_dataset/regression/sin_data_few_shot.pt')
x_train = sin_data['x_train']
x_test = sin_data['x_test']
y_train = sin_data['y_train']
y_test = sin_data['y_test']
y_true_train = sin_data['y_true_train']
y_true_test = sin_data['y_true_test']

x_train, x_test, y_train = x_train.to(device), x_test.to(device), y_train.to(device)


# %%
model = RegressionNN(input_dim=1, emb_size=40, hidden_size=40, output_dim=500, activation_function='erf')
model.train()
model = model.to(device)
# Use the adam optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)


# %%
batch_size = 64
training_iter = 100000
p_bar = tqdm(total=training_iter)
p_bar.set_description(f'Begin training')
for i in range(training_iter):
    idx = torch.randperm(len(x_train))[:batch_size]
    mini_batch_x, mini_batch_y = x_train[idx], y_train[idx]
    optimizer.zero_grad()
    output = model(mini_batch_x)
    loss = F.mse_loss(output, mini_batch_y)
    loss.backward()
    # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
    desc = f'iter {i + 1} - loss {loss.item():.4f}'
    p_bar.set_description(desc)
    p_bar.update(1)
p_bar.refresh()
p_bar.close()



# %%
model.eval()
all_mse = []
for task_id in range(500):
    n_shots = 10
    idx = torch.randperm(len(x_test))
    support_idx, _ = torch.sort(idx[:n_shots])
    query_idx, _ = torch.sort(idx[n_shots:])
    x_support = x_test[support_idx]
    y_support = y_test[support_idx][:, task_id]
    x_query = x_test[query_idx]
    y_query = y_test[query_idx][:, task_id]
    y_true_query = y_true_test[query_idx][:, task_id]

    feature_support = model.extract_feature(x_support)
    feature_query = model.extract_feature(x_query)
    feature_support, feature_query = feature_support.detach().cpu().numpy(), feature_query.detach().cpu().numpy()
    y_support, y_query = y_support.cpu().numpy(), y_query.cpu().numpy()

    """
    clf = Ridge(alpha=1.0)
    clf.fit(feature_support, y_support)
    pred_y = clf.predict(feature_query)
    """

    clf = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True, alpha_init=1.0, lambda_init=0.01)
    clf.fit(feature_support, y_support)
    pred_y, std_y = clf.predict(feature_query, return_std=True)

    mse = mean_squared_error(y_query, pred_y)
    all_mse.append(mse)
    # print('MSE: ', mse)


# %%
mse_all_np = np.array(all_mse)
print('Mean MSE: ', mse_all_np.mean())
print('MSE std: ', mse_all_np.std())

# %%
fig, ax = plt.subplots()
ax.plot(x_support.cpu(), y_support, 'kx', label='few-shot train')
ax.plot(x_test.cpu(), y_true_test[:, -1], label='true function')
ax.plot(x_query.cpu(), pred_y, '.', label='predicted')
ax.fill_between(x_query.cpu().reshape(-1), pred_y-std_y, pred_y+std_y,
                color="pink", alpha=0.5, label="predict std")
ax.legend()
plt.show()