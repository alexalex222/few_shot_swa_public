# %%
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--activation_function', type=str, default='erf', help='activation function name')
    parser.add_argument('--trial', type=str, default='1', help='the experiment id')
    args = parser.parse_args()

    # %%
    model_save_root = './pretrained_models/sin_{}'.format(
        args.activation_function)
    if not os.path.isdir(model_save_root):
        os.makedirs(model_save_root)
    model_path = os.path.join(model_save_root, 'sin_model_{}.pth'.format(args.trial))
    swa_model_path = os.path.join(model_save_root, 'sin_swa_model_{}.pth'.format(args.trial))

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
    model = RegressionNN(input_dim=1, emb_size=40, hidden_size=40, output_dim=500,
                         activation_function=args.activation_function)
    model.train()
    model = model.to(device)
    # Use the adam optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    swa_model = AveragedModel(model)
    swa_model = swa_model.to(device)
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    new_model = RegressionNN(input_dim=1, emb_size=40, hidden_size=40, output_dim=500,
                             activation_function=args.activation_function)
    new_model = new_model.to(device)
    new_swa_model = AveragedModel(new_model)
    new_swa_mode = new_swa_model.to(device)

    # %%
    batch_size = 64
    training_iter = 100000
    swa_start = 80000
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

        if i == swa_start:
            torch.save(model.state_dict(), model_path)

        if i >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        desc = f'iter {i + 1} - loss {loss.item():.4f}'
        p_bar.set_description(desc)
        p_bar.update(1)

    p_bar.refresh()
    p_bar.close()

    torch.save(swa_model.state_dict(), swa_model_path)

    # %%
    new_model.load_state_dict(torch.load(model_path))
    new_swa_model.load_state_dict(torch.load(swa_model_path))
    new_model.eval()
    new_swa_model.eval()
    all_mse = []
    all_mse_swa = []
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

        feature_support = new_model.extract_feature(x_support)
        feature_query = new_model.extract_feature(x_query)
        feature_support, feature_query = feature_support.detach().cpu().numpy(), feature_query.detach().cpu().numpy()
        feature_support_swa = new_swa_model.module.extract_feature(x_support)
        feature_query_swa = new_swa_model.module.extract_feature(x_query)
        feature_support_swa = feature_support_swa.detach().cpu().numpy()
        feature_query_swa = feature_query_swa.detach().cpu().numpy()
        y_support, y_query = y_support.cpu().numpy(), y_query.cpu().numpy()

        """
        clf = Ridge(alpha=1.0)
        clf.fit(feature_support, y_support)
        pred_y = clf.predict(feature_query)
        mse = mean_squared_error(y_query, pred_y)
        all_mse.append(mse)

        clf_swa = Ridge(alpha=1.0)
        clf_swa.fit(feature_support_swa, y_support)
        pred_y_swa = clf_swa.predict(feature_query_swa)
        mse_swa = mean_squared_error(y_query, pred_y_swa)
        all_mse_swa.append(mse_swa)
        """

        clf = BayesianRidge(tol=1e-6, alpha_init=1.0, lambda_init=0.01)
        clf.fit(feature_support, y_support)
        pred_y, std_y = clf.predict(feature_query, return_std=True)
        mse = mean_squared_error(y_query, pred_y)
        all_mse.append(mse)

        clf_swa = BayesianRidge(tol=1e-6, alpha_init=1.0, lambda_init=0.01)
        clf_swa.fit(feature_support_swa, y_support)
        pred_y_swa, std_y_swa = clf_swa.predict(feature_query_swa, return_std=True)
        mse_swa = mean_squared_error(y_query, pred_y_swa)
        all_mse_swa.append(mse_swa)


    # %%
    mse_all_np = np.array(all_mse)
    print('SGD')
    print('Mean MSE: ', mse_all_np.mean())
    print('MSE std: ', mse_all_np.std())

    mse_swa_all_np = np.array(all_mse_swa)
    print('SWA')
    print('Mean MSE: ', mse_swa_all_np.mean())
    print('MSE std: ', mse_swa_all_np.std())



# %%
fig, ax = plt.subplots()
ax.plot(x_support.cpu(), y_support, 'kx', label='few-shot train')
ax.plot(x_test.cpu(), y_true_test[:, -1], label='true function')
ax.plot(x_query.cpu(), pred_y, '.', label='predicted')
ax.legend()
plt.show()
