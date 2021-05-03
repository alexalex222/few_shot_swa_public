# %%
import os
import sys
import torch
from torch.optim.swa_utils import AveragedModel
import matplotlib
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge

sys.path.append(os.getcwd() + '/..')
from models.nn_regression import RegressionNN
from models.hierarchical_bayesian_linear_regression import HierarchicalBayesianRegression

plt.style.use('ggplot')
matplotlib.rcParams['figure.dpi'] = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
args = lambda x: None
args.activation_function = 'erf'

# %%
if os.name == 'nt':
    sin_data = torch.load('D:\\Temp\\torch_dataset\\regression\\sin_data_few_shot.pt')
else:
    sin_data = torch.load('/media/kuilin/research/temp/torch_dataset/regression/sin_data_few_shot.pt')
x_test = sin_data['x_test']
y_test = sin_data['y_test']
y_true_test = sin_data['y_true_test']

x_test = x_test.to(device)

# %%
model = RegressionNN(input_dim=1, emb_size=40, hidden_size=40, output_dim=500,
                     activation_function=args.activation_function)
model = model.to(device)
# Use the adam optimizer
swa_model = AveragedModel(model)
swa_model = swa_model.to(device)
model.eval()
swa_model.eval()


# %%
model_save_root = './pretrained_models/sin_{}'.format(
        args.activation_function)
model_file = model_save_root + '/sin_swa_model_1.pth'

# %%
n_shots = 10
# task_id = 17
# support_idx = torch.tensor([2, 12,  53,  61,  83, 99, 122, 141, 168, 194])
task_id = 17
support_idx = torch.tensor([2, 12,  53,  61,  83, 99, 122, 141, 168, 194])

swa_model.load_state_dict(torch.load(model_file))

x_support = x_test[support_idx]
y_support = y_test[support_idx][:, task_id]
x_query = x_test
y_query = y_test[:, task_id]
y_true_query = y_true_test[:, task_id]

feature_support_swa = swa_model.module.extract_feature(x_support)
feature_query_swa = swa_model.module.extract_feature(x_query)

feature_support_swa = feature_support_swa.detach().cpu()
feature_query_swa = feature_query_swa.detach().cpu()
y_support, y_query = y_support.cpu(), y_query.cpu()


bayesian_model = BayesianRidge()
bayesian_model.fit(feature_support_swa, y_support)
pred_y, pred_y_std = bayesian_model.predict(feature_query_swa.numpy(), return_std=True)

# %%
fig, ax = plt.subplots()
ax.plot(x_support.cpu(), y_support, 'kx', label='noisy train sample')
ax.plot(x_query.cpu(), pred_y, label='predicted')
ax.plot(x_test.cpu(), y_true_test[:, task_id], label='true function')
ax.fill_between(x_query.cpu().reshape(-1), pred_y - 2*pred_y_std, pred_y + 2*pred_y_std,
                color="pink", alpha=0.5, label="Uncertainty")
ax.set_title("Hierarchical Bayesian linear regression")
ax.legend()
plt.show()
fig.savefig('./figs/sin_swa.pdf')



















"""
# %%
input_dim = feature_support_swa.shape[-1]
mcmc_regression_model = HierarchicalBayesianRegression(input_dim=input_dim)
mcmc_regression_model.train_model(feature_support_swa, y_support)
pred_y_samples = mcmc_regression_model.make_prediction(feature_query_swa)

fig, ax = plt.subplots()
ax.plot(x_support.cpu(), y_support, 'kx', label='few-shot train')
ax.plot(x_test.cpu(), y_true_test[:, task_id], label='true function')
ax.plot(x_query.cpu(), pred_y_samples.mean(dim=1), '.', label='predicted')
ax.fill_between(x_query.cpu().reshape(-1),
                pred_y_samples.mean(dim=1) - 2*pred_y_samples.std(dim=1),
                pred_y_samples.mean(dim=1) + 2*pred_y_samples.std(dim=1),
                color="pink", alpha=0.5, label="Uncertainty")
ax.set_title("MCMC Hierarchical Bayesian regression")
ax.legend()
plt.show()

# %%
feature_support_swa = feature_support_swa.numpy()
feature_query_swa = feature_query_swa.numpy()
y_support, y_query = y_support.numpy(), y_query.numpy()


# %%
alpha_value = 4
clf_swa = Ridge(alpha=alpha_value)
clf_swa.fit(feature_support_swa, y_support)
pred_y_swa = clf_swa.predict(feature_query_swa)

fig, ax = plt.subplots()
ax.plot(x_support.cpu(), y_support, 'kx', label='few-shot train')
ax.plot(x_test.cpu(), y_true_test[:, task_id], label='true function')
ax.plot(x_query.cpu(), pred_y_swa, '.', label='predicted')
ax.set_title("lambda = {}".format(alpha_value))
ax.legend()
plt.show()
"""
