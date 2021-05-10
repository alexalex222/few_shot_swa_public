# %%
import pickle
import numpy as np
from PIL import Image
import torch
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from models.convnet import ConvNetRegression

plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% load model
model = ConvNetRegression(num_tasks=24)
model_swa = AveragedModel(model)
model.eval()
model_swa.eval()
model.load_state_dict(torch.load('./pretrained_models/qmul_regression/conv_reg_adam_bn.pt'))
model_swa.load_state_dict(torch.load('./pretrained_models/qmul_regression/conv_reg_swa.pt'))
model = model.to(device)
model_swa = model_swa.to(device)

# %% load data
test_file_path = './QMUL//qmul_test.pickle'

test_file = open(test_file_path, 'rb')
test_data = pickle.load(test_file)
test_images = test_data['data']
test_labels = test_data['labels']
test_tasks = test_images.shape[0]

transform = transforms.Compose([
        transforms.ToTensor(),
    ])

n_shots = 10
all_mse = []
all_mse_swa = []

# %%
for person in tqdm(range(1000)):
    person = person % test_tasks
    samples_in_task = test_images.shape[1]
    query_x = torch.empty(samples_in_task, 3, 100, 100)
    query_y = torch.empty(samples_in_task, )
    for i in range(samples_in_task):
        img = np.asarray(test_images[person, i]).astype('uint8')
        img = Image.fromarray(img)
        img = transform(img)
        label = test_labels[person, i]
        query_x[i] = img
        query_y[i] = torch.tensor(label, dtype=torch.float32)

    idx = torch.randperm(samples_in_task)
    support_idx, _ = torch.sort(idx[:n_shots])
    query_idx, _ = torch.sort(idx[n_shots:])

    support_x = query_x[support_idx]
    support_y = query_y[support_idx]
    query_x = query_x[query_idx]
    query_y = query_y[query_idx]

    support_x, query_x = support_x.to(device), query_x.to(device)

    with torch.no_grad():
        support_feature, _ = model(support_x, is_feat=True)
        query_feature, _ = model(query_x, is_feat=True)
        support_feature_swa, _ = model_swa(support_x, is_feat=True)
        query_feature_swa, _ = model_swa(query_x, is_feat=True)

    support_feature = support_feature[0].detach().cpu().numpy()
    query_feature = query_feature[0].detach().cpu().numpy()
    support_feature_swa = support_feature_swa[0].detach().cpu().numpy()
    query_feature_swa = query_feature_swa[0].detach().cpu().numpy()
    support_y = support_y.cpu().numpy()
    query_y = query_y.cpu().numpy()

    clf = BayesianRidge(tol=1e-6, fit_intercept=True, alpha_init=1.0, lambda_init=0.01)
    clf.fit(support_feature, support_y)
    pred_y, std_y = clf.predict(query_feature, return_std=True)

    clf_swa = BayesianRidge(tol=1e-6, fit_intercept=True, alpha_init=1.0, lambda_init=0.01)
    clf_swa.fit(support_feature_swa, support_y)
    pred_y_swa, std_y_swa = clf_swa.predict(query_feature_swa, return_std=True)

    mse = mean_squared_error(query_y, pred_y)
    mse_swa = mean_squared_error(query_y, pred_y_swa)
    all_mse.append(mse)
    all_mse_swa.append(mse_swa)
    # print('MSE: ', mse)


# %%
mse_all_np = np.array(all_mse)
print('SGD')
print('Mean MSE: ', mse_all_np.mean())
print('MSE std: ', mse_all_np.std())

mse_all_swa_np = np.array(all_mse_swa)
print('SWA')
print('Mean MSE: ', mse_all_swa_np.mean())
print('MSE std: ', mse_all_swa_np.std())
