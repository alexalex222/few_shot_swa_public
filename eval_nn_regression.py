# %%
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
from models.nn_regression import RegressionNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
if __name__ == '__main__':
    # parser = argparse.ArgumentParser('argument for training')
    # parser.add_argument('--activation_function', type=str, default='erf', help='activation function name')
    # args = parser.parse_args()

    args = lambda x: None
    args.activation_function = 'erf'

    # %%
    sin_data = torch.load('./data/regression/sin_data_few_shot.pt')
    x_test = sin_data['x_test']
    y_test = sin_data['y_test']
    y_true_test = sin_data['y_true_test']

    x_test = x_test.to(device)

    # %%
    model = RegressionNN(input_dim=1, emb_size=40, hidden_size=40, output_dim=500,
                         activation_function=args.activation_function)
    model = model.to(device)
    swa_model = AveragedModel(model)
    swa_model = swa_model.to(device)
    model.eval()
    swa_model.eval()


    # %%
    model_save_root = './pretrained_models/sin_{}'.format(
        args.activation_function)
    model_path = model_save_root + '/sin_model_2.pth'
    swa_model_path = model_save_root + '/sin_swa_model_2.pth'
    n_shots = 10

    all_mse = []
    all_mse_swa = []
    for i in tqdm(range(y_test.shape[-1] * 1)):
        task_id = i % y_test.shape[-1]
        idx = torch.randperm(len(x_test))
        support_idx, _ = torch.sort(idx[:n_shots])
        query_idx, _ = torch.sort(idx[n_shots:])

        all_pred = np.zeros(shape=(10, len(query_idx)))

        model.load_state_dict(torch.load(model_path))
        swa_model.load_state_dict(torch.load(swa_model_path))

        x_support = x_test[support_idx]
        y_support = y_test[support_idx][:, task_id]
        x_query = x_test[query_idx]
        y_query = y_test[query_idx][:, task_id]
        y_true_query = y_true_test[query_idx][:, task_id]

        with torch.no_grad():
            feature_support = model.extract_feature(x_support)
            feature_query = model.extract_feature(x_query)
            feature_support_swa = swa_model.module.extract_feature(x_support)
            feature_query_swa = swa_model.module.extract_feature(x_query)
        feature_support = feature_support.detach().cpu().numpy()
        feature_query = feature_query.detach().cpu().numpy()
        feature_support_swa = feature_support_swa.detach().cpu().numpy()
        feature_query_swa = feature_query_swa.detach().cpu().numpy()
        y_support, y_query = y_support.cpu().numpy(), y_query.cpu().numpy()

        clf = BayesianRidge(tol=1e-6, alpha_init=1.0, lambda_init=0.01)
        clf.fit(feature_support, y_support)
        pred_y = clf.predict(feature_query)

        clf_swa = BayesianRidge(tol=1e-6, alpha_init=1.0, lambda_init=0.01)
        clf_swa.fit(feature_support_swa, y_support)
        pred_y_swa = clf_swa.predict(feature_query_swa)

        mse_swa = mean_squared_error(y_query, pred_y_swa)
        mse = mean_squared_error(y_query, pred_y)
        all_mse.append(mse)
        all_mse_swa.append(mse_swa)

    mse_swa_all_np = np.array(all_mse_swa)
    print('SWA Mean MSE: ', mse_swa_all_np.mean())
    print('SWA MSE std: ', mse_swa_all_np.std())

    mse_all_np = np.array(all_mse)
    print('Mean MSE: ', mse_all_np.mean())
    print('MSE std: ', mse_all_np.std())

