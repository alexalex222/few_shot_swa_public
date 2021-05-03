# %%
import torch
from torch import nn

from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import softmax
import numpy as np


class RidgeLogisticRegression(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 num_class: int,
                 c: float = 1.0):
        super(RidgeLogisticRegression, self).__init__()

        self.classifier = nn.Linear(feature_dim, num_class, bias=True)
        self.scale_factor = nn.Parameter(torch.tensor([30.0], dtype=torch.float32))
        # self.scale_factor = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c = c

    def forward(self, x):
        scores = self.classifier(x)
        scores = self.scale_factor * scores
        return scores

    def fit(self, x, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        ce_loss = nn.CrossEntropyLoss()
        self.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        self.train()
        for epoch in range(300):
            # compute output
            output = self(x)
            loss = self.c * ce_loss(output, y) + 0.5 * torch.sum(self.classifier.weight * self.classifier.weight)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def fit_lbfgs(self, x, y):
        optimizer = torch.optim.LBFGS(self.parameters(), lr=1, max_iter=1000)
        ce_loss = nn.CrossEntropyLoss()
        self.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        self.train()

        def closure():
            optimizer.zero_grad()
            output = self(x)
            loss = self.c * ce_loss(output, y) + 0.5 * torch.sum(self.classifier.weight * self.classifier.weight)
            # compute gradient and do SGD step
            loss.backward()
            return loss

        optimizer.step(closure)

    def predict(self, x):
        self.eval()
        x = x.to(self.device)
        scores = self(x)

        return scores


# %%
if __name__ == '__main__':
    file_path = '/media/kuilin/research/temp/torch_dataset/classification/one_task_data.pt'
    tensor_dict = torch.load(file_path)
    x_train = tensor_dict['support_features']
    y_train = tensor_dict['support_ys'].view(-1).cuda()
    x_test = tensor_dict['query_features']
    y_test = tensor_dict['query_ys'].view(-1).cuda()

    num_classes = 5

    clf = RidgeLogisticRegression(feature_dim=x_train.shape[-1], num_class=num_classes)
    # clf.fit(x_train, y_train)
    clf.fit(x_train, y_train)
    pred_scores = clf(x_test)
    soft_max = torch.nn.Softmax(dim=1)
    pred_prob = soft_max(pred_scores)
    _, pred_idx = torch.max(pred_prob, dim=1)
    accuracy = torch.sum(pred_idx == y_test) / x_test.shape[0]
    print('Pytorch accuracy: \n', accuracy)
    print('Pytorch probability: \n', pred_prob)

    #  compare with scikit learn
    x_train = x_train.cpu().numpy()
    y_train = y_train.cpu().numpy()
    x_test = x_test.cpu().numpy()
    y_test = y_test.cpu().numpy()
    clf_sk = LogisticRegression(penalty='l2',
                                random_state=0,
                                C=1.0,         # small values lead to stronger regularization; 20 best calib
                                solver='lbfgs',
                                max_iter=1000,
                                multi_class='multinomial',
                                fit_intercept=True)
    clf_sk.fit(x_train, y_train)
    pred_idx_sk = clf_sk.predict(x_test)
    pred_scores_sk = clf_sk.decision_function(x_test)
    pred_prob_sk = softmax(1.0 * pred_scores_sk, copy=False)
    accuracy_sk = np.sum(pred_idx_sk == y_test) / x_test.shape[0]
    print('sk accuracy: \n', accuracy_sk)
    print('sk probability: \n', pred_prob_sk)
