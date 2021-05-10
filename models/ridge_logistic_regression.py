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

