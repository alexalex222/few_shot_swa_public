import torch
from torch import nn
import torch.optim as optim


class LogisticRegressionTemperature(nn.Module):
    def __init__(self, num_classes=5, feature_dim=1024):
        super(LogisticRegressionTemperature, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.scale_factor = nn.Parameter(torch.tensor([10.0], dtype=torch.float32))

    def forward(self, x):
        # scores = self.scale_factor * self.classifier(x)
        scores = self.classifier(x)
        return scores

    def fit(self, x, y, max_iter=1000):
        self.train()
        self.to(x.get_device())
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        for i in range(max_iter):
            output = self(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.eval()

def fit_classifier(classifier, x, y, max_iter=1000):
    classifier.train()
    classifier.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for i in range(max_iter):
        output = classifier(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    classifier.eval()
    # print(classifier.scale_factor)


@variational_estimator
class BayesianLogisticClassification(nn.Module):
    def __init__(self, num_classes=5, feature_dim=1024):
        super().__init__()
        self.fc = BayesianLinear(feature_dim, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out


def fit_bayesian_classifier(classifier, x, y, max_iter=1000):
    classifier.train()
    classifier.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = classifier.sample_elbo(inputs=x,
                                      labels=y,
                                      criterion=criterion,
                                      sample_nbr=3,
                                      complexity_cost_weight=1.0 / (x.shape[0]))

        loss.backward()
        optimizer.step()

