# %%
import torch
from torch import nn
from torch.nn.utils.weight_norm import WeightNorm


class CosineClassifier(nn.Module):
    def __init__(self, feature_dim, num_class):
        super(CosineClassifier, self).__init__()

        self.classifier = nn.Linear(feature_dim, num_class, bias=False)
        WeightNorm.apply(self.classifier, 'weight', dim=0)
        self.scale_factor = 2.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        scores = self.scale_factor * self.classifier(x)
        return scores

    def fit(self, x, y):
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_func = nn.CrossEntropyLoss()
        self.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        self.train()
        for epoch in range(200):
            # compute output
            output = self(x)
            loss = loss_func(output, y)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    clf = CosineClassifier(feature_dim=x_train.shape[-1], num_class=num_classes)
    clf.fit(x_train, y_train)
    pred_scores = clf(x_test)
    soft_max = torch.nn.Softmax(dim=1)
    pred_prob = soft_max(pred_scores)
    _, pred_idx = torch.max(pred_prob, dim=1)
    accuracy = torch.sum(pred_idx == y_test) / x_test.shape[0]
    print(accuracy)
