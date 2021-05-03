import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import softmax

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .top_layer import BayesianLogisticClassification, fit_bayesian_classifier
from models.cos_classifier import CosineClassifier
from models.ridge_logistic_regression import RidgeLogisticRegression


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def meta_test(net, testloader, use_logit=False, is_norm=True, classifier='LR', opt=None):
    net = net.eval()
    acc = []
    y_true = []
    y_pred = []
    y_prob = []
    train_scores_all = []
    train_y_true = []

    idx = 0

    if True:
        for idx, data in tqdm(enumerate(testloader)):
            support_xs, support_ys, query_xs, query_ys = data
            support_ys = support_ys.view(-1)
            query_ys = query_ys.view(-1)
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, channel, height, width = support_xs.size()
            support_xs = support_xs.view(-1, channel, height, width)
            query_xs = query_xs.view(-1, channel, height, width)

            with torch.no_grad():
                if use_logit:
                    support_features = net(support_xs).view(support_xs.size(0), -1)
                    query_features = net(query_xs).view(query_xs.size(0), -1)
                else:
                    feat_support, _ = net(support_xs, is_feat=True)
                    support_features = feat_support[-1].view(support_xs.size(0), -1)
                    feat_query, _ = net(query_xs, is_feat=True)
                    query_features = feat_query[-1].view(query_xs.size(0), -1)


            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            # ---- Tukey's transform
            if opt.model == 'wrn_28_10':
                beta = 0.5
                support_features = torch.pow(torch.abs(support_features), beta)
                query_features = torch.pow(torch.abs(query_features), beta)


            """
            if idx == 0:
                m = {'support_features': support_features,
                     'query_features': query_features,
                     'support_ys': support_ys,
                     'query_ys': query_ys}
                torch.save(m, 'pretrained_models/one_task_data.pt')
            """

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.numpy()
            query_ys = query_ys.numpy()


            #  clf = SVC(gamma='auto', C=0.1)
            if classifier == 'LR':
                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=opt.reg,       # small values lead to stronger regularization; 20 best calib
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial',
                                         fit_intercept=True
                                         )

                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
                pred_scores = clf.decision_function(query_features)
                train_scores = clf.decision_function(support_features)
                # query_ys_prob = clf.predict_proba(query_features)
                query_ys_prob = softmax(opt.temp*pred_scores, copy=False)
            elif classifier == 'SVM':
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto',
                                                          C=1.0,
                                                          kernel='linear',
                                                          decision_function_shape='ovr'))
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
                query_ys_prob = np.zeros(query_ys_pred.shape)
            elif classifier == 'CosineClassifier':
                support_features = torch.tensor(support_features, dtype=torch.float)
                query_features = torch.tensor(query_features, dtype=torch.float)
                support_ys = torch.tensor(support_ys, dtype=torch.long)
                query_ys = torch.tensor(query_ys, dtype=torch.long)

                clf = CosineClassifier(feature_dim=support_features.shape[-1], num_class=5)
                clf.fit(support_features, support_ys)
                pred_scores = clf.predict(query_features)
                train_scores = clf.predict(support_features)
                query_ys_prob = softmax(pred_scores.detach().cpu().numpy(), copy=False)
                query_ys_pred = np.argmax(query_ys_prob, axis=-1)
            elif classifier == 'LR_torch':
                support_features = torch.tensor(support_features, dtype=torch.float)
                query_features = torch.tensor(query_features, dtype=torch.float)
                support_ys = torch.tensor(support_ys, dtype=torch.long)
                query_ys = torch.tensor(query_ys, dtype=torch.long)

                clf = RidgeLogisticRegression(feature_dim=support_features.shape[-1], num_class=5)
                clf.fit(support_features, support_ys)
                pred_scores = clf.predict(query_features)
                query_ys_prob = softmax(pred_scores.detach().cpu().numpy(), copy=False)
                query_ys_pred = np.argmax(query_ys_prob, axis=-1)
                print('scale factor: ', clf.scale_factor.item())
            elif classifier == 'NN':
                query_ys_pred = nearest_neighbour(support_features, support_ys, query_features)
                query_ys_prob = np.zeros(query_ys_pred.shape)
            elif classifier == 'Cosine':
                query_ys_pred = cosine_similarity(support_features, support_ys, query_features)
                query_ys_prob = np.zeros(query_ys_pred.shape)
            elif classifier == 'Proto':
                query_ys_pred, distance = proto(support_features, support_ys, query_features, opt)
                query_ys_prob = softmax(distance, copy=False)
            else:
                raise NotImplementedError('classifier not supported: {}'.format(classifier))

            # print(query_ys)
            # print(query_ys.shape)
            # print(query_ys_pred)
            # print(query_ys_pred.shape)
            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
            y_pred.append(query_ys_pred)
            y_true.append(query_ys)
            y_prob.append(query_ys_prob)
            train_scores_all.append(train_scores)
            train_y_true.append(support_ys)

    return mean_confidence_interval(acc), y_pred, y_prob, y_true, train_scores_all, train_y_true


def meta_test_lr_torch(net, test_loader):
    net = net.eval()
    acc = []
    y_true = []
    y_pred = []
    y_prob = []


    for idx, data in tqdm(enumerate(test_loader)):
        support_xs, support_ys, query_xs, query_ys = data
        support_xs = support_xs.cuda()
        query_xs = query_xs.cuda()
        batch_size, _, channel, height, width = support_xs.size()
        support_xs = support_xs.view(-1, channel, height, width)
        query_xs = query_xs.view(-1, channel, height, width)

        with torch.no_grad():
            feat_support, _ = net(support_xs, is_feat=True)
            feat_query, _ = net(query_xs, is_feat=True)
        support_features = feat_support[-1].view(support_xs.size(0), -1)
        query_features = feat_query[-1].view(query_xs.size(0), -1)

        support_features = normalize(support_features)
        query_features = normalize(query_features)

        support_features = support_features.detach()
        query_features = query_features.detach()

        support_ys = support_ys.reshape(-1)
        support_ys = support_ys.cuda()
        query_ys = query_ys.reshape(-1)

        feature_dim = support_features.shape[-1]

        # cls = LogisticRegressionTemperature(num_classes=5, feature_dim=feature_dim)
        # fit_classifier(cls, support_features, support_ys)

        cls = BayesianLogisticClassification(num_classes=5, feature_dim=feature_dim)
        fit_bayesian_classifier(cls, support_features, support_ys)

        scores = cls(query_features)
        # a = torch.nn.functional.softmax(scores, dim=-1)
        # print(a)
        query_ys_prob = torch.nn.functional.softmax(scores, dim=-1).detach().cpu().numpy()
        query_ys_pred = torch.nn.functional.softmax(scores, dim=-1).argmax(-1).detach().cpu().numpy()


        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
        y_pred.append(query_ys_pred)
        y_true.append(query_ys)
        y_prob.append(query_ys_prob)
    return mean_confidence_interval(acc), y_pred, y_prob, y_true


def proto(support, support_ys, query, opt):
    """Protonet classifier"""
    nc = support.shape[-1]
    support = np.reshape(support, (-1, 1, opt.n_ways, opt.n_shots, nc))
    support = support.mean(axis=3)
    batch_size = support.shape[0]
    query = np.reshape(query, (batch_size, -1, 1, nc))
    logits = - ((query - support)**2).sum(-1)
    pred = np.argmax(logits, axis=-1)
    pred = np.reshape(pred, (-1,))
    return pred, logits


def nearest_neighbour(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred, distance


def cosine_similarity(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred
