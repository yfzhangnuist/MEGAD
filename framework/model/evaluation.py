import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import  roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from detectors import LSHiForest

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

def multiple_regression_analysis(X, y):
    y = torch.tensor(y, dtype=torch.float32, device='cuda')
    X = torch.cat((torch.ones(X.shape[0], 1, device='cuda', requires_grad=False), X), dim=1)
    model = nn.Linear(X.shape[1], 1).to('cuda')
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(3): 
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y.view(-1, 1))
        loss.backward(retain_graph=True)
        for param in model.parameters():
            param.data = torch.clamp(param.data, min=0)
        optimizer.step()
    return model.weight.detach()[:, 1:]



def eva(k, labels, emb):
    labels = np.array([1 if x != 0 else x for x in labels])
    embeddings = emb.cpu().data.numpy()
    num_ensemblers=100
    clf= LSHiForest('L2SH', num_ensemblers)
    clf.fit(embeddings)
    y_pred = clf.decision_function(embeddings)
    if  y_pred.mean()>-0.5:
        y_pred=-y_pred
    AUCROC= roc_auc_score(labels, y_pred)
    AUCPR= average_precision_score(labels, y_pred)   
    if k<2:
        dimension_variance_weight=multiple_regression_analysis(emb, y_pred)
        return AUCROC, AUCPR, dimension_variance_weight
    else:
        return AUCROC, AUCPR



