# eval.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

@torch.no_grad()
def evaluate_fullgraph(model, x, y, edge_index_list, triangles_list, mask, criterion, device):
    model.eval()
    logits = model(x, edge_index_list, triangles_list)
    logits_m = logits[mask]; y_m = y[mask]
    loss = criterion(logits_m, y_m).item()

    probs = F.softmax(logits_m, dim=1)[:, 1].detach().cpu().numpy()
    preds = logits_m.argmax(dim=1).detach().cpu().numpy()
    labels = y_m.detach().cpu().numpy()

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average='macro')             
    prec = precision_score(labels, preds, average='macro')
    rec  = recall_score(labels, preds, average='macro')
    auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    return loss, acc, f1, prec, rec, auc

@torch.no_grad()
def evaluate_loader(model, loader, criterion, device, build_triangles_fn):
    model.eval()
    losses, y_all, pred_all, prob_all = [], [], [], []
    for batch in loader:
        batch = batch.to(device)

        ei_fol  = batch.edge_index_dict[("user","follower","user")]
        ei_fing = batch.edge_index_dict[("user","following","user")]
        edge_list = [ei_fol, ei_fing]

        tri_list = []
        dev = batch["user"].x.device
        for ei in edge_list:
            undir = torch.cat([ei, ei.flip(0)], dim=1).unique(dim=1)
            tri = build_triangles_fn(undir.cpu()).to(dev)
            tri_list.append(tri)

        logits = model(batch, edge_list, tri_list)
        bs = int(batch["user"].batch_size)
        logits_t = logits[:bs]
        y = batch["user"].y[:bs].long()

        loss = criterion(logits_t, y).item()
        losses.append(loss)

        probs = torch.softmax(logits_t, dim=1)[:, 1].cpu().numpy()
        preds = logits_t.argmax(dim=1).cpu().numpy()
        y_np  = y.cpu().numpy()
        y_all.append(y_np); pred_all.append(preds); prob_all.append(probs)

    if not y_all:
        return 0, 0, 0, 0, 0, 0
    y_all   = np.concatenate(y_all)
    pred_all= np.concatenate(pred_all)
    prob_all= np.concatenate(prob_all)
    acc  = accuracy_score(y_all, pred_all)
    f1   = f1_score(y_all, pred_all)            
    prec = precision_score(y_all, pred_all)
    rec  = recall_score(y_all, pred_all)
    auc  = roc_auc_score(y_all, prob_all) if len(np.unique(y_all)) > 1 else 0.0
    return float(np.mean(losses)), acc, f1, prec, rec, auc
