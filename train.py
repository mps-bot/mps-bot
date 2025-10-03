import argparse, time, copy
import numpy as np
import torch
import torch.nn as nn

from datasets import load_mgtab, load_twibot22, make_twibot22_loaders
from models import build_simplexes, MGTABModel, TwiBot22Model
from eval import evaluate_fullgraph, evaluate_loader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train_mgtab(args, device):

    d = load_mgtab()
    x = d["x"].to(device)
    y = d["y"].to(device)
    edge_index = d["edge_index"].to(device)
    edge_type  = d["edge_type"].to(device)

    num_nodes = x.size(0)
    rng = np.random.RandomState(args.random_seed)
    perm = rng.permutation(np.arange(num_nodes))
    n_tr = int(0.7 * num_nodes)
    n_va = int(0.9 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device); train_mask[perm[:n_tr]]       = True
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool, device=device); val_mask[perm[n_tr:n_va]]    = True
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool, device=device); test_mask[perm[n_va:]]       = True
    
    rel_ids = [0, 1]
    rel_edge_index_list = [edge_index[:, edge_type == rid] for rid in rel_ids]

    rel_triangles_cpu = []
    for ei in rel_edge_index_list:
        undir = torch.cat([ei, ei.flip(0)], dim=1).unique(dim=1).cpu()
        rel_triangles_cpu.append(build_simplexes(undir))

    model = MGTABModel(
            in_dim=x.size(1),
            hidden=args.mgtab_hidden,   
            proj_channels=32,           
            dropout=args.dropout,
            num_rel=2
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    best_val_acc = 0.0
    best_state = None

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        tris_gpu = [t.to(device) for t in rel_triangles_cpu]
        logits = model(x, rel_edge_index_list, tris_gpu)
        loss_tr = criterion(logits[train_mask], y[train_mask])
        loss_tr.backward()
        optimizer.step()

        loss_va, acc_va, *_ = evaluate_fullgraph(model, x, y, rel_edge_index_list, tris_gpu, val_mask, criterion, device)
        loss_te, acc_te, _, _, _, auc_te = evaluate_fullgraph(model, x, y, rel_edge_index_list, tris_gpu, test_mask, criterion, device)

        if acc_va > best_val_acc:
            best_val_acc = acc_va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 1 or epoch == args.epochs:
            print(f"[MGTAB] Epoch {epoch:03d} | loss_train: {loss_tr.item():.4f} "
                  f"val_acc: {acc_va:.4f} test_acc: {acc_te:.4f} test_auc: {auc_te:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    tris_gpu = [t.to(device) for t in rel_triangles_cpu]
    loss_te, acc_te, f1_te, prec_te, rec_te, auc_te = evaluate_fullgraph(model, x, y, rel_edge_index_list, tris_gpu, test_mask, criterion, device)
    print("\n>>> [MGTAB] Test set best results:",
          f"test_accuracy={acc_te:.4f} ",
          f"precision={prec_te:.4f} ",
          f"recall={rec_te:.4f} ",
          f"f1_score={f1_te:.4f} ",
          f"auc={auc_te:.4f}")
    print(f"Total time: {time.time() - t0:.1f}s")


def train_twibot22(args, device):
    data = load_twibot22()
    train_loader, val_loader, test_loader = make_twibot22_loaders(
        data, batch_size=args.batch_size, test_batch_size=args.test_batch_size, num_neighbors=args.num_neighbors
    )

    model = TwiBot22Model(
        cat_num=args.cat_num, numeric_num=args.numeric_num,
        tweet_channel=args.tweet_channel, des_channel=args.des_channel,
        hidden=args.twibot22_hidden, proj_channels=64,          
        dropout=args.dropout,num_rel=2
    ).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=0)

    best_val_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(device)

            ei_fol  = batch.edge_index_dict[("user","follower","user")]
            ei_fing = batch.edge_index_dict[("user","following","user")]
            edge_list = [ei_fol, ei_fing]
            tri_list = []
            dev = batch["user"].x.device
            for ei in edge_list:
                undir = torch.cat([ei, ei.flip(0)], dim=1).unique(dim=1)
                tri = build_simplexes(undir.cpu()).to(dev)
                tri_list.append(tri)

            optimizer.zero_grad()
            logits = model(batch, edge_list, tri_list)
            bs = int(batch["user"].batch_size)
            loss = criterion(logits[:bs], batch["user"].y[:bs])
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        loss_va, acc_va, *_ = evaluate_loader(model, val_loader, criterion, device, build_simplexes)
        loss_te, acc_te, _, _, _, auc_te = evaluate_loader(model, test_loader, criterion, device, build_simplexes)

        if acc_va > best_val_acc:
            best_val_acc = acc_va
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 1 or epoch == args.epochs:
            print(f"[TwiBot-22] Epoch {epoch:03d} | loss_train: {np.mean(epoch_losses) if epoch_losses else 0.0:.4f} "
                  f"val_acc: {acc_va:.4f} test_acc: {acc_te:.4f} test_auc: {auc_te:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    loss_te, acc_te, f1_te, prec_te, rec_te, auc_te = evaluate_loader(model, test_loader, criterion, device, build_simplexes)
    print("\n>>> [TwiBot-22] Test set best results:",
          f"test_accuracy={acc_te:.4f} ",
          f"precision={prec_te:.4f} ",
          f"recall={rec_te:.4f} ",
          f"f1_score={f1_te:.4f} ",
          f"auc={auc_te:.4f}")
    print(f"Total time: {time.time() - t0:.1f}s")

def main():
    ap = argparse.ArgumentParser("MPS-Bot")
    ap.add_argument("--dataset", choices=["mgtab", "twibot22"], required=True)

    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2_reg", type=float, default=3e-5)
    ap.add_argument("--random_seed", type=int, default=422)

    # MGTAB 
    ap.add_argument("--mgtab_hidden", type=int, default=32)

    # TwiBot-22 
    ap.add_argument("--twibot22_hidden", type=int, default=128)
    ap.add_argument("--numeric_num",   type=int, default=5)
    ap.add_argument("--cat_num",       type=int, default=3)
    ap.add_argument("--des_channel",   type=int, default=768)
    ap.add_argument("--tweet_channel", type=int, default=768)
    ap.add_argument("--batch_size",    type=int, default=256)
    ap.add_argument("--test_batch_size", type=int, default=200)
    ap.add_argument("--num_neighbors", type=int, default=256)
    ap.add_argument("--tmax", type=int, default=16)


    args = ap.parse_args()
    torch.manual_seed(args.random_seed); np.random.seed(args.random_seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mgtab":
        train_mgtab(args, device)
    else:
        train_twibot22(args, device)

if __name__ == "__main__":
    main()

