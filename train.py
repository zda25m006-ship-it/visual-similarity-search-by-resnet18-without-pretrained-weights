import argparse
import os
import random
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import build_cifar100_datasets, build_dataloaders
from model import ResNet18Embedder, build_optimizer, CosineWarmupScheduler
import sys
import importlib
import loss as _loss_mod
importlib.reload(_loss_mod)
TripletLoss = getattr(_loss_mod, "TripletLoss", getattr(_loss_mod, "TripletMarginLossFromScratch", None))
NTXentLoss = getattr(_loss_mod, "NTXentLoss", getattr(_loss_mod, "NTXentLossFromScratch", None))
evaluate_validation_triplets = _loss_mod.evaluate_validation_triplets
from retrieval import extract_embeddings_to_numpy, BruteForceRetriever, NumPyKDTree, benchmark_search_latency
from eval import (
    evaluate_embeddings,
    evaluate_with_reranking,
    generate_ablation_finding,
    format_ablation_markdown_table,
    plot_tsne_gallery,
)


def set_global_seed(seed=42):
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_retrieval_grid(
    query_dataset,
    gallery_dataset,
    query_embeddings,
    gallery_embeddings,
    gallery_labels,
    class_names,
    save_path="artifacts/top5_retrievals.png",
    num_queries=5,
    top_k=5,
    exclude_self=True,
):
    """Save a figure showing query images and their top-5 retrieved gallery matches."""
    retriever = BruteForceRetriever(gallery_embeddings, gallery_labels, class_names)
    rng = np.random.RandomState(42)
    q_indices = rng.choice(len(query_embeddings), size=min(num_queries, len(query_embeddings)), replace=False)

    exclude = q_indices if exclude_self else None
    results = retriever.search_l2(query_embeddings[q_indices], top_k=top_k, exclude_self_indices=exclude)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(len(q_indices), top_k + 1, figsize=(2.4 * (top_k + 1), 2.5 * len(q_indices)), dpi=150)
    if len(q_indices) == 1:
        axes = axes[None, :]

    for r_i, q_idx in enumerate(q_indices):
        q_img = query_dataset.get_raw_image(int(q_idx))
        q_lbl = int(query_dataset.labels[int(q_idx)])
        q_name = class_names[q_lbl]

        ax_q = axes[r_i, 0]
        ax_q.imshow(q_img)
        ax_q.set_title(f"Query: {q_name}", fontsize=9, fontweight="bold")
        ax_q.axis("off")

        for c_j in range(top_k):
            g_idx = int(results["indices"][r_i, c_j])
            dist = float(results["distances"][r_i, c_j])
            cos_s = float(results["cosine_similarities"][r_i, c_j])
            g_lbl = int(results["labels"][r_i, c_j])
            g_name = class_names[g_lbl]
            g_img = gallery_dataset.get_raw_image(g_idx)

            match = g_lbl == q_lbl
            color = "green" if match else "crimson"

            ax = axes[r_i, c_j + 1]
            ax.imshow(g_img)
            ax.set_title(f"#{c_j + 1} {g_name}\nL2={dist:.3f} cos={cos_s:.2f}", fontsize=8, color=color)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def train_single_experiment(
    datasets,
    class_names,
    device,
    epochs=35,
    batch_size=128,
    k_samples=4,
    lr=1e-3,
    optimizer_name="adam",
    margin=0.3,
    embedding_dim=128,
    mining="hard",
    loss_type="triplet",
    weight_decay=1e-4,
    seed=42,
    save_checkpoint_path=None,
    max_batches_per_epoch=None,
):
    """Train the embedding model and evaluate on validation and test splits."""
    set_global_seed(seed)

    use_pk = mining == "hard" or loss_type == "ntxent"
    loaders = build_dataloaders(
        datasets,
        batch_size=batch_size,
        k_samples=k_samples,
        num_workers=2 if torch.cuda.is_available() else 0,
        seed=seed,
        use_pk_sampler=use_pk,
    )

    model = ResNet18Embedder(embedding_dim=embedding_dim, cifar_stem=True).to(device)
    optimizer = build_optimizer(model, optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, base_lr=lr, total_epochs=epochs, warmup_epochs=min(3, max(1, epochs // 10)))

    criterion = NTXentLoss(temperature=0.1) if loss_type == "ntxent" else TripletLoss(margin=margin, mining=mining)

    best_val_acc = -1.0
    best_state = None
    history = []

    print(f"Training: loss={loss_type} mining={mining} dim={embedding_dim} margin={margin} opt={optimizer_name} lr={lr} bs={batch_size} epochs={epochs}")

    for epoch in range(epochs):
        t0 = time.time()
        cur_lr = scheduler.step(epoch)
        model.train()

        run_loss = 0.0
        run_acc = 0.0
        steps = 0

        for b_idx, (images, labels, _) in enumerate(loaders["train"]):
            if max_batches_per_epoch is not None and b_idx >= max_batches_per_epoch:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            emb = model(images)
            loss, stats = criterion(emb, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            run_acc += stats["triplet_acc"]
            steps += 1

        tr_loss = run_loss / max(1, steps)
        tr_acc = (run_acc / max(1, steps)) * 100.0

        val_stats = evaluate_validation_triplets(model, loaders["val_triplet"], margin=margin, device=device)
        val_hard = val_stats["val_hard_triplet_acc"] * 100.0
        val_all = val_stats["val_all_triplet_acc"] * 100.0
        val_score = max(val_hard, val_all)

        if val_score > best_val_acc:
            best_val_acc = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_checkpoint_path is not None:
                os.makedirs(os.path.dirname(os.path.abspath(save_checkpoint_path)), exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "embedding_dim": embedding_dim,
                        "margin": margin,
                        "lr": lr,
                        "batch_size": batch_size,
                        "optimizer": optimizer_name,
                        "val_triplet_acc": best_val_acc,
                        "class_names": class_names,
                    },
                    save_checkpoint_path,
                )

        dt = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "lr": cur_lr,
            "train_loss": tr_loss,
            "train_triplet_acc": tr_acc,
            "val_loss": val_stats["val_loss"],
            "val_hard_triplet_acc": val_hard,
            "val_all_triplet_acc": val_all,
        })
        if (epoch + 1) % max(1, epochs // 7) == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(f"Epoch {epoch + 1:02d}/{epochs:02d} | lr={cur_lr:.5f} | loss={tr_loss:.4f} | tr_acc={tr_acc:.1f}% | val_hard={val_hard:.1f}% | val_all={val_all:.1f}% | {dt:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_emb, test_lbl, test_idx = extract_embeddings_to_numpy(model, loaders["test"], device)
    test_metrics = evaluate_embeddings(test_emb, test_lbl, ks=(1, 5, 10))

    return {
        "model": model,
        "loaders": loaders,
        "history": history,
        "val_triplet_acc": best_val_acc,
        "test_embeddings": test_emb,
        "test_labels": test_lbl,
        "test_indices": test_idx,
        "test_metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--ablation-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "nadam", "sgd"])
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--mining", type=str, default="hard", choices=["hard", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--run-stretch", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets, class_names = build_cifar100_datasets(data_root=args.data_root, seed=args.seed)
    max_batches = 8 if args.quick_test else None
    ckpt_path = os.path.join(args.output_dir, "best_resnet18_triplet.pt")

    primary = train_single_experiment(
        datasets=datasets,
        class_names=class_names,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k_samples=args.k_samples,
        lr=args.lr,
        optimizer_name=args.optimizer,
        margin=args.margin,
        embedding_dim=args.embedding_dim,
        mining=args.mining,
        seed=args.seed,
        save_checkpoint_path=ckpt_path,
        max_batches_per_epoch=max_batches,
    )

    emb_prefix = os.path.join(args.output_dir, "test")
    np.save(f"{emb_prefix}_embeddings.npy", primary["test_embeddings"])
    np.save(f"{emb_prefix}_labels.npy", primary["test_labels"])
    np.save(f"{emb_prefix}_indices.npy", primary["test_indices"])

    m = primary["test_metrics"]
    print(f"Val Triplet Acc: {primary['val_triplet_acc']:.2f}%")
    print(f"Recall@1: {m['Recall@1']:.2f}% | Recall@5: {m['Recall@5']:.2f}% | Recall@10: {m['Recall@10']:.2f}% | mAP: {m['mAP']:.2f}%")

    save_retrieval_grid(
        datasets["test"],
        datasets["test"],
        primary["test_embeddings"],
        primary["test_embeddings"],
        primary["test_labels"],
        class_names,
        save_path=os.path.join(args.output_dir, "top5_retrievals.png"),
    )
    plot_tsne_gallery(
        primary["test_embeddings"],
        primary["test_labels"],
        class_names,
        save_path=os.path.join(args.output_dir, "tsne_test_gallery.png"),
    )

    if args.run_ablations:
        records = []
        ab_epochs = 2 if args.quick_test else args.ablation_epochs

        m_res, m_runs = {}, {}
        for mar in [0.1, 0.3, 0.5]:
            run = primary if (np.isclose(mar, args.margin) and args.embedding_dim == 128 and args.mining == "hard") else train_single_experiment(
                datasets, class_names, device, epochs=ab_epochs, batch_size=args.batch_size, k_samples=args.k_samples,
                lr=args.lr, optimizer_name=args.optimizer, margin=mar, embedding_dim=128, mining="hard",
                seed=args.seed, max_batches_per_epoch=max_batches
            )
            m_res[str(mar)] = run["test_metrics"]
            m_runs[str(mar)] = run
        m_find = generate_ablation_finding("Margin", m_res)
        for k_str, run in m_runs.items():
            tm = run["test_metrics"]
            records.append({
                "ablation": "Margin", "variant": k_str, "lr": args.lr, "batch_size": args.batch_size,
                "margin": float(k_str), "embedding_dim": 128, "mining": "hard", "val_triplet_acc": run["val_triplet_acc"],
                "Recall@1": tm["Recall@1"], "Recall@5": tm["Recall@5"], "Recall@10": tm["Recall@10"], "mAP": tm["mAP"],
                "finding": m_find,
            })

        d_res, d_runs = {}, {}
        for dim in [64, 128, 256]:
            run = primary if (dim == args.embedding_dim and np.isclose(args.margin, 0.3) and args.mining == "hard") else train_single_experiment(
                datasets, class_names, device, epochs=ab_epochs, batch_size=args.batch_size, k_samples=args.k_samples,
                lr=args.lr, optimizer_name=args.optimizer, margin=0.3, embedding_dim=dim, mining="hard",
                seed=args.seed, max_batches_per_epoch=max_batches
            )
            d_res[str(dim)] = run["test_metrics"]
            d_runs[str(dim)] = run
        d_find = generate_ablation_finding("Embedding dim", d_res)
        for k_str, run in d_runs.items():
            tm = run["test_metrics"]
            records.append({
                "ablation": "Embedding dim", "variant": k_str, "lr": args.lr, "batch_size": args.batch_size,
                "margin": 0.3, "embedding_dim": int(k_str), "mining": "hard", "val_triplet_acc": run["val_triplet_acc"],
                "Recall@1": tm["Recall@1"], "Recall@5": tm["Recall@5"], "Recall@10": tm["Recall@10"], "mAP": tm["mAP"],
                "finding": d_find,
            })

        s_res, s_runs = {}, {}
        for strat in ["random", "hard"]:
            run = primary if (strat == args.mining and args.embedding_dim == 128 and np.isclose(args.margin, 0.3)) else train_single_experiment(
                datasets, class_names, device, epochs=ab_epochs, batch_size=args.batch_size, k_samples=args.k_samples,
                lr=args.lr, optimizer_name=args.optimizer, margin=0.3, embedding_dim=128, mining=strat,
                seed=args.seed, max_batches_per_epoch=max_batches
            )
            s_res[strat] = run["test_metrics"]
            s_runs[strat] = run
        s_find = generate_ablation_finding("Mining strategy", s_res)
        for k_str, run in s_runs.items():
            tm = run["test_metrics"]
            records.append({
                "ablation": "Mining strategy", "variant": k_str, "lr": args.lr, "batch_size": args.batch_size,
                "margin": 0.3, "embedding_dim": 128, "mining": k_str, "val_triplet_acc": run["val_triplet_acc"],
                "Recall@1": tm["Recall@1"], "Recall@5": tm["Recall@5"], "Recall@10": tm["Recall@10"], "mAP": tm["mAP"],
                "finding": s_find,
            })

        table_str = format_ablation_markdown_table(records)
        print(table_str)
        with open(os.path.join(args.output_dir, "ablation_table.md"), "w", encoding="utf-8") as f:
            f.write(table_str + "\n")

    if args.run_stretch:
        q_emb, q_lbl, _ = extract_embeddings_to_numpy(primary["model"], primary["loaders"]["test_query"], device)
        g_emb, g_lbl, _ = extract_embeddings_to_numpy(primary["model"], primary["loaders"]["test_gallery"], device)
        rr = evaluate_with_reranking(q_emb, q_lbl, g_emb, g_lbl)
        print("Baseline mAP:", rr["baseline"]["mAP"], "Reranked mAP:", rr["reranked"]["mAP"])
        lat = benchmark_search_latency(q_emb, g_emb, top_k=5)
        print("Latency (ms):", lat)


if __name__ == "__main__":
    main()
