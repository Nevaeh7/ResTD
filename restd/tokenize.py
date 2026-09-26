"""Step 2: item embeddings, RQ training, final SIDs, and residual export."""

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import config_for, read_json, seed_everything, sha256, write_json
from .rq.datasets import EmbDataset
from .rq.rqvae import RQVAE
from .rq.export import export, build_model


def train_index(embedding, output, cfg, device, seed, max_steps=None):
    output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(**cfg, data_path=str(embedding), seed=seed)
    dataset = EmbDataset(embedding)
    model = RQVAE(
        in_dim=dataset.dim,
        **{
            key: cfg[key]
            for key in (
                "num_emb_list",
                "e_dim",
                "layers",
                "dropout_prob",
                "bn",
                "loss_type",
                "quant_loss_weight",
                "kmeans_init",
                "kmeans_iters",
                "sk_epsilons",
                "sk_iters",
            )
        },
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    # Stream the encoder in batches: full-catalog activations need not reside on the GPU.
    if cfg["kmeans_init"]:
        with torch.no_grad():
            encoded = []
            model.eval()
            loader = DataLoader(
                dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0
            )
            for values, _ in loader:
                encoded.append(model.encoder(values.to(device)).cpu())
            model.rq.vq_ini(torch.cat(encoded).to(device))
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    labels = {str(i): [] for i in range(len(cfg["num_emb_list"]))}
    steps = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        count = 0
        for values, item_ids in loader:
            values = values.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, quantized = model(values, labels)
            loss, _, _, _ = model.compute_loss(
                reconstructed, quant_loss, item_ids, quantized, xs=values
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite RQ loss")
            loss.backward()
            optimizer.step()
            steps += 1
            total += float(loss.detach()) * len(values)
            count += len(values)
            if max_steps and steps >= max_steps:
                break
        record = {"epoch": epoch + 1, "steps": steps, "loss": total / count}
        print(json.dumps(record), flush=True)
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        # Retain one optimizer checkpoint to bound storage during index training.
        temporary = output / "last.tmp.pt"
        torch.save(
            {
                "args": args,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "steps": steps,
            },
            temporary,
        )
        temporary.replace(output / "last.pt")
        if max_steps and steps >= max_steps:
            break
    (output / "last.pt").rename(output / "model.pt")
    write_json(
        output / "manifest.json",
        {
            "embedding_sha256": sha256(embedding),
            "config": cfg,
            "seed": seed,
            "steps": steps,
            "smoke": max_steps is not None,
            "checkpoint_sha256": sha256(output / "model.pt"),
        },
    )
    return output / "model.pt"


@torch.no_grad()
def generate_sids(
    checkpoint, embedding, output, device, batch_size=2048, iterations=20
):
    if output.exists():
        raise FileExistsError(output)
    dataset = EmbDataset(embedding)
    model, _ = build_model(checkpoint, dataset.dim)
    model.to(device).eval()
    labels = {str(i): [] for i in range(len(model.rq.vq_layers))}
    rows = []
    for values, _ in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        rows.append(
            model.get_indices(values.to(device), labels, use_sk=False).cpu().numpy()
        )
    numeric = np.concatenate(rows)
    for layer in model.rq.vq_layers[:-1]:
        layer.sk_epsilon = 0.0
    model.rq.vq_layers[-1].sk_epsilon = model.rq.vq_layers[-1].sk_epsilon or 0.003
    for iteration in range(iterations):
        groups = defaultdict(list)
        for index, sid in enumerate(numeric):
            groups[tuple(sid)].append(index)
        collisions = [ids for ids in groups.values() if len(ids) > 1]
        if not collisions:
            break
        print(
            f"Collision refinement {iteration + 1}: {len(collisions)} groups",
            flush=True,
        )
        for ids in collisions:
            values, _ = dataset[ids]
            numeric[ids] = (
                model.get_indices(values.to(device), labels, use_sk=True).cpu().numpy()
            )
    unique = len({tuple(sid) for sid in numeric})
    write_json(
        output,
        {
            str(i): [f"<{chr(97 + j)}_{int(code)}>" for j, code in enumerate(sid)]
            for i, sid in enumerate(numeric)
        },
    )
    write_json(
        output.with_suffix(".audit.json"),
        {
            "items": len(numeric),
            "unique_sids": unique,
            "residual_collisions": len(numeric) - unique,
            "collision_rate": 1 - unique / len(numeric),
            "rq_sha256": sha256(checkpoint),
            "embedding_sha256": sha256(embedding),
            "sid_sha256": sha256(output),
        },
    )


def run(args):
    if args.bundle:
        from .bundle import install

        print(install(args.bundle, args.locale, args.data_root, "tokenization"))
        return
    if args.rq_checkpoint and not Path(args.rq_checkpoint).is_file():
        raise FileNotFoundError(f"RQ checkpoint does not exist: {args.rq_checkpoint}")
    cfg = config_for(args.locale, args.config)
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    directory = Path(args.data_root) / f"esci_{args.locale}"
    dataset = directory.name
    embedding = directory / f"{dataset}.emb-bert.npy"
    if not embedding.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "restd.embeddings",
                "--dataset",
                dataset,
                "--root",
                str(args.data_root),
                "--model_name",
                args.embedding_model or cfg["embedding_model"],
                "--batch_size",
                str(args.embedding_batch_size),
                "--gpu_id",
                "-1" if args.device == "cpu" else args.device.split(":")[-1],
            ],
            check=True,
        )
    rq_cfg = cfg["rq"].copy()
    if args.rq_epochs is not None:
        rq_cfg["epochs"] = args.rq_epochs
    output = Path(args.output_root) / args.locale / "tokenizer"
    checkpoint = Path(args.rq_checkpoint) if args.rq_checkpoint else output / "model.pt"
    if checkpoint.exists() and args.rq_checkpoint is None:
        manifest = read_json(output / "manifest.json")
        if (
            manifest["embedding_sha256"] != sha256(embedding)
            or manifest["config"] != rq_cfg
            or manifest["seed"] != args.seed
            or manifest["smoke"] != (args.max_steps is not None)
        ):
            raise ValueError(
                "Existing RQ checkpoint differs from this run; use a fresh output root"
            )
    if not checkpoint.exists():
        checkpoint = train_index(
            embedding, output, rq_cfg, args.device, args.seed, args.max_steps
        )
    index = directory / f"{dataset}.index.json"
    if index.exists():
        audit = read_json(index.with_suffix(".audit.json"))
        if (
            audit["rq_sha256"] != sha256(checkpoint)
            or audit["embedding_sha256"] != sha256(embedding)
            or audit["sid_sha256"] != sha256(index)
        ):
            raise ValueError("Existing SID index belongs to different inputs")
    else:
        generate_sids(checkpoint, embedding, index, args.device)
    residual = directory / "residuals"
    if residual.exists():
        meta = read_json(residual / "residual_meta.json")
        report = read_json(residual / "reconstruction_report.json")
        if (
            report["status"] != "accepted"
            or meta["rq_checkpoint_hash"] != sha256(checkpoint)
            or meta["sid_file_hash"] != sha256(index)
            or meta["embedding_file_hash"] != sha256(embedding)
        ):
            raise ValueError(
                "Existing residual export is incomplete or belongs to different inputs"
            )
    else:
        export(
            SimpleNamespace(
                rq_checkpoint=str(checkpoint),
                embedding_data_path=str(embedding),
                final_sid_json=str(index),
                product_id_to_index=str(directory / "product_id_to_index.json"),
                output_dir=str(residual),
                device=args.device,
                batch_size=2048,
                num_workers=0,
                teacher_temperature=cfg["restd"]["teacher_temperature"],
                reconstruction_samples=100,
                reconstruction_tolerance=1e-3,
                seed=args.seed,
            )
        )
    print(f"Tokenization complete: {index} and {residual}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locale", choices=["us", "es", "jp"], required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--config")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--embedding-model")
    p.add_argument("--embedding-batch-size", type=int, default=64)
    p.add_argument("--rq-checkpoint")
    p.add_argument("--bundle", help="Install the supplied SID index and residual cache")
    p.add_argument("--rq-epochs", type=int)
    p.add_argument(
        "--max-steps",
        type=int,
        help="Limit index training to this number of optimizer updates",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
