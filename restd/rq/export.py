#!/usr/bin/env python3
"""Export final-SID-aligned residual trajectories from a trained RQ-VAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SID_TOKEN_RE = re.compile(r"^<([a-z])_([0-9]+)>$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_final_sids(path: Path, codebook_sizes: list[int]) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    expected_keys = {str(index) for index in range(len(raw))}
    if set(raw) != expected_keys:
        raise ValueError("final SID JSON keys must be contiguous item indices")
    numeric = np.empty((len(raw), len(codebook_sizes)), dtype=np.int16)
    for item_idx in range(len(raw)):
        tokens = raw[str(item_idx)]
        if not isinstance(tokens, list) or len(tokens) != len(codebook_sizes):
            raise ValueError(f"item {item_idx} has an invalid SID length")
        for level, (token, codebook_size) in enumerate(zip(tokens, codebook_sizes)):
            match = SID_TOKEN_RE.fullmatch(token)
            expected_prefix = chr(ord("a") + level)
            if match is None or match.group(1) != expected_prefix:
                raise ValueError(f"item {item_idx} has invalid SID token {token!r}")
            code = int(match.group(2))
            if not 0 <= code < codebook_size:
                raise ValueError(
                    f"item {item_idx} code {code} is outside level {level}"
                )
            if code > np.iinfo(np.int16).max:
                raise ValueError(
                    "codebook index does not fit the specified int16 cache"
                )
            numeric[item_idx, level] = code
    return numeric


def build_model(checkpoint_path: Path, embedding_dim: int):
    # Load the model implementation only when reconstructing a checkpoint.
    from .rqvae import RQVAE

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model = RQVAE(
        in_dim=embedding_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, args


class TeacherStatistics:
    def __init__(self, num_codebooks: int):
        self.count = [0] * num_codebooks
        self.top1 = [0] * num_codebooks
        self.top5 = [0] * num_codebooks
        self.reciprocal_rank = [0.0] * num_codebooks
        self.entropy = [0.0] * num_codebooks

    def update(
        self,
        level: int,
        residual: torch.Tensor,
        codebook: torch.Tensor,
        target: torch.Tensor,
        temperature: float,
    ) -> None:
        residual = residual.float()
        codebook = codebook.float()
        distance = (
            residual.square().sum(-1, keepdim=True)
            + codebook.square().sum(-1).unsqueeze(0)
            - 2.0 * residual @ codebook.t()
        ).clamp_min(0.0)
        order = distance.argsort(dim=-1)
        matches = order.eq(target.unsqueeze(1))
        ranks = matches.float().argmax(dim=1) + 1
        probabilities = torch.softmax(-distance / temperature, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        self.count[level] += target.numel()
        self.top1[level] += int(ranks.eq(1).sum())
        self.top5[level] += int(ranks.le(min(5, codebook.shape[0])).sum())
        self.reciprocal_rank[level] += float((1.0 / ranks.float()).sum())
        self.entropy[level] += float(entropy.sum())

    def as_dict(self) -> dict:
        layers = []
        for level, count in enumerate(self.count):
            layers.append(
                {
                    "level": level,
                    "count": count,
                    "top1": self.top1[level] / count,
                    "top5": self.top5[level] / count,
                    "mrr": self.reciprocal_rank[level] / count,
                    "entropy": self.entropy[level] / count,
                }
            )
        return {"layers": layers}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def reconstruction_error(current, codeword, observed, absolute_tolerance):
    """Check the recurrence allowing two independently rounded FP16 states.

    Each cached state incurs magnitude-dependent rounding. A fixed absolute
    threshold alone incorrectly rejects valid trajectories whose values are large.
    """
    if absolute_tolerance < 0:
        raise ValueError("Reconstruction tolerance must be nonnegative")
    if not all(np.isfinite(value).all() for value in (current, codeword, observed)):
        raise ValueError("Non-finite values in the residual reconstruction")
    error = np.abs(current - codeword - observed)
    budget = absolute_tolerance + np.finfo(np.float16).eps * (
        np.abs(current) + np.abs(observed)
    )
    return float(error.max()), float((error / np.maximum(budget, 1e-12)).max())


def export(args) -> None:
    from .datasets import EmbDataset

    checkpoint_path = Path(args.rq_checkpoint).resolve(strict=True)
    embedding_path = Path(args.embedding_data_path).resolve(strict=True)
    sid_path = Path(args.final_sid_json).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to reuse non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = EmbDataset(str(embedding_path))
    model, checkpoint_args = build_model(checkpoint_path, dataset.dim)
    codebooks = [
        layer.embedding.weight.detach().cpu().float() for layer in model.rq.vq_layers
    ]
    codebook_sizes = [int(codebook.shape[0]) for codebook in codebooks]
    sid_numeric = parse_final_sids(sid_path, codebook_sizes)
    if len(dataset) != sid_numeric.shape[0]:
        raise ValueError("embedding count and final SID count differ")

    device = torch.device(args.device)
    model = model.to(device).eval()
    rq_dim = int(codebooks[0].shape[1])
    trace_path = output_dir / "residual_trace.fp16.npy"
    sid_numeric_path = output_dir / "sid_numeric.int16.npy"
    trace = np.lib.format.open_memmap(
        trace_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), len(codebooks) + 1, rq_dim),
    )
    np.save(sid_numeric_path, sid_numeric, allow_pickle=False)
    statistics = TeacherStatistics(len(codebooks))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for embeddings, item_indices in tqdm(loader, desc="export ResTD residuals"):
            embeddings = embeddings.to(device, non_blocking=True)
            encoded = model.encoder(embeddings)
            numeric = torch.from_numpy(
                np.array(sid_numeric[item_indices.numpy()], dtype=np.int64, copy=True)
            ).to(device)
            residual = encoded
            states = [residual]
            for level, codebook_cpu in enumerate(codebooks):
                codebook = codebook_cpu.to(device)
                target = numeric[:, level]
                statistics.update(
                    level,
                    residual,
                    codebook,
                    target,
                    args.teacher_temperature,
                )
                residual = residual - codebook[target]
                states.append(residual)
            batch_trace = torch.stack(states, dim=1).cpu().numpy().astype(np.float16)
            trace[item_indices.numpy()] = batch_trace
    trace.flush()
    del trace

    torch.save(codebooks, output_dir / "codebooks.pt")
    metadata = {
        "schema": "restd.residual_cache.v1",
        "num_items": len(dataset),
        "num_codebooks": len(codebooks),
        "rq_dim": rq_dim,
        "codebook_sizes": codebook_sizes,
        "storage_dtype": "float16",
        "rq_checkpoint_hash": sha256_file(checkpoint_path),
        "sid_file_hash": sha256_file(sid_path),
        "embedding_file_hash": sha256_file(embedding_path),
        "teacher_temperature": args.teacher_temperature,
    }
    atomic_json(output_dir / "residual_meta.json", metadata)
    diagnostics = statistics.as_dict()
    diagnostics.update(
        {
            "schema": "restd.teacher_diagnostics.v1",
            "teacher_temperature": args.teacher_temperature,
        }
    )
    atomic_json(output_dir / "teacher_diagnostics.json", diagnostics)

    receipt = {
        "schema": "restd.item_index_receipt.v1",
        "item_count": len(dataset),
        "item_order": "contiguous numeric keys in final SID JSON",
        "final_sid_json": str(sid_path),
        "sid_file_hash": metadata["sid_file_hash"],
    }
    if args.product_id_to_index:
        mapping_path = Path(args.product_id_to_index).resolve(strict=True)
        with mapping_path.open("r", encoding="utf-8") as handle:
            mapping = json.load(handle)
        if set(map(int, mapping.values())) != set(range(len(dataset))):
            raise ValueError("product_id_to_index is not a catalog permutation")
        receipt["product_id_to_index"] = str(mapping_path)
        receipt["product_id_to_index_hash"] = sha256_file(mapping_path)
    atomic_json(output_dir / "item_index_receipt.json", receipt)

    # Re-open and reconstruct a deterministic sample from the stored FP16 cache.
    cached = np.load(trace_path, mmap_mode="r", allow_pickle=False)
    sample_count = min(args.reconstruction_samples, len(dataset))
    generator = np.random.default_rng(args.seed)
    sample_indices = generator.choice(len(dataset), sample_count, replace=False)
    max_error = 0.0
    max_error_ratio = 0.0
    tolerance = args.reconstruction_tolerance
    for item_idx in sample_indices:
        current = cached[item_idx, 0].astype(np.float32)
        for level, codebook in enumerate(codebooks):
            codeword = codebook[int(sid_numeric[item_idx, level])].numpy()
            observed = cached[item_idx, level + 1].astype(np.float32)
            error, ratio = reconstruction_error(current, codeword, observed, tolerance)
            max_error = max(max_error, error)
            max_error_ratio = max(max_error_ratio, ratio)
            current = observed
    tolerance = args.reconstruction_tolerance
    reconstruction = {
        "schema": "restd.reconstruction.v1",
        "sample_count": sample_count,
        "max_absolute_error": max_error,
        "tolerance": tolerance,
        "roundoff_allowance": "float16_epsilon * (abs(current) + abs(next))",
        "max_error_budget_ratio": max_error_ratio,
        "status": "accepted" if max_error_ratio <= 1.0 else "rejected",
    }
    atomic_json(output_dir / "reconstruction_report.json", reconstruction)
    if max_error_ratio > 1.0:
        raise RuntimeError(
            f"Residual reconstruction exceeds its FP16 error budget by a factor of {max_error_ratio}"
        )
    print(
        json.dumps(
            {"metadata": metadata, "reconstruction": reconstruction}, sort_keys=True
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq_checkpoint", required=True)
    parser.add_argument("--embedding_data_path", required=True)
    parser.add_argument("--final_sid_json", required=True)
    parser.add_argument("--product_id_to_index")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--teacher_temperature", type=float, default=0.2)
    parser.add_argument("--reconstruction_samples", type=int, default=100)
    parser.add_argument("--reconstruction_tolerance", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
