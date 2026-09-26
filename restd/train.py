"""Step 3: train the retriever, continue with ResTD, and evaluate."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import torch
from torch.utils.data import Subset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from .common import config_for, read_json, seed_everything, sha256, write_json
from .data import Catalog, RetrievalDataset, ResidualCache, Collator, query_split
from .modeling import ResTDT5, ResTDMT5


class ResTDTrainer(Trainer):
    def __init__(self, *args, auxiliary=True, gradient_check=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Loss is already a batch mean. Trainer must perform accumulation scaling.
        self.model_accepts_loss_kwargs = False
        self.auxiliary = auxiliary
        self.gradient_check = gradient_check
        self.gradient_checked = False
        self.logged_steps = set()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        actual = model.module if hasattr(model, "module") else model
        actual.set_restd_progress(self.state.global_step, self.state.max_steps)
        output = model(**inputs, restd_enabled=self.auxiliary)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite training loss")
        if (
            self.gradient_check
            and self.auxiliary
            and model.training
            and not self.gradient_checked
            and self.state.global_step > 0
        ):
            parameter = actual.decoder.block[-1].layer[0].SelfAttention.o.weight
            gradient = torch.autograd.grad(
                actual._last_loss_restd, parameter, retain_graph=True
            )[0]
            norm = float(gradient.float().norm())
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError(
                    "The ResTD objective did not reach the shared decoder"
                )
            write_json(
                Path(self.args.output_dir) / "gradient_check.json",
                {
                    "auxiliary_decoder_gradient_norm": norm,
                    "step": self.state.global_step,
                },
            )
            self.gradient_checked = True
        step = self.state.global_step
        if (
            model.training
            and step not in self.logged_steps
            and (
                step % self.args.logging_steps == 0 or step + 1 == self.state.max_steps
            )
        ):
            record = {
                "step": step,
                "total_loss": float(output.loss.detach()),
                "sid_loss": float(actual._last_loss_lm.detach()),
                "restd_loss": float(actual._last_loss_restd.detach()),
                "weighted_restd_loss": float(actual._last_loss_restd_weighted.detach()),
                **{k: float(v) for k, v in actual._last_restd_metrics.items()},
            }
            if self.is_world_process_zero():
                with (Path(self.args.output_dir) / "losses.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            self.logged_steps.add(step)
        return (output.loss, output) if return_outputs else output.loss


def configure_model(source, catalog, cache, cfg, auxiliary, adapter=None):
    config = AutoConfig.from_pretrained(source)
    source_has_contract = hasattr(config, "restd_contract")
    source_has_distillation = bool(getattr(config, "restd_enabled", False))
    if config.model_type not in ("t5", "mt5"):
        raise ValueError("Only T5 and mT5 backbones are supported")
    cls = ResTDT5 if config.model_type == "t5" else ResTDMT5
    tokenizer = AutoTokenizer.from_pretrained(
        source, model_max_length=cfg["training"]["max_length"]
    )
    new_tokens = tokenizer.add_tokens(catalog.new_tokens())
    if source_has_contract:
        if config.restd_contract != catalog.contract():
            raise ValueError("Checkpoint and catalog contracts differ")
        if new_tokens:
            raise ValueError(
                "Trained checkpoint is missing required SID/category tokens"
            )
    for name, value in cfg["restd"].items():
        setattr(config, "restd_" + name, value)
    config.latent_token_length = cfg["training"]["latent_steps"]
    config.num_categories_list = catalog.category_sizes
    config.alpha, config.beta = cfg["training"]["alpha"], cfg["training"]["beta"]
    model, info = cls.from_pretrained(
        source, config=config, restd_codebooks=cache.codebooks, output_loading_info=True
    )
    # Public T5/mT5 initializes only new category/projection heads. Existing retrieval
    # weights must match; never silently ignore a size mismatch.
    if info.get("mismatched_keys"):
        raise ValueError(f"Checkpoint tensor shape mismatch: {info['mismatched_keys']}")
    # Portable inference checkpoints intentionally omit training-only heads and
    # codebooks. Restore those from fresh initialization/the fixed residual cache,
    # while still refusing to silently initialize missing retriever weights.
    allowed_missing = (
        () if source_has_distillation else ("restd_horizon_heads.", "restd_codebook_")
    )
    if not source_has_contract:
        allowed_missing += ("category_heads.",)
    missing = [
        key
        for key in info.get("missing_keys", ())
        if not key.startswith(allowed_missing)
    ]
    if missing or info.get("unexpected_keys"):
        raise ValueError(
            f"Incomplete retrieval checkpoint: missing={missing}, "
            f"unexpected={sorted(info.get('unexpected_keys', ()))}"
        )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    model.config.restd_contract = catalog.contract()
    model.config.restd_backbone_source = str(source)
    model.config.restd_locale = cfg["locale"]
    if adapter:
        payload = torch.load(adapter, map_location="cpu", weights_only=True)
        model.category_heads[-1].load_state_dict(
            payload.get("state_dict", payload), strict=True
        )
        model.config.restd_category_adapter_sha256 = sha256(adapter)
    for parameter in model.category_heads.parameters():
        parameter.requires_grad_(not auxiliary)
    # Allocate all four heads in the base stage, without optimizing them until continuation.
    for parameter in model.restd_horizon_heads.parameters():
        parameter.requires_grad_(auxiliary)
    for actual, expected in zip(model.get_restd_codebooks(), cache.codebooks):
        if not torch.equal(actual.cpu(), expected.float().cpu()):
            raise ValueError("Loaded codebooks differ from the frozen index")
    return model, tokenizer


def train_phase(source, output, dataset, cache, cfg, args, auxiliary):
    seed_everything(args.seed)
    output = Path(output)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    micro = (
        cfg["training"]["micro_batch_size"]
        if args.micro_batch_size is None
        else args.micro_batch_size
    )
    effective = cfg["training"]["effective_batch_size"]
    if micro < 1 or world < 1 or effective % (micro * world):
        raise ValueError(
            "Effective batch size must be divisible by positive microbatch size × GPU count"
        )
    source_dir = Path(source)
    source_files = (
        sorted(source_dir.glob("*.safetensors"))
        + sorted(source_dir.glob("pytorch_model*.bin"))
        if source_dir.is_dir()
        else []
    )
    manifest = {
        "locale": cfg["locale"],
        "seed": args.seed,
        "phase": "restd" if auxiliary else "base",
        "source": str(source),
        "source_weights": {path.name: sha256(path) for path in source_files},
        "config": cfg,
        "catalog": dataset.catalog.contract(),
        "residual_cache": cache.contract,
        "effective_batch_size": effective,
        "micro_batch_size": micro,
        "world_size": world,
        "smoke_steps": args.max_steps,
        "category_adapter_sha256": sha256(args.category_adapter)
        if auxiliary and args.category_adapter
        else None,
    }
    if output.exists() and any(output.iterdir()):
        previous = output / "run.json"
        if not previous.exists() or read_json(previous) != manifest:
            raise ValueError(
                "Existing run differs from requested inputs or configuration; use a fresh output directory"
            )
        if (output / "_SUCCESS").exists():
            return output
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Incomplete run: {output}. Use --resume or a fresh output root."
        )
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = configure_model(
        source,
        dataset.catalog,
        cache,
        cfg,
        auxiliary,
        args.category_adapter if auxiliary else None,
    )
    training = cfg["training"]
    phase = cfg["restd"] if auxiliary else cfg["base"]
    max_steps = args.max_steps or (phase["steps"] if auxiliary else -1)
    train_data, valid_data = dataset, None
    if not auxiliary:
        train_ids, valid_ids = query_split(dataset, seed=42)
        train_data, valid_data = Subset(dataset, train_ids), Subset(dataset, valid_ids)
    validation = not auxiliary and args.max_steps is None
    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=effective // (micro * world),
        learning_rate=phase["learning_rate"],
        num_train_epochs=phase.get("epochs", 1),
        max_steps=max_steps,
        weight_decay=training["weight_decay"],
        lr_scheduler_type="cosine",
        warmup_ratio=phase["warmup_ratio"],
        max_grad_norm=training["max_grad_norm"],
        bf16=args.device != "cpu",
        fp16=False,
        use_cpu=args.device == "cpu",
        optim="adamw_torch",
        eval_strategy="epoch" if validation else "no",
        save_strategy="no" if args.max_steps else ("epoch" if validation else "steps"),
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=validation,
        metric_for_best_model="eval_loss" if validation else None,
        greater_is_better=False,
        logging_steps=1 if args.max_steps else 10,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        prediction_loss_only=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True if world > 1 else None,
        dataloader_num_workers=0,
        dataloader_pin_memory=args.device != "cpu",
    )
    trainer = ResTDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=Collator(
            tokenizer, cache, training["latent_steps"], training["max_length"]
        ),
        processing_class=tokenizer,
        auxiliary=auxiliary,
        gradient_check=bool(args.max_steps),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        if validation
        else [],
    )
    frozen = (
        [p.detach().cpu().clone() for p in model.category_heads.parameters()]
        if auxiliary
        else None
    )
    if trainer.is_world_process_zero():
        write_json(output / "run.json", manifest)
    from transformers.trainer_utils import get_last_checkpoint

    resume = get_last_checkpoint(str(output)) if args.resume else None
    trainer.train(resume_from_checkpoint=resume)
    if auxiliary and any(
        not torch.equal(before, after.detach().cpu())
        for before, after in zip(frozen, model.category_heads.parameters())
    ):
        raise RuntimeError("Frozen category adapter changed during continuation")
    if (
        args.max_steps
        and auxiliary
        and args.max_steps > 1
        and not trainer.gradient_checked
    ):
        raise RuntimeError("Short run did not complete the decoder gradient check")
    trainer.save_model(str(output))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output)
        write_json(
            output / "_SUCCESS",
            {
                "steps": trainer.state.global_step,
                "phase": manifest["phase"],
                "smoke": args.max_steps is not None,
                "category_frozen": auxiliary,
            },
        )
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def run(args):
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required; use --device cpu only for small tests"
            )
        torch.cuda.set_device(
            int(os.environ.get("LOCAL_RANK", args.device.split(":")[-1]))
        )
    cfg = config_for(args.locale, args.config)
    if args.base_epochs is not None:
        cfg["base"]["epochs"] = args.base_epochs
    if args.bundle:
        from .bundle import verify
        from .evaluate import evaluate

        bundle, manifest = verify(args.bundle, args.locale, "checkpoint/")
        catalog = Catalog(Path(args.data_root) / f"esci_{args.locale}")
        if catalog.contract() != manifest["catalog"]:
            raise ValueError("Run bundle preparation and tokenization before GR")
        ev = manifest["evaluation"]
        return evaluate(
            bundle / "checkpoint",
            catalog,
            Path(args.output_root) / args.locale / "metrics.json",
            device=args.device,
            num_beams=ev["num_beams"],
            category_width=ev["category_width"],
            limit=args.eval_queries,
            expected_queries=ev["expected_queries"],
            constraint_weight=ev["constraint_weight"],
            decoding=ev["decoding"],
            relevance=ev["relevance"],
        )
    catalog = Catalog(
        Path(args.data_root) / f"esci_{args.locale}",
        args.index_file,
        cfg["training"]["latent_steps"],
    )
    cache = ResidualCache(
        args.residual_cache or catalog.directory / "residuals", catalog
    )
    dataset = RetrievalDataset(catalog)
    output = Path(args.output_root) / args.locale / f"seed{args.seed}"
    source = args.base_checkpoint
    if not source:
        source = train_phase(
            args.base_model or cfg["backbone"],
            output / "base",
            dataset,
            cache,
            cfg,
            args,
            auxiliary=False,
        )
    final = train_phase(
        source, output / "restd", dataset, cache, cfg, args, auxiliary=True
    )
    if not args.skip_evaluation and int(os.environ.get("RANK", "0")) == 0:
        from .evaluate import evaluate

        evaluate(
            final,
            catalog,
            output / "metrics.json",
            device=args.device,
            num_beams=cfg["evaluation"]["num_beams"],
            category_width=cfg["evaluation"]["category_width"],
            limit=args.eval_queries,
            expected_queries=cfg["evaluation"]["expected_queries"]
            if not args.eval_queries
            else None,
            constraint_weight=cfg["evaluation"]["constraint_weight"],
            decoding=cfg["evaluation"]["decoding"],
            relevance=cfg["evaluation"]["relevance"],
        )
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locale", required=True, choices=["us", "es", "jp"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--config")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--micro-batch-size", type=int)
    p.add_argument(
        "--base-model",
        help="Public T5/mT5 model ID or local snapshot for base training",
    )
    p.add_argument("--bundle", help="Evaluate the supplied checkpoint bundle")
    p.add_argument(
        "--base-checkpoint",
        help="Prepared initial retrieval checkpoint; skips base training",
    )
    p.add_argument(
        "--category-adapter",
        help="Optional last-category-head state dict for matched continuation",
    )
    p.add_argument("--base-epochs", type=int)
    p.add_argument("--index-file", help="Override the final SID JSON path")
    p.add_argument("--residual-cache", help="Override the residual cache directory")
    p.add_argument(
        "--max-steps",
        type=int,
        help="Limit each training phase to this number of optimizer updates",
    )
    p.add_argument(
        "--eval-queries",
        type=int,
        default=0,
        help="0 evaluates the complete official split",
    )
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.max_steps is not None and args.max_steps < 2:
        p.error("Short runs require at least two optimizer updates")
    run(args)


if __name__ == "__main__":
    main()
