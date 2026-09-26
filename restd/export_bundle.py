"""Export a portable checkpoint with its category adapter included.

The source weights are read-only. Destination bundles contain no machine paths.
"""

import argparse
from pathlib import Path
import shutil
from transformers import AutoConfig
from .common import config_for, read_json, sha256, write_json
from .data import Catalog, ResidualCache
from .evaluate import load_model


def export_bundle(args):
    destination = Path(args.output_root) / args.locale
    destination.mkdir(parents=True, exist_ok=False)
    cfg = config_for(args.locale)
    catalog = Catalog(args.data_dir, args.index_file)
    model, tokenizer = load_model(args.checkpoint, catalog, args.category_adapter)
    # Retain native model configuration fields and the minimal ResTD catalog contract.
    clean = AutoConfig.for_model(model.config.model_type).to_dict()
    clean = {
        key: value
        for key, value in model.config.to_dict().items()
        if key in clean
        and key not in ("_name_or_path", "architectures", "transformers_version")
    }
    # Transformers 5 moves generation defaults out of the base config, but the
    # latent decoder also consumes these IDs directly before generate().
    for name in (
        "decoder_start_token_id",
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
    ):
        clean[name] = getattr(
            model.config, name, getattr(model.generation_config, name, None)
        )
    model.config = AutoConfig.for_model(
        model.config.model_type, **{k: v for k, v in clean.items() if k != "model_type"}
    )
    model.config.latent_token_length = catalog.latent_steps
    model.config.num_categories_list = catalog.category_sizes
    model.config.restd_contract = catalog.contract()
    model.config.restd_locale = args.locale
    model.config.alpha = model.alpha
    model.config.beta = model.beta
    inference_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("restd_")
    }
    model.save_pretrained(
        destination / "checkpoint", max_shard_size="5GB", state_dict=inference_state
    )
    exported_config = destination / "checkpoint" / "config.json"
    payload = read_json(exported_config)
    payload["architectures"] = [
        "RetrievalT5" if model.config.model_type == "t5" else "RetrievalMT5"
    ]
    write_json(exported_config, payload)
    tokenizer.init_kwargs.pop("name_or_path", None)
    tokenizer.save_pretrained(destination / "checkpoint")
    data = destination / "data" / catalog.dataset
    data.mkdir(parents=True)
    for name in [
        f"{catalog.dataset}.item.json",
        f"{catalog.dataset}.train.json",
        f"{catalog.dataset}.test.seen.json",
        "product_id_to_index.json",
        "product_categories.json",
    ]:
        shutil.copyfile(catalog.directory / name, data / name)
    tokenization = destination / "tokenization"
    tokenization.mkdir()
    shutil.copyfile(catalog.index_path, tokenization / f"{catalog.dataset}.index.json")
    if args.residual_cache:
        ResidualCache(args.residual_cache, catalog)
        residual = tokenization / "residuals"
        residual.mkdir()
        for name in [
            "residual_trace.fp16.npy",
            "sid_numeric.int16.npy",
            "codebooks.pt",
            "residual_meta.json",
            "reconstruction_report.json",
        ]:
            shutil.copyfile(Path(args.residual_cache) / name, residual / name)
        # Only hashes and item order are portable; no filesystem locations.
        write_json(
            residual / "item_index_receipt.json",
            {
                "item_count": len(catalog.indices),
                "item_order": "contiguous numeric rows",
                "sid_file_hash": sha256(catalog.index_path),
                "product_id_to_index_hash": sha256(
                    catalog.directory / "product_id_to_index.json"
                ),
            },
        )
    files = {
        str(path.relative_to(destination)): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    write_json(
        destination / "manifest.json",
        {
            "schema": "restd.bundle.v1",
            "locale": args.locale,
            "files": files,
            "catalog": catalog.contract(),
            "evaluation": cfg["evaluation"],
            "source_checkpoint_sha256": sha256(
                Path(args.checkpoint) / "model.safetensors"
            ),
            "source_adapter_sha256": sha256(args.category_adapter)
            if args.category_adapter
            else None,
        },
    )
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locale", choices=["us", "es", "jp"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--category-adapter")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--index-file", required=True)
    p.add_argument("--residual-cache")
    p.add_argument("--output-root", required=True)
    print(export_bundle(p.parse_args()))


if __name__ == "__main__":
    main()
