"""Evaluate retrieval with category tries and optional prefix score guidance."""

import argparse
import json
from collections import OrderedDict
import math
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, T5Tokenizer, LogitsProcessorList
from .common import config_for, sha256, write_json
from .data import Catalog, evaluation_queries, relevance_targets
from .modeling import ResTDT5, ResTDMT5, RetrievalT5, RetrievalMT5
from .trie import Trie, prefix_allowed_tokens_fn
from .constraints import LexicalCatalog
from .decoding import (
    build_prefix_assignments,
    aggregate_prefix_feasibility,
    PrefixScoreLogitsProcessor,
)


def metrics(predictions, targets):
    """Deduplicate SIDs before rank cutoffs. Targets map complete SIDs to gains."""
    predictions = list(dict.fromkeys(predictions))
    if not targets:
        raise ValueError("A query must have at least one relevant SID")
    result = {}
    for k in (5, 10, 100):
        result[f"recall@{k}"] = (
            100 * sum(sid in targets for sid in predictions[:k]) / len(targets)
        )
    ideal = sorted(targets.values(), reverse=True)
    for k in (10, 100):
        dcg = sum(
            targets.get(sid, 0.0) / math.log2(rank + 2)
            for rank, sid in enumerate(predictions[:k])
        )
        idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal[:k]))
        result[f"ndcg@{k}"] = 100 * dcg / idcg if idcg else 0.0
    return result


def load_model(checkpoint, catalog, adapter=None):
    config = AutoConfig.from_pretrained(checkpoint)
    is_residual = bool(getattr(config, "restd_enabled", False))
    if config.model_type == "t5":
        cls = ResTDT5 if is_residual else RetrievalT5
        tokenizer = T5Tokenizer.from_pretrained(checkpoint, model_max_length=512)
    elif config.model_type == "mt5":
        cls = ResTDMT5 if is_residual else RetrievalMT5
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512)
    else:
        raise ValueError("Expected T5 or mT5 checkpoint")
    if tokenizer.add_tokens(catalog.new_tokens()):
        raise ValueError(
            "Checkpoint does not contain the complete SID/category vocabulary"
        )
    if (
        hasattr(config, "restd_contract")
        and config.restd_contract != catalog.contract()
    ):
        raise ValueError("Checkpoint belongs to a different catalog")
    model, info = cls.from_pretrained(
        checkpoint,
        latent_token_length=catalog.latent_steps,
        num_categories_list=catalog.category_sizes,
        output_loading_info=True,
    )
    if (
        info.get("missing_keys")
        or info.get("mismatched_keys")
        or info.get("unexpected_keys")
    ):
        raise ValueError(f"Incomplete or mismatched retrieval checkpoint: {info}")
    if adapter:
        payload = torch.load(adapter, map_location="cpu", weights_only=True)
        anchor = payload.get("anchor_sha256")
        if anchor and anchor != sha256(Path(checkpoint) / "model.safetensors"):
            raise ValueError("Category adapter belongs to a different checkpoint")
        model.category_heads[-1].load_state_dict(
            payload.get("state_dict", payload), strict=True
        )
    return model, tokenizer


@torch.no_grad()
def evaluate(
    checkpoint,
    catalog,
    output,
    *,
    device="cuda:0",
    num_beams=100,
    category_width=3,
    limit=0,
    expected_queries=None,
    adapter=None,
    decoding="guided",
    relevance="graded",
    constraint_weight=2.0,
    lexical_weight=0.25,
    max_lexical_features=2,
    max_lexical_df_ratio=0.02,
):
    if num_beams < 1 or category_width < 1:
        raise ValueError("Beam and category width must be positive")
    model, tokenizer = load_model(checkpoint, catalog, adapter)
    model.to(device).eval()
    queries = evaluation_queries(catalog)
    if expected_queries and len(queries) != expected_queries:
        raise ValueError(
            f"Expected {expected_queries} test queries, got {len(queries)}"
        )
    if limit:
        queries = queries[:limit]
    if not queries:
        raise ValueError("No eligible evaluation queries")
    gains = relevance_targets(catalog, relevance)
    sid_ids = np.asarray(
        [
            tokenizer.convert_tokens_to_ids(catalog.indices[str(i)])
            for i in range(len(catalog.indices))
        ],
        dtype=np.int32,
    )
    if (sid_ids == tokenizer.unk_token_id).any():
        raise ValueError("Unknown SID tokens")
    lexical = None
    if decoding == "guided":
        lexical = LexicalCatalog(
            str(catalog.directory / f"{catalog.dataset}.item.json"),
            max_lexical_features=max_lexical_features,
            lexical_weight=lexical_weight,
            max_lexical_df_ratio=max_lexical_df_ratio,
            unicode_nfkc=True,
            enable_measurements=True,
            enable_negation=False,
        )
        assignments = build_prefix_assignments(sid_ids)
        if lexical.item_count != len(sid_ids):
            raise ValueError("Item metadata and SID index differ")
    cache = OrderedDict()
    valid_sids = {tuple(sid) for sid in catalog.indices.values()}
    records = []
    for query, _ in tqdm(queries, desc="ResTD retrieval"):
        inputs = tokenizer(
            "<retrieval>" + query, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        categories = (
            model.predict_latent_categories(**inputs, top_k=category_width)[-1][0]
            .cpu()
            .tolist()
        )
        key = tuple(sorted(set(categories)))
        if key not in cache:
            candidates = set()
            for category in key:
                for index in catalog.category_items.get(category, []):
                    candidates.add(
                        (
                            model.config.decoder_start_token_id,
                            *map(int, sid_ids[index]),
                            tokenizer.eos_token_id,
                        )
                    )
            if not candidates:
                records.append(
                    {
                        "query": query,
                        "predictions": [],
                        "metrics": metrics([], gains[query]),
                    }
                )
                continue
            cache[key] = (
                prefix_allowed_tokens_fn(
                    Trie([list(sid) for sid in sorted(candidates)])
                ),
                len(candidates),
            )
            if len(cache) > 256:
                cache.popitem(last=False)
        allowed, candidate_count = cache[key]
        cache.move_to_end(key)
        processors = LogitsProcessorList()
        if lexical:
            constraints = lexical.compile(query)
            if constraints.count:
                item_scores = lexical.item_scores(
                    constraints, lexical_weight=lexical_weight
                )
                scores = aggregate_prefix_feasibility(item_scores, assignments)
                processors.append(
                    PrefixScoreLogitsProcessor(
                        assignments=assignments,
                        prefix_scores=scores,
                        allowed_tokens_fn=allowed,
                        weight=constraint_weight,
                        decoder_start_token_id=model.config.decoder_start_token_id,
                    )
                )
        # Keep the configured beam width; filter invalid and duplicate completions below.
        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            prefix_allowed_tokens_fn=allowed,
            logits_processor=processors,
            early_stopping=True,
        )
        ranked = []
        for row in generated.tolist():
            codes = row[1 : 1 + catalog.levels]
            tokens = tuple(tokenizer.convert_ids_to_tokens(codes))
            if (
                tokens in valid_sids
                and tokens not in ranked
                and all(
                    token.startswith("<" + chr(97 + i) + "_")
                    for i, token in enumerate(tokens)
                )
                and len(tokens) == catalog.levels
            ):
                ranked.append(tokens)
        records.append(
            {
                "query": query,
                "predictions": ["".join(sid) for sid in ranked],
                "metrics": metrics(ranked, gains[query]),
                "candidate_count": candidate_count,
            }
        )
    means = {
        name: sum(row["metrics"][name] for row in records) / len(records)
        for name in records[0]["metrics"]
    }
    report = {
        "dataset": catalog.dataset,
        "queries": len(records),
        "partial": bool(limit),
        "metrics": means,
        "protocol": {
            "decoding": decoding,
            "relevance": relevance,
            "beam": num_beams,
            "category_width": category_width,
            "constraint_weight": constraint_weight,
            "lexical_weight": lexical_weight,
            "unicode_nfkc": True,
        },
        "catalog": catalog.contract(),
        "checkpoint_sha256": sha256(Path(checkpoint) / "model.safetensors"),
        "per_query": records,
    }
    write_json(output, report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2),
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locale", required=True, choices=["us", "es", "jp"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--index-file")
    p.add_argument("--category-adapter")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num-beams", type=int, default=100)
    p.add_argument("--decoding", choices=["guided", "standard"], default="guided")
    p.add_argument("--relevance", choices=["graded", "binary"], default="graded")
    p.add_argument("--custom-population", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    cfg = config_for(args.locale)
    ev = cfg["evaluation"]
    catalog = Catalog(Path(args.data_root) / f"esci_{args.locale}", args.index_file)
    evaluate(
        args.checkpoint,
        catalog,
        args.output,
        device=args.device,
        num_beams=args.num_beams,
        category_width=ev["category_width"],
        adapter=args.category_adapter,
        limit=args.limit,
        expected_queries=None if args.custom_population else ev["expected_queries"],
        decoding=args.decoding,
        relevance=args.relevance,
        constraint_weight=ev["constraint_weight"],
    )


if __name__ == "__main__":
    main()
