#!/usr/bin/env python3
"""Prepare ESCI-US, ESCI-ES, and ESCI-JP for generative retrieval.

Join the small-version ESCI records with ESCI-S categories by product ID.
Keep categories whose tab-joined path contains at least three characters,
then build training pairs, seen-test pairs, and stable catalog mappings.
Dataset statistics are validated before the output directories are published.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq
import zstandard as zstd


LOCALES = ("us", "es", "jp")
PAIR_COLUMNS = (
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
)
ITEM_COLUMNS = (
    "product_id",
    "product_locale",
    "small_version",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
)
ITEM_FEATURES = ITEM_COLUMNS[3:]
LABEL_TO_SHORT = {
    "Exact": "E",
    "Substitute": "S",
    "Complement": "C",
    "Irrelevant": "I",
    "E": "E",
    "S": "S",
    "C": "C",
    "I": "I",
}
RELEVANCE = {"E": 3, "S": 2, "C": 1}
POSITIVE_LABELS = frozenset(RELEVANCE)
EXPECTED_STATISTICS = {
    "us": {
        "products": 288_372,
        "train_queries": 20_109,
        "train_pairs": 292_354,
        "test_queries": 6_137,
        "test_pairs": 29_971,
    },
    "es": {
        "products": 101_957,
        "train_queries": 5_421,
        "train_pairs": 108_587,
        "test_queries": 1_877,
        "test_pairs": 14_554,
    },
    "jp": {
        "products": 119_052,
        "train_queries": 6_543,
        "train_pairs": 127_788,
        "test_queries": 2_163,
        "test_pairs": 15_861,
    },
}


class PreparationError(RuntimeError):
    """Raised when inputs do not satisfy the preprocessing requirements."""


class JsonObjectWriter:
    """Stream a JSON object without materializing its values in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        self.stream = self.temporary.open("x", encoding="utf-8", newline="\n")
        self.stream.write("{")
        self.first = True
        self.closed = False

    def write(self, key: str, value: Any) -> None:
        if self.closed:
            raise RuntimeError("cannot write a closed JSON object")
        if not self.first:
            self.stream.write(",")
        self.first = False
        self.stream.write(json.dumps(str(key), ensure_ascii=False))
        self.stream.write(":")
        self.stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def close(self) -> None:
        if self.closed:
            return
        self.stream.write("}\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.temporary, self.path)
        self.closed = True

    def abort(self) -> None:
        if not self.stream.closed:
            self.stream.close()
        self.closed = True


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parquet_shards(hf_data_dir: Path, split: str) -> list[Path]:
    shards = sorted(hf_data_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no {split} parquet shards under {hf_data_dir}")
    return shards


def load_small_split(hf_data_dir: Path, split: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for shard in parquet_shards(hf_data_dir, split):
        parquet = pq.ParquetFile(shard)
        missing = set(PAIR_COLUMNS) - set(parquet.schema.names)
        if missing:
            raise PreparationError(f"{shard} misses columns: {sorted(missing)}")
        for batch in parquet.iter_batches(
            batch_size=131_072, columns=list(PAIR_COLUMNS)
        ):
            frame = batch.to_pandas()
            frame = frame.loc[frame["small_version"].eq(1)].copy()
            if not frame.empty:
                pieces.append(frame)
    if not pieces:
        raise PreparationError(f"{split} has no small_version=1 rows")
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.loc[
        frame["query"].notna()
        & frame["product_id"].notna()
        & frame["product_locale"].isin(LOCALES)
    ].copy()
    frame["query"] = frame["query"].astype(str)
    frame["product_id"] = frame["product_id"].astype(str)
    frame["product_locale"] = frame["product_locale"].astype(str).str.lower()
    mapped_labels = frame["esci_label"].map(LABEL_TO_SHORT)
    if mapped_labels.isna().any():
        unknown = sorted(set(frame.loc[mapped_labels.isna(), "esci_label"].astype(str)))
        raise PreparationError(f"unknown ESCI labels in {split}: {unknown}")
    frame["esci_label"] = mapped_labels
    return frame


def load_esci_s_categories(
    source: Path,
    needed_asins: set[str],
    log_every: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    mapping: dict[str, list[str]] = {}
    counters: Counter[str] = Counter()
    with source.open("rb") as compressed:
        decompressor = zstd.ZstdDecompressor(max_window_size=2**31)
        with decompressor.stream_reader(compressed, read_across_frames=True) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    counters["records"] += 1
                    if log_every and line_number % log_every == 0:
                        print(
                            f"ESCI-S: {line_number:,} records, "
                            f"{len(mapping):,} needed ASINs with categories",
                            flush=True,
                        )
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        counters["json_errors"] += 1
                        continue
                    if item.get("type") != "product":
                        counters["non_product"] += 1
                        continue
                    asin = item.get("asin")
                    categories = item.get("category")
                    if not asin or str(asin) not in needed_asins:
                        counters["not_needed"] += 1
                        continue
                    if not isinstance(categories, list) or not categories:
                        counters["empty_category"] += 1
                        continue
                    if not all(isinstance(level, str) for level in categories):
                        raise PreparationError(
                            f"ESCI-S line {line_number} has a non-string category level"
                        )
                    # For duplicate ASINs, retain the last non-empty category path.
                    # Category lookup uses the ASIN without a locale component.
                    mapping[str(asin)] = list(categories)
    compatible = {
        asin: path for asin, path in mapping.items() if len("\t".join(path)) >= 3
    }
    statistics = {
        "records": counters["records"],
        "json_errors": counters["json_errors"],
        "non_product_records": counters["non_product"],
        "needed_official_asins": len(needed_asins),
        "needed_asins_with_nonempty_category": len(mapping),
        "needed_asins_passing_category_filter": len(compatible),
    }
    return compatible, statistics


def group_pairs(frame: pd.DataFrame) -> dict[str, list[list[str]]]:
    output: dict[str, list[list[str]]] = {}
    for query, product_id, label in frame[
        ["query", "product_id", "esci_label"]
    ].itertuples(index=False, name=None):
        output.setdefault(str(query), []).append([str(product_id), str(label)])
    return output


def write_qrels(
    path: Path,
    frame: pd.DataFrame,
    product_id_to_index: Mapping[str, int],
) -> int:
    positive = frame.loc[frame["esci_label"].isin(POSITIVE_LABELS)]
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        for query, product_id, label in positive[
            ["query", "product_id", "esci_label"]
        ].itertuples(index=False, name=None):
            stream.write(
                f"{query}\t{product_id_to_index[str(product_id)]}"
                f"\t{RELEVANCE[str(label)]}\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return len(positive)


def json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes, list, dict)):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def write_item_files(
    hf_data_dir: Path,
    temp_dirs: Mapping[str, Path],
    product_mappings: Mapping[str, Mapping[str, int]],
) -> None:
    remaining = {locale: set(mapping) for locale, mapping in product_mappings.items()}
    writers = {
        locale: JsonObjectWriter(temp_dirs[locale] / f"esci_{locale}.item.json")
        for locale in LOCALES
    }
    try:
        for shard in parquet_shards(hf_data_dir, "train"):
            parquet = pq.ParquetFile(shard)
            missing = set(ITEM_COLUMNS) - set(parquet.schema.names)
            if missing:
                raise PreparationError(f"{shard} misses columns: {sorted(missing)}")
            for batch in parquet.iter_batches(
                batch_size=32_768, columns=list(ITEM_COLUMNS)
            ):
                frame = batch.to_pandas()
                frame = frame.loc[frame["small_version"].eq(1)]
                for row in frame.itertuples(index=False, name=None):
                    product_id = str(row[0])
                    locale = str(row[1]).lower()
                    if locale not in remaining or product_id not in remaining[locale]:
                        continue
                    record = {
                        feature: json_scalar(value)
                        for feature, value in zip(ITEM_FEATURES, row[3:])
                    }
                    writers[locale].write(
                        str(product_mappings[locale][product_id]), record
                    )
                    remaining[locale].remove(product_id)
        missing_summary = {
            locale: len(product_ids)
            for locale, product_ids in remaining.items()
            if product_ids
        }
        if missing_summary:
            raise PreparationError(
                f"failed to recover item features for catalog products: {missing_summary}"
            )
        for writer in writers.values():
            writer.close()
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise


def artifact_checksums(directory: Path) -> str:
    lines = []
    for path in sorted(directory.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name in {"artifacts.sha256", "_SUCCESS"}:
            continue
        lines.append(f"{sha256_file(path)}  {path.name}\n")
    return "".join(lines)


def prepare(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    hf_data_dir = args.hf_data_dir.resolve()
    esci_s_source = args.esci_s_json_zst.resolve()
    output_root = args.output_root.resolve()
    if not esci_s_source.is_file():
        raise FileNotFoundError(esci_s_source)
    output_root.mkdir(parents=True, exist_ok=True)

    final_dirs = {locale: output_root / f"esci_{locale}" for locale in LOCALES}
    nonempty_existing = [
        str(path)
        for path in final_dirs.values()
        if path.exists() and (not path.is_dir() or any(path.iterdir()))
    ]
    if nonempty_existing:
        raise PreparationError(
            "refusing to overwrite non-empty dataset paths: "
            + ", ".join(nonempty_existing)
        )
    temp_dirs = {
        locale: output_root / f".esci_{locale}.tmp-{os.getpid()}" for locale in LOCALES
    }
    for directory in temp_dirs.values():
        directory.mkdir(mode=0o750)

    print("Loading Hugging Face ESCI small_version=1 rows...", flush=True)
    train_all = load_small_split(hf_data_dir, "train")
    test_all = load_small_split(hf_data_dir, "test")
    needed_asins = set(train_all["product_id"]) | set(test_all["product_id"])
    print(
        f"HF rows: train={len(train_all):,}, test={len(test_all):,}, "
        f"needed ASINs={len(needed_asins):,}",
        flush=True,
    )

    categories, category_statistics = load_esci_s_categories(
        esci_s_source, needed_asins, args.log_every
    )
    eligible_asins = set(categories)
    print(
        f"ESCI-S compatible ASINs needed by ESCI: {len(eligible_asins):,}",
        flush=True,
    )

    locale_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    product_mappings: dict[str, dict[str, int]] = {}
    observed_statistics: dict[str, dict[str, int]] = {}

    for locale in LOCALES:
        train = train_all.loc[
            train_all["product_locale"].eq(locale)
            & train_all["product_id"].isin(eligible_asins)
        ].drop_duplicates(["query", "product_id", "esci_label"], keep="first")
        test = test_all.loc[
            test_all["product_locale"].eq(locale)
            & test_all["product_id"].isin(eligible_asins)
        ].drop_duplicates(["query", "product_id", "esci_label"], keep="first")

        train_positive = train.loc[train["esci_label"].isin(POSITIVE_LABELS)]
        catalog_products = set(train["product_id"])
        if locale == "us":
            # Restrict US test pairs to positive products observed in training.
            seen_products = set(train_positive["product_id"])
            test_seen = test.loc[
                test["product_id"].isin(seen_products)
                & test["esci_label"].isin(POSITIVE_LABELS)
            ].copy()
            test_seen_protocol = "positive_train_products_and_positive_test_pairs"
        else:
            # ES/JP retain I-labelled test rows for
            # query counting and remove I only when qrels are written.
            seen_products = catalog_products
            test_seen = test.loc[test["product_id"].isin(seen_products)].copy()
            test_seen_protocol = "all_label_train_products"

        observed = {
            "products": len(catalog_products),
            "train_queries": int(train["query"].nunique()),
            "train_pairs": int(train_positive.shape[0]),
            "test_queries": int(test_seen["query"].nunique()),
            "test_pairs": int(test_seen["esci_label"].isin(POSITIVE_LABELS).sum()),
        }
        expected = EXPECTED_STATISTICS[locale]
        if not getattr(args, "allow_custom_data", False) and observed != expected:
            raise PreparationError(
                f"ESCI-{locale} dataset statistics mismatch: expected={expected}, observed={observed}"
            )
        label = (
            "custom data"
            if getattr(args, "allow_custom_data", False)
            else "dataset statistics verified"
        )
        print(f"ESCI-{locale}: {observed} ({label})", flush=True)

        product_ids = sorted(str(product_id) for product_id in catalog_products)
        product_mapping = {
            product_id: index for index, product_id in enumerate(product_ids)
        }
        product_mappings[locale] = product_mapping
        locale_frames[locale] = (train, test_seen)
        observed_statistics[locale] = observed

        dataset = f"esci_{locale}"
        directory = temp_dirs[locale]
        write_json(directory / "product_id_to_index.json", product_mapping)
        write_json(
            directory / "product_categories.json",
            {product_id: categories[product_id] for product_id in product_ids},
        )
        train_pairs = group_pairs(train)
        test_pairs = group_pairs(test_seen)
        write_json(directory / f"{dataset}.train.json", train_pairs)
        write_json(directory / f"{dataset}.test.json", test_pairs)
        write_json(directory / f"{dataset}.test.seen.json", test_pairs)
        write_json(
            directory / "train_query_to_index.json",
            {query: index for index, query in enumerate(sorted(train_pairs))},
        )
        qrels_pairs = write_qrels(directory / "qrels.txt", train, product_mapping)
        test_qrels_pairs = write_qrels(
            directory / "test_qrels.txt", test_seen, product_mapping
        )
        if qrels_pairs != observed["train_pairs"]:
            raise PreparationError(f"{locale} qrels pair count mismatch")
        if test_qrels_pairs != observed["test_pairs"]:
            raise PreparationError(f"{locale} test_qrels pair count mismatch")

        depth_histogram = Counter(
            len(categories[product_id]) for product_id in product_ids
        )
        manifest = {
            "schema": "restd.esci",
            "schema_version": 1,
            "status": "prepared_pending_item_export",
            "dataset": dataset,
            "locale": locale,
            "statistics": observed,
            "expected_statistics": expected,
            "category_depth_histogram": {
                str(depth): count for depth, count in sorted(depth_histogram.items())
            },
            "protocol": {
                "primary_data": "tasksource/esci joined official ESCI export",
                "small_version": 1,
                "category_source": "ESCI-S",
                "category_join_key": "product_id=asin (locale intentionally omitted)",
                "category_filter": "len('\\t'.join(category_path)) >= 3",
                "labels_in_train_json": ["E", "S", "C", "I"],
                "labels_in_qrels": ["E", "S", "C"],
                "test_seen_protocol": test_seen_protocol,
                "check_expected_statistics": not getattr(
                    args, "allow_custom_data", False
                ),
            },
            "category_extraction": category_statistics,
            "sources": {
                "hf_data_dir": str(hf_data_dir),
                "esci_s_json_zst": str(esci_s_source),
                "esci_s_size_bytes": esci_s_source.stat().st_size,
            },
        }
        write_json(directory / "dataset_manifest.json", manifest)

    print("Streaming catalog item features from HF train shards...", flush=True)
    write_item_files(hf_data_dir, temp_dirs, product_mappings)

    for locale in LOCALES:
        directory = temp_dirs[locale]
        manifest_path = directory / "dataset_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest["status"] = "complete"
        write_json(manifest_path, manifest)
        checksums = artifact_checksums(directory)
        checksum_path = directory / "artifacts.sha256"
        checksum_path.write_text(checksums, encoding="utf-8", newline="\n")
        with checksum_path.open("rb") as stream:
            os.fsync(stream.fileno())
        write_json(
            directory / "_SUCCESS",
            {
                "status": "complete",
                "dataset": f"esci_{locale}",
                "statistics": observed_statistics[locale],
                "artifacts_sha256": sha256_file(checksum_path),
            },
        )

    for locale in LOCALES:
        if final_dirs[locale].exists():
            # A pre-created empty destination is safe to replace. Recheck it at
            # publication time so a concurrent writer cannot be overwritten.
            if not final_dirs[locale].is_dir() or any(final_dirs[locale].iterdir()):
                raise PreparationError(
                    f"destination became non-empty during preparation: "
                    f"{final_dirs[locale]}"
                )
            final_dirs[locale].rmdir()
        os.replace(temp_dirs[locale], final_dirs[locale])
        print(f"Published {final_dirs[locale]}", flush=True)
    return observed_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-data-dir", type=Path, required=True)
    parser.add_argument("--esci-s-json-zst", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=200_000)
    parser.add_argument(
        "--allow-custom-data",
        action="store_true",
        help="Skip expected dataset statistics checks for custom inputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics = prepare(args)
    print(json.dumps(statistics, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
