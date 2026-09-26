"""ESCI query/product data, category mapping, and SID-aligned residual batches."""

from pathlib import Path
import random
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import read_json, sha256

POSITIVE = frozenset(("E", "S", "C"))
TOKEN = re.compile(r"^<([a-z])_(\d+)>$")


class Catalog:
    def __init__(self, directory, index_file=None, latent_steps=3):
        self.directory = Path(directory)
        self.dataset = self.directory.name
        self.index_path = (
            Path(index_file)
            if index_file
            else self.directory / f"{self.dataset}.index.json"
        )
        self.indices = read_json(self.index_path)
        self.product_to_index = read_json(self.directory / "product_id_to_index.json")
        self.category_path = self.directory / "product_categories.json"
        categories = read_json(self.category_path)
        self.latent_steps = latent_steps
        size = len(self.indices)
        if set(self.indices) != {str(i) for i in range(size)}:
            raise ValueError("SID index keys must be contiguous integers")
        if (
            set(self.product_to_index.values()) != set(range(size))
            or len(self.product_to_index) != size
        ):
            raise ValueError("Product-to-row mapping must be a catalog permutation")
        # Assign category IDs in insertion order for consistent checkpoint loading.
        self.category_maps = []
        self.categories = {}
        for product, names in categories.items():
            path = []
            for level, name in enumerate(names):
                if level == len(self.category_maps):
                    self.category_maps.append({})
                mapping = self.category_maps[level]
                if name not in mapping:
                    mapping[name] = len(mapping)
                path.append(mapping[name])
            self.categories[product] = path
        if set(self.product_to_index) - set(self.categories):
            raise ValueError("Catalog products are missing category metadata")
        self.category_sizes = [len(x) for x in self.category_maps[:latent_steps]]
        if len(self.category_sizes) < latent_steps:
            raise ValueError(
                "Dataset has fewer category levels than the latent reasoning depth"
            )
        lengths = {len(sid) for sid in self.indices.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("SIDs must have one common positive length")
        self.levels = next(iter(lengths))
        self.numeric = np.empty((size, self.levels), dtype=np.int64)
        for index, sid in self.indices.items():
            for level, token in enumerate(sid):
                match = TOKEN.fullmatch(token)
                if match is None or match[1] != chr(97 + level):
                    raise ValueError(f"Invalid SID token {token}")
                self.numeric[int(index), level] = int(match[2])
        self.category_items = {}
        # Assign repeated SIDs to their last catalog category when building tries.
        sid_owners = {}
        for product, category in self.categories.items():
            row = self.product_to_index[product]
            sid_owners[tuple(self.indices[str(row)])] = (row, category)
        for row, category in sid_owners.values():
            if len(category) >= latent_steps:
                self.category_items.setdefault(category[latent_steps - 1], []).append(
                    row
                )

    def new_tokens(self):
        tokens = {token for sid in self.indices.values() for token in sid}
        for level, count in enumerate(self.category_sizes):
            tokens.update(f"<class_{level}_{i}>" for i in range(count))
        return sorted(tokens | {"<retrieval>"})

    def sid(self, product):
        return tuple(self.indices[str(self.product_to_index[str(product)])])

    def contract(self):
        return {
            "sid_sha256": sha256(self.index_path),
            "categories_sha256": sha256(self.category_path),
            "products_sha256": sha256(self.directory / "product_id_to_index.json"),
        }


class RetrievalDataset(Dataset):
    def __init__(self, catalog):
        self.catalog = catalog
        self.examples = []
        self.query_categories = {}
        pairs = read_json(catalog.directory / f"{catalog.dataset}.train.json")
        for query, products in pairs.items():
            positive_categories = []
            for product, label in products:
                if label not in POSITIVE:
                    continue
                category = catalog.categories[str(product)]
                if category not in positive_categories:
                    positive_categories.append(category)
                self.examples.append(
                    {
                        "query": query,
                        "item_idx": catalog.product_to_index[str(product)],
                        "cate": category,
                    }
                )
            self.query_categories[query] = positive_categories
        if not self.examples:
            raise ValueError("Training split contains no positive query/product pairs")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return {
            **example,
            "cates": self.query_categories[example["query"]],
            "sid": self.catalog.indices[str(example["item_idx"])],
        }


def query_split(dataset, seed=42, fraction=0.1):
    queries = sorted({row["query"] for row in dataset.examples})
    if len(queries) < 2 or not 0 < fraction < 1:
        raise ValueError(
            "Validation needs at least two queries and a fraction in (0,1)"
        )
    random.Random(seed).shuffle(queries)
    selected = set(
        queries[: min(len(queries) - 1, max(1, round(len(queries) * fraction)))]
    )
    train, validation = [], []
    for index, example in enumerate(dataset.examples):
        (validation if example["query"] in selected else train).append(index)
    return train, validation


class ResidualCache:
    def __init__(self, directory, catalog):
        directory = Path(directory)
        self.residuals = np.load(
            directory / "residual_trace.fp16.npy", mmap_mode="r", allow_pickle=False
        )
        self.sids = np.load(
            directory / "sid_numeric.int16.npy", mmap_mode="r", allow_pickle=False
        )
        self.codebooks = torch.load(
            directory / "codebooks.pt", map_location="cpu", weights_only=True
        )
        metadata = read_json(directory / "residual_meta.json")
        if metadata["sid_file_hash"] != sha256(catalog.index_path):
            raise ValueError("Residual cache belongs to a different SID index")
        expected = (
            len(catalog.indices),
            catalog.levels + 1,
            self.codebooks[0].shape[1],
        )
        if self.residuals.shape != expected or self.sids.shape != catalog.numeric.shape:
            raise ValueError("Residual/SID cache dimensions do not match the catalog")
        if not np.array_equal(self.sids, catalog.numeric):
            raise ValueError("Numeric SID rows do not match the catalog")
        if len(self.codebooks) != catalog.levels:
            raise ValueError("Codebook count differs from SID length")
        for level, codebook in enumerate(self.codebooks):
            if (
                not torch.isfinite(codebook).all()
                or codebook.ndim != 2
                or codebook.shape[1] != expected[2]
            ):
                raise ValueError("Invalid codebook values or shape")
            if (self.sids[:, level] < 0).any() or (
                self.sids[:, level] >= len(codebook)
            ).any():
                raise ValueError("SID is outside its codebook")
        for start in range(0, len(self.sids), 8192):
            if not np.isfinite(self.residuals[start : start + 8192]).all():
                raise ValueError("Non-finite residual cache")
        receipt = directory / "item_index_receipt.json"
        if receipt.exists():
            product_hash = read_json(receipt).get("product_id_to_index_hash")
            if product_hash and product_hash != sha256(
                catalog.directory / "product_id_to_index.json"
            ):
                raise ValueError(
                    "Residual cache uses a different product-to-row mapping"
                )
        self.contract = {
            name: sha256(directory / name)
            for name in (
                "residual_trace.fp16.npy",
                "sid_numeric.int16.npy",
                "codebooks.pt",
            )
        }


class Collator:
    def __init__(self, tokenizer, cache, latent_steps=3, max_length=128):
        self.tokenizer, self.cache = tokenizer, cache
        self.latent_steps, self.max_length = latent_steps, max_length

    def __call__(self, examples):
        batch = self.tokenizer(
            ["<retrieval>" + row["query"] for row in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = []
        for row in examples:
            ids = self.tokenizer.convert_tokens_to_ids(row["sid"])
            if self.tokenizer.unk_token_id in ids or len(ids) != len(row["sid"]):
                raise ValueError("Every SID code must be one registered token")
            labels.append(ids + [self.tokenizer.eos_token_id])
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        size, steps = len(examples), self.latent_steps
        category_labels = torch.full((size, steps), -100, dtype=torch.long)
        # Keep repeated coarse categories and cap each group by the latent depth.
        groups = [
            [
                [path[level] for path in row["cates"] if len(path) > level][:steps]
                for level in range(steps)
            ]
            for row in examples
        ]
        width = max(1, max(len(ids) for row in groups for ids in row))
        group_labels = torch.full((size, steps, width), -100, dtype=torch.long)
        for i, row in enumerate(examples):
            category = row["cate"][:steps]
            category_labels[i, : len(category)] = torch.tensor(
                category, dtype=torch.long
            )
            for level, ids in enumerate(groups[i]):
                group_labels[i, level, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        batch["category_labels"], batch["group_category_labels"] = (
            category_labels,
            group_labels,
        )
        indices = np.asarray([row["item_idx"] for row in examples], dtype=np.int64)
        batch["restd_residual_targets"] = torch.from_numpy(
            np.array(self.cache.residuals[indices], dtype=np.float32, copy=True)
        )
        batch["restd_sid_targets"] = torch.from_numpy(
            np.array(self.cache.sids[indices], dtype=np.int64, copy=True)
        )
        batch["restd_valid_sid_mask"] = torch.ones_like(
            batch["restd_sid_targets"], dtype=torch.bool
        )
        return batch


def evaluation_queries(catalog):
    pairs = read_json(catalog.directory / f"{catalog.dataset}.test.seen.json")
    result = []
    for query, products in pairs.items():
        positive = [
            (str(product), label) for product, label in products if label in POSITIVE
        ]
        if not positive:
            continue
        # Include queries whose first positive has a complete category path.
        if len(catalog.categories[positive[0][0]]) < catalog.latent_steps:
            continue
        targets = {catalog.sid(product) for product, _ in positive}
        result.append((query, targets))
    return result


def relevance_targets(catalog, relevance="graded"):
    """Preserve the fixed checkpoint qrels rule; binary judgments all have gain one."""
    if relevance not in {"graded", "binary"}:
        raise ValueError("Relevance must be graded or binary")
    gains = {}
    for query, products in read_json(
        catalog.directory / f"{catalog.dataset}.test.seen.json"
    ).items():
        values = {}
        for product, label in products:
            if label not in POSITIVE:
                continue
            # Resolve repeated SID judgments by their last occurrence.
            # For binary relevance every positive has value one, equal to max pooling.
            values[catalog.sid(product)] = (
                1 if relevance == "binary" else {"E": 3, "S": 2, "C": 1}[label]
            )
        gains[query] = values
    return gains
