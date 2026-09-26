"""Verify and install checkpoint, dataset, and index bundles."""

from pathlib import Path
import shutil
from .common import read_json, sha256


def verify(bundle, locale, prefix=None):
    root = Path(bundle).resolve() / locale
    manifest = read_json(root / "manifest.json")
    if manifest["schema"] != "restd.bundle.v1" or manifest["locale"] != locale:
        raise ValueError("Artifact bundle schema or locale does not match")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact path escapes bundle root")
        if prefix and not name.startswith(prefix):
            continue
        if sha256(path) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return root, manifest


def install(bundle, locale, data_root, stage):
    prefix = "data/" if stage == "data" else "tokenization/"
    root, manifest = verify(bundle, locale, prefix)
    dataset = f"esci_{locale}"
    destination = Path(data_root) / dataset
    source = root / "data" / dataset if stage == "data" else root / "tokenization"
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        out = destination / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if sha256(out) != sha256(path):
                raise ValueError(f"Existing artifact differs: {out}")
        else:
            shutil.copyfile(path, out)
    return destination
