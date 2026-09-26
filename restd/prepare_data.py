"""Step 1: prepare all ESCI locales from public inputs or a verified data bundle."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import urllib.request
from .bundle import install
from .prepare import prepare

ESCI_REVISION = "8113b17a5d4099e20243282c926f1bc1a08a4d13"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=Path("raw"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--hf-data-dir", type=Path)
    p.add_argument("--esci-s-json-zst", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--locale", choices=["us", "es", "jp", "all"], default="all")
    args = p.parse_args()
    if args.bundle:
        for locale in ["us", "es", "jp"] if args.locale == "all" else [args.locale]:
            print(install(args.bundle, locale, args.data_root, "data"))
        return
    if args.locale != "all":
        p.error("Raw preparation builds all three locales together; omit --locale")
    raw = args.raw_root
    raw.mkdir(parents=True, exist_ok=True)
    hf = args.hf_data_dir
    if hf is None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            "tasksource/esci",
            repo_type="dataset",
            revision=ESCI_REVISION,
            allow_patterns=["data/*.parquet"],
            local_dir=raw / "esci",
        )
        hf = Path(snapshot) / "data"
    metadata = args.esci_s_json_zst or raw / "esci.json.zst"
    if not metadata.exists():
        temporary = metadata.with_suffix(metadata.suffix + ".part")
        metadata.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading ESCI-S metadata (approximately 3.4 GB)...", flush=True)
        with (
            urllib.request.urlopen(
                "https://esci-s.s3.amazonaws.com/esci.json.zst", timeout=60
            ) as response,
            temporary.open("wb") as out,
        ):
            import shutil

            shutil.copyfileobj(response, out, 1 << 20)
        temporary.replace(metadata)
    prepare(
        SimpleNamespace(
            hf_data_dir=hf,
            esci_s_json_zst=metadata,
            output_root=args.data_root,
            log_every=200000,
            allow_custom_data=False,
        )
    )


if __name__ == "__main__":
    main()
