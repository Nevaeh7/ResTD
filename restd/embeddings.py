"""
Generate BERT embeddings for ESCI items.

Read the selected locale's item metadata and encode it with BERT.
Each output row follows the mapping in product_id_to_index.json.
"""

import argparse
import json
import os
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(raw_text):
    """Clean text by removing HTML tags and special characters."""
    import html
    import re

    if raw_text is None:
        return ""

    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r"</?\w+[^>]*>", "", raw)
            raw = re.sub(r'["\n\r]*', "", raw)
            new_raw_text.append(raw.strip())
        cleaned_text = " ".join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)

    # Ensure text ends with period
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."

    # Discard oversized metadata fields.
    if len(cleaned_text) >= 2000:
        cleaned_text = ""

    return cleaned_text


def set_device(gpu_id):
    """Set device for computation."""
    if gpu_id == -1:
        return torch.device("cpu")
    else:
        return torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu"
        )


def load_data_esci(item_json_path):
    """Load ESCI item data."""
    item2feature = load_json(item_json_path)
    return item2feature


def generate_text(item2feature, features):
    """Generate text from item features."""
    item_text_list = []
    for item_idx in item2feature:
        data = item2feature[item_idx]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                if meta_value.strip():
                    text.append(f"{meta_key}:{meta_value.strip()}")

        item_text_list.append([int(item_idx), " ".join(text)])

    return item_text_list


def preprocess_text(item_json_path):
    """Preprocess text data."""
    print("Processing text data...")
    item2feature = load_data_esci(item_json_path)
    # Use product_title, product_brand, product_color, product_description
    item_text_list = generate_text(
        item2feature,
        ["product_title", "product_brand", "product_color", "product_description"],
    )
    return item_text_list


def generate_item_embedding(item_text_list, tokenizer, model, device, batch_size=32):
    """Generate BERT embeddings for items."""
    print(f"Generating BERT embeddings...")

    items, texts = zip(*item_text_list)

    # Create ordered text list indexed by item index
    max_item_idx = max(items)
    order_texts = [""] * (max_item_idx + 1)
    for item, text in zip(items, texts):
        order_texts[item] = text

    # Verify all items have text
    for i, text in enumerate(order_texts):
        if not text:
            print(f"Warning: Item {i} has no text")

    embeddings = []
    start = 0

    print(f"Total items: {len(order_texts)}")

    with torch.no_grad():
        for start in tqdm(
            range(0, len(order_texts), batch_size), desc="Generating embeddings"
        ):
            end = min(start + batch_size, len(order_texts))
            batch_texts = order_texts[start:end]
            # Tokenize batch
            encoded_sentences = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Get model output
            outputs = model(**encoded_sentences, output_hidden_states=True)

            # Use last hidden state with attention mask for mean pooling
            last_hidden_state = outputs.last_hidden_state
            attention_mask = encoded_sentences["attention_mask"]

            # Apply attention mask and compute mean
            masked_output = last_hidden_state * attention_mask.unsqueeze(-1)
            mean_output = masked_output.sum(dim=1) / attention_mask.sum(
                dim=-1, keepdim=True
            )

            embeddings.append(mean_output.cpu())

    # Concatenate all embeddings
    embeddings = torch.cat(embeddings, dim=0).to(torch.float32).numpy()
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Generate BERT embeddings for ESCI items"
    )
    parser.add_argument("--dataset", type=str, default="esci_jp", help="Dataset name")
    parser.add_argument("--root", type=str, default="data", help="Root directory")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="BERT model name from HuggingFace",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for processing"
    )

    args = parser.parse_args()

    # Set up paths
    dataset_dir = os.path.join(args.root, args.dataset)
    item_json_path = os.path.join(dataset_dir, f"{args.dataset}.item.json")
    product_id_to_index_path = os.path.join(dataset_dir, "product_id_to_index.json")
    output_emb_path = os.path.join(dataset_dir, f"{args.dataset}.emb-bert.npy")

    print(f"Dataset directory: {dataset_dir}")
    print(f"Item JSON path: {item_json_path}")
    print(f"Product ID to index path: {product_id_to_index_path}")
    print(f"Output embedding path: {output_emb_path}")

    # Check if files exist
    if not os.path.exists(item_json_path):
        raise FileNotFoundError(f"Item JSON file not found: {item_json_path}")
    if not os.path.exists(product_id_to_index_path):
        raise FileNotFoundError(
            f"Product ID to index file not found: {product_id_to_index_path}"
        )
    if os.path.exists(output_emb_path):
        raise FileExistsError(
            f"Refusing to overwrite existing embedding file: {output_emb_path}"
        )

    product_id_to_index = load_json(product_id_to_index_path)
    num_products = len(product_id_to_index)
    if set(product_id_to_index.values()) != set(range(num_products)):
        raise ValueError(
            "product_id_to_index values must be contiguous integers from 0 to N-1"
        )

    # Set device
    device = set_device(args.gpu_id)
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading BERT model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()

    # Preprocess text
    item_text_list = preprocess_text(item_json_path)
    print(f"Loaded {len(item_text_list)} items")

    # Generate embeddings
    embeddings = generate_item_embedding(
        item_text_list, tokenizer, model, device, batch_size=args.batch_size
    )

    if embeddings.shape[0] != num_products:
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match product count "
            f"({num_products})"
        )

    # Publish atomically so an interrupted process cannot leave a truncated
    # final .npy file that looks complete to downstream RQ-VAE jobs.
    temporary_emb_path = os.path.join(
        dataset_dir,
        f".{args.dataset}.emb-bert.npy.tmp-{os.getpid()}",
    )
    with open(temporary_emb_path, "xb") as stream:
        np.save(stream, embeddings)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # Publish without overwriting a file created by another process.
        os.link(temporary_emb_path, output_emb_path)
    except FileExistsError:
        raise FileExistsError(
            f"Embedding output appeared while this job was running: {output_emb_path}"
        )
    finally:
        if os.path.exists(temporary_emb_path):
            os.unlink(temporary_emb_path)
    print(f"Saved embeddings to {output_emb_path}")

    # Verify embeddings match product_id_to_index
    print(f"\nVerification:")
    print(f"  Number of products in product_id_to_index: {num_products}")
    print(f"  Embedding matrix shape: {embeddings.shape}")
    print(f"  Expected shape: ({num_products}, embedding_dim)")

    if embeddings.shape[0] == num_products:
        print("✓ Embedding matrix rows match product count")
    else:
        print(
            f"✗ Warning: Mismatch between embedding rows ({embeddings.shape[0]}) and product count ({num_products})"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
