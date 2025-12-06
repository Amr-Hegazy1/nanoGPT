# streams a small slice of The Stack v2 deduplicated dataset and writes tokenized binaries
# only the first N rows are fetched to avoid bulk downloads

import os
import random
import itertools
import importlib.util
from typing import Optional

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# how many rows to pull from the train split (useful for smoke tests)
N = 1000

# validation fraction from the sampled subset
VAL_FRACTION = 0.01

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

enc = tiktoken.get_encoding("gpt2")


def require_dependency(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"Missing dependency: pip install {name} to stream this dataset.")


def build_s3_client():
    require_dependency("boto3")
    import boto3

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    return session.client("s3")


def fetch_content(s3_client, blob_id: str, src_encoding: Optional[str]) -> str:
    require_dependency("smart_open")
    from smart_open import open  # type: ignore

    s3_url = f"s3://softwareheritage/content/{blob_id}"
    encoding = src_encoding or "utf-8"
    # dataset stores gzip-compressed contents
    with open(s3_url, "rb", compression=".gz", transport_params={"client": s3_client}) as fin:
        raw = fin.read()
    return raw.decode(encoding, errors="replace")


def write_bin(split: str, sequences: list[list[int]]):
    arr_len = np.sum([len(seq) for seq in sequences], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

    idx = 0
    for seq in tqdm(sequences, desc=f"writing {filename}"):
        arr[idx : idx + len(seq)] = seq
        idx += len(seq)
    arr.flush()


if __name__ == "__main__":
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to download The Stack v2 contents.")

    s3_client = build_s3_client()

    dataset = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train",
        streaming=True,
    )

    samples = []
    for ex in tqdm(itertools.islice(dataset, N), total=N, desc="fetch + encode"):
        content = fetch_content(s3_client, ex["blob_id"], ex.get("src_encoding"))
        ids = enc.encode_ordinary(content)
        ids.append(enc.eot_token)
        samples.append(ids)

    random.Random(2357).shuffle(samples)
    val_size = max(1, int(len(samples) * VAL_FRACTION))
    val_seqs = samples[:val_size]
    train_seqs = samples[val_size:]

    write_bin("train", train_seqs)
    write_bin("val", val_seqs)
