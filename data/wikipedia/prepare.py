# saves a slice of the English Wikipedia dataset to binary files for training
# only the first N rows are streamed to keep footprint small

import os
import random
import itertools
import importlib.util
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# how many rows to pull from the train split (useful for smoke tests)
N = 1000

# validation fraction from the sampled subset
VAL_FRACTION = 0.01

enc = tiktoken.get_encoding("gpt2")


def write_bin(split, sequences):
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
    if importlib.util.find_spec("zstandard") is None:
        raise RuntimeError("Missing dependency: pip install zstandard to stream this dataset.")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    samples = []
    for ex in tqdm(itertools.islice(dataset, N), total=N, desc="encoding"):
        ids = enc.encode_ordinary(ex["text"])
        ids.append(enc.eot_token)
        samples.append(ids)

    random.Random(2357).shuffle(samples)
    val_size = max(1, int(len(samples) * VAL_FRACTION))
    val_seqs = samples[:val_size]
    train_seqs = samples[val_size:]

    write_bin("train", train_seqs)
    write_bin("val", val_seqs)
