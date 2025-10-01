# saves the openwebtext dataset to a binary file for training.
# adapted from: https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, DatasetDict

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # how many training docs to keep
    N_train = 500
    N_val = 10
    seed = 2357

    # takes 54GB in huggingface .cache dir, about 8M documents
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # sample exactly N docs for training
    sub = dataset["train"].train_test_split(
        train_size=min(N_train, len(dataset["train"])),  # clamp to dataset size
        test_size=N_val,  # clamp to dataset size
        seed=seed,
        shuffle=True
    )
    train_small = sub["train"]

    # make a small validation set (can adjust number here)
    val_small = sub["test"]

    split_dataset = DatasetDict({
        "train": train_small,
        "val": val_small
    })

    # encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores special tokens
        ids.append(enc.eot_token)  # add end-of-text token
        return {'ids': ids, 'len': len(ids)}

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # enc.max_token_value == 50256 < 2**16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        # choose number of shards not larger than the number of examples to avoid IndexError
        max_shards = 1024
        num_examples = len(dset)
        if num_examples == 0:
            # nothing to write for this split
            arr.flush()
            continue
        total_batches = min(max_shards, num_examples)

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin and val.bin are written
    # to read later:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
