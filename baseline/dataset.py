"""Data loader."""
import argparse
import pickle
import logging
import pathlib
import pprint
import sys

import numpy as np
import muspy
import torch
import torch.utils.data
from tqdm import tqdm

import representation_mmm
import representation_remi
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd"),
        required=True,
        help="dataset key",
    )
    parser.add_argument(
        "-r",
        "--representation",
        choices=("mmm", "remi"),
        required=True,
        help="representation key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    # Data
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=8,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--aug",
        action='store_true',
        default=True,
        help="whether to use data augmentation",
    )
    parser.add_argument(
        "--max_seq_len",
        default=1024,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--max_beat",
        default=64,
        type=int,
        help="maximum number of beats",
    )
    # Others
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="number of jobs (deafult to `min(batch_size, 8)`)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def pad(data, maxlen=None):
    if maxlen is None:
        max_len = max(len(x) for x in data)
    else:
        for x in data:
            assert len(x) <= max_len
    if data[0].ndim == 1:
        padded = [np.pad(x, (0, max_len - len(x))) for x in data]
    elif data[0].ndim == 2:
        padded = [np.pad(x, ((0, max_len - len(x)), (0, 0))) for x in data]
    else:
        raise ValueError("Got 3D data.")
    return np.stack(padded)


def get_mask(data):
    max_seq_len = max(len(sample) for sample in data)
    mask = torch.zeros((len(data), max_seq_len), dtype=torch.bool)
    for i, seq in enumerate(data):
        mask[i, : len(seq)] = 1
    return mask

class MusicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        data_dir,
        encoding,
        indexer,
        encode_fn,
        representation,
        max_seq_len=None,
        max_beat=None,
        use_csv=False,
        use_augmentation=False,
    ):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        with open(filename, 'r') as f:
            self.names = [line.strip() for line in f if line]
        self.encoding = encoding
        self.indexer = indexer
        self.encode_fn = encode_fn
        self.representation = representation
        self.max_seq_len = max_seq_len
        self.max_beat = max_beat
        self.use_csv = use_csv
        self.use_augmentation = use_augmentation
        self.valid_name_indices = []
        self.caches = dict()
        self.load_caches()

    def load_caches(self):
        cache_path = self.data_dir / (self.representation + ".pickle")
        if cache_path.is_file():
            print('Found pickled cache, using it.')
            with open(cache_path, 'rb') as cache_file:
                obj = pickle.load(cache_file)
                self.caches = obj[0]
                self.valid_name_indices = obj[1]
        else:
            for i, name in tqdm(enumerate(self.names), desc='Caching codes'):
                music = muspy.load(self.data_dir / 'json' / f"{name}.json")
                try:
                    if self.representation == 'mmm':
                        # the "codes" here is actually track_list
                        codes = self.encode_fn(music, self.encoding, self.indexer)
                    else:
                        pass
                except AssertionError:
                    continue
                self.valid_name_indices.append(i)
                self.caches[name]=codes

    def __len__(self):
        return len(self.valid_name_indices)

    def __getitem__(self, idx):
        # Get the name
        name = self.names[self.valid_name_indices[idx]]

        # Get the code
        if self.representation == 'mmm':
            track_list = self.caches[name]
            seq = representation_mmm.track_list_to_code(track_list, self.indexer)
        else:
            pass

        # Trim sequence to max_seq_len
        if self.max_seq_len is not None and len(seq) > self.max_seq_len:
            seq = np.concatenate((seq[: self.max_seq_len - 1], seq[-1:]))

        return {"name": name, "seq": seq}

    @classmethod
    def collate(cls, data):
        seq = [sample["seq"] for sample in data]
        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(seq), dtype=torch.long),
            "seq_len": torch.tensor([len(s) for s in seq], dtype=torch.long),
            "mask": get_mask(seq),
        }


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(
                f"data/{args.dataset}/processed/names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/")
    if args.jobs is None:
        args.jobs = min(args.batch_size, 8)
    if args.representation == "mmm":
        representation = representation_mmm
    elif args.representation == "remi":
        representation = representation_remi

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Load the encoding
    encoding = representation.get_encoding()

    # Get the indexer
    indexer = representation.Indexer(encoding["event_code_map"])

    # Create the dataset and data loader
    dataset = MusicDataset(
        args.names,
        args.in_dir,
        encoding=encoding,
        indexer=indexer,
        encode_fn=representation.encode,
        representation=args.representation,
        # max_seq_len=args.max_seq_len,
        max_seq_len=None,
        # max_beat=args.max_beat,
        max_beat=None,
        use_csv=args.use_csv,
        # use_augmentation=args.aug,
        use_augmentation=False
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, args.batch_size, True, collate_fn=MusicDataset.collate
    )

    # Store caches as pickle file
    cache_path = pathlib.Path(f"data/{args.dataset}/processed/{args.representation}.pickle")
    with open(cache_path, 'wb') as cache_file:
        pickle.dump(
            (dataset.caches, dataset.valid_name_indices),
            cache_file
        )

    # Iterate over the loader
    n_batches = 0
    n_samples = 0
    seq_lens = []
    note_nums = []
    for i, batch in tqdm(enumerate(data_loader)):
        n_batches += 1
        n_samples += len(batch["name"])
        seq_lens.extend(int(l) for l in batch["seq_len"])
        for seq in batch["seq"].numpy():
            decoded_music = representation.decode(seq, encoding, encoding["code_event_map"])
            note_nums.append(sum([len(track) for track in decoded_music]))

        if i == 0:
            logging.info("Example:")
            for key, value in batch.items():
                if key == "name":
                    continue
                logging.info(f"Shape of {key}: {value.shape}")
            logging.info(f"Name: {batch['name'][0]}")
    logging.info(
        f"Successfully loaded {n_batches} batches ({n_samples} samples)."
    )

    # Print sequence length statistics
    logging.info(f"Avg sequence length: {np.mean(seq_lens):4f}")
    logging.info(f"Min sequence length: {min(seq_lens)}")
    logging.info(f"Max sequence length: {max(seq_lens)}")
    logging.info(f"Tot note number: {np.sum(note_nums)}")
    logging.info(f"Avg note number: {np.mean(note_nums):4f}")


if __name__ == "__main__":
    main()
