#!/usr/bin/env python3
"""
dataset.py — Download Salesforce/wikitext-2-raw-v1 → Data/
============================================================
Downloads three splits (train / validation / test) as raw text files
that Train.cpp reads directly.

Usage:
    pip install datasets huggingface_hub
    python dataset.py
    python dataset.py --data_dir /custom/path
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Data",
                        help="Directory to write train.txt / validation.txt / test.txt")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("[dataset.py] 'datasets' not found. Installing...")
        os.system(f"{sys.executable} -m pip install datasets huggingface_hub -q")
        from datasets import load_dataset

    os.makedirs(args.data_dir, exist_ok=True)

    print("[dataset.py] Downloading Salesforce/wikitext  wikitext-2-raw-v1 ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    splits = {
        "train":      "train.txt",
        "validation": "validation.txt",
        "test":       "test.txt",
    }

    for split, filename in splits.items():
        out_path = os.path.join(args.data_dir, filename)
        rows = ds[split]["text"]          # list[str]
        n_chars = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for line in rows:
                if line.strip():          # skip blank lines
                    f.write(line.rstrip("\n") + "\n")
                    n_chars += len(line)
        print(f"[dataset.py]   {split:12s} → {out_path}  "
              f"({len(rows):,} rows, {n_chars/1e6:.2f} MB)")

    print(f"[dataset.py] Done. Files written to '{args.data_dir}/'")
    print(f"[dataset.py] Run training with:")
    print(f"             g++ -std=c++17 -O3 -march=native -I. Train.cpp -o train -lm -lpthread")
    print(f"             ./train")

if __name__ == "__main__":
    main()
