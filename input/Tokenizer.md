# Tokenizer

Converts raw Amharic / Ge'ez text into token IDs using the **EthioBBPE** pretrained model (16 000-token vocabulary trained on Amharic biblical and Synaxarium corpora).

---

## Install

```bash
pip install -r requirements.txt
```

The model is downloaded automatically from Hugging Face on first use and cached locally.

---

## Usage

```python
from tokenizer import Tokenizer

tok = Tokenizer()

# Single string → List[int]
ids = tok.encode("ሰላም ዓለም")
print(ids)

# Batch → List[List[int]]
batch_ids = tok.encode_batch(["ሰላም", "ዓለም", "ሐዋርያ"])
print(batch_ids)

# Callable shorthand (same as above)
ids       = tok("ሰላም ዓለም")
batch_ids = tok(["ሰላም", "ዓለም"])

# With truncation
ids = tok.encode("long text ...", truncation=True, max_length=256)

# Vocabulary size
print(tok.vocab_size)  # 16000
```

---

## Scope

| In scope | Out of scope |
|---|---|
| Text → token IDs | Embeddings |
| Batch tokenization | Attention masks / type IDs |
| Truncation | Model inference |

---

## Source

- Library: [EthioBBPE](https://github.com/nexuss0781/Ethio_BBPE)
- Model: [Nexuss0781/Ethio-BBPE](https://huggingface.co/Nexuss0781/Ethio-BBPE)


Path to look at(this is to avoid confusion)
Transformer/SRC/Phase-1_Input-Processing$ find . type f
.
./Component-1.1_Tokenizer
./Component-1.1_Tokenizer/README.md    # == Tokenizer.md
./Component-1.1_Tokenizer/requirements.txt
./Component-1.1_Tokenizer/tokenizer.py
./Component-1.2_Token-Embedding         # You're here
./Component-1.2_Token-Embedding/input
./Component-1.2_Token-Embedding/input/Tokenizer.md
./Component-1.2_Token-Embedding/SPEC.md

import by ..tokenizer relatively


