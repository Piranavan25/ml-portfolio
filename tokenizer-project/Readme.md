# Custom BPE Tokenizer for Wikitext-103

This project demonstrates how to train a **Byte-Pair Encoding (BPE) tokenizer** on the [Wikitext-103](https://huggingface.co/datasets/wikitext) dataset using the Hugging Face `tokenizers` library.

---

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Output](#output)
- [License](#license)

---

## Overview

The script performs the following steps:

1. Loads the **Wikitext-103** dataset from Hugging Face.
2. Cleans and preprocesses the text by removing extra spaces and newlines.
3. Trains a **BPE tokenizer** with a vocabulary size of 30,000.
4. Saves the tokenizer as `wikitext_tokenizer.json`.
5. Demonstrates encoding a sample text.

---

## Requirements

- Python 3.8+
- Hugging Face `datasets` library
- Hugging Face `tokenizers` library
- `re` (regular expressions, included in standard library)

Install dependencies via pip:

```bash
pip install datasets tokenizers
````


## Usage

Run the script:

```bash
python train_tokenizer.py
```

This will:

1. Load the Wikitext-103 dataset.
2. Clean and preprocess the text.
3. Train a custom BPE tokenizer.
4. Save the trained tokenizer to `wikitext_tokenizer.json`.
5. Print a sample encoding.

Example output:

```text
Loading Wikitext-103...
Total samples: 1800000
Training tokenizer...
Tokenizer saved as wikitext_tokenizer.json
['Hello', ',', 'Ġthis', 'Ġis', 'Ġmy', 'Ġcustom', 'Ġtokenizer', '!']
[31373, 11, 232, 318, 672, 14126, 23298, 0]
```

---

## Output

* **`wikitext_tokenizer.json`** – The trained tokenizer file.
* Encoded tokens and their corresponding IDs for sample text.

---

