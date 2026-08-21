# GPT-Style Language Model from Scratch Using NumPy

A GPT-style autoregressive language model implemented from scratch using **NumPy**, without using deep-learning frameworks such as PyTorch or TensorFlow for the model architecture.

The project implements the major components of a Transformer-based language model, including BPE tokenization, token and positional embeddings, causal multi-head self-attention, feed-forward networks, layer normalization, backpropagation, training, checkpointing, evaluation, and autoregressive text generation.

## Project Highlights

- Implemented Byte Pair Encoding (BPE) tokenizer from scratch
- Implemented token embeddings and positional embeddings
- Implemented causal self-attention from scratch
- Implemented multi-head attention with 4 attention heads
- Implemented Transformer block
- Implemented GELU activation
- Implemented layer normalization
- Implemented feed-forward neural network
- Implemented cross-entropy loss
- Implemented manual backward propagation and parameter updates
- Implemented model checkpoint saving and loading
- Implemented greedy and Top-K text generation
- Trained the model on Sherlock Holmes text

## Architecture

The model follows a simplified GPT-style decoder architecture:

```text
Input Text
    ↓
BPE Tokenizer
    ↓
Token IDs
    ↓
Token Embeddings
    +
Positional Embeddings
    ↓
Transformer Block
    ├── Layer Normalization
    ├── Causal Multi-Head Self-Attention
    ├── Residual Connection
    ├── Layer Normalization
    ├── Feed-Forward Network
    └── Residual Connection
    ↓
Final Layer Normalization
    ↓
Vocabulary Projection
    ↓
Next-Token Logits
```

## Model Configuration

| Parameter | Value |
|---|---:|
| Vocabulary Size | 239 |
| Embedding Dimension | 256 |
| Transformer Blocks | 1 |
| Attention Heads | 4 |
| Context Length | 16 |
| Training Stride | 4 |
| BPE Merges | 150 |
| Learning Rate | 0.001 |
| Trained Epochs | 15 |

## Tokenization

A custom Byte Pair Encoding tokenizer was implemented instead of using an existing tokenizer library.

The tokenizer:

1. Splits the input text using regular expressions.
2. Counts token-pair frequencies.
3. Iteratively merges frequently occurring adjacent pairs.
4. Stores the order of learned BPE merges.
5. Converts tokens into integer IDs.
6. Decodes generated token IDs back into text.

Example:

```python
encoded = tokenizer.encode("Sherlock Holmes was ")
decoded = tokenizer.decode(encoded)
```

Example token IDs:

```text
[45, 115, 139, 82, 2, 33, 2, 212, 2]
```

## Training

The model is trained using next-token prediction.

For a sequence:

```text
x1 x2 x3 x4
```

the corresponding target is:

```text
x2 x3 x4 x5
```

Cross-entropy loss is calculated between the model's vocabulary logits and the target token IDs.

The backward pass was implemented manually using NumPy, including gradients through:

- Output projection
- Layer normalization
- Feed-forward network
- GELU
- Multi-head attention
- Individual self-attention heads

Model parameters are updated using gradient descent.

## Evaluation

Evaluation of the trained Epoch-15 checkpoint produced:

| Metric | Result |
|---|---:|
| Overall Accuracy | 41.29% |
| Non-Space Accuracy | 24.35% |
| Space Prediction Percentage | 34.70% |

The non-space accuracy is reported separately because whitespace is a frequent token in the character/subword-level training data.

## Text Generation

The model supports:

- Greedy decoding
- Temperature-controlled sampling
- Top-K sampling

Example:

```python
text = generate(
    gpt,
    tokenizer,
    "Sherlock Holmes was ",
    max_new_tokens=30,
    temperature=0.8,
    do_sample=True,
    top_k=10
)

print(text)
```

Example generated text:

```text
Sherlock Holmes was a discoloured in my pipe clearly, and that he c
```

Because the model is intentionally small and trained on a limited corpus, generated text demonstrates learned language structure but is not expected to have the coherence of large pretrained language models.

## Project Structure

```text
GPT-From-Scratch/
│
├── tokenizer.py
│   └── BPE tokenizer and vocabulary construction
│
├── data.py
│   └── Dataset and DataLoader
│
├── layers.py
│   ├── LayerNormalization
│   ├── GELU
│   └── FeedForward
│
├── attention.py
│   ├── SelfAttentionHead
│   └── MultiHeadAttention
│
├── transformer.py
│   └── TransformerBlock
│
├── model.py
│   └── GPT model and checkpoint handling
│
├── loss.py
│   └── CrossEntropyLoss
│
├── generate.py
│   └── Autoregressive text generation
│
├── train.py
│   └── Model training loop
│
├── evaluate.py
│   └── Model evaluation
│
├── LLM.ipynb
│   └── Experiments and development notebook
│
├── requirements.txt
│
└── README.md
```

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the tokenizer

```python
from tokenizer import build_tokenizer

tokenizer, vocab, merge_rank = build_tokenizer(
    raw_data,
    num_merges=150
)
```

### 3. Create the model

```python
from model import GPT
import numpy as np

embedding_matrix = (
    np.random.randn(
        len(vocab),
        256
    ) * 0.02
)

gpt = GPT(
    num_layers=1,
    embed_dim=256,
    num_heads=4,
    sequence_length=16,
    embedding_matrix=embedding_matrix
)
```

### 4. Load a trained checkpoint

```python
gpt.load_weights(
    "gpt_stride4_epoch_15.pkl"
)
```

### 5. Generate text

```python
from generate import generate

output = generate(
    gpt,
    tokenizer,
    "Sherlock Holmes was ",
    max_new_tokens=30,
    temperature=0.8,
    do_sample=True,
    top_k=10
)

print(output)
```

## Why Build GPT From Scratch?

The primary objective of this project was not to compete with pretrained language models, but to understand how Transformer language models work internally.

Rather than relying on high-level deep-learning APIs, the core mathematical operations were implemented using NumPy. This provides hands-on understanding of:

- Tokenization
- Embedding representations
- Query, Key, and Value projections
- Scaled dot-product attention
- Causal masking
- Multi-head attention
- Residual connections
- Layer normalization
- Feed-forward networks
- Cross-entropy loss
- Backpropagation
- Autoregressive decoding

## Limitations

This is an educational implementation of a small GPT-style model.

Current limitations include:

- Small training corpus
- Short context length
- Single Transformer block
- Small vocabulary
- Basic gradient-descent optimization
- Limited semantic coherence for long generations
- CPU/NumPy-based training is significantly slower than optimized deep-learning frameworks

These limitations are intentional trade-offs that keep the complete Transformer implementation understandable and implementable from first principles.

## Future Improvements

Potential extensions include:

- Increasing the number of Transformer blocks
- Increasing context length
- Training on a larger corpus
- Implementing Adam/AdamW optimization
- Adding dropout
- Implementing mini-batch shuffling
- Adding validation loss and perplexity
- Improving BPE vocabulary construction
- Implementing learning-rate scheduling
- Comparing the NumPy implementation with an equivalent PyTorch model

## Tech Stack

- Python
- NumPy
- Regular Expressions
- Jupyter Notebook

## Author

**Tammana Venkata Karthik**

M.Tech, IIT (BHU) Varanasi