# Triton-Optimized Cross-Entropy Kernel

A high-performance, memory-efficient cross-entropy loss implementation using [Triton](https://github.com/openai/triton) for CUDA GPUs. Significantly faster than PyTorch's native cross-entropy, especially for large vocabulary sizes in large language models.

> **Attribution:**  
> This implementation is adapted from [Unsloth's cross-entropy kernel](https://github.com/unslothai/unsloth/blob/1898b6d049d606ec88f3f9307172373776eec0f6/unsloth/kernels/cross_entropy_loss.py).

---

## Features

- **Memory Efficient:** Fused kernel reduces memory footprint.
- **High Performance:** Optimized for large vocabulary sizes with Triton JIT.
- **Causal LM Compatible:** Handles shifted logits/labels for autoregressive language modeling.
- **Ignore Index Support:** Configurable ignore index for masking tokens (default: `-100`).
- **CUDA Accelerated:** Fully utilizes CUDA GPUs for maximum throughput.
- **Autograd Compatible:** Exposes a PyTorch-compatible `autograd.Function` and `nn.Module`.

---

## Requirements

- PyTorch (CUDA-enabled)
- Triton
- CUDA-compatible GPU

---

## Installation

Install from PyPI:

```bash
pip install crossentropy-triton
```

Or install with specific PyTorch/Triton versions:

```bash
pip install crossentropy-triton torch triton
```

---

## Usage

### Basic Usage (Autograd Function)

```python
import torch
from src import CrossEntropyFunction

device = torch.device('cuda')

# Create sample data [batch, seq, vocab_size]
logits = torch.randn(2, 10, 32000, device=device, requires_grad=True)
labels = torch.randint(0, 32000, (2, 10), device=device)

# Forward pass with ignore_index=-100 (default for masked tokens)
loss = CrossEntropyFunction.apply(logits, labels, -100)
print(f"Loss: {loss.item():.4f}")

# Backward pass
loss.backward()
print(f"Gradients computed - shape: {logits.grad.shape}")
```

### Using the Causal LM Loss Module

```python
import torch
from src import TritonCausalLMLoss

device = torch.device('cuda')
vocab_size = 32000

# Initialize the loss function
loss_fn = TritonCausalLMLoss(vocab_size)

# Create sample data
logits = torch.randn(2, 10, vocab_size, device=device, requires_grad=True)
labels = torch.randint(0, vocab_size, (2, 10), device=device)

# Forward and backward pass
loss = loss_fn(logits, labels)
print(f"Causal LM loss: {loss.item():.4f}")

loss.backward()
print(f"Backward pass completed")
```

---

## Performance Characteristics

- **Optimized Block Size:** Chooses optimal kernel block sizes up to 32,768.
- **Memory Fusion:** Fuses softmax and gradient computation in a single kernel.
- **Efficient Masking:** Ignore index is handled directly in the kernel.
- **Gradient Scaling:** Proper normalization by non-ignored tokens.

---

## Technical Details

### Kernel Implementation

- **`cross_entropy_kernel`:** Computes the forward pass (loss) and gradients in the logits tensor.
- **`element_mul_kernel`:** Scales in-place gradients by gradient outputs during backward.

### Memory and Numerical Stability

- Supports both contiguous and non-contiguous tensors.
- In-place gradient computation for minimal overhead.
- Log-sum-exp trick for stable softmax.

### Shifted Sequence Handling

- Causal/auto-regressive shifts are built in for next-token prediction.

---

## License

MIT License
