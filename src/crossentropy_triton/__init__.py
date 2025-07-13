"""
Triton-Optimized Cross-Entropy Kernel

A high-performance, memory-efficient cross-entropy loss implementation using Triton for CUDA GPUs.
"""

from .crossentropy_tl import CrossEntropyFunction
from .causalLMloss import TritonCausalLMLoss

__version__ = "0.1.2"
__author__ = "Daniel Castillo"
__email__ = "d.castillocastagneto@gmail.com"

__all__ = [
    "CrossEntropyFunction",
    "TritonCausalLMLoss",
]
