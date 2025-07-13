import torch
from .crossentropy_tl import CrossEntropyFunction

class TritonCausalLMLoss(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
    
    def forward(
        self, 
        logits: torch.FloatTensor, 
        labels: torch.LongTensor
    ) -> torch.FloatTensor:
        return CrossEntropyFunction.apply(logits, labels, -100)
