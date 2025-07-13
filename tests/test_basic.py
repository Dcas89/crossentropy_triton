import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src import TritonCausalLMLoss

# Reference PyTorch Module
class CausalLMLoss(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self, 
        logits: torch.FloatTensor, 
        labels: torch.LongTensor
        ) -> torch.FloatTensor:
        
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        labels = labels.to(logits.device)

        loss = F.cross_entropy(logits, labels, ignore_index=-100)

        return loss


# CUDA Timing Helper
def cuda_time(fn, warmup: int = 10, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


class TestTritonCausalLMLoss:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = torch.device('cuda')

    def test_basic_functionality(self):
        batch, seq, vocab = 2, 8, 100000
        loss_fn = TritonCausalLMLoss(vocab).to(self.device)

        logits = torch.randn(batch, seq, vocab, device=self.device, requires_grad=True)
        labels = torch.randint(0, vocab, (batch, seq), device=self.device)
        labels[0, -2:] = -100
        labels[1, -1] = -100

        loss = loss_fn(logits, labels)
        loss.backward()

        assert loss.requires_grad and not torch.isnan(loss)
        assert logits.grad is not None and not torch.isnan(logits.grad).any()


@pytest.mark.parametrize("batch,seq,vocab", [
    (1, 500, 60000),
    (2, 1000, 80000),
    (4, 500, 100000),
])
def test_triton_vs_pytorch_equivalence(batch, seq, vocab):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    base_logits = torch.randn(batch, seq, vocab, device=device)
    base_labels = torch.randint(0, vocab, (batch, seq), device=device)
    base_labels[:, -1] = -100
    if seq > 2:
        base_labels[0, -2] = -100

    def run_triton():
        l = base_logits.clone().detach().requires_grad_(True)
        lab = base_labels.clone()
        if l.grad is not None:
            l.grad.zero_()
        loss = TritonCausalLMLoss(vocab)(l, lab)
        loss.backward()

    def run_pytorch():
        l = base_logits.clone().detach().requires_grad_(True)
        lab = base_labels.clone()
        if l.grad is not None:
            l.grad.zero_()
        loss = CausalLMLoss(vocab)(l, lab)
        loss.backward()

    t_triton = cuda_time(run_triton, warmup=20, repeat=100)
    t_pt = cuda_time(run_pytorch, warmup=20, repeat=100)

    l_tr = base_logits.clone().detach().requires_grad_(True)
    lb_tr = base_labels.clone()
    l_pt = base_logits.clone().detach().requires_grad_(True)
    lb_pt = base_labels.clone()

    loss_triton = TritonCausalLMLoss(vocab)(l_tr, lb_tr)
    loss_triton.backward()
    grad_triton = l_tr.grad.detach().clone()

    loss_ref = CausalLMLoss(vocab)(l_pt, lb_pt)
    loss_ref.backward()
    grad_ref = l_pt.grad.detach().clone()

    # --- Assertions ---
    assert torch.allclose(loss_triton, loss_ref, atol=1e-6, rtol=1e-6), \
        f"Loss mismatch: {loss_triton.item():.6f} vs {loss_ref.item():.6f}"
    assert grad_triton.shape == grad_ref.shape
    max_diff = (grad_triton - grad_ref).abs().max()
    assert max_diff < 1e-6, f"Grad max diff too large: {max_diff:.2e}"

    # --- Report ---
    print(f"\nTest[batch={batch} seq={seq} vocab={vocab}]")
    print(f"  Triton:   {t_triton:.3f} ms")
    print(f"  PyTorch:  {t_pt:.3f} ms")
    print(f"  Speedup:  {t_pt/t_triton:.2f}×")
