import torch
import triton
import triton.language as tl
from typing import Optional

MAX_FUSED_SIZE = 32768

@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride
    grad_output = tl.load(grad_output_ptr)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)

@triton.jit
def cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        
        softmax_block = tl.exp(X_block - m) / d
        grad_block = tl.where(X_offsets != y, softmax_block, softmax_block - 1.0)
        grad_block = grad_block / n_non_ignore
        tl.store(X_ptr + X_offsets, grad_block, mask=X_offsets < n_cols)

    tl.debug_barrier()
    loss = lse - ori_X_y
    loss = loss / n_non_ignore
    tl.store(loss_ptr, loss)

def cross_entropy_forward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not logits.is_cuda:
        raise RuntimeError("Triton kernels require CUDA tensors.")
    
    BT, V = logits.shape
    n_rows = BT
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    loss_1d = torch.zeros(n_rows, dtype=logits.dtype, device=logits.device)
    target_mask = labels != ignore_index
    n_non_ignore = target_mask.sum().item()
    
    if n_non_ignore == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype), logits

    valid_targets = labels * target_mask
    assert valid_targets.max() < V
    assert valid_targets.min() >= 0

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if labels.stride(-1) != 1:
        labels = labels.contiguous()

    grid = (n_rows,)
    cross_entropy_kernel[grid](
        X_ptr=logits,
        X_stride=logits.stride(-2),
        Y_ptr=labels,
        Y_stride=labels.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        n_cols=V,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    total_loss = torch.sum(loss_1d)
    return total_loss, logits

def cross_entropy_backward(
    logits_with_grads: torch.Tensor, 
    grad_output: torch.Tensor
) -> torch.Tensor:
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return logits_with_grads
    
    if grad_output.ndim == 0:
        BT, V = logits_with_grads.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            logits_with_grads,
            logits_with_grads.stride(-2),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
        return logits_with_grads
    else:
        return logits_with_grads * grad_output.unsqueeze(dim=1)

class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        logits: torch.Tensor,
        labels: torch.Tensor, 
        ignore_index: int = -100
    ) -> torch.Tensor:
        original_logits_shape = logits.shape
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
        flat_labels = shifted_labels.view(-1)
        flat_labels = flat_labels.to(flat_logits.device)
        
        loss, logits_with_grads = cross_entropy_forward(
            flat_logits, flat_labels, ignore_index
        )
        
        ctx.save_for_backward(logits_with_grads.detach())
        ctx.original_shape = original_logits_shape
        return loss
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[Optional[torch.Tensor], None, None]:
        (logits_with_grads,) = ctx.saved_tensors
        original_shape = ctx.original_shape
        scaled_grads = cross_entropy_backward(logits_with_grads, grad_output)
        *batch_dims, seq_len, vocab_size = original_shape
        shifted_shape = (*batch_dims, seq_len - 1, vocab_size)
        reshaped_grads = scaled_grads.view(shifted_shape)
        zeros_shape = (*batch_dims, 1, vocab_size)
        zeros_pad = torch.zeros(
            zeros_shape, 
            dtype=reshaped_grads.dtype, 
            device=reshaped_grads.device
        )
        full_grads = torch.cat([reshaped_grads, zeros_pad], dim=-2)
        return full_grads, None, None