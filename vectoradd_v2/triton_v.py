#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpu B200

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def vector_add_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """grid-stride vector add: C = A + B"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < N
    a = tl.load(A_ptr + block_start, mask=mask)
    b = tl.load(B_ptr + block_start, mask=mask)
    c = a + b
    tl.store(C_ptr + block_start, c, mask=mask)


def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    N = A.numel()

    # Triton 自动调 grid，一个 program 处理 BLOCK_SIZE 个元素
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](A, B, output, N, BLOCK_SIZE=BLOCK_SIZE)
    return output
