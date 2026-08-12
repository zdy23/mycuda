[GEMM](https://leetgpu.com/challenges/general-matrix-multiplication-gemm)

### Description

#### The basic idea of GEMM

GEMM = **GE**nearl **M**atrix **M**ultiply. It computes:

$$
C = \alpha \cdot A \times B + \beta \cdot C
$$

Each output element is one **dot product**, a row of $A$ times a column of $B$, adding up $K$ numbers:

$$
C[i][j] = \alpha \cdot \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j] + \beta \cdot C[i][j]
$$

For 1024×1024×1024, that's about 2 billion multiply-adds.

The real enemy is not the math — it's memory: the naive way re-fetches the same numbers from slow global memory again and again. This kernel is all about fetching each number once and reusing it as much as possible.

#### How this CUDA version makes it fast

**1. Tiltin - work on small blocks, reuse data**

Like cooking, don't run to the supermarket global memory for every dish. Bring one tray of ingredients to the counter shared memory, and cook many dishes from it.

- Each block computes a **128x128** output tile.
- It loads **128×64 slice of A** and a **64×128 slice of B** into shared memory.
- Every loaded number is then reused ~128 times before being thrown away.

**2. Tensor cores - let the hardware multiply little matrices**

The `mma.sync.m16n8k16` instruction makes one warp compute a **16×8 output tile** from 16×16 and 16×8 inputs in a single instruction.

- 128 threads = 4 warps, arranged **2×2**. Each warp owns a **64×64** sub-tile.
- so, it makes 32 MMA calls per K-step.
- Each thread keeps **128 partial sums** in registers `reg_c[128]` and accumulates over the entire K loop.
- Global memory is touched only once at the very end.

**3. Software pipeline - load the next tile while computes the current one**

`cp.async` copies data from global memory to shared memory in the background, without blocking math. With `NPIPE = 3`, shared memory works like a 3-slot conveyor belt:

- slot 0: being computed using `mma.sync`
- slot 1: already loaded (ready for next bound)
- slot 2: loading right now using `cp.async`

While doing math on tile **t**, the GPU is already fetching tiles **t+1** and **t+2**. `cp.async`.

`wait_group + __syncthreads()` make sure a slot is full before anyone reads it.

**4. ldmatrix + XOR swizzle - fast, conflict-free shared memory reads**

Tensor cores want data in a weird per-thread "fragment" layout. `ldmatrix.x4` loads four 8×8 pieces from shared memory **directly into that layout** in one instruction and`ldmatrix.trans` does the transposed version for B.
The XOR swizzle — **(row % 8) ^ col** — scrambles which shared-memory **bank** each column lands in, so 32 threads reading at once never queue up on the same bank (no bank conflicts).

**5. Register-level pipelining too**

Even the fragment loads are double-buffered that while MMA for step **kk** runs, the fragments for **kk + 1** are already being fetched `load_a_frags(..., kk_next)`. The register arrays alternate between two halves (stage & 1).

**6. Smart epilogue**

At the end: registers → shared memory (reused, now as a float buffer) → multiply by **alpha** → **vectorized float4** reads → add **beta · old C** → convert to half → write back 8 bytes at a time.

#### Two kernels

- **M = N = K = 1024** → `gemm_tc_kernel`: the full tensor-core pipeline above.
- Any other size → `gemm_naive_kernel`: each thread computes an 8×8 patch of outputs straight from global memory. Slow, but simple and correct for arbitrary shapes.

#### Summary

- load data in big async chunks
- keep partial sums in registers
- let tensor-core instructions do the multiplication

### Math

**Helpers**

big chunk into small:

$$
C_{\text{tile}}^{(128 \times 128)} = \sum_{t=0}^{T-1} A_{\text{tile},t}^{(128 \times 64)} \cdot B_{\text{tile},t}^{(64 \times 128)}
$$

`m16n8k16` instruction:

$$
D^{(16 \times 8)} = A^{(16 \times 16)} \cdot B^{(16 \times 8)} + C^{(16 \times 8)}
$$

epilogue:

$$
out[i][j] = \mathrm{half}\Big(\alpha \cdot acc[i][j] + \beta \cdot old[i][j]\Big)
$$

**Total Computation**

Each multiplication and addition counts as one floating-point operation:

$$
\text{FLOPs} = 2 \cdot M \cdot N \cdot K
$$

For $M=N=K=1024$

$$
2 \cdot 1024^3 \approx 2.15 \times 10^9
$$	

So the operation performs approximately 2.15 billion floating-point operations.

**Data Movement Per Tile**

Each tile loads one tile from $A$ and one tile from $B$:

$$
\text{Bytes}=(BM \cdot BK + BK \cdot BN)\cdot \mathrm{sizeof}(f16)
$$

The arithmetic intensity measures how much computation is performed for each byte loaded:

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs per tile}}{\text{Bytes per tile}}
$$

For $BM=BN=128$, $BK=64$, and each f16 using 2 bytes:

$$
\frac{128 \times 128 \times 64 \times 2}
{(128 \times 64 + 64 \times 128) \times 2}
\approx 64\ \text{FLOP/Byte}
$$

This means that for every byte loaded from memory, the GPU performs about 64 floating-point operations. Higher arithmetic intensity usually means better use of memory bandwidth and better performance.


### Performance

The optimized GEMM achieves 109% of cuBLAS at 1024x1024x1024.

A significant gain.

