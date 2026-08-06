好，教你。核心思路：**你会 1D softmax（reduce → exp → normalize），attention 就是每行 query 做一次"带权重的 softmax"**。拆成 4 步：

## 数学结构（先搞清形状）

```
Q[M,d] × Kᵀ[N,d] → S[M,N]     # score
P[i,j] = softmax(S[i,j] / √d)  # 对每个 i，沿 j(N) 方向
O[i,k] = Σⱼ P[i,j] · V[j,k]    # 加权求和，得到 O[M,d]
```

关键：softmax 那部分你全会了。多出来的只有两件事：
1. score 不是现成的数组，是 `Q[i]·K[j]` 点积算出来的
2. 归一化后不是写回，而是拿去和 V 加权求和

## 分 4 pass（先写单 block 版，你已有的代码全能用）

每个 block 负责一个 query i：

**pass 1: 算 scores + max**
- 循环 j = 0..N，thread 分组，每个 thread 算部分点积 `s[j] = dot(Q[i,:], K[j,:])`
- `warp_reduce_max` / `block_reduce_max` 求行内最大值

**pass 2: exp + sum**
- 难点：pass 1 的 score 丢了。两条路：
  - (a) 存 smem（N 小的话，`N*4` 字节）
  - (b) 重算点积
- `block_reduce_sum` 求分母

**pass 3: 归一化 + 加权求和**
- `O[i,k] = inv * Σⱼ exp(S[j]-m) · V[j,k]`
- 这个求和是 d 维的向量和，每个 thread 负责一部分 k，循环累加

**pass 4（其实是 pass 3 的一部分）：** 没有 pass 4，3 步完了

## 你的任务

先把 `softmax_attn_single` 写出来（一个 block 一个 query，scores 存 smem 的 (a) 方案），写完给我看，我帮你改错。写完这个，grid-stride 版（`softmax_attn_kernel`）只是把 query 循环变成 `i = blockIdx.x; i < M; i += gridDim.x` 的事。

写吧，卡住随时问。