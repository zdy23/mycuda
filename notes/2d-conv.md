[2D Convolution](https://leetgpu.com/challenges/2d-convolution)

### Description

#### The basic idea of 2D convolution

Imagine you have a big image (the input), and a small stamp (the kernel, e.g 3x3). 
You slide the stamp over the image. At each position, you multiply the stamp's values by the image values underneath it, add them all up, and that sum becomes **one pixel of the output**. Then you move the stamp one step and repeat.

#### How this CUDA version makes it fast

**1. Each thread computes 16 outputs at once**

One thread is responsible for a vertical strip: 16 output pixels stacked in one column. `RS = 16, CS = 1`. 
A block has 32x8 threads, and the grid is sized so all strips together
cover the whole output image.

**2. The "sliding window" trick**

The naive way is that for each of the 16 outputs, you'd re-read all kernel-height values from slow global memory. That's a lot of repeated reading, because neighbouring outputs overlap heavily.

This Code's trick is keeping a tiny cache in fast registers, which holds a window of 16 input values in a column:

- **Warmup:** load the first 15 values of the column into the cache.
- **Each step:** load just **1 new value** at the bottom, multiply all 16 cached values by the matching kernel weight, add into the running sums, then **slide the window down by one** (shift values up, drop the oldest). 

### Math

output pixels:

$$
O[i, j] = \sum_{r=0}^{K_h-1} \sum_{c=0}^{K_w-1} I[i+r,\; j+c] \cdot K[r, c]
$$

compute size:

$$
H_{out} = H - K_h + 1, \qquad W_{out} = W - K_w + 1
$$

sliding window:

$$
sum[i] = \sum_{c=0}^{K_w-1} \sum_{r=0}^{K_h-1} I[row+i+r,\; col+c] \cdot K[r, c], \quad i = 0, 1, \dots, 15
$$
