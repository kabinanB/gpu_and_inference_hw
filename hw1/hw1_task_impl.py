import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # TODO (1 line): implement a lowest-AI op
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = torch.ones_like(x)
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    # TODO (1 line): return either `fn` or `torch.compile(fn)` based on `compiled`
    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # TODO: time `rep` runs using CUDA events and return median latency (ms)
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # TODO: compute total FLOPs, arithmetic intensity, and achieved FLOP/s
    total_flops = num_elements * num_ops * 2
    
    if variant == "compiled":
        # Fused kernel: read once, write once
        total_bytes = num_elements * 2 * bytes_per_element
    else:
        # Eager: separate ops, each reads and writes intermediates
        # Approximate as: read input, write for each of the num_ops operations
        total_bytes = num_elements * bytes_per_element * (2 * num_ops + 2)
    
    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# A1. From the actual data: compiled ops 1-64 have AI ranging from 0.25 to 16
# FLOP/B, runtimes stay nearly constant (~0.214ms), but achieved FLOP/s scales
# from ~616 GFLOP/s to ~40 TFLOP/s. This is because we're memory-bandwidth
# limited throughout this range (left of the ridge point at ~20 FLOP/B). The
# kernel hits the 3.35 TB/s memory ceiling each time, so runtime stays constant
# as we add more work-per-byte. The achieved FLOP/s increases simply because
# we compute more FLOPs in the same wall time.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# A2. From data: 1024x1024 matmul achieved 31.9 TFLOP/s while 128 ops compiled
# achieved 52.9 TFLOP/s. Two reasons: (1) Matmul has higher arithmetic intensity
# (170.7) so it's compute-bound, but the problem size is moderate and can't fully
# saturate H100's 67 TFLOP/s peak. (2) The fused element-wise operation is a
# single optimally-compiled kernel with zero overhead, while matmul may have
# suboptimal kernel selection or memory layout for this moderate size. Smaller
# matmuls often have poor FLOP/s utilization.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# A3. From data: 64 ops runs in 0.214ms (AI=16) but 128 ops runs in 0.325ms
# (AI=32), a 52% increase. The ridge point is ~20 FLOP/B. At 64 ops we're still
# memory-bound; at 128 ops we've crossed into the compute-bound regime. Once
# compute becomes the bottleneck (right of ridge), adding more FLOPs directly
# increases runtime since we can't exceed 67 TFLOP/s compute throughput. The
# runtime increase reflects the transition from being bandwidth-limited to
# compute-limited.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# A4. From the plot: eager ops (orange) stay horizontal with low AI (~0.12-0.25)
# and runtimes grow linearly (0.62ms to 67ms), while compiled ops (blue) move
# rightward (AI 0.25-32) with nearly constant runtime (~0.214ms). Eager PyTorch
# launches separate kernels for each operation in each iteration, materializing
# all intermediate tensors. For `acc = acc * x + x` with num_ops iterations, eager
# performs ~(2*num_ops + 2) reads/writes per element vs. compiled's 2 reads/1 write.
# This crushes the arithmetic intensity, keeping eager operations memory-bound no
# matter how many ops, and making runtime grow linearly with num_ops. Fusion is
# the key difference.
