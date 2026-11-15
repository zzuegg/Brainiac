// Brainiac OpenCL kernels. The host prepends a prelude that defines, for one number format:
//   T             the storage type (float, double, half or ushort)
//   ACC           the type computed in (float, or double for float64)
//   LOAD(p, i)    reads element i of a T buffer as ACC
//   STORE(p, i, v) writes v (an ACC) to element i, rounding to the storage type
//   RAW           a type of the same width as T that can be copied without arithmetic
//   TS            the matrix tile size
// Operation codes are the ordinals of the Java enums UnaryOp, ActivationOp, BinaryOp and ReduceOp.

#define SQRT_2_OVER_PI 0.7978845608028654
#define INV_SQRT_2 0.7071067811865476
#define INV_SQRT_2PI 0.3989422804014327
#define GELU_CUBIC 0.044715
#define SELU_ALPHA 1.6732632423543772
#define SELU_SCALE 1.0507009873554805

ACC sigmoid_(ACC x) {
    if (x >= (ACC) 0) {
        return (ACC) 1 / ((ACC) 1 + exp(-x));
    }
    ACC e = exp(x);
    return e / ((ACC) 1 + e);
}

ACC softplus_(ACC x) {
    return fmax(x, (ACC) 0) + log1p(exp(-fabs(x)));
}

ACC unary_op(int op, ACC v) {
    switch (op) {
        case 0: return -v;
        case 1: return fabs(v);
        case 2: return exp(v);
        case 3: return log(v);
        case 4: return sqrt(v);
        case 5: return v * v;
        case 6: return tanh(v);
        case 7: return sigmoid_(v);
        case 8: return v > (ACC) 0 ? v : (ACC) 0;
        case 9: return v > (ACC) 0 ? (ACC) 1 : (ACC) 0;
        case 10: return v > (ACC) 0 ? (ACC) 1 : v < (ACC) 0 ? (ACC) -1 : v;
        case 11: return erf(v);
        default: return softplus_(v);
    }
}

ACC hard_sigmoid_(ACC x) {
    return fmin(fmax(x / (ACC) 6 + (ACC) 0.5, (ACC) 0), (ACC) 1);
}

ACC activation_op(int op, ACC p, ACC x) {
    switch (op) {
        case 0: return x > (ACC) 0 ? x : (ACC) 0;
        case 1: return sigmoid_(x);
        case 2: return tanh(x);
        case 3: return softplus_(x);
        case 4: return x * sigmoid_(x);
        case 5: return x * (ACC) 0.5 * ((ACC) 1 + erf(x * (ACC) INV_SQRT_2));
        case 6: return (ACC) 0.5 * x * ((ACC) 1 + tanh((ACC) SQRT_2_OVER_PI * (x + (ACC) GELU_CUBIC * x * x * x)));
        case 7: return x * tanh(softplus_(x));
        case 8: return x > (ACC) 0 ? x : p * x;
        case 9: return x > (ACC) 0 ? x : p * expm1(x);
        case 10: return (ACC) SELU_SCALE * (x > (ACC) 0 ? x : (ACC) SELU_ALPHA * expm1(x));
        case 11: return fmin(fmax(x, (ACC) 0), (ACC) 6);
        case 12: return fmin(fmax(x, (ACC) -1), (ACC) 1);
        case 13: return hard_sigmoid_(x);
        case 14: return x * hard_sigmoid_(x);
        default: return x / ((ACC) 1 + fabs(x));
    }
}

ACC activation_slope(int op, ACC p, ACC x) {
    switch (op) {
        case 0: return x > (ACC) 0 ? (ACC) 1 : (ACC) 0;
        case 1: { ACC s = sigmoid_(x); return s * ((ACC) 1 - s); }
        case 2: { ACC t = tanh(x); return (ACC) 1 - t * t; }
        case 3: return sigmoid_(x);
        case 4: { ACC s = sigmoid_(x); return s * ((ACC) 1 + x * ((ACC) 1 - s)); }
        case 5: {
            ACC e = exp((ACC) -0.5 * x * x);
            return (ACC) 0.5 * ((ACC) 1 + erf(x * (ACC) INV_SQRT_2)) + x * (ACC) INV_SQRT_2PI * e;
        }
        case 6: {
            ACC t = tanh((ACC) SQRT_2_OVER_PI * (x + (ACC) GELU_CUBIC * x * x * x));
            return (ACC) 0.5 * ((ACC) 1 + t) + (ACC) 0.5 * x * ((ACC) 1 - t * t) * (ACC) SQRT_2_OVER_PI
                    * ((ACC) 1 + (ACC) 3 * (ACC) GELU_CUBIC * x * x);
        }
        case 7: { ACC t = tanh(softplus_(x)); return t + x * ((ACC) 1 - t * t) * sigmoid_(x); }
        case 8: return x > (ACC) 0 ? (ACC) 1 : p;
        case 9: return x > (ACC) 0 ? (ACC) 1 : p * exp(x);
        case 10: return (ACC) SELU_SCALE * (x > (ACC) 0 ? (ACC) 1 : (ACC) SELU_ALPHA * exp(x));
        case 11: return x > (ACC) 0 && x < (ACC) 6 ? (ACC) 1 : (ACC) 0;
        case 12: return x > (ACC) -1 && x < (ACC) 1 ? (ACC) 1 : (ACC) 0;
        case 13: return x > (ACC) -3 && x < (ACC) 3 ? (ACC) 1 / (ACC) 6 : (ACC) 0;
        case 14: return x <= (ACC) -3 ? (ACC) 0 : x >= (ACC) 3 ? (ACC) 1 : ((ACC) 2 * x + (ACC) 3) / (ACC) 6;
        default: { ACC d = (ACC) 1 + fabs(x); return (ACC) 1 / (d * d); }
    }
}

ACC binary_op(int op, ACC a, ACC b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return pow(a, b);
        case 5: return isnan(a) || isnan(b) ? a + b : fmax(a, b);
        case 6: return isnan(a) || isnan(b) ? a + b : fmin(a, b);
        case 7: return a > b ? (ACC) 1 : (ACC) 0;
        case 8: return a == b ? (ACC) 1 : (ACC) 0;
        default: return b != (ACC) 0 ? a : (ACC) 0;
    }
}

__kernel void fill(__global T* out, float v, long n) {
    long i = get_global_id(0);
    if (i < n) STORE(out, i, (ACC) v);
}

__kernel void unary(int op, __global const T* x, __global T* y, long n) {
    long i = get_global_id(0);
    if (i < n) STORE(y, i, unary_op(op, LOAD(x, i)));
}

__kernel void activation(int op, float p, __global const T* x, __global T* y, long n) {
    long i = get_global_id(0);
    if (i < n) STORE(y, i, activation_op(op, (ACC) p, LOAD(x, i)));
}

__kernel void activation_backward(int op, float p, __global const T* x, __global const T* g, __global T* z, long n) {
    long i = get_global_id(0);
    if (i < n) STORE(z, i, LOAD(g, i) * activation_slope(op, (ACC) p, LOAD(x, i)));
}

__kernel void binary_contiguous(int op, __global const T* x, __global const T* y, __global T* z, long n) {
    long i = get_global_id(0);
    if (i < n) STORE(z, i, binary_op(op, LOAD(x, i), LOAD(y, i)));
}

// meta: shape[rank], aStrides[rank], bStrides[rank]
__kernel void binary_strided(int op, __global const T* x, __global const T* y, __global T* z, long n, int rank,
                             __global const long* meta) {
    long i = get_global_id(0);
    if (i >= n) return;
    long rest = i;
    long ia = 0;
    long ib = 0;
    for (int d = rank - 1; d >= 0; d--) {
        long idx = rest % meta[d];
        rest /= meta[d];
        ia += idx * meta[rank + d];
        ib += idx * meta[2 * rank + d];
    }
    STORE(z, i, binary_op(op, LOAD(x, ia), LOAD(y, ib)));
}

// meta: shape[rank], inStrides[rank], outStrides[rank]
__kernel void copy_strided(__global const RAW* x, long inOffset, __global RAW* z, long outOffset, long n, int rank,
                           __global const long* meta) {
    long i = get_global_id(0);
    if (i >= n) return;
    long rest = i;
    long ia = inOffset;
    long iz = outOffset;
    for (int d = rank - 1; d >= 0; d--) {
        long idx = rest % meta[d];
        rest /= meta[d];
        ia += idx * meta[rank + d];
        iz += idx * meta[2 * rank + d];
    }
    z[iz] = x[ia];
}

// Reductions; op 0 sum, 1 mean, 2 max, 3 min. Generated (fused) reductions use the same order.
ACC reduce_combine(int op, ACC a, ACC b) {
    if (op == 2) return isnan(a) || isnan(b) ? a + b : fmax(a, b);
    if (op == 3) return isnan(a) || isnan(b) ? a + b : fmin(a, b);
    return a + b;
}

ACC reduce_init(int op) {
    return op == 2 ? (ACC) -INFINITY : op == 3 ? (ACC) INFINITY : (ACC) 0;
}

// One work item per output, looping over its row.
__kernel void reduce_rows(int op, __global const T* x, __global T* y, long n, long len, long inner) {
    long r = get_global_id(0);
    if (r >= n) return;
    long base = (r / inner) * len * inner + r % inner;
    ACC acc = reduce_init(op);
    for (long t = 0; t < len; t++) {
        acc = reduce_combine(op, acc, LOAD(x, base + t * inner));
    }
    STORE(y, r, op == 1 ? acc / (ACC) len : acc);
}

// One work-group of RG items per output, for long rows.
__kernel void reduce_group(int op, __global const T* x, __global T* y, long n, long len, long inner) {
    __local ACC part[RG];
    long r = get_group_id(0);
    int lid = get_local_id(0);
    long base = (r / inner) * len * inner + r % inner;
    ACC acc = reduce_init(op);
    for (long t = lid; t < len; t += RG) {
        acc = reduce_combine(op, acc, LOAD(x, base + t * inner));
    }
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = RG / 2; s > 0; s >>= 1) {
        if (lid < s) part[lid] = reduce_combine(op, part[lid], part[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        ACC total = part[0];
        STORE(y, r, op == 1 ? total / (ACC) len : total);
    }
}

__kernel void argmax(__global const T* x, __global float* y, long outer, long len, long inner) {
    long r = get_global_id(0);
    if (r >= outer * inner) return;
    long base = (r / inner) * len * inner + r % inner;
    long best = 0;
    ACC bestValue = LOAD(x, base);
    for (long i = 1; i < len; i++) {
        ACC v = LOAD(x, base + i * inner);
        if (v > bestValue) {
            best = i;
            bestValue = v;
        }
    }
    y[r] = (float) best;
}

// z[b] = a[b] x b[b] for [m, k] x [k, n] matrices; strides in elements between batch entries.
// a(i, j) is a[aOff + batch * strideA + i * aRow + j * aCol], b likewise, so transposed or
// sliced operands are read in place.
__kernel void matmul(__global const T* a, __global const T* b, __global T* z, int m, int k, int n,
                     long strideA, long strideB, long strideZ, long aOff, long aRow, long aCol,
                     long bOff, long bRow, long bCol) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    long batch = get_global_id(2);
    int lc = get_local_id(0);
    int lr = get_local_id(1);
    __local ACC tileA[TS][TS];
    __local ACC tileB[TS][TS];
    long aBase = aOff + batch * strideA;
    long bBase = bOff + batch * strideB;
    ACC acc = (ACC) 0;
    for (int t = 0; t < k; t += TS) {
        int ac = t + lc;
        int br = t + lr;
        tileA[lr][lc] = row < m && ac < k ? LOAD(a, aBase + (long) row * aRow + (long) ac * aCol) : (ACC) 0;
        tileB[lr][lc] = br < k && col < n ? LOAD(b, bBase + (long) br * bRow + (long) col * bCol) : (ACC) 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TS; i++) {
            acc += tileA[lr][i] * tileB[i][lc];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < m && col < n) {
        STORE(z, batch * strideZ + (long) row * n + col, acc);
    }
}

// table[p * ks + k]: the image cell of kernel cell k at window position p, or -1 for padding
__kernel void im2col(__global const T* x, __global const int* table, __global T* z, int channels, int positions,
                     int ks, int img, float pad, long total) {
    long idx = get_global_id(0);
    if (idx >= total) return;
    long width = (long) channels * ks;
    long row = idx / width;
    int col = (int) (idx % width);
    long nb = row / positions;
    int p = (int) (row % positions);
    int ch = col / ks;
    int k = col % ks;
    int cell = table[p * ks + k];
    STORE(z, idx, cell < 0 ? (ACC) pad : LOAD(x, (nb * channels + ch) * img + cell));
}

// For every image cell, offsets/entries list the (p * ks + k) that read it.
__kernel void col2im(__global const T* cols, __global const int* offsets, __global const int* entries,
                     __global T* z, int channels, int positions, int ks, int img, long total) {
    long idx = get_global_id(0);
    if (idx >= total) return;
    long plane = idx / img;
    int cell = (int) (idx % img);
    long nb = plane / channels;
    int ch = (int) (plane % channels);
    long width = (long) channels * ks;
    ACC acc = (ACC) 0;
    for (int e = offsets[cell]; e < offsets[cell + 1]; e++) {
        int p = entries[e] / ks;
        int k = entries[e] % ks;
        acc += LOAD(cols, (nb * positions + p) * width + (long) ch * ks + k);
    }
    STORE(z, idx, acc);
}

__kernel void gather(__global const RAW* x, __global const int* indices, __global RAW* z, long len, long inner, int count,
                     long total) {
    long idx = get_global_id(0);
    if (idx >= total) return;
    long o = idx / ((long) count * inner);
    long rest = idx % ((long) count * inner);
    long i = rest / inner;
    long j = rest % inner;
    z[idx] = x[(o * len + indices[i]) * inner + j];
}

// For every target l, offsets/entries list the source rows i with indices[i] == l.
__kernel void scatter_add(__global const T* src, __global const int* offsets, __global const int* entries,
                          __global T* z, long len, long inner, int count, long total) {
    long idx = get_global_id(0);
    if (idx >= total) return;
    long o = idx / (len * inner);
    long rest = idx % (len * inner);
    long l = rest / inner;
    long j = rest % inner;
    ACC acc = (ACC) 0;
    for (int e = offsets[l]; e < offsets[l + 1]; e++) {
        acc += LOAD(src, (o * count + entries[e]) * inner + j);
    }
    STORE(z, idx, acc);
}

__kernel void to_float(__global const T* x, __global float* y, long n) {
    long i = get_global_id(0);
    if (i < n) y[i] = (float) LOAD(x, i);
}

__kernel void from_float(__global const float* x, __global T* y, long n) {
    long i = get_global_id(0);
    if (i < n) STORE(y, i, (ACC) x[i]);
}
