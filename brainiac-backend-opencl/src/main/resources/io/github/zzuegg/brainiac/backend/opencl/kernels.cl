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
        case 12: return softplus_(v);
        default: return rint(v);
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

// Element i of a counter-based random sequence, uniform in [0, 1): CounterRandom.uniform in Java.
float random_at(long seed, long i) {
    ulong z = (ulong) seed + ((ulong) i + 1UL) * 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z ^= z >> 31;
    return (float) (z >> 40) * 0x1.0p-24f;
}

__kernel void random(__global float* out, long seed, long n) {
    long i = get_global_id(0);
    if (i >= n) return;
    out[i] = random_at(seed, i);
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
// sliced operands are read in place. With splits > 1, the third dimension also splits k into
// pieces of chunk (a multiple of TS) summed separately: entry s of batch b is written to
// z[(b * splits + s) * strideZ ...], and the pieces are added up afterwards.
// the work-group size is fixed at compile time, so the compiler can budget registers for it
__kernel __attribute__((reqd_work_group_size(TS, RTS, 1))) void matmul(__global const T* a, __global const T* b, __global T* z, int m, int k, int n,
                     long strideA, long strideB, long strideZ, long aOff, long aRow, long aCol,
                     long bOff, long bRow, long bCol, int splits, int chunk) {
    // each work item computes WPT outputs of a column, RTS rows apart, so a value loaded from
    // the B tile serves WPT products; every output still sums k in order
    int col = get_global_id(0);
    long piece = get_global_id(2);
    long batch = piece / splits;
    int from = (int) (piece % splits) * chunk;
    int to = min(k, from + chunk);
    int lc = get_local_id(0);
    int lr = get_local_id(1);
    int row0 = get_group_id(1) * TS + lr;
    __local ACC tileA[TS][TS];
    __local ACC tileB[TS][TS];
    long aBase = aOff + batch * strideA;
    long bBase = bOff + batch * strideB;
    ACC acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = (ACC) 0;
    }
    for (int t = from; t < to; t += TS) {
        int ac = t + lc;
        for (int w = 0; w < WPT; w++) {
            int row = row0 + w * RTS;
            int br = t + lr + w * RTS;
            tileA[lr + w * RTS][lc] = row < m && ac < k ? LOAD(a, aBase + (long) row * aRow + (long) ac * aCol)
                    : (ACC) 0;
            tileB[lr + w * RTS][lc] = br < k && col < n ? LOAD(b, bBase + (long) br * bRow + (long) col * bCol)
                    : (ACC) 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TS; i++) {
            ACC bv = tileB[i][lc];
            for (int w = 0; w < WPT; w++) {
                acc[w] = fma(tileA[lr + w * RTS][i], bv, acc[w]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPT; w++) {
        int row = row0 + w * RTS;
        if (row < m && col < n) {
            STORE(z, piece * strideZ + (long) row * n + col, acc[w]);
        }
    }
}

// The same product for larger matrices: a work-group of 16 x 16 items computes a TILE x TILE
// tile of outputs (TILE = 16 * MW), each item MW x MW of them, 16 apart, so every value loaded
// from local memory serves MW products instead of 1 (and the k loop is shared by MW * MW
// outputs). Each output still sums k in order, as matmul does. matmul_big has 64 x 64 tiles
// (MW 4), matmul_mid 32 x 32 (MW 2), for products with too few big tiles to fill the device.
#define TILED_MATMUL(NAME, MW) \
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void NAME(__global const T* a, __global const T* b, __global T* z, int m, int k, int n, \
                   long strideA, long strideB, long strideZ, long aOff, long aRow, long aCol, \
                   long bOff, long bRow, long bCol, int splits, int chunk) { \
    int lc = get_local_id(0); \
    int lr = get_local_id(1); \
    int tid = lr * 16 + lc; \
    int colBase = get_group_id(0) * (16 * MW); \
    int rowBase = get_group_id(1) * (16 * MW); \
    long piece = get_global_id(2); \
    long batch = piece / splits; \
    int from = (int) (piece % splits) * chunk; \
    int to = min(k, from + chunk); \
    __local ACC tileA[16][16 * MW + 1]; \
    __local ACC tileB[16][16 * MW]; \
    long aBase = aOff + batch * strideA; \
    long bBase = bOff + batch * strideB; \
    ACC acc[MW][MW]; \
    for (int wm = 0; wm < MW; wm++) { \
        for (int wn = 0; wn < MW; wn++) { \
            acc[wm][wn] = (ACC) 0; \
        } \
    } \
    for (int t = from; t < to; t += 16) { \
        for (int l = 0; l < MW; l++) { \
            int idx = l * 256 + tid; \
            int row = rowBase + (idx >> 4); \
            int ac = t + (idx & 15); \
            tileA[idx & 15][idx >> 4] = row < m && ac < k ? LOAD(a, aBase + (long) row * aRow + (long) ac * aCol) \
                    : (ACC) 0; \
            int br = t + idx / (16 * MW); \
            int col = colBase + idx % (16 * MW); \
            tileB[idx / (16 * MW)][idx % (16 * MW)] = br < k && col < n \
                    ? LOAD(b, bBase + (long) br * bRow + (long) col * bCol) : (ACC) 0; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        for (int q = 0; q < 16; q++) { \
            ACC ra[MW]; \
            ACC rb[MW]; \
            for (int w = 0; w < MW; w++) { \
                ra[w] = tileA[q][lr + w * 16]; \
                rb[w] = tileB[q][lc + w * 16]; \
            } \
            for (int wm = 0; wm < MW; wm++) { \
                for (int wn = 0; wn < MW; wn++) { \
                    acc[wm][wn] = fma(ra[wm], rb[wn], acc[wm][wn]); \
                } \
            } \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (int wm = 0; wm < MW; wm++) { \
        int row = rowBase + lr + wm * 16; \
        for (int wn = 0; wn < MW; wn++) { \
            int col = colBase + lc + wn * 16; \
            if (row < m && col < n) { \
                STORE(z, piece * strideZ + (long) row * n + col, acc[wm][wn]); \
            } \
        } \
    } \
}
TILED_MATMUL(matmul_big, 4)
TILED_MATMUL(matmul_mid, 2)

// Attention in one pass per query row, without storing the [queries, keys] weights: the
// softmax is computed online (running maximum and sum), as in FlashAttention. q is
// [batch, heads, tq, dk], k [batch, kvHeads, tk, dk], v [batch, kvHeads, tk, dv]; heads
// share key/value heads in groups. Causal: query i sees keys up to i + tk - tq. lse keeps
// each row's log-sum-exp of the scaled scores for the backward pass. dk, dv <= ATT_D.
#define ATT_D 64
__kernel void attention(__global const T* q, __global const T* k, __global const T* v, __global T* out,
                        __global T* lse, int heads, int kvHeads, int tq, int tk, int dk, int dv, float scale,
                        int causal, long total) {
    long g = get_global_id(0);
    if (g >= total) return;
    int i = (int) (g % tq);
    long bh = g / tq;
    int h = (int) (bh % heads);
    long b = bh / heads;
    long kv = b * kvHeads + h / (heads / kvHeads);
    __global const T* kb = k;
    long kOff = kv * tk * dk;
    long vOff = kv * tk * dv;
    ACC qv[ATT_D];
    ACC acc[ATT_D];
    for (int d = 0; d < dk; d++) {
        qv[d] = LOAD(q, g * dk + d);
    }
    for (int d = 0; d < dv; d++) {
        acc[d] = (ACC) 0;
    }
    int last = causal ? min(tk - 1, i + tk - tq) : tk - 1;
    ACC m = -INFINITY;
    ACC l = (ACC) 0;
    for (int j = 0; j <= last; j++) {
        ACC s = (ACC) 0;
        for (int d = 0; d < dk; d++) {
            s += qv[d] * LOAD(kb, kOff + (long) j * dk + d);
        }
        s *= scale;
        ACC mn = fmax(m, s);
        ACC corr = exp(m - mn);
        ACC p = exp(s - mn);
        l = l * corr + p;
        for (int d = 0; d < dv; d++) {
            acc[d] = acc[d] * corr + p * LOAD(v, vOff + (long) j * dv + d);
        }
        m = mn;
    }
    for (int d = 0; d < dv; d++) {
        STORE(out, g * dv + d, acc[d] / l);
    }
    STORE(lse, g, m + log(l));
}

// Gradient of attention for the queries, one query row per item, recomputing each weight
// from lse: dq = scale * sum_j p_j (dO . v_j - D) k_j with D = dO . out. Also stores D.
__kernel void attention_dq(__global const T* q, __global const T* k, __global const T* v, __global const T* out,
                           __global const T* lse, __global const T* dout, __global T* dq, __global T* dsum,
                           int heads, int kvHeads, int tq, int tk, int dk, int dv, float scale, int causal,
                           long total) {
    long g = get_global_id(0);
    if (g >= total) return;
    int i = (int) (g % tq);
    long bh = g / tq;
    int h = (int) (bh % heads);
    long b = bh / heads;
    long kv = b * kvHeads + h / (heads / kvHeads);
    long kOff = kv * tk * dk;
    long vOff = kv * tk * dv;
    ACC qv[ATT_D];
    ACC dov[ATT_D];
    ACC acc[ATT_D];
    for (int d = 0; d < dk; d++) {
        qv[d] = LOAD(q, g * dk + d);
        acc[d] = (ACC) 0;
    }
    ACC dd = (ACC) 0;
    for (int d = 0; d < dv; d++) {
        dov[d] = LOAD(dout, g * dv + d);
        dd += dov[d] * LOAD(out, g * dv + d);
    }
    ACC row = LOAD(lse, g);
    int last = causal ? min(tk - 1, i + tk - tq) : tk - 1;
    for (int j = 0; j <= last; j++) {
        ACC s = (ACC) 0;
        for (int d = 0; d < dk; d++) {
            s += qv[d] * LOAD(k, kOff + (long) j * dk + d);
        }
        ACC p = exp(s * scale - row);
        ACC dp = (ACC) 0;
        for (int d = 0; d < dv; d++) {
            dp += dov[d] * LOAD(v, vOff + (long) j * dv + d);
        }
        ACC ds = p * (dp - dd) * scale;
        for (int d = 0; d < dk; d++) {
            acc[d] += ds * LOAD(k, kOff + (long) j * dk + d);
        }
    }
    for (int d = 0; d < dk; d++) {
        STORE(dq, g * dk + d, acc[d]);
    }
    STORE(dsum, g, dd);
}

// Gradient of attention for the keys and values, one key row per item, summing over the
// queries of every head that shares it: dv_j = sum p dO, dk_j = scale * sum p (dO . v_j - D) q.
__kernel void attention_dkv(__global const T* q, __global const T* k, __global const T* v,
                            __global const T* lse, __global const T* dout, __global const T* dsum,
                            __global T* dk_, __global T* dv_, int heads, int kvHeads, int tq, int tk, int dk,
                            int dv, float scale, int causal, long total) {
    long g = get_global_id(0);
    if (g >= total) return;
    int j = (int) (g % tk);
    long bkv = g / tk;
    int kh = (int) (bkv % kvHeads);
    long b = bkv / kvHeads;
    int group = heads / kvHeads;
    ACC kv_[ATT_D];
    ACC vv[ATT_D];
    ACC gk[ATT_D];
    ACC gv[ATT_D];
    for (int d = 0; d < dk; d++) {
        kv_[d] = LOAD(k, g * dk + d);
        gk[d] = (ACC) 0;
    }
    for (int d = 0; d < dv; d++) {
        vv[d] = LOAD(v, g * dv + d);
        gv[d] = (ACC) 0;
    }
    int first = causal ? max(0, j - (tk - tq)) : 0;
    for (int hh = 0; hh < group; hh++) {
        long qh = b * heads + kh * group + hh;
        for (int i = first; i < tq; i++) {
            long r = qh * tq + i;
            ACC s = (ACC) 0;
            for (int d = 0; d < dk; d++) {
                s += LOAD(q, r * dk + d) * kv_[d];
            }
            ACC p = exp(s * scale - LOAD(lse, r));
            ACC dp = (ACC) 0;
            for (int d = 0; d < dv; d++) {
                ACC o = LOAD(dout, r * dv + d);
                gv[d] += p * o;
                dp += o * vv[d];
            }
            ACC ds = p * (dp - LOAD(dsum, r)) * scale;
            for (int d = 0; d < dk; d++) {
                gk[d] += ds * LOAD(q, r * dk + d);
            }
        }
    }
    for (int d = 0; d < dk; d++) {
        STORE(dk_, g * dk + d, gk[d]);
    }
    for (int d = 0; d < dv; d++) {
        STORE(dv_, g * dv + d, gv[d]);
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

// Packed (quantized) matrices: levels row-major [k, n], one signed byte each, or with
// nibbles = 1 two per byte, the byte at (r / 2) * n + c holding row r (even) in its low four
// bits and row r + 1 in its high four bits; one float scale per column for every `group` rows.
int packed_level(__global const char* q, int r, int c, int n, int nibbles) {
    if (!nibbles) {
        return q[(long) r * n + c];
    }
    char b = q[(long) (r >> 1) * n + c];
    return (r & 1) ? (b >> 4) : ((char) (b << 4) >> 4);
}

__kernel void unpack_packed(__global const char* q, __global const float* scales, __global T* z, int k, int n,
                            int group, int nibbles) {
    long idx = get_global_id(0);
    if (idx >= (long) k * n) return;
    int r = idx / n;
    int c = idx % n;
    STORE(z, idx, (ACC) (packed_level(q, r, c, n, nibbles) * scales[(long) (r / group) * n + c]));
}

#define PACKED_ROWS 8
#define PACKED_QUADS 32
#define PACKED_SLICES 8

// the levels of columns c .. c + 3 at row r, read as one 32-bit word when n is a multiple of 4
ACC4 packed_levels(__global const uint* q, uint r, uint c, uint n, int nibbles) {
    uint b = nibbles ? (r >> 1) * n + c : r * n + c;
    if (n % 4 == 0) {
        int w = (int) q[b >> 2];
        if (nibbles) {
            int s = (r & 1) * 4;
            uint u = (uint) w;
            return (ACC4) ((int) (u << (28 - s)) >> 28, (int) (u << (20 - s)) >> 28, (int) (u << (12 - s)) >> 28,
                           (int) (u << (4 - s)) >> 28);
        }
        uint u = (uint) w;
        return (ACC4) ((int) (u << 24) >> 24, (int) (u << 16) >> 24, (int) (u << 8) >> 24, w >> 24);
    }
    __global const char* bytes = (__global const char*) q;
    ACC v[4];
    for (int j = 0; j < 4; j++) {
        v[j] = c + j < n ? (ACC) packed_level(bytes, r, c + j, n, nibbles) : (ACC) 0;
    }
    return (ACC4) (v[0], v[1], v[2], v[3]);
}

// z = a x packed for up to PACKED_ROWS rows of a. Each work item computes four adjacent
// columns; a work group covers 4 * PACKED_QUADS columns with PACKED_SLICES work items per
// column quad, each over its own run of rows, and dimension 2 picks a piece of `chunk` rows.
// Sums go to z[piece * m * n + ...], added up afterwards when there are several pieces.
__kernel __attribute__((reqd_work_group_size(PACKED_QUADS, PACKED_SLICES, 1)))
void matmul_packed(__global const T* a, __global const uint* q, __global const float* scales, __global T* z,
                   int m, int k, int n, int group, int nibbles, int chunk) {
    __local ACC4 sums[PACKED_SLICES][PACKED_QUADS];
    uint quad = get_local_id(0);
    uint slice = get_local_id(1);
    uint c = get_global_id(0) * 4;
    uint piece = get_global_id(2);
    uint from = piece * chunk;
    uint to = min((uint) k, from + chunk);
    uint run = ((to - from + PACKED_SLICES - 1) / PACKED_SLICES + 1) & ~1u;
    uint start = min(to, from + slice * run);
    uint end = min(to, start + run);
    ACC4 total[PACKED_ROWS];
    for (int i = 0; i < PACKED_ROWS; i++) {
        total[i] = (ACC4) 0;
    }
    if (c < (uint) n) {
        uint r = start;
        while (r < end) {
            uint g = r / group;
            uint stop = min(end, (g + 1) * group);
            ACC4 part[PACKED_ROWS];
            for (int i = 0; i < PACKED_ROWS; i++) {
                part[i] = (ACC4) 0;
            }
            for (; r < stop; r++) {
                ACC4 l = packed_levels(q, r, c, n, nibbles);
                for (int i = 0; i < PACKED_ROWS; i++) {
                    if (i < m) part[i] = fma((ACC4) LOAD(a, (long) i * k + r), l, part[i]);
                }
            }
            long s = (long) g * n + c;
            ACC4 sc = (ACC4) (scales[s], c + 1 < n ? scales[s + 1] : 0, c + 2 < n ? scales[s + 2] : 0,
                              c + 3 < n ? scales[s + 3] : 0);
            for (int i = 0; i < PACKED_ROWS; i++) {
                if (i < m) total[i] = fma(part[i], sc, total[i]);
            }
        }
    }
    long out0 = (long) piece * m * n;
    for (int i = 0; i < PACKED_ROWS; i++) {
        if (i >= m) break;
        sums[slice][quad] = total[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slice == 0 && c < (uint) n) {
            ACC4 t = sums[0][quad];
            for (int s = 1; s < PACKED_SLICES; s++) {
                t += sums[s][quad];
            }
            long o = out0 + (long) i * n + c;
            STORE(z, o, t.x);
            if (c + 1 < n) STORE(z, o + 1, t.y);
            if (c + 2 < n) STORE(z, o + 2, t.z);
            if (c + 3 < n) STORE(z, o + 3, t.w);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
