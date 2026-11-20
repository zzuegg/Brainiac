// Brainiac's HIP kernels, compiled at run time by hipRTC for the GPU in use. A port of the CUDA
// backend's kernels.cu: the same code apart from the half conversions and warp shuffles. Written
// for 32-lane wavefronts (RDNA GPUs, which HIP compiles for in wave32).
//
// Every kernel exists once per number format, with the suffix _f32, _f64, _f16 or _bf16.
// float16 and bfloat16 are stored as 16-bit integers and computed in float; float64 in double.
// The math follows the CPU backend (and the Vulkan backend's math.glsl): operation codes are
// the ordinals of the Java enums UnaryOp, ActivationOp, BinaryOp and ReduceOp.
//
// No headers: hipRTC provides the math functions; bfloat16 conversions are written out.

typedef unsigned short u16;
typedef long long i64;

#define GROUP 256
#define MAX_AXES 8

// ---------------------------------------------------------------- number formats

// the GPU's own half conversions, rounding to nearest even
__device__ __forceinline__ float h2f(u16 h) {
    _Float16 v;
    __builtin_memcpy(&v, &h, 2);
    return (float) v;
}

__device__ __forceinline__ u16 f2h(float f) {
    _Float16 v = (_Float16) f;
    u16 h;
    __builtin_memcpy(&h, &v, 2);
    return h;
}

// shuffles within the 32 lanes of a wavefront (CUDA's __shfl_*_sync over a full warp)
#define SHFL_DOWN(v, o) __shfl_down(v, o, 32)
#define SHFL_XOR(v, o) __shfl_xor(v, o, 32)

__device__ __forceinline__ float bf2f(u16 b) {
    return __uint_as_float(((unsigned) b) << 16);
}

// rounds to nearest even, as DType.floatToBfloat16 does
__device__ __forceinline__ u16 f2bf(float f) {
    unsigned bits = __float_as_uint(f);
    if (f != f) {
        return (u16) ((bits >> 16) | 0x40u);
    }
    return (u16) ((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

// T: how values are stored; A: what they are computed in
struct F32 {
    typedef float T;
    typedef float A;
    static __device__ __forceinline__ A ld(const T* p, i64 i) { return p[i]; }
    static __device__ __forceinline__ void st(T* p, i64 i, A v) { p[i] = v; }
    static __device__ __forceinline__ A round(A v) { return v; }
};

struct F64 {
    typedef double T;
    typedef double A;
    static __device__ __forceinline__ A ld(const T* p, i64 i) { return p[i]; }
    static __device__ __forceinline__ void st(T* p, i64 i, A v) { p[i] = v; }
    static __device__ __forceinline__ A round(A v) { return v; }
};

struct F16 {
    typedef u16 T;
    typedef float A;
    static __device__ __forceinline__ A ld(const T* p, i64 i) { return h2f(p[i]); }
    static __device__ __forceinline__ void st(T* p, i64 i, A v) { p[i] = f2h(v); }
    static __device__ __forceinline__ A round(A v) { return h2f(f2h(v)); }
};

struct BF16 {
    typedef u16 T;
    typedef float A;
    static __device__ __forceinline__ A ld(const T* p, i64 i) { return bf2f(p[i]); }
    static __device__ __forceinline__ void st(T* p, i64 i, A v) { p[i] = f2bf(v); }
    static __device__ __forceinline__ A round(A v) { return bf2f(f2bf(v)); }
};

// ---------------------------------------------------------------- math in float and double

__device__ __forceinline__ float m_exp(float x) { return expf(x); }
__device__ __forceinline__ double m_exp(double x) { return exp(x); }
__device__ __forceinline__ float m_log(float x) { return logf(x); }
__device__ __forceinline__ double m_log(double x) { return log(x); }
__device__ __forceinline__ float m_sqrt(float x) { return sqrtf(x); }
__device__ __forceinline__ double m_sqrt(double x) { return sqrt(x); }
__device__ __forceinline__ float m_tanh(float x) { return tanhf(x); }
__device__ __forceinline__ double m_tanh(double x) { return tanh(x); }
__device__ __forceinline__ float m_sin(float x) { return sinf(x); }
__device__ __forceinline__ double m_sin(double x) { return sin(x); }
__device__ __forceinline__ float m_cos(float x) { return cosf(x); }
__device__ __forceinline__ double m_cos(double x) { return cos(x); }
__device__ __forceinline__ float m_atan(float x) { return atanf(x); }
__device__ __forceinline__ double m_atan(double x) { return atan(x); }
__device__ __forceinline__ float m_abs(float x) { return fabsf(x); }
__device__ __forceinline__ double m_abs(double x) { return fabs(x); }
__device__ __forceinline__ float m_floor(float x) { return floorf(x); }
__device__ __forceinline__ double m_floor(double x) { return floor(x); }
__device__ __forceinline__ float m_rint(float x) { return rintf(x); }
__device__ __forceinline__ double m_rint(double x) { return rint(x); }
__device__ __forceinline__ float m_pow(float a, float b) { return powf(a, b); }
__device__ __forceinline__ double m_pow(double a, double b) { return pow(a, b); }
__device__ __forceinline__ float m_fmod(float a, float b) { return fmodf(a, b); }
__device__ __forceinline__ double m_fmod(double a, double b) { return fmod(a, b); }
__device__ __forceinline__ float m_expm1(float x) { return expm1f(x); }
__device__ __forceinline__ double m_expm1(double x) { return expm1(x); }
__device__ __forceinline__ float m_log1p(float x) { return log1pf(x); }
__device__ __forceinline__ double m_log1p(double x) { return log1p(x); }

template <class A> __device__ __forceinline__ bool is_nan(A x) { return x != x; }
template <class A> __device__ __forceinline__ A pos_inf();
template <> __device__ __forceinline__ float pos_inf<float>() { return __int_as_float(0x7f800000); }
template <> __device__ __forceinline__ double pos_inf<double>() { return __longlong_as_double(0x7ff0000000000000LL); }
template <class A> __device__ __forceinline__ A nan_();
template <> __device__ __forceinline__ float nan_<float>() { return __int_as_float(0x7fc00000); }
template <> __device__ __forceinline__ double nan_<double>() { return __longlong_as_double(0x7ff8000000000000LL); }

template <class A> __device__ __forceinline__ A max_(A a, A b) { return is_nan(a) || is_nan(b) ? a + b : (a > b ? a : b); }
template <class A> __device__ __forceinline__ A min_(A a, A b) { return is_nan(a) || is_nan(b) ? a + b : (a < b ? a : b); }
template <class A> __device__ __forceinline__ A clamp_(A x, A lo, A hi) { return x < lo ? lo : (x > hi ? hi : x); }

template <class A> __device__ A sigmoid_(A x) {
    if (x >= A(0)) {
        return A(1) / (A(1) + m_exp(-x));
    }
    A e = m_exp(x);
    return e / (A(1) + e);
}

template <class A> __device__ A softplus_(A x) {
    return (x > A(0) ? x : A(0)) + m_log1p(m_exp(-m_abs(x)));
}

// Abramowitz and Stegun 7.1.26, as the CPU backend computes it
template <class A> __device__ A erf_(A x) {
    A z = m_abs(x);
    A t = A(1) / (A(1) + A(0.3275911) * z);
    A poly = t * (A(0.254829592) + t * (A(-0.284496736) + t * (A(1.421413741) + t * (A(-1.453152027)
            + t * A(1.061405429)))));
    A r = A(1) - poly * m_exp(-z * z);
    return x >= A(0) ? r : -r;
}

template <class A> __device__ A pow_(A a, A b) {
    if (b == A(0)) {
        return A(1);
    }
    if (is_nan(a) || is_nan(b)) {
        return a + b;
    }
    if (a < A(0)) {
        if (b != m_floor(b)) {
            return nan_<A>();
        }
        A r = m_pow(-a, b);
        return m_fmod(m_abs(b), A(2)) == A(1) ? -r : r;
    }
    if (a == A(0)) {
        return b > A(0) ? A(0) : pos_inf<A>();
    }
    return m_pow(a, b);
}

template <class A> __device__ A unary_op(int op, A v) {
    switch (op) {
        case 0: return -v;
        case 1: return m_abs(v);
        case 2: return m_exp(v);
        case 3: return m_log(v);
        case 4: return m_sqrt(v);
        case 5: return v * v;
        case 6: return m_tanh(v);
        case 7: return sigmoid_(v);
        case 8: return v > A(0) ? v : A(0);
        case 9: return v > A(0) ? A(1) : A(0);
        case 10: return v > A(0) ? A(1) : (v < A(0) ? A(-1) : v);
        case 11: return erf_(v);
        case 12: return softplus_(v);
        case 13: return m_rint(v);
        case 14: return m_sin(v);
        case 15: return m_cos(v);
        default: return m_atan(v);
    }
}

#define SQRT_2_OVER_PI A(0.7978845608028654)
#define INV_SQRT_2 A(0.7071067811865476)
#define INV_SQRT_2PI A(0.3989422804014327)
#define GELU_CUBIC A(0.044715)
#define SELU_ALPHA A(1.6732632423543772)
#define SELU_SCALE A(1.0507009873554805)

template <class A> __device__ A hard_sigmoid_(A x) {
    return clamp_(x / A(6) + A(0.5), A(0), A(1));
}

template <class A> __device__ A activation_op(int op, A p, A x) {
    switch (op) {
        case 0: return x > A(0) ? x : A(0);
        case 1: return sigmoid_(x);
        case 2: return m_tanh(x);
        case 3: return softplus_(x);
        case 4: return x * sigmoid_(x);
        case 5: return x * A(0.5) * (A(1) + erf_(x * INV_SQRT_2));
        case 6: return A(0.5) * x * (A(1) + m_tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)));
        case 7: return x * m_tanh(softplus_(x));
        case 8: return x > A(0) ? x : p * x;
        case 9: return x > A(0) ? x : p * m_expm1(x);
        case 10: return SELU_SCALE * (x > A(0) ? x : SELU_ALPHA * m_expm1(x));
        case 11: return clamp_(x, A(0), A(6));
        case 12: return clamp_(x, A(-1), A(1));
        case 13: return hard_sigmoid_(x);
        case 14: return x * hard_sigmoid_(x);
        default: return x / (A(1) + m_abs(x));
    }
}

template <class A> __device__ A activation_slope(int op, A p, A x) {
    switch (op) {
        case 0: return x > A(0) ? A(1) : A(0);
        case 1: { A s = sigmoid_(x); return s * (A(1) - s); }
        case 2: { A t = m_tanh(x); return A(1) - t * t; }
        case 3: return sigmoid_(x);
        case 4: { A s = sigmoid_(x); return s * (A(1) + x * (A(1) - s)); }
        case 5: {
            A e = m_exp(A(-0.5) * x * x);
            return A(0.5) * (A(1) + erf_(x * INV_SQRT_2)) + x * INV_SQRT_2PI * e;
        }
        case 6: {
            A t = m_tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
            return A(0.5) * (A(1) + t) + A(0.5) * x * (A(1) - t * t) * SQRT_2_OVER_PI
                    * (A(1) + A(3) * GELU_CUBIC * x * x);
        }
        case 7: { A t = m_tanh(softplus_(x)); return t + x * (A(1) - t * t) * sigmoid_(x); }
        case 8: return x > A(0) ? A(1) : p;
        case 9: return x > A(0) ? A(1) : p * m_exp(x);
        case 10: return SELU_SCALE * (x > A(0) ? A(1) : SELU_ALPHA * m_exp(x));
        case 11: return x > A(0) && x < A(6) ? A(1) : A(0);
        case 12: return x > A(-1) && x < A(1) ? A(1) : A(0);
        case 13: return x > A(-3) && x < A(3) ? A(1) / A(6) : A(0);
        case 14: return x <= A(-3) ? A(0) : (x >= A(3) ? A(1) : (A(2) * x + A(3)) / A(6));
        default: { A d = A(1) + m_abs(x); return A(1) / (d * d); }
    }
}

template <class A> __device__ A binary_op(int op, A a, A b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return pow_(a, b);
        case 5: return max_(a, b);
        case 6: return min_(a, b);
        case 7: return a > b ? A(1) : A(0);
        case 8: return a == b ? A(1) : A(0);
        default: return b != A(0) ? a : A(0);
    }
}

// ---------------------------------------------------------------- indexing

// A shape of up to MAX_AXES axes with the strides of two operands (in elements; 0 broadcasts)
struct Strided {
    int rank;
    int pad_;
    i64 shape[MAX_AXES];
    i64 s0[MAX_AXES];
    i64 s1[MAX_AXES];
};

#define FOR_EACH(i, n) \
    for (i64 i = (i64) blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (i64) gridDim.x * blockDim.x)

__device__ __forceinline__ void offsets(const Strided& d, i64 i, i64& o0, i64& o1) {
    i64 rest = i;
    i64 a = 0;
    i64 b = 0;
    for (int axis = d.rank - 1; axis >= 0; axis--) {
        i64 size = d.shape[axis];
        i64 idx = rest % size;
        rest /= size;
        a += idx * d.s0[axis];
        b += idx * d.s1[axis];
    }
    o0 = a;
    o1 = b;
}

// ---------------------------------------------------------------- element-wise

template <class D> __device__ void unary_k(const typename D::T* x, typename D::T* y, i64 n, int op) {
    typedef typename D::A A;
    FOR_EACH(i, n) {
        D::st(y, i, unary_op<A>(op, D::ld(x, i)));
    }
}

template <class D> __device__ void activation_k(const typename D::T* x, typename D::T* y, i64 n, int op, float p) {
    typedef typename D::A A;
    FOR_EACH(i, n) {
        D::st(y, i, activation_op<A>(op, A(p), D::ld(x, i)));
    }
}

template <class D> __device__ void activation_backward_k(const typename D::T* x, const typename D::T* g,
        typename D::T* y, i64 n, int op, float p) {
    typedef typename D::A A;
    FOR_EACH(i, n) {
        D::st(y, i, D::ld(g, i) * activation_slope<A>(op, A(p), D::ld(x, i)));
    }
}

template <class D> __device__ void binary_contiguous_k(const typename D::T* a, const typename D::T* b,
        typename D::T* z, i64 n, int op) {
    typedef typename D::A A;
    FOR_EACH(i, n) {
        D::st(z, i, binary_op<A>(op, D::ld(a, i), D::ld(b, i)));
    }
}

template <class D> __device__ void binary_k(const typename D::T* a, const typename D::T* b, typename D::T* z, i64 n,
        int op, Strided d) {
    typedef typename D::A A;
    FOR_EACH(i, n) {
        i64 ia;
        i64 ib;
        offsets(d, i, ia, ib);
        D::st(z, i, binary_op<A>(op, D::ld(a, ia), D::ld(b, ib)));
    }
}

// in: strides s0; out: strides s1
template <class D> __device__ void copy_k(const typename D::T* x, typename D::T* z, i64 n, Strided d) {
    FOR_EACH(i, n) {
        i64 ix;
        i64 iz;
        offsets(d, i, ix, iz);
        z[iz] = x[ix];
    }
}

template <class D> __device__ void fill_k(typename D::T* y, i64 n, double v) {
    typedef typename D::A A;
    FOR_EACH(i, n) {
        D::st(y, i, A(v));
    }
}

template <class D> __device__ void from_float_k(const float* x, typename D::T* y, i64 n) {
    typedef typename D::A A;
    FOR_EACH(i, n) {
        D::st(y, i, A(x[i]));
    }
}

template <class D> __device__ void to_float_k(const typename D::T* x, float* y, i64 n) {
    FOR_EACH(i, n) {
        y[i] = (float) D::ld(x, i);
    }
}

// ---------------------------------------------------------------- reductions (op: 0 sum, 1 mean, 2 max, 3 min)

template <class A> __device__ __forceinline__ A reduce_start(int op) {
    return op == 2 ? -pos_inf<A>() : (op == 3 ? pos_inf<A>() : A(0));
}

template <class A> __device__ __forceinline__ A combine(int op, A a, A b) {
    return op == 2 ? max_(a, b) : (op == 3 ? min_(a, b) : a + b);
}

template <class A> __device__ __forceinline__ A warp_reduce(int op, A v) {
    for (int o = 16; o > 0; o >>= 1) {
        v = combine(op, v, SHFL_DOWN(v, o));
    }
    return v;
}

// one thread per output, looping over its row: rows of [outer, length, inner] as outer * inner
template <class D> __device__ void reduce_k(const typename D::T* x, typename D::T* y, i64 rows, i64 len, i64 inner,
        int op) {
    typedef typename D::A A;
    FOR_EACH(r, rows) {
        i64 base = (r / inner) * len * inner + r % inner;
        A acc = reduce_start<A>(op);
        for (i64 t = 0; t < len; t++) {
            acc = combine(op, acc, D::ld(x, base + t * inner));
        }
        D::st(y, r, op == 1 ? acc / A(len) : acc);
    }
}

// contiguous rows (inner 1): one warp per row, GROUP / 32 rows per block
template <class D> __device__ void reduce_warp_k(const typename D::T* x, typename D::T* y, i64 rows, i64 len, int op) {
    typedef typename D::A A;
    int lane = threadIdx.x & 31;
    for (i64 r = (i64) blockIdx.x * (GROUP / 32) + (threadIdx.x >> 5); r < rows; r += (i64) gridDim.x * (GROUP / 32)) {
        const typename D::T* row = x + r * len;
        A acc = reduce_start<A>(op);
        for (i64 t = lane; t < len; t += 32) {
            acc = combine(op, acc, D::ld(row, t));
        }
        acc = warp_reduce(op, acc);
        if (lane == 0) {
            D::st(y, r, op == 1 ? acc / A(len) : acc);
        }
    }
}

// few long rows: one block per output
template <class D> __device__ void reduce_block_k(const typename D::T* x, typename D::T* y, i64 rows, i64 len,
        i64 inner, int op) {
    typedef typename D::A A;
    __shared__ A part[GROUP / 32];
    for (i64 r = blockIdx.x; r < rows; r += gridDim.x) {
        i64 base = (r / inner) * len * inner + r % inner;
        A acc = reduce_start<A>(op);
        for (i64 t = threadIdx.x; t < len; t += GROUP) {
            acc = combine(op, acc, D::ld(x, base + t * inner));
        }
        acc = warp_reduce(op, acc);
        if ((threadIdx.x & 31) == 0) {
            part[threadIdx.x >> 5] = acc;
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            acc = threadIdx.x < GROUP / 32 ? part[threadIdx.x] : reduce_start<A>(op);
            acc = warp_reduce(op, acc);
            if (threadIdx.x == 0) {
                D::st(y, r, op == 1 ? acc / A(len) : acc);
            }
        }
        __syncthreads();
    }
}

// the index of each row's first largest value, stored as a number
template <class D> __device__ void argmax_k(const typename D::T* x, typename D::T* y, i64 rows, i64 len, i64 inner) {
    typedef typename D::A A;
    FOR_EACH(r, rows) {
        i64 base = (r / inner) * len * inner + r % inner;
        i64 best = 0;
        A bestValue = D::ld(x, base);
        for (i64 t = 1; t < len; t++) {
            A v = D::ld(x, base + t * inner);
            if (v > bestValue) {
                best = t;
                bestValue = v;
            }
        }
        D::st(y, r, A(best));
    }
}

// argmax of few long rows (a vocabulary's logits): a block per row, each thread scanning every
// GROUP-th value, then the threads' bests combined, the first index winning a tie
template <class D> __device__ void argmax_block_k(const typename D::T* x, typename D::T* y, i64 rows, i64 len,
        i64 inner) {
    typedef typename D::A A;
    __shared__ A values[GROUP];
    __shared__ i64 indices[GROUP];
    for (i64 r = blockIdx.x; r < rows; r += gridDim.x) {
        i64 base = (r / inner) * len * inner + r % inner;
        i64 best = -1;
        A bestValue = A(0);
        for (i64 t = threadIdx.x; t < len; t += GROUP) {
            A v = D::ld(x, base + t * inner);
            if (best < 0 || v > bestValue) {
                best = t;
                bestValue = v;
            }
        }
        values[threadIdx.x] = bestValue;
        indices[threadIdx.x] = best;
        __syncthreads();
        for (int s = GROUP / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                i64 oi = indices[threadIdx.x + s];
                i64 mi = indices[threadIdx.x];
                A ov = values[threadIdx.x + s];
                A mv = values[threadIdx.x];
                if (oi >= 0 && (mi < 0 || ov > mv || (ov == mv && oi < mi))) {
                    values[threadIdx.x] = ov;
                    indices[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            D::st(y, r, A(indices[0]));
        }
        __syncthreads();
    }
}

// argmax of few long rows split over blocks: block (p, r) finds the best of piece p of row r and
// writes (value, index) to part; the row's last block to finish (counted in counters[r], left at
// 0 again) picks among the pieces in order, so the first index still wins a tie
template <class D> __device__ void argmax_split_k(const typename D::T* x, typename D::T* y, double* part,
        int* counters, i64 rows, i64 len, i64 inner, int pieces) {
    typedef typename D::A A;
    __shared__ A values[GROUP];
    __shared__ i64 indices[GROUP];
    __shared__ int last;
    i64 r = blockIdx.y;
    int p = blockIdx.x;
    i64 chunk = (len + pieces - 1) / pieces;
    i64 t0 = p * chunk;
    i64 t1 = min(len, t0 + chunk);
    i64 base = (r / inner) * len * inner + r % inner;
    i64 best = -1;
    A bestValue = A(0);
    for (i64 t = t0 + threadIdx.x; t < t1; t += GROUP) {
        A v = D::ld(x, base + t * inner);
        if (best < 0 || v > bestValue) {
            best = t;
            bestValue = v;
        }
    }
    values[threadIdx.x] = bestValue;
    indices[threadIdx.x] = best;
    __syncthreads();
    for (int s = GROUP / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            i64 oi = indices[threadIdx.x + s];
            i64 mi = indices[threadIdx.x];
            A ov = values[threadIdx.x + s];
            A mv = values[threadIdx.x];
            if (oi >= 0 && (mi < 0 || ov > mv || (ov == mv && oi < mi))) {
                values[threadIdx.x] = ov;
                indices[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        part[(r * pieces + p) * 2] = (double) values[0];
        part[(r * pieces + p) * 2 + 1] = (double) indices[0];
        __threadfence();
        last = atomicAdd(&counters[r], 1) == pieces - 1;
    }
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence();
        volatile double* pr = part + r * pieces * 2;
        double bv = 0;
        double bi = -1;
        for (int q = 0; q < pieces; q++) {
            double vi = pr[q * 2 + 1];
            double vv = pr[q * 2];
            if (vi >= 0 && (bi < 0 || vv > bv)) {
                bv = vv;
                bi = vi;
            }
        }
        D::st(y, r, A(bi));
        counters[r] = 0;
    }
}

// RMSNorm, one block per row: y = round(x / sqrt(mean(x^2) + eps)) * w, in float
template <class D> __device__ void rms_norm_k(const typename D::T* x, const typename D::T* w, typename D::T* y,
        i64 rows, int cols, float eps) {
    __shared__ float part[GROUP / 32];
    for (i64 r = blockIdx.x; r < rows; r += gridDim.x) {
        i64 base = r * cols;
        float acc = 0;
        for (int t = threadIdx.x; t < cols; t += GROUP) {
            float v = (float) D::ld(x, base + t);
            acc += v * v;
        }
        acc = warp_reduce(0, acc);
        if ((threadIdx.x & 31) == 0) {
            part[threadIdx.x >> 5] = acc;
        }
        __syncthreads();
        float total = 0;
        for (int i = 0; i < GROUP / 32; i++) {
            total += part[i];
        }
        float d = sqrtf(total / (float) cols + eps);
        for (int t = threadIdx.x; t < cols; t += GROUP) {
            float n = (float) D::round((typename D::A) ((float) D::ld(x, base + t) / d));
            D::st(y, base + t, (typename D::A) (n * (float) D::ld(w, t)));
        }
        __syncthreads();
    }
}

// a residual add and the RMSNorm after it, one block per row: sum = x + r as an add stores it,
// y the RMSNorm of the stored sum. Each thread reads back only the sums it wrote.
template <class D> __device__ void add_rms_norm_k(const typename D::T* x, const typename D::T* res,
        const typename D::T* w, typename D::T* sum, typename D::T* y, i64 rows, int cols, float eps) {
    typedef typename D::A A;
    __shared__ float part[GROUP / 32];
    for (i64 r = blockIdx.x; r < rows; r += gridDim.x) {
        i64 base = r * cols;
        float acc = 0;
        for (int t = threadIdx.x; t < cols; t += GROUP) {
            D::st(sum, base + t, (A) D::ld(x, base + t) + (A) D::ld(res, base + t));
            float v = (float) D::ld(sum, base + t);
            acc += v * v;
        }
        acc = warp_reduce(0, acc);
        if ((threadIdx.x & 31) == 0) {
            part[threadIdx.x >> 5] = acc;
        }
        __syncthreads();
        float total = 0;
        for (int i = 0; i < GROUP / 32; i++) {
            total += part[i];
        }
        float d = sqrtf(total / (float) cols + eps);
        for (int t = threadIdx.x; t < cols; t += GROUP) {
            float n = (float) D::round((A) ((float) D::ld(sum, base + t) / d));
            D::st(y, base + t, (A) (n * (float) D::ld(w, t)));
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- windows and indices

// table[pos * ks + k]: the image cell kernel cell k reads at window position pos, or -1
template <class D> __device__ void im2col_k(const typename D::T* x, const int* table, typename D::T* z, int channels,
        int positions, int ks, i64 img, float pad, i64 total) {
    typedef typename D::A A;
    i64 width = (i64) channels * ks;
    FOR_EACH(idx, total) {
        i64 row = idx / width;
        i64 col = idx % width;
        i64 nb = row / positions;
        i64 pos = row % positions;
        i64 ch = col / ks;
        i64 k = col % ks;
        int cell = table[pos * ks + k];
        D::st(z, idx, cell < 0 ? A(pad) : D::ld(x, (nb * channels + ch) * img + cell));
    }
}

// offsets[cell] .. offsets[cell + 1]: the entries (pos * ks + k) of the table that read image cell `cell`
template <class D> __device__ void col2im_k(const typename D::T* cols, const int* offsets, const int* entries,
        typename D::T* z, int channels, int positions, int ks, i64 img, i64 total) {
    typedef typename D::A A;
    i64 width = (i64) channels * ks;
    FOR_EACH(idx, total) {
        i64 plane = idx / img;
        i64 cell = idx % img;
        i64 nb = plane / channels;
        i64 ch = plane % channels;
        A acc = A(0);
        int end = offsets[cell + 1];
        for (int e = offsets[cell]; e < end; e++) {
            i64 en = entries[e];
            i64 pos = en / ks;
            i64 k = en % ks;
            acc += D::ld(cols, (nb * positions + pos) * width + ch * ks + k);
        }
        D::st(z, idx, acc);
    }
}

template <class D> __device__ void gather_k(const typename D::T* x, const float* indices, typename D::T* z, i64 len,
        i64 inner, i64 count, i64 total) {
    FOR_EACH(idx, total) {
        i64 o = idx / (count * inner);
        i64 rest = idx % (count * inner);
        i64 i = rest / inner;
        i64 j = rest % inner;
        z[idx] = x[(o * len + (i64) indices[i]) * inner + j];
    }
}

// offsets[l] .. offsets[l + 1]: the source rows i whose index is l, in order
template <class D> __device__ void scatter_add_k(const typename D::T* src, const int* offsets, const int* entries,
        typename D::T* z, i64 len, i64 inner, i64 count, i64 total) {
    typedef typename D::A A;
    FOR_EACH(idx, total) {
        i64 o = idx / (len * inner);
        i64 rest = idx % (len * inner);
        i64 l = rest / inner;
        i64 j = rest % inner;
        A acc = A(0);
        int end = offsets[l + 1];
        for (int e = offsets[l]; e < end; e++) {
            acc += D::ld(src, (o * count + entries[e]) * inner + j);
        }
        D::st(z, idx, acc);
    }
}

// ---------------------------------------------------------------- matrix products without rocBLAS

#define TILE 16

// c[b] = a[b] x b[b], row-major [m, k] x [k, n]; strides between matrices, 0 repeats one.
// Blocks of TILE x TILE threads (launched as GROUP = 256 in one dimension), one output each.
template <class D> __device__ void matmul_k(const typename D::T* a, const typename D::T* b, typename D::T* c, int m,
        int k, int n, i64 sa, i64 sb, i64 sc) {
    typedef typename D::A A;
    __shared__ A ta[TILE][TILE];
    __shared__ A tb[TILE][TILE + 1];
    int tx = threadIdx.x % TILE;
    int ty = threadIdx.x / TILE;
    const typename D::T* pa = a + blockIdx.z * sa;
    const typename D::T* pb = b + blockIdx.z * sb;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    A acc = A(0);
    for (int t = 0; t < k; t += TILE) {
        int ac = t + tx;
        int br = t + ty;
        ta[ty][tx] = row < m && ac < k ? D::ld(pa, (i64) row * k + ac) : A(0);
        tb[ty][tx] = br < k && col < n ? D::ld(pb, (i64) br * n + col) : A(0);
        __syncthreads();
        for (int i = 0; i < TILE; i++) {
            acc += ta[ty][i] * tb[i][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        D::st(c + blockIdx.z * sc, (i64) row * n + col, acc);
    }
}

// ---------------------------------------------------------------- products of few rows

// The weights of a few-row product: a dense row-major [k, n] matrix
template <class D> struct DenseWeights {
    const typename D::T* p;
    i64 n;
    __device__ __forceinline__ typename D::A load(i64 r, i64 c) const { return D::ld(p, r * n + c); }
};

#define FEW_ROWS 16
#define FEW_TILE 64

// a partial sum another block wrote, read past this compute unit's cache
__device__ __forceinline__ float load_part(const float* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// out[m, n] = a[m, k] x w[k, n] for m <= FEW_ROWS: a thread per column, each reading its column
// of w once for every row of a. Blocks over columns (x) and pieces of k (y); with several pieces
// each writes float partial sums to part[piece, m, n], added up in order by the last block of
// its column block to finish (counted in counters[blockIdx.x], left at 0 again), or by few_sum
// when counters is null.
template <class D, class W> __device__ void few_k(const typename D::T* a, W w, float* part, typename D::T* out,
        int m, int k, int n, int chunk, int* counters) {
    typedef typename D::A A;
    __shared__ A as[FEW_ROWS][FEW_TILE];
    i64 c = (i64) blockIdx.x * GROUP + threadIdx.x;
    int k0 = blockIdx.y * chunk;
    int k1 = min(k, k0 + chunk);
    A acc[FEW_ROWS];
#pragma unroll
    for (int r = 0; r < FEW_ROWS; r++) {
        acc[r] = A(0);
    }
    for (int t = k0; t < k1; t += FEW_TILE) {
        int len = min(FEW_TILE, k1 - t);
        for (int i = threadIdx.x; i < FEW_ROWS * FEW_TILE; i += GROUP) {
            int r = i / FEW_TILE;
            int kk = i % FEW_TILE;
            as[r][kk] = r < m && kk < len ? D::ld(a, (i64) r * k + t + kk) : A(0);
        }
        __syncthreads();
        if (c < n) {
            for (int kk = 0; kk < len; kk++) {
                A wv = (A) w.load(t + kk, c);
#pragma unroll
                for (int r = 0; r < FEW_ROWS; r++) {
                    acc[r] += as[r][kk] * wv;
                }
            }
        }
        __syncthreads();
    }
    if (c < n) {
#pragma unroll
        for (int r = 0; r < FEW_ROWS; r++) {
            if (r >= m) {
                break;
            }
            if (gridDim.y == 1) {
                D::st(out, (i64) r * n + c, acc[r]);
            } else {
                part[((i64) blockIdx.y * m + r) * n + c] = (float) acc[r];
            }
        }
    }
    if (gridDim.y > 1 && counters != nullptr) {
        __shared__ int last;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            last = atomicAdd(&counters[blockIdx.x], 1) == (int) gridDim.y - 1;
        }
        __syncthreads();
        if (last) {
            __threadfence();
            if (c < n) {
                for (int r = 0; r < m; r++) {
                    A sum = A(0);
                    for (int q = 0; q < (int) gridDim.y; q++) {
                        sum += (A) load_part(part + ((i64) q * m + r) * n + c);
                    }
                    D::st(out, (i64) r * n + c, sum);
                }
            }
            if (threadIdx.x == 0) {
                counters[blockIdx.x] = 0;
            }
        }
    }
}

// out[m, n] = a[m, k] x e[n, k]^T for m <= FEW_ROWS, e row-major with rows ld apart (a tied
// output head while decoding, read in place): a warp per row of e, which it reads once and
// contiguously, 4 values per lane at a time when `vec` (16-bit formats, rows 8-byte aligned).
template <class D> __device__ void few_t_k(const typename D::T* a, const typename D::T* e, typename D::T* out,
        int m, int k, int n, i64 ld, int vec) {
    typedef typename D::A A;
    typedef typename D::T T;
    int lane = threadIdx.x % 32;
    i64 j = (i64) blockIdx.x * (GROUP / 32) + threadIdx.x / 32;
    if (j >= n) {
        return;
    }
    const T* row = e + j * ld;
    A acc[FEW_ROWS];
#pragma unroll
    for (int r = 0; r < FEW_ROWS; r++) {
        acc[r] = A(0);
    }
    int done = 0;
    if (vec) {
        int k4 = k & ~3;
        for (int i = lane * 4; i < k4; i += 128) {
            uint2 q = *(const uint2*) (row + i);
            const T* p = (const T*) &q;
            A w0 = D::ld(p, 0);
            A w1 = D::ld(p, 1);
            A w2 = D::ld(p, 2);
            A w3 = D::ld(p, 3);
#pragma unroll
            for (int r = 0; r < FEW_ROWS; r++) {
                if (r < m) {
                    const T* x = a + (i64) r * k + i;
                    acc[r] += w0 * D::ld(x, 0) + w1 * D::ld(x, 1) + w2 * D::ld(x, 2) + w3 * D::ld(x, 3);
                }
            }
        }
        done = k4;
    }
    for (int i = done + lane; i < k; i += 32) {
        A w = D::ld(row, i);
#pragma unroll
        for (int r = 0; r < FEW_ROWS; r++) {
            if (r < m) {
                acc[r] += w * D::ld(a, (i64) r * k + i);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < FEW_ROWS; r++) {
        if (r < m) {
            A v = acc[r];
            for (int o = 16; o > 0; o >>= 1) {
                v += SHFL_XOR(v, o);
            }
            if (lane == 0) {
                D::st(out, (i64) r * n + j, v);
            }
        }
    }
}

template <class D> __device__ void few_sum_k(const float* part, typename D::T* out, i64 mn, int pieces) {
    typedef typename D::A A;
    FOR_EACH(i, mn) {
        A acc = A(0);
        for (int p = 0; p < pieces; p++) {
            acc += (A) part[(i64) p * mn + i];
        }
        D::st(out, i, acc);
    }
}

// ---------------------------------------------------------------- packed weights

// A packed matrix as HipBackend.Packed lays it out (as the Vulkan backend's does). q holds one
// signed byte per value, row-major, or with flag 1 two per byte (rows 2i and 2i + 1 of a column
// in the low and high half of byte i * n + c). With plane bits (5- and 6-bit levels) q holds the
// low four bits of level + 2^(bits-1) as nibbles and hi the rest, 8 / planeBits rows of a column
// per byte at (r / (8 / planeBits)) * n + c. With flag 32 q holds 4-bit indices (as nibbles)
// into a table of 16 signed bytes in hi. The scale of group g in column c is its signed byte
// code (flag 4; else 1) times supers[(g / perSuper) * n + c], float32 or with flag 16 float16;
// the offset (flag 2) likewise from its codes (flag 8) and supers, plus the scale times shift.
// Flags bits 8-11: the plane bits; bits 12-15: the bits per level.
struct PackedLayout {
    const unsigned* q;
    const unsigned* hi;
    const unsigned* scaleCodes;
    const unsigned* offsetCodes;
    const unsigned* scales;
    const unsigned* offsets;
    int n;
    int group;
    int flags;
    int perSuper;
    float shift;
};

// b bits of w from bit s, unsigned and sign-extended
__device__ __forceinline__ unsigned ubits(unsigned w, int s, int b) { return (w >> s) & ((1u << b) - 1u); }
__device__ __forceinline__ int sbits(unsigned w, int s, int b) { return ((int) (w << (32 - s - b))) >> (32 - b); }

__device__ __forceinline__ int pk_table(const PackedLayout& p, int i) { return sbits(p.hi[i >> 2], (i & 3) * 8, 8); }

// the level at row r, column c
__device__ int pk_level(const PackedLayout& p, i64 r, i64 c) {
    int sh = (int) (r & 1) * 4;
    if (p.flags & 32) {
        i64 b = (r >> 1) * p.n + c;
        return pk_table(p, (int) ubits(p.q[b >> 2], (int) (b & 3) * 8 + sh, 4));
    }
    int planeBits = (p.flags >> 8) & 15;
    if (planeBits != 0) {
        i64 lb = (r >> 1) * p.n + c;
        unsigned low = ubits(p.q[lb >> 2], (int) (lb & 3) * 8 + sh, 4);
        int perByte = 8 / planeBits;
        i64 hb = (r / perByte) * p.n + c;
        unsigned high = ubits(p.hi[hb >> 2], (int) (hb & 3) * 8 + (int) (r % perByte) * planeBits, planeBits);
        return (int) (low | (high << 4)) - (1 << (((p.flags >> 12) & 15) - 1));
    }
    if (p.flags & 1) {
        i64 b = (r >> 1) * p.n + c;
        return sbits(p.q[b >> 2], (int) (b & 3) * 8 + sh, 4);
    }
    i64 b = r * p.n + c;
    return sbits(p.q[b >> 2], (int) (b & 3) * 8, 8);
}

__device__ __forceinline__ float pk_super(const PackedLayout& p, const unsigned* w, i64 i) {
    if (p.flags & 16) {
        return h2f((u16) (w[i >> 1] >> ((i & 1) * 16)));
    }
    return __uint_as_float(w[i]);
}

__device__ __forceinline__ float pk_code(const unsigned* w, i64 i) {
    return (float) sbits(w[i >> 2], (int) (i & 3) * 8, 8);
}

// the scale of group g in column c
__device__ float pk_scale(const PackedLayout& p, i64 g, i64 c) {
    float s = pk_super(p, p.scales, (g / p.perSuper) * p.n + c);
    return (p.flags & 4) ? pk_code(p.scaleCodes, g * p.n + c) * s : s;
}

// the offset of group g in column c, whose scale is s
__device__ float pk_offset(const PackedLayout& p, i64 g, i64 c, float s) {
    float o = pk_super(p, p.offsets, (g / p.perSuper) * p.n + c);
    o = (p.flags & 8) ? pk_code(p.offsetCodes, g * p.n + c) * o : o;
    return o + s * p.shift;
}

template <class D> __device__ void unpack_packed_k(PackedLayout p, typename D::T* out, i64 rows) {
    typedef typename D::A A;
    FOR_EACH(i, rows * p.n) {
        i64 r = i / p.n;
        i64 c = i % p.n;
        i64 g = r / p.group;
        float s = pk_scale(p, g, c);
        float v = (float) pk_level(p, r, c) * s;
        if (p.flags & 2) {
            v += pk_offset(p, g, c, s);
        }
        D::st(out, i, (A) v);
    }
}

// the scales (or offsets, from supers and codes) of group g in columns c .. c + 3, n a multiple of 4
__device__ __forceinline__ void pk_factors4(const PackedLayout& p, const unsigned* supers, const unsigned* codes,
        bool hasCodes, i64 g, int c, float* v) {
    i64 i = (g / p.perSuper) * p.n + c;
    if (p.flags & 16) {
        unsigned w0 = supers[i >> 1];
        unsigned w1 = supers[(i >> 1) + 1];
        v[0] = h2f((u16) w0);
        v[1] = h2f((u16) (w0 >> 16));
        v[2] = h2f((u16) w1);
        v[3] = h2f((u16) (w1 >> 16));
    } else {
        for (int j = 0; j < 4; j++) {
            v[j] = __uint_as_float(supers[i + j]);
        }
    }
    if (hasCodes) {
        unsigned w = codes[(g * p.n + c) >> 2];
        for (int j = 0; j < 4; j++) {
            v[j] *= (float) sbits(w, 8 * j, 8);
        }
    }
}

// the four columns' levels of one row: nibbles at ls of each byte of lw, pb high bits at hs of each byte of hw
#define PK_PLANES(lw, ls, hw, hs, pb) \
    (int) (ubits(lw, ls, 4) | (ubits(hw, hs, pb) << 4)) - bias, \
    (int) (ubits(lw, 8 + (ls), 4) | (ubits(hw, 8 + (hs), pb) << 4)) - bias, \
    (int) (ubits(lw, 16 + (ls), 4) | (ubits(hw, 16 + (hs), pb) << 4)) - bias, \
    (int) (ubits(lw, 24 + (ls), 4) | (ubits(hw, 24 + (hs), pb) << 4)) - bias
// the four signed fields of b bits at s, s + 8, s + 16 and s + 24 of w
#define PK_FIELDS(w, s, b) sbits(w, s, b), sbits(w, 8 + (s), b), sbits(w, 16 + (s), b), sbits(w, 24 + (s), b)
#define PK_TABLE(w, s) table[ubits(w, s, 4)], table[ubits(w, 8 + (s), 4)], table[ubits(w, 16 + (s), 4)], \
    table[ubits(w, 24 + (s), 4)]

// unpack_packed for n a multiple of 4: a thread per word of levels, four columns of one row
// (8-bit levels) or of two rows (the 4-bit layouts, whose groups have an even number of rows)
template <class D> __device__ void unpack_words_k(PackedLayout p, typename D::T* out, int rows) {
    typedef typename D::A A;
    __shared__ int table[16];
    int planeBits = (p.flags >> 8) & 15;
    int mode = (p.flags & 32) ? 3 : planeBits != 0 ? 2 : (p.flags & 1) ? 1 : 0;
    int bias = 1 << (((p.flags >> 12) & 15) - 1);
    if (mode == 3 && threadIdx.x < 16) {
        table[threadIdx.x] = pk_table(p, threadIdx.x);
    }
    __syncthreads();
    int n = p.n;
    int quads = n >> 2;
    i64 units = (i64) (mode == 0 ? rows : rows / 2) * quads;
    for (i64 u = (i64) blockIdx.x * blockDim.x + threadIdx.x; u < units; u += (i64) gridDim.x * blockDim.x) {
        int unit = (int) (u / quads);
        int c = (int) (u % quads) * 4;
        int r = mode == 0 ? unit : unit * 2;
        unsigned w = p.q[(i64) unit * quads + (c >> 2)];
        int l0[4];
        int l1[4] = {0, 0, 0, 0};
        if (mode == 0) {
            const int v[4] = {PK_FIELDS(w, 0, 8)};
            for (int j = 0; j < 4; j++) l0[j] = v[j];
        } else if (mode == 1) {
            const int v[4] = {PK_FIELDS(w, 0, 4)};
            const int x[4] = {PK_FIELDS(w, 4, 4)};
            for (int j = 0; j < 4; j++) { l0[j] = v[j]; l1[j] = x[j]; }
        } else if (mode == 3) {
            const int v[4] = {PK_TABLE(w, 0)};
            const int x[4] = {PK_TABLE(w, 4)};
            for (int j = 0; j < 4; j++) { l0[j] = v[j]; l1[j] = x[j]; }
        } else {
            int perByte = 8 / planeBits;
            unsigned hw = p.hi[((i64) (r / perByte) * n + c) >> 2];
            int hs = (r % perByte) * planeBits;
            const int v[4] = {PK_PLANES(w, 0, hw, hs, planeBits)};
            const int x[4] = {PK_PLANES(w, 4, hw, hs + planeBits, planeBits)};
            for (int j = 0; j < 4; j++) { l0[j] = v[j]; l1[j] = x[j]; }
        }
        int g = r / p.group;
        float s[4];
        float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        pk_factors4(p, p.scales, p.scaleCodes, (p.flags & 4) != 0, g, c, s);
        if (p.flags & 2) {
            pk_factors4(p, p.offsets, p.offsetCodes, (p.flags & 8) != 0, g, c, o);
            for (int j = 0; j < 4; j++) {
                o[j] += s[j] * p.shift;
            }
        }
        i64 at = (i64) r * n + c;
        for (int j = 0; j < 4; j++) {
            D::st(out, at + j, (A) ((float) l0[j] * s[j] + o[j]));
        }
        if (mode != 0) {
            for (int j = 0; j < 4; j++) {
                D::st(out, at + n + j, (A) ((float) l1[j] * s[j] + o[j]));
            }
        }
    }
}

#define PK_QUADS 32
#define PK_SLICES 8

// adds row rr's levels of the four columns, times each row of a, to part (and a to sum)
#define PK_ACC(rr, ...) { \
    const int lv_[4] = {__VA_ARGS__}; \
    _Pragma("unroll") for (int i = 0; i < ROWS; i++) { \
        A x = D::ld(a, rowOf[i] + (rr)); \
        part[i][0] += x * (A) lv_[0]; part[i][1] += x * (A) lv_[1]; \
        part[i][2] += x * (A) lv_[2]; part[i][3] += x * (A) lv_[3]; \
        sum[i] += x; } }

// out[m, n] = a[m, k] x packed[k, n] for m <= ROWS (rows past m repeat row 0 and are not
// stored), as the Vulkan backend's matmul_packed. Each thread computes four adjacent columns,
// reading their levels of a row as one 32-bit word when n is a multiple of 4, so neighbouring
// lanes read neighbouring words. A block covers 4 * PK_QUADS columns with PK_SLICES threads
// (wavefronts) per column quad, each over its own run of rows; blockIdx.y picks a piece of
// `chunk` rows. With several pieces each writes float sums to part[piece, m, n], added up by
// the last block of its column block to finish (counted in counters[blockIdx.x], left at 0).
// Levels are summed per scale group and scaled once per group; with offsets each group also
// adds its offset times the sum of a over its rows.
template <class D, int ROWS> __device__ void packed_mv_k(const typename D::T* a, PackedLayout p, float* partial,
        typename D::T* out, int m, int k, int chunk, int* counters) {
    typedef typename D::A A;
    __shared__ A sums[PK_SLICES][PK_QUADS * 4];
    __shared__ int table[16];
    __shared__ int last;
    int quad = threadIdx.x % PK_QUADS;
    int slice = threadIdx.x / PK_QUADS;
    int n = p.n;
    int c = (blockIdx.x * PK_QUADS + quad) * 4;
    int piece = blockIdx.y;
    bool words = (n & 3) == 0;
    int from = piece * chunk;
    int to = min(k, from + chunk);
    int run = ((to - from + PK_SLICES - 1) / PK_SLICES + 1) & ~1;
    int start = min(to, from + slice * run);
    int end = min(to, start + run);
    i64 rowOf[ROWS];
    A total[ROWS][4];
#pragma unroll
    for (int i = 0; i < ROWS; i++) {
        rowOf[i] = (i64) (i < m ? i : 0) * k;
        total[i][0] = total[i][1] = total[i][2] = total[i][3] = A(0);
    }
    int planeBits = (p.flags >> 8) & 15;
    int mode = (p.flags & 32) ? 3 : planeBits != 0 ? 2 : (p.flags & 1) ? 1 : 0;
    int bias = 1 << (((p.flags >> 12) & 15) - 1);
    if (mode == 3 && threadIdx.x < 16) {
        table[threadIdx.x] = pk_table(p, threadIdx.x);
    }
    __syncthreads();
    if (c < n) {
        const unsigned* q = p.q;
        unsigned stride = (unsigned) n >> 2;   // words per row (per two rows with 4-bit levels)
        int r = start;
        while (r < end) {
            int g = r / p.group;
            int stop = min(end, (g + 1) * p.group);
            A part[ROWS][4];
            A sum[ROWS];
#pragma unroll
            for (int i = 0; i < ROWS; i++) {
                part[i][0] = part[i][1] = part[i][2] = part[i][3] = sum[i] = A(0);
            }
            // a loop per layout, each reading a word of levels once for the rows it holds (r
            // and stop are even in the 4-bit layouts: groups and runs have an even number of rows)
            if (words && mode == 1) {
                for (; r + 4 <= stop; r += 4) {
                    unsigned at = ((unsigned) (r >> 1) * n + c) >> 2;
                    unsigned w = q[at];
                    unsigned v = q[at + stride];
                    PK_ACC(r, PK_FIELDS(w, 0, 4));
                    PK_ACC(r + 1, PK_FIELDS(w, 4, 4));
                    PK_ACC(r + 2, PK_FIELDS(v, 0, 4));
                    PK_ACC(r + 3, PK_FIELDS(v, 4, 4));
                }
                for (; r < stop; r += 2) {
                    unsigned w = q[((unsigned) (r >> 1) * n + c) >> 2];
                    PK_ACC(r, PK_FIELDS(w, 0, 4));
                    PK_ACC(r + 1, PK_FIELDS(w, 4, 4));
                }
            } else if (words && mode == 2) {
                int perByte = 8 / planeBits;
                if (planeBits == 2) {
                    // 6-bit levels: a word of high bits holds four rows; to a multiple of four
                    // rows, then four at a time from two words of nibbles and one of high bits
                    for (; r < stop && (r & 3) != 0; r += 2) {
                        unsigned lw = q[((unsigned) (r >> 1) * n + c) >> 2];
                        unsigned hw = p.hi[((unsigned) (r >> 2) * n + c) >> 2];
                        PK_ACC(r, PK_PLANES(lw, 0, hw, 4, 2));
                        PK_ACC(r + 1, PK_PLANES(lw, 4, hw, 6, 2));
                    }
                    for (; r + 4 <= stop; r += 4) {
                        unsigned at = ((unsigned) (r >> 1) * n + c) >> 2;
                        unsigned lw = q[at];
                        unsigned lv = q[at + stride];
                        unsigned hw = p.hi[((unsigned) (r >> 2) * n + c) >> 2];
                        PK_ACC(r, PK_PLANES(lw, 0, hw, 0, 2));
                        PK_ACC(r + 1, PK_PLANES(lw, 4, hw, 2, 2));
                        PK_ACC(r + 2, PK_PLANES(lv, 0, hw, 4, 2));
                        PK_ACC(r + 3, PK_PLANES(lv, 4, hw, 6, 2));
                    }
                }
                for (; r < stop; r += 2) {
                    unsigned lw = q[((unsigned) (r >> 1) * n + c) >> 2];
                    unsigned hw = p.hi[((unsigned) (r / perByte) * n + c) >> 2];
                    int hs = (r % perByte) * planeBits;
                    PK_ACC(r, PK_PLANES(lw, 0, hw, hs, planeBits));
                    PK_ACC(r + 1, PK_PLANES(lw, 4, hw, hs + planeBits, planeBits));
                }
            } else if (words && mode == 3) {
                for (; r < stop; r += 2) {
                    unsigned w = q[((unsigned) (r >> 1) * n + c) >> 2];
                    PK_ACC(r, PK_TABLE(w, 0));
                    PK_ACC(r + 1, PK_TABLE(w, 4));
                }
            } else if (words) {
                for (; r + 4 <= stop; r += 4) {
                    unsigned at = ((unsigned) r * n + c) >> 2;
                    unsigned w0 = q[at];
                    unsigned w1 = q[at + stride];
                    unsigned w2 = q[at + 2 * stride];
                    unsigned w3 = q[at + 3 * stride];
                    PK_ACC(r, PK_FIELDS(w0, 0, 8));
                    PK_ACC(r + 1, PK_FIELDS(w1, 0, 8));
                    PK_ACC(r + 2, PK_FIELDS(w2, 0, 8));
                    PK_ACC(r + 3, PK_FIELDS(w3, 0, 8));
                }
                for (; r < stop; r++) {
                    unsigned w = q[((unsigned) r * n + c) >> 2];
                    PK_ACC(r, PK_FIELDS(w, 0, 8));
                }
            } else {
                for (; r < stop; r++) {
                    PK_ACC(r, pk_level(p, r, c), c + 1 < n ? pk_level(p, r, c + 1) : 0,
                            c + 2 < n ? pk_level(p, r, c + 2) : 0, c + 3 < n ? pk_level(p, r, c + 3) : 0);
                }
            }
            float s[4];
            float o[4];
            if (words) {
                pk_factors4(p, p.scales, p.scaleCodes, (p.flags & 4) != 0, g, c, s);
            } else {
                for (int j = 0; j < 4; j++) {
                    s[j] = c + j < n ? pk_scale(p, g, c + j) : 0.0f;
                }
            }
            if (p.flags & 2) {
                if (words) {
                    pk_factors4(p, p.offsets, p.offsetCodes, (p.flags & 8) != 0, g, c, o);
                    for (int j = 0; j < 4; j++) {
                        o[j] += s[j] * p.shift;
                    }
                } else {
                    for (int j = 0; j < 4; j++) {
                        o[j] = c + j < n ? pk_offset(p, g, c + j, s[j]) : 0.0f;
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < 4; j++) {
                    total[i][j] += part[i][j] * (A) s[j];
                    if (p.flags & 2) {
                        total[i][j] += sum[i] * (A) o[j];
                    }
                }
            }
        }
    }
    bool single = gridDim.y == 1;
#pragma unroll
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < 4; j++) {
            sums[slice][quad * 4 + j] = total[i][j];
        }
        __syncthreads();
        // the block's 128 sums of row i, one per thread of the first half
        if (threadIdx.x < PK_QUADS * 4 && i < m) {
            int col = blockIdx.x * PK_QUADS * 4 + threadIdx.x;
            if (col < n) {
                A t = sums[0][threadIdx.x];
                for (int s = 1; s < PK_SLICES; s++) {
                    t += sums[s][threadIdx.x];
                }
                if (single) {
                    D::st(out, (i64) i * n + col, t);
                } else {
                    partial[((i64) piece * m + i) * n + col] = (float) t;
                }
            }
        }
        __syncthreads();
    }
    if (single) {
        return;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(&counters[blockIdx.x], 1) == (int) gridDim.y - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    int col = blockIdx.x * PK_QUADS * 4 + (threadIdx.x % (PK_QUADS * 4));
    if (col < n) {
        // two rows at a time: the block has twice as many threads as columns
        for (int i = threadIdx.x / (PK_QUADS * 4); i < m; i += 2) {
            A t = A(0);
            for (int q = 0; q < (int) gridDim.y; q++) {
                t += (A) load_part(partial + ((i64) q * m + i) * n + col);
            }
            D::st(out, (i64) i * n + col, t);
        }
    }
    if (threadIdx.x == 0) {
        counters[blockIdx.x] = 0;   // ready for the next product
    }
}

// ---------------------------------------------------------------- attention

// q [batch, heads, tq, dk], k [batch, kvHeads, tk, dk], v [batch, kvHeads, tk, dv];
// query head h reads key/value head h / (heads / kvHeads); causal: query i sees keys
// j <= i + tk - tq. A warp per query row (and piece of its keys), lane l holding dimensions
// l, l + 32, ... of q and of the output, so heads up to 32 * ATT_LANES.
#define ATT_LANES 8
#define ATT_KEYS 4

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        v += SHFL_XOR(v, o);
    }
    return v;
}

__device__ __forceinline__ float neg_inf() { return -__int_as_float(0x7f800000); }

struct AttShape {
    int heads;
    int kvHeads;
    int tq;
    int tk;
    int dk;
    int dv;
    float scale;
    int causal;
};

// the last key query i sees, plus one
__device__ __forceinline__ int key_end(AttShape s, int i) {
    return s.causal ? max(0, min(s.tk, i + s.tk - s.tq + 1)) : s.tk;
}

// one row's pieces (max, sum, acc[dv] each) rescaled to a common maximum and added, by a warp;
// read past this compute unit's cache, as other blocks may have written them
template <class D> __device__ void attention_row(const float* pr, typename D::T* out, typename D::T* lse, i64 r,
        int parts, int dv, int lane) {
    float m = neg_inf();
    for (int p = 0; p < parts; p++) {
        if (load_part(pr + p * (dv + 2) + 1) > 0) {
            m = fmaxf(m, load_part(pr + p * (dv + 2)));
        }
    }
    float l = 0.0f;
    for (int p = 0; p < parts; p++) {
        float lp = load_part(pr + p * (dv + 2) + 1);
        if (lp > 0) {
            l += lp * expf(load_part(pr + p * (dv + 2)) - m);
        }
    }
    for (int d = lane; d < dv; d += 32) {
        float acc = 0.0f;
        for (int p = 0; p < parts; p++) {
            const float* pp = pr + p * (dv + 2);
            float lp = load_part(pp + 1);
            if (lp > 0) {
                acc += load_part(pp + 2 + d) * expf(load_part(pp) - m);
            }
        }
        D::st(out, r * dv + d, l > 0 ? acc / l : 0.0f);
    }
    if (lane == 0) {
        D::st(lse, r, l > 0 ? m + logf(l) : neg_inf());
    }
}

// rows * parts warps; parts == 1 writes out and lse, otherwise partial (max, sum, acc[dv]) per
// piece, combined by the row's last warp to finish (counted in counters[r], left at 0 again),
// or by attention_combine when counters is null
template <class D> __device__ void attention_k(const typename D::T* q, const typename D::T* k,
        const typename D::T* v, typename D::T* out, typename D::T* lse, const typename D::T* bias, float* part,
        AttShape s, i64 rows, int hasBias, i64 b0, i64 b1, i64 b2, i64 b3, int parts, int chunk, int* counters) {
    int lane = threadIdx.x & 31;
    i64 w = ((i64) blockIdx.x * GROUP + threadIdx.x) >> 5;
    if (w >= rows * parts) {
        return;
    }
    i64 r = w / parts;
    int p = (int) (w % parts);
    int i = (int) (r % s.tq);
    i64 bh = r / s.tq;
    int h = (int) (bh % s.heads);
    i64 b = bh / s.heads;
    i64 kvRow = (b * s.kvHeads + h / (s.heads / s.kvHeads)) * s.tk;
    float qv[ATT_LANES];
    float acc[ATT_LANES];
#pragma unroll
    for (int t = 0; t < ATT_LANES; t++) {
        int d = lane + 32 * t;
        qv[t] = d < s.dk ? (float) D::ld(q, r * s.dk + d) : 0.0f;
        acc[t] = 0.0f;
    }
    float m = neg_inf();
    float l = 0.0f;
    int end = min(key_end(s, i), (p + 1) * chunk);
    // ATT_KEYS keys at a time: their rows loaded and their dot products summed together, so
    // the loads and shuffles of one key overlap the others' (one at a time, a warp mostly waited)
    for (int j0 = p * chunk; j0 < end; j0 += ATT_KEYS) {
        float score[ATT_KEYS];
#pragma unroll
        for (int u = 0; u < ATT_KEYS; u++) {
            int j = min(j0 + u, end - 1);
            const typename D::T* kr = k + (kvRow + j) * s.dk;
            float dot = 0.0f;
#pragma unroll
            for (int t = 0; t < ATT_LANES; t++) {
                int d = lane + 32 * t;
                if (d < s.dk) {
                    dot += qv[t] * (float) D::ld(kr, d);
                }
            }
            score[u] = dot;
        }
        float mNew = m;
#pragma unroll
        for (int u = 0; u < ATT_KEYS; u++) {
            int j = j0 + u;
            float sc = warp_sum(score[u]) * s.scale;
            if (j >= end) {
                sc = neg_inf();
            } else if (hasBias) {
                sc += (float) D::ld(bias, b * b0 + h * b1 + i * b2 + j * b3);
            }
            score[u] = sc;
            mNew = fmaxf(mNew, sc);
        }
        if (mNew == neg_inf()) {
            continue;   // every key so far masked out
        }
        float correction = expf(m - mNew);
        float weight[ATT_KEYS];
        l *= correction;
#pragma unroll
        for (int u = 0; u < ATT_KEYS; u++) {
            weight[u] = score[u] == neg_inf() ? 0.0f : expf(score[u] - mNew);
            l += weight[u];
        }
#pragma unroll
        for (int t = 0; t < ATT_LANES; t++) {
            int d = lane + 32 * t;
            if (d < s.dv) {
                float a = acc[t] * correction;
#pragma unroll
                for (int u = 0; u < ATT_KEYS; u++) {
                    int j = min(j0 + u, end - 1);
                    a += weight[u] * (float) D::ld(v + (kvRow + j) * s.dv, d);
                }
                acc[t] = a;
            }
        }
        m = mNew;
    }
    if (parts == 1) {
#pragma unroll
        for (int t = 0; t < ATT_LANES; t++) {
            int d = lane + 32 * t;
            if (d < s.dv) {
                D::st(out, r * s.dv + d, l > 0 ? acc[t] / l : 0.0f);
            }
        }
        if (lane == 0) {
            D::st(lse, r, l > 0 ? m + logf(l) : neg_inf());
        }
    } else {
        float* pp = part + w * (s.dv + 2);
#pragma unroll
        for (int t = 0; t < ATT_LANES; t++) {
            int d = lane + 32 * t;
            if (d < s.dv) {
                pp[2 + d] = acc[t];
            }
        }
        if (lane == 0) {
            pp[0] = m;
            pp[1] = l;
        }
        if (counters != nullptr) {
            __threadfence();
            int last = 0;
            if (lane == 0) {
                last = atomicAdd(&counters[r], 1) == parts - 1;
            }
            last = __shfl(last, 0, 32);
            if (last) {
                __threadfence();
                attention_row<D>(part + r * parts * (s.dv + 2), out, lse, r, parts, s.dv, lane);
                if (lane == 0) {
                    counters[r] = 0;
                }
            }
        }
    }
}

// a warp per row: the pieces' partial results rescaled to a common maximum and added
template <class D> __device__ void attention_combine_k(const float* part, typename D::T* out, typename D::T* lse,
        i64 rows, int parts, int dv) {
    int lane = threadIdx.x & 31;
    i64 r = ((i64) blockIdx.x * GROUP + threadIdx.x) >> 5;
    if (r >= rows) {
        return;
    }
    attention_row<D>(part + r * parts * (dv + 2), out, lse, r, parts, dv, lane);
}

// delta[r] = grad[r] . out[r], a warp per row
template <class D> __device__ void attention_delta_k(const typename D::T* out, const typename D::T* grad,
        float* delta, i64 rows, int dv) {
    int lane = threadIdx.x & 31;
    i64 r = ((i64) blockIdx.x * GROUP + threadIdx.x) >> 5;
    if (r >= rows) {
        return;
    }
    float acc = 0.0f;
    for (int d = lane; d < dv; d += 32) {
        acc += (float) D::ld(out, r * dv + d) * (float) D::ld(grad, r * dv + d);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        delta[r] = acc;
    }
}

// dq[i] = scale * sum_j p_ij (grad_i . v_j - delta_i) k_j, a warp per query row
template <class D> __device__ void attention_dq_k(const typename D::T* q, const typename D::T* k,
        const typename D::T* v, const typename D::T* lse, const typename D::T* grad, const float* delta,
        typename D::T* dq, AttShape s, i64 rows) {
    int lane = threadIdx.x & 31;
    i64 r = ((i64) blockIdx.x * GROUP + threadIdx.x) >> 5;
    if (r >= rows) {
        return;
    }
    int i = (int) (r % s.tq);
    i64 bh = r / s.tq;
    int h = (int) (bh % s.heads);
    i64 b = bh / s.heads;
    i64 kvRow = (b * s.kvHeads + h / (s.heads / s.kvHeads)) * s.tk;
    float qv[ATT_LANES];
    float gv[ATT_LANES];
    float acc[ATT_LANES];
#pragma unroll
    for (int t = 0; t < ATT_LANES; t++) {
        int d = lane + 32 * t;
        qv[t] = d < s.dk ? (float) D::ld(q, r * s.dk + d) : 0.0f;
        gv[t] = d < s.dv ? (float) D::ld(grad, r * s.dv + d) : 0.0f;
        acc[t] = 0.0f;
    }
    float l = (float) D::ld(lse, r);
    float dl = delta[r];
    int end = key_end(s, i);
    for (int j = 0; j < end; j++) {
        const typename D::T* kr = k + (kvRow + j) * s.dk;
        const typename D::T* vr = v + (kvRow + j) * s.dv;
        float dot = 0.0f;
        float dp = 0.0f;
#pragma unroll
        for (int t = 0; t < ATT_LANES; t++) {
            int d = lane + 32 * t;
            if (d < s.dk) {
                dot += qv[t] * (float) D::ld(kr, d);
            }
            if (d < s.dv) {
                dp += gv[t] * (float) D::ld(vr, d);
            }
        }
        float pw = expf(warp_sum(dot) * s.scale - l);
        float ds = pw * (warp_sum(dp) - dl);
#pragma unroll
        for (int t = 0; t < ATT_LANES; t++) {
            int d = lane + 32 * t;
            if (d < s.dk) {
                acc[t] += ds * (float) D::ld(kr, d);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < ATT_LANES; t++) {
        int d = lane + 32 * t;
        if (d < s.dk) {
            D::st(dq, r * s.dk + d, acc[t] * s.scale);
        }
    }
}

// dk[j], dv[j] over every query head of the group and every query that sees key j, a warp per key row
template <class D> __device__ void attention_dkv_k(const typename D::T* q, const typename D::T* k,
        const typename D::T* v, const typename D::T* lse, const typename D::T* grad, const float* delta,
        typename D::T* dk, typename D::T* dvOut, AttShape s, i64 keyRows) {
    int lane = threadIdx.x & 31;
    i64 kr = ((i64) blockIdx.x * GROUP + threadIdx.x) >> 5;
    if (kr >= keyRows) {
        return;
    }
    int j = (int) (kr % s.tk);
    i64 bkv = kr / s.tk;
    int kvh = (int) (bkv % s.kvHeads);
    i64 b = bkv / s.kvHeads;
    int group = s.heads / s.kvHeads;
    float kv[ATT_LANES];
    float vv[ATT_LANES];
    float accK[ATT_LANES];
    float accV[ATT_LANES];
#pragma unroll
    for (int t = 0; t < ATT_LANES; t++) {
        int d = lane + 32 * t;
        kv[t] = d < s.dk ? (float) D::ld(k, kr * s.dk + d) : 0.0f;
        vv[t] = d < s.dv ? (float) D::ld(v, kr * s.dv + d) : 0.0f;
        accK[t] = 0.0f;
        accV[t] = 0.0f;
    }
    // query i sees key j when j <= i + tk - tq
    int first = s.causal ? max(0, j - (s.tk - s.tq)) : 0;
    for (int h = kvh * group; h < (kvh + 1) * group; h++) {
        i64 rowBase = (b * s.heads + h) * s.tq;
        for (int i = first; i < s.tq; i++) {
            i64 r = rowBase + i;
            const typename D::T* qr = q + r * s.dk;
            const typename D::T* gr = grad + r * s.dv;
            float dot = 0.0f;
            float dp = 0.0f;
            float g[ATT_LANES];
            float qq[ATT_LANES];
#pragma unroll
            for (int t = 0; t < ATT_LANES; t++) {
                int d = lane + 32 * t;
                qq[t] = d < s.dk ? (float) D::ld(qr, d) : 0.0f;
                g[t] = d < s.dv ? (float) D::ld(gr, d) : 0.0f;
                dot += qq[t] * kv[t];
                dp += g[t] * vv[t];
            }
            float pw = expf(warp_sum(dot) * s.scale - (float) D::ld(lse, r));
            float ds = pw * (warp_sum(dp) - delta[r]);
#pragma unroll
            for (int t = 0; t < ATT_LANES; t++) {
                accV[t] += pw * g[t];
                accK[t] += ds * qq[t];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < ATT_LANES; t++) {
        int d = lane + 32 * t;
        if (d < s.dk) {
            D::st(dk, kr * s.dk + d, accK[t] * s.scale);
        }
        if (d < s.dv) {
            D::st(dvOut, kr * s.dv + d, accV[t]);
        }
    }
}

// ---------------------------------------------------------------- Fourier transforms

// A block per row of n complex values (a power of two up to 2048), radix 2 in shared memory,
// as Fft.radix2 computes it; the inverse uses the conjugate exponent and divides by n
#ifdef BRAINIAC_FFT
extern "C" __global__ void fft_rows(const float* in, float* out, i64 rows, int n, int logN, int inverse) {
    __shared__ float re[2048];
    __shared__ float im[2048];
    for (i64 r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* x = in + r * n * 2;
        for (int i = threadIdx.x; i < n; i += GROUP) {
            int j = (int) (__brev((unsigned) i) >> (32 - logN));
            re[j] = x[2 * i];
            im[j] = x[2 * i + 1];
        }
        __syncthreads();
        float sign = inverse ? 1.0f : -1.0f;
        for (int len = 2; len <= n; len <<= 1) {
            int half = len >> 1;
            for (int t = threadIdx.x; t < n / 2; t += GROUP) {
                int kk = t % half;
                int i = (t / half) * len + kk;
                int j = i + half;
                float wr;
                float wi;
                sincospif(sign * 2.0f * kk / len, &wi, &wr);
                float xr = re[j] * wr - im[j] * wi;
                float xi = re[j] * wi + im[j] * wr;
                re[j] = re[i] - xr;
                im[j] = im[i] - xi;
                re[i] += xr;
                im[i] += xi;
            }
            __syncthreads();
        }
        float scale = inverse ? 1.0f / n : 1.0f;
        float* y = out + r * n * 2;
        for (int i = threadIdx.x; i < n; i += GROUP) {
            y[2 * i] = re[i] * scale;
            y[2 * i + 1] = im[i] * scale;
        }
        __syncthreads();
    }
}
#endif

// ---------------------------------------------------------------- entry points, one per format

#define KERNELS(S, D)                                                                                             \
extern "C" __global__ void unary_##S(const D::T* x, D::T* y, i64 n, int op) { unary_k<D>(x, y, n, op); }        \
extern "C" __global__ void activation_##S(const D::T* x, D::T* y, i64 n, int op, float p) {                      \
    activation_k<D>(x, y, n, op, p);                                                                              \
}                                                                                                                 \
extern "C" __global__ void activation_backward_##S(const D::T* x, const D::T* g, D::T* y, i64 n, int op,         \
        float p) {                                                                                                \
    activation_backward_k<D>(x, g, y, n, op, p);                                                                  \
}                                                                                                                 \
extern "C" __global__ void binary_contiguous_##S(const D::T* a, const D::T* b, D::T* z, i64 n, int op) {         \
    binary_contiguous_k<D>(a, b, z, n, op);                                                                       \
}                                                                                                                 \
extern "C" __global__ void binary_##S(const D::T* a, const D::T* b, D::T* z, i64 n, int op, Strided d) {         \
    binary_k<D>(a, b, z, n, op, d);                                                                               \
}                                                                                                                 \
extern "C" __global__ void copy_##S(const D::T* x, D::T* z, i64 n, Strided d) { copy_k<D>(x, z, n, d); }         \
extern "C" __global__ void fill_##S(D::T* y, i64 n, double v) { fill_k<D>(y, n, v); }                            \
extern "C" __global__ void from_float_##S(const float* x, D::T* y, i64 n) { from_float_k<D>(x, y, n); }          \
extern "C" __global__ void to_float_##S(const D::T* x, float* y, i64 n) { to_float_k<D>(x, y, n); }              \
extern "C" __global__ void reduce_##S(const D::T* x, D::T* y, i64 rows, i64 len, i64 inner, int op) {            \
    reduce_k<D>(x, y, rows, len, inner, op);                                                                      \
}                                                                                                                 \
extern "C" __global__ void reduce_warp_##S(const D::T* x, D::T* y, i64 rows, i64 len, int op) {                  \
    reduce_warp_k<D>(x, y, rows, len, op);                                                                        \
}                                                                                                                 \
extern "C" __global__ void reduce_block_##S(const D::T* x, D::T* y, i64 rows, i64 len, i64 inner, int op) {      \
    reduce_block_k<D>(x, y, rows, len, inner, op);                                                                \
}                                                                                                                 \
extern "C" __global__ void argmax_##S(const D::T* x, D::T* y, i64 rows, i64 len, i64 inner) {                    \
    argmax_k<D>(x, y, rows, len, inner);                                                                          \
}                                                                                                                 \
extern "C" __global__ void argmax_split_##S(const D::T* x, D::T* y, double* part, int* counters, i64 rows,      \
        i64 len, i64 inner, int pieces) {                                                                         \
    argmax_split_k<D>(x, y, part, counters, rows, len, inner, pieces);                                            \
}                                                                                                                 \
extern "C" __global__ void argmax_block_##S(const D::T* x, D::T* y, i64 rows, i64 len, i64 inner) {              \
    argmax_block_k<D>(x, y, rows, len, inner);                                                                    \
}                                                                                                                 \
extern "C" __global__ void rms_norm_##S(const D::T* x, const D::T* w, D::T* y, i64 rows, int cols, float eps) {  \
    rms_norm_k<D>(x, w, y, rows, cols, eps);                                                                      \
}                                                                                                                 \
extern "C" __global__ void add_rms_norm_##S(const D::T* x, const D::T* r, const D::T* w, D::T* sum, D::T* y,     \
        i64 rows, int cols, float eps) {                                                                          \
    add_rms_norm_k<D>(x, r, w, sum, y, rows, cols, eps);                                                          \
}                                                                                                                 \
extern "C" __global__ void im2col_##S(const D::T* x, const int* table, D::T* z, int channels, int positions,     \
        int ks, i64 img, float pad, i64 total) {                                                                  \
    im2col_k<D>(x, table, z, channels, positions, ks, img, pad, total);                                           \
}                                                                                                                 \
extern "C" __global__ void col2im_##S(const D::T* cols, const int* offsets, const int* entries, D::T* z,         \
        int channels, int positions, int ks, i64 img, i64 total) {                                                \
    col2im_k<D>(cols, offsets, entries, z, channels, positions, ks, img, total);                                  \
}                                                                                                                 \
extern "C" __global__ void gather_##S(const D::T* x, const float* indices, D::T* z, i64 len, i64 inner,          \
        i64 count, i64 total) {                                                                                   \
    gather_k<D>(x, indices, z, len, inner, count, total);                                                         \
}                                                                                                                 \
extern "C" __global__ void scatter_add_##S(const D::T* src, const int* offsets, const int* entries, D::T* z,     \
        i64 len, i64 inner, i64 count, i64 total) {                                                               \
    scatter_add_k<D>(src, offsets, entries, z, len, inner, count, total);                                         \
}                                                                                                                 \
extern "C" __global__ void matmul_##S(const D::T* a, const D::T* b, D::T* c, int m, int k, int n, i64 sa,        \
        i64 sb, i64 sc) {                                                                                         \
    matmul_k<D>(a, b, c, m, k, n, sa, sb, sc);                                                                    \
}

// Compiled in parts, each at its first use, with -DBRAINIAC_<PART>: CORE (the kernels above
// in every format, and random_values), FEW_<format>, ATTENTION_<format> and FFT; all in one
// program, NVRTC took half a minute (in the CUDA backend).
#ifdef BRAINIAC_CORE
KERNELS(f32, F32)
KERNELS(f64, F64)
KERNELS(f16, F16)
KERNELS(bf16, BF16)
#endif

#define FEW_KERNELS(S, D)                                                                                         \
extern "C" __global__ void few_dense_##S(const D::T* a, const D::T* w, float* part, D::T* out, int m, int k,     \
        int n, int chunk, int* counters) {                                                                        \
    DenseWeights<D> dw = {w, n};                                                                                  \
    few_k<D>(a, dw, part, out, m, k, n, chunk, counters);                                                         \
}                                                                                                                 \
extern "C" __global__ void few_t_##S(const D::T* a, const D::T* e, D::T* out, int m, int k, int n, i64 ld,       \
        int vec) {                                                                                                \
    few_t_k<D>(a, e, out, m, k, n, ld, vec);                                                                      \
}                                                                                                                 \
extern "C" __global__ void few_sum_##S(const float* part, D::T* out, i64 mn, int pieces) {                        \
    few_sum_k<D>(part, out, mn, pieces);                                                                          \
}                                                                                                                 \
extern "C" __global__ void unpack_packed_##S(const unsigned* q, const unsigned* hi, const unsigned* sc, const unsigned* oc,                \
        const unsigned* s, const unsigned* o, int n, int group, int flags, int perSuper, float shift,                                \
        D::T* out, int rows) {                                                                                     \
    PackedLayout p = {q, hi, sc, oc, s, o, n, group, flags, perSuper, shift};                                                \
    unpack_packed_k<D>(p, out, rows);                                                                             \
}                                                                                                                 \
extern "C" __global__ void unpack_words_##S(const unsigned* q, const unsigned* hi, const unsigned* sc,             \
        const unsigned* oc, const unsigned* s, const unsigned* o, int n, int group, int flags, int perSuper,      \
        float shift, D::T* out, int rows) {                                                                       \
    PackedLayout p = {q, hi, sc, oc, s, o, n, group, flags, perSuper, shift};                                     \
    unpack_words_k<D>(p, out, rows);                                                                              \
}                                                                                                                 \
extern "C" __global__ void packed_mv1_##S(const D::T* a, const unsigned* q, const unsigned* hi, const unsigned* sc, const unsigned* oc,                \
        const unsigned* s, const unsigned* o, int n, int group, int flags, int perSuper, float shift,                    \
        float* part, D::T* out, int m, int k, int chunk, int* counters) {                                          \
    PackedLayout p = {q, hi, sc, oc, s, o, n, group, flags, perSuper, shift};                                                \
    packed_mv_k<D, 1>(a, p, part, out, m, k, chunk, counters);                                                 \
}                                                                                                                 \
extern "C" __global__ void packed_mv2_##S(const D::T* a, const unsigned* q, const unsigned* hi, const unsigned* sc, const unsigned* oc,                \
        const unsigned* s, const unsigned* o, int n, int group, int flags, int perSuper, float shift,                    \
        float* part, D::T* out, int m, int k, int chunk, int* counters) {                                          \
    PackedLayout p = {q, hi, sc, oc, s, o, n, group, flags, perSuper, shift};                                                \
    packed_mv_k<D, 2>(a, p, part, out, m, k, chunk, counters);                                                 \
}                                                                                                                 \
extern "C" __global__ void packed_mv4_##S(const D::T* a, const unsigned* q, const unsigned* hi, const unsigned* sc, const unsigned* oc,                \
        const unsigned* s, const unsigned* o, int n, int group, int flags, int perSuper, float shift,                    \
        float* part, D::T* out, int m, int k, int chunk, int* counters) {                                          \
    PackedLayout p = {q, hi, sc, oc, s, o, n, group, flags, perSuper, shift};                                                \
    packed_mv_k<D, 4>(a, p, part, out, m, k, chunk, counters);                                                 \
}                                                                                                                 \
extern "C" __global__ void packed_mv8_##S(const D::T* a, const unsigned* q, const unsigned* hi, const unsigned* sc, const unsigned* oc,                \
        const unsigned* s, const unsigned* o, int n, int group, int flags, int perSuper, float shift,                    \
        float* part, D::T* out, int m, int k, int chunk, int* counters) {                                          \
    PackedLayout p = {q, hi, sc, oc, s, o, n, group, flags, perSuper, shift};                                                \
    packed_mv_k<D, 8>(a, p, part, out, m, k, chunk, counters);                                                 \
}

#ifdef BRAINIAC_FEW_F32
FEW_KERNELS(f32, F32)
#endif
#ifdef BRAINIAC_FEW_F64
FEW_KERNELS(f64, F64)
#endif
#ifdef BRAINIAC_FEW_F16
FEW_KERNELS(f16, F16)
#endif
#ifdef BRAINIAC_FEW_BF16
FEW_KERNELS(bf16, BF16)
#endif

// Multi-head latent attention over cached latents (Backend.latentAttention, DeepSeek): every
// query head reads the batch's one head of latents c [batch, tk, dc], which are the keys (with
// the rotary keys r [batch, tk, dr] adding to the scores) and also the values, so the output
// stays in latent space: softmax(scale (qc c^T + qr r^T) + bias) c. A warp per query row, lane l
// holding latent components l, l + 32, ... (up to 32 * LAT_LANES) and rotary ones up to
// 32 * ATT_LANES.
#define LAT_LANES 16

template <class D> __device__ void attention_latent_k(const typename D::T* qc, const typename D::T* qr,
        const typename D::T* c, const typename D::T* r, const typename D::T* bias, typename D::T* out, int heads,
        int tq, int tk, int dc, int dr, float scale, i64 rows, int hasBias, i64 b0, i64 b1, i64 b2, i64 b3) {
    int lane = threadIdx.x & 31;
    i64 row = ((i64) blockIdx.x * GROUP + threadIdx.x) >> 5;
    if (row >= rows) {
        return;
    }
    int i = (int) (row % tq);
    int h = (int) ((row / tq) % heads);
    i64 b = row / ((i64) tq * heads);
    float qv[LAT_LANES];
    float acc[LAT_LANES];
    float rv[ATT_LANES];
#pragma unroll
    for (int t = 0; t < LAT_LANES; t++) {
        int d = lane + 32 * t;
        qv[t] = d < dc ? (float) D::ld(qc, row * dc + d) : 0.0f;
        acc[t] = 0.0f;
    }
#pragma unroll
    for (int t = 0; t < ATT_LANES; t++) {
        int d = lane + 32 * t;
        rv[t] = d < dr ? (float) D::ld(qr, row * dr + d) : 0.0f;
    }
    float m = neg_inf();
    float l = 0.0f;
    for (int j = 0; j < tk; j++) {
        const typename D::T* cr = c + (b * tk + j) * dc;
        const typename D::T* rr = r + (b * tk + j) * dr;
        float cv[LAT_LANES];
        float dot = 0.0f;
#pragma unroll
        for (int t = 0; t < LAT_LANES; t++) {
            int d = lane + 32 * t;
            cv[t] = d < dc ? (float) D::ld(cr, d) : 0.0f;
            dot += qv[t] * cv[t];
        }
#pragma unroll
        for (int t = 0; t < ATT_LANES; t++) {
            int d = lane + 32 * t;
            if (d < dr) {
                dot += rv[t] * (float) D::ld(rr, d);
            }
        }
        float score = warp_sum(dot) * scale;
        if (hasBias) {
            score += (float) D::ld(bias, b * b0 + h * b1 + i * b2 + j * b3);
        }
        if (score == neg_inf()) {
            continue;
        }
        float mNew = fmaxf(m, score);
        float correction = expf(m - mNew);
        float weight = expf(score - mNew);
        l = l * correction + weight;
#pragma unroll
        for (int t = 0; t < LAT_LANES; t++) {
            acc[t] = acc[t] * correction + weight * cv[t];
        }
        m = mNew;
    }
#pragma unroll
    for (int t = 0; t < LAT_LANES; t++) {
        int d = lane + 32 * t;
        if (d < dc) {
            D::st(out, row * dc + d, l > 0 ? acc[t] / l : 0.0f);
        }
    }
}

#define ATTENTION_KERNELS(S, D)                                                                                   \
extern "C" __global__ void attention_##S(const D::T* q, const D::T* k, const D::T* v, D::T* out, D::T* lse,       \
        const D::T* bias, float* part, AttShape s, i64 rows, int hasBias, i64 b0, i64 b1, i64 b2, i64 b3,         \
        int parts, int chunk, int* counters) {                                                                    \
    attention_k<D>(q, k, v, out, lse, bias, part, s, rows, hasBias, b0, b1, b2, b3, parts, chunk, counters);     \
}                                                                                                                 \
extern "C" __global__ void attention_combine_##S(const float* part, D::T* out, D::T* lse, i64 rows, int parts,    \
        int dv) {                                                                                                 \
    attention_combine_k<D>(part, out, lse, rows, parts, dv);                                                      \
}                                                                                                                 \
extern "C" __global__ void attention_delta_##S(const D::T* out, const D::T* grad, float* delta, i64 rows,         \
        int dv) {                                                                                                 \
    attention_delta_k<D>(out, grad, delta, rows, dv);                                                             \
}                                                                                                                 \
extern "C" __global__ void attention_dq_##S(const D::T* q, const D::T* k, const D::T* v, const D::T* lse,         \
        const D::T* grad, const float* delta, D::T* dq, AttShape s, i64 rows) {                                   \
    attention_dq_k<D>(q, k, v, lse, grad, delta, dq, s, rows);                                                    \
}                                                                                                                 \
extern "C" __global__ void attention_dkv_##S(const D::T* q, const D::T* k, const D::T* v, const D::T* lse,        \
        const D::T* grad, const float* delta, D::T* dk, D::T* dv, AttShape s, i64 keyRows) {                      \
    attention_dkv_k<D>(q, k, v, lse, grad, delta, dk, dv, s, keyRows);                                            \
}                                                                                                                 \
extern "C" __global__ void attention_latent_##S(const D::T* qc, const D::T* qr, const D::T* c, const D::T* r,     \
        const D::T* bias, D::T* out, int heads, int tq, int tk, int dc, int dr, float scale, i64 rows, int hasBias,\
        i64 b0, i64 b1, i64 b2, i64 b3) {                                                                         \
    attention_latent_k<D>(qc, qr, c, r, bias, out, heads, tq, tk, dc, dr, scale, rows, hasBias, b0, b1, b2, b3);  \
}

#ifdef BRAINIAC_ATTENTION_F32
ATTENTION_KERNELS(f32, F32)
#endif
#ifdef BRAINIAC_ATTENTION_F16
ATTENTION_KERNELS(f16, F16)
#endif
#ifdef BRAINIAC_ATTENTION_BF16
ATTENTION_KERNELS(bf16, BF16)
#endif

// uniform in [0, 1): CounterRandom.uniform(seed, i), float32 only
#ifdef BRAINIAC_CORE
extern "C" __global__ void random_values(float* y, i64 n, i64 seed) {
    FOR_EACH(i, n) {
        unsigned long long z = (unsigned long long) seed + ((unsigned long long) i + 1ull) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        y[i] = (float) (unsigned) (z >> 40) * (1.0f / 16777216.0f);
    }
}
#endif
