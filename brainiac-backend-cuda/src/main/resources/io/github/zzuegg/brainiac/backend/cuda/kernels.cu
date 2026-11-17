// Brainiac's CUDA kernels, compiled at run time by NVRTC for the device in use.
//
// Every kernel exists once per number format, with the suffix _f32, _f64, _f16 or _bf16.
// float16 and bfloat16 are stored as 16-bit integers and computed in float; float64 in double.
// The math follows the CPU backend (and the Vulkan backend's math.glsl): operation codes are
// the ordinals of the Java enums UnaryOp, ActivationOp, BinaryOp and ReduceOp.
//
// No headers: NVRTC provides the math functions; half and bfloat16 conversions are written out.

typedef unsigned short u16;
typedef long long i64;

#define GROUP 256
#define MAX_AXES 8

// ---------------------------------------------------------------- number formats

__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

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
        v = combine(op, v, __shfl_down_sync(0xffffffffu, v, o));
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

// ---------------------------------------------------------------- matrix products without cuBLAS

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
extern "C" __global__ void rms_norm_##S(const D::T* x, const D::T* w, D::T* y, i64 rows, int cols, float eps) {  \
    rms_norm_k<D>(x, w, y, rows, cols, eps);                                                                      \
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

KERNELS(f32, F32)
KERNELS(f64, F64)
KERNELS(f16, F16)
KERNELS(bf16, BF16)

// uniform in [0, 1): CounterRandom.uniform(seed, i), float32 only
extern "C" __global__ void random_values(float* y, i64 n, i64 seed) {
    FOR_EACH(i, n) {
        unsigned long long z = (unsigned long long) seed + ((unsigned long long) i + 1ull) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        y[i] = (float) (unsigned) (z >> 40) * (1.0f / 16777216.0f);
    }
}
