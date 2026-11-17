// Element-wise functions in ACC precision, matching the CPU backend. Operation codes are the
// ordinals of the Java enums UnaryOp, ActivationOp and BinaryOp. GLSL has no double exp or
// log, so float64 brings its own (accurate to a few units in the last place).

#if defined(TYPE_F64)
#define SQRT_2_OVER_PI 0.7978845608028654LF
#define INV_SQRT_2 0.7071067811865476LF
#define INV_SQRT_2PI 0.3989422804014327LF
#define GELU_CUBIC 0.044715LF
#define SELU_ALPHA 1.6732632423543772LF
#define SELU_SCALE 1.0507009873554805LF
#define ERF_P 0.3275911LF
#define ERF_A1 0.254829592LF
#define ERF_A2 -0.284496736LF
#define ERF_A3 1.421413741LF
#define ERF_A4 -1.453152027LF
#define ERF_A5 1.061405429LF
#else
#define SQRT_2_OVER_PI 0.7978845608028654
#define INV_SQRT_2 0.7071067811865476
#define INV_SQRT_2PI 0.3989422804014327
#define GELU_CUBIC 0.044715
#define SELU_ALPHA 1.6732632423543772
#define SELU_SCALE 1.0507009873554805
#define ERF_P 0.3275911
#define ERF_A1 0.254829592
#define ERF_A2 -0.284496736
#define ERF_A3 1.421413741
#define ERF_A4 -1.453152027
#define ERF_A5 1.061405429
#endif

#if defined(TYPE_F64)
double exp_(double x) {
    if (isnan(x)) return x;
    if (x > 709.782712893384LF) return POS_INF;
    if (x < -745.1332191019412LF) return 0.0LF;
    double k = floor(x * 1.4426950408889634LF + 0.5LF);
    double r = (x - k * 0.693147180369123816490LF) - k * 1.90821492927058770002e-10LF;
    double p = 1.0LF;
    for (int i = 17; i >= 1; i--) {
        p = 1.0LF + p * r / double(i);
    }
    // 2^k in two steps, so k down to -1075 does not underflow the scale itself
    int ki = int(k);
    int half_ = ki / 2;
    return ldexp(ldexp(p, half_), ki - half_);
}

double log_(double x) {
    if (isnan(x) || x < 0.0LF) return NAN_;
    if (x == 0.0LF) return NEG_INF;
    if (isinf(x)) return x;
    int e;
    double m = frexp(x, e);
    if (m < 0.7071067811865476LF) {
        m *= 2.0LF;
        e -= 1;
    }
    double s = (m - 1.0LF) / (m + 1.0LF);
    double s2 = s * s;
    double p = 0.0LF;
    for (int i = 12; i >= 0; i--) {
        p = 1.0LF / double(2 * i + 1) + s2 * p;
    }
    return 2.0LF * s * p + double(e) * 0.6931471805599453094LF;
}

double pow_native(double a, double b) {
    return exp_(b * log_(a));
}
#else
float exp_(float x) { return exp(x); }
float log_(float x) { return log(x); }
float pow_native(float a, float b) { return pow(a, b); }
#endif

ACC sigmoid_(ACC x) {
    if (x >= K(0.0)) {
        return K(1.0) / (K(1.0) + exp_(-x));
    }
    ACC e = exp_(x);
    return e / (K(1.0) + e);
}

// tanh without overflow: some drivers compute it as a ratio of exponentials
ACC tanh_(ACC x) {
    ACC c = clamp(x, K(-20.0), K(20.0));
    if (abs(c) < K(0.0004)) {
        ACC c2 = c * c;
        return c - c * c2 / K(3.0) + K(2.0) * c * c2 * c2 / K(15.0);
    }
    ACC e = exp_(K(2.0) * c);
    return (e - K(1.0)) / (e + K(1.0));
}

ACC expm1_(ACC x) {
    if (abs(x) < K(1e-3)) {
        return x + x * x * (K(0.5) + x * (K(1.0) / K(6.0) + x / K(24.0)));
    }
    return exp_(x) - K(1.0);
}

ACC log1p_(ACC x) {
    ACC u = K(1.0) + x;
    // log(u) * x / (u - 1) corrects the rounding of 1 + x
    return u == K(1.0) ? x : log_(u) * x / (u - K(1.0));
}

ACC softplus_(ACC x) {
    return max(x, K(0.0)) + log1p_(exp_(-abs(x)));
}

// Abramowitz and Stegun 7.1.26, as the CPU backend computes it
ACC erf_(ACC x) {
    ACC z = abs(x);
    ACC t = K(1.0) / (K(1.0) + ERF_P * z);
    ACC poly = t * (ERF_A1 + t * (ERF_A2 + t * (ERF_A3 + t * (ERF_A4 + t * ERF_A5))));
    ACC r = K(1.0) - poly * exp_(-z * z);
    return x >= K(0.0) ? r : -r;
}

ACC pow_(ACC a, ACC b) {
    if (b == K(0.0)) {
        return K(1.0);
    }
    if (isnan(a) || isnan(b)) {
        return a + b;
    }
    if (a < K(0.0)) {
        if (b != floor(b)) {
            return NAN_;
        }
        ACC r = pow_native(-a, b);
        return mod(b, K(2.0)) == K(1.0) ? -r : r;
    }
    if (a == K(0.0)) {
        return b > K(0.0) ? K(0.0) : POS_INF;
    }
    return pow_native(a, b);
}

ACC unary_op(uint op, ACC v) {
    switch (op) {
        case 0: return -v;
        case 1: return abs(v);
        case 2: return exp_(v);
        case 3: return log_(v);
        case 4: return sqrt(v);
        case 5: return v * v;
        case 6: return tanh_(v);
        case 7: return sigmoid_(v);
        case 8: return v > K(0.0) ? v : K(0.0);
        case 9: return v > K(0.0) ? K(1.0) : K(0.0);
        case 10: return v > K(0.0) ? K(1.0) : v < K(0.0) ? K(-1.0) : v;
        case 11: return erf_(v);
        case 12: return softplus_(v);
        default: return softplus_(v);
    }
}

ACC hard_sigmoid_(ACC x) {
    return min(max(x / K(6.0) + K(0.5), K(0.0)), K(1.0));
}

ACC activation_op(uint op, ACC p, ACC x) {
    switch (op) {
        case 0: return x > K(0.0) ? x : K(0.0);
        case 1: return sigmoid_(x);
        case 2: return tanh_(x);
        case 3: return softplus_(x);
        case 4: return x * sigmoid_(x);
        case 5: return x * K(0.5) * (K(1.0) + erf_(x * INV_SQRT_2));
        case 6: return K(0.5) * x * (K(1.0) + tanh_(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)));
        case 7: return x * tanh_(softplus_(x));
        case 8: return x > K(0.0) ? x : p * x;
        case 9: return x > K(0.0) ? x : p * expm1_(x);
        case 10: return SELU_SCALE * (x > K(0.0) ? x : SELU_ALPHA * expm1_(x));
        case 11: return min(max(x, K(0.0)), K(6.0));
        case 12: return min(max(x, K(-1.0)), K(1.0));
        case 13: return hard_sigmoid_(x);
        case 14: return x * hard_sigmoid_(x);
        default: return x / (K(1.0) + abs(x));
    }
}

ACC activation_slope(uint op, ACC p, ACC x) {
    switch (op) {
        case 0: return x > K(0.0) ? K(1.0) : K(0.0);
        case 1: { ACC s = sigmoid_(x); return s * (K(1.0) - s); }
        case 2: { ACC t = tanh_(x); return K(1.0) - t * t; }
        case 3: return sigmoid_(x);
        case 4: { ACC s = sigmoid_(x); return s * (K(1.0) + x * (K(1.0) - s)); }
        case 5: {
            ACC e = exp_(K(-0.5) * x * x);
            return K(0.5) * (K(1.0) + erf_(x * INV_SQRT_2)) + x * INV_SQRT_2PI * e;
        }
        case 6: {
            ACC t = tanh_(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
            return K(0.5) * (K(1.0) + t) + K(0.5) * x * (K(1.0) - t * t) * SQRT_2_OVER_PI
                    * (K(1.0) + K(3.0) * GELU_CUBIC * x * x);
        }
        case 7: { ACC t = tanh_(softplus_(x)); return t + x * (K(1.0) - t * t) * sigmoid_(x); }
        case 8: return x > K(0.0) ? K(1.0) : p;
        case 9: return x > K(0.0) ? K(1.0) : p * exp_(x);
        case 10: return SELU_SCALE * (x > K(0.0) ? K(1.0) : SELU_ALPHA * exp_(x));
        case 11: return x > K(0.0) && x < K(6.0) ? K(1.0) : K(0.0);
        case 12: return x > K(-1.0) && x < K(1.0) ? K(1.0) : K(0.0);
        case 13: return x > K(-3.0) && x < K(3.0) ? K(1.0) / K(6.0) : K(0.0);
        case 14: return x <= K(-3.0) ? K(0.0) : x >= K(3.0) ? K(1.0) : (K(2.0) * x + K(3.0)) / K(6.0);
        default: { ACC d = K(1.0) + abs(x); return K(1.0) / (d * d); }
    }
}

ACC binary_op(uint op, ACC a, ACC b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return pow_(a, b);
        case 5: return isnan(a) || isnan(b) ? a + b : max(a, b);
        case 6: return isnan(a) || isnan(b) ? a + b : min(a, b);
        case 7: return a > b ? K(1.0) : K(0.0);
        case 8: return a == b ? K(1.0) : K(0.0);
        default: return b != K(0.0) ? a : K(0.0);
    }
}
