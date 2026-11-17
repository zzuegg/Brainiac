// Element-wise functions, matching the CPU backend. Operation codes are the ordinals of the
// Java enums UnaryOp, ActivationOp and BinaryOp.

#define SQRT_2_OVER_PI 0.7978845608028654
#define INV_SQRT_2 0.7071067811865476
#define INV_SQRT_2PI 0.3989422804014327
#define GELU_CUBIC 0.044715
#define SELU_ALPHA 1.6732632423543772
#define SELU_SCALE 1.0507009873554805

float sigmoid_(float x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    float e = exp(x);
    return e / (1.0 + e);
}

// tanh without overflow: some drivers compute it as a ratio of exponentials
float tanh_(float x) {
    float c = clamp(x, -15.0, 15.0);
    if (abs(c) < 0.0004) {
        return c - c * c * c / 3.0;
    }
    float e = exp(2.0 * c);
    return (e - 1.0) / (e + 1.0);
}

float expm1_(float x) {
    if (abs(x) < 1e-3) {
        return x + 0.5 * x * x + x * x * x / 6.0;
    }
    return exp(x) - 1.0;
}

float log1p_(float x) {
    float u = 1.0 + x;
    // log(u) * x / (u - 1) corrects the rounding of 1 + x
    return u == 1.0 ? x : log(u) * x / (u - 1.0);
}

float softplus_(float x) {
    return max(x, 0.0) + log1p_(exp(-abs(x)));
}

// Abramowitz and Stegun 7.1.26, as the CPU backend computes it
float erf_(float x) {
    float z = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * z);
    float poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    float r = 1.0 - poly * exp(-z * z);
    return x >= 0.0 ? r : -r;
}

float pow_(float a, float b) {
    if (b == 0.0) {
        return 1.0;
    }
    if (a < 0.0) {
        if (b != floor(b)) {
            return uintBitsToFloat(0x7fc00000u);
        }
        float r = pow(-a, b);
        return mod(b, 2.0) == 1.0 ? -r : r;
    }
    if (a == 0.0) {
        return b > 0.0 ? 0.0 : POS_INF;
    }
    return pow(a, b);
}

float unary_op(uint op, float v) {
    switch (op) {
        case 0: return -v;
        case 1: return abs(v);
        case 2: return exp(v);
        case 3: return log(v);
        case 4: return sqrt(v);
        case 5: return v * v;
        case 6: return tanh_(v);
        case 7: return sigmoid_(v);
        case 8: return v > 0.0 ? v : 0.0;
        case 9: return v > 0.0 ? 1.0 : 0.0;
        case 10: return v > 0.0 ? 1.0 : v < 0.0 ? -1.0 : v;
        case 11: return erf_(v);
        case 12: return softplus_(v);
        default: return roundEven(v);
    }
}

float hard_sigmoid_(float x) {
    return min(max(x / 6.0 + 0.5, 0.0), 1.0);
}

float activation_op(uint op, float p, float x) {
    switch (op) {
        case 0: return x > 0.0 ? x : 0.0;
        case 1: return sigmoid_(x);
        case 2: return tanh_(x);
        case 3: return softplus_(x);
        case 4: return x * sigmoid_(x);
        case 5: return x * 0.5 * (1.0 + erf_(x * INV_SQRT_2));
        case 6: return 0.5 * x * (1.0 + tanh_(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)));
        case 7: return x * tanh_(softplus_(x));
        case 8: return x > 0.0 ? x : p * x;
        case 9: return x > 0.0 ? x : p * expm1_(x);
        case 10: return SELU_SCALE * (x > 0.0 ? x : SELU_ALPHA * expm1_(x));
        case 11: return min(max(x, 0.0), 6.0);
        case 12: return min(max(x, -1.0), 1.0);
        case 13: return hard_sigmoid_(x);
        case 14: return x * hard_sigmoid_(x);
        default: return x / (1.0 + abs(x));
    }
}

float activation_slope(uint op, float p, float x) {
    switch (op) {
        case 0: return x > 0.0 ? 1.0 : 0.0;
        case 1: { float s = sigmoid_(x); return s * (1.0 - s); }
        case 2: { float t = tanh_(x); return 1.0 - t * t; }
        case 3: return sigmoid_(x);
        case 4: { float s = sigmoid_(x); return s * (1.0 + x * (1.0 - s)); }
        case 5: {
            float e = exp(-0.5 * x * x);
            return 0.5 * (1.0 + erf_(x * INV_SQRT_2)) + x * INV_SQRT_2PI * e;
        }
        case 6: {
            float t = tanh_(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
            return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x * x);
        }
        case 7: { float t = tanh_(softplus_(x)); return t + x * (1.0 - t * t) * sigmoid_(x); }
        case 8: return x > 0.0 ? 1.0 : p;
        case 9: return x > 0.0 ? 1.0 : p * exp(x);
        case 10: return SELU_SCALE * (x > 0.0 ? 1.0 : SELU_ALPHA * exp(x));
        case 11: return x > 0.0 && x < 6.0 ? 1.0 : 0.0;
        case 12: return x > -1.0 && x < 1.0 ? 1.0 : 0.0;
        case 13: return x > -3.0 && x < 3.0 ? 1.0 / 6.0 : 0.0;
        case 14: return x <= -3.0 ? 0.0 : x >= 3.0 ? 1.0 : (2.0 * x + 3.0) / 6.0;
        default: { float d = 1.0 + abs(x); return 1.0 / (d * d); }
    }
}

float binary_op(uint op, float a, float b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return pow_(a, b);
        case 5: return isnan(a) || isnan(b) ? a + b : max(a, b);
        case 6: return isnan(a) || isnan(b) ? a + b : min(a, b);
        case 7: return a > b ? 1.0 : 0.0;
        case 8: return a == b ? 1.0 : 0.0;
        default: return b != 0.0 ? a : 0.0;
    }
}
