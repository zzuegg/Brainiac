package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

/**
 * Fused activation functions and their derivatives, one pass each. Functions built only
 * from arithmetic are computed in float, step by step exactly like the SIMD versions in
 * {@link VectorActivationKernels}, so both give identical bits; the others are computed
 * in double.
 */
final class ActivationKernels {

    static final double SELU_ALPHA = 1.6732632423543772;
    static final double SELU_SCALE = 1.0507009873554805;
    private static final double SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
    private static final double INV_SQRT_2 = 1 / Math.sqrt(2);
    private static final double INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
    private static final double GELU_CUBIC = 0.044715;

    private ActivationKernels() {
    }

    /** Relative cost per element, for splitting work across threads. */
    static long cost(ActivationOp op) {
        return op.transcendental() ? 30 : 2;
    }

    static void forward(ActivationOp op, float p, float[] x, float[] y, int from, int to) {
        switch (op) {
            case RELU -> { for (int i = from; i < to; i++) y[i] = x[i] > 0f ? x[i] : 0f; }
            case LEAKY_RELU -> { for (int i = from; i < to; i++) y[i] = x[i] > 0f ? x[i] : p * x[i]; }
            case RELU6 -> { for (int i = from; i < to; i++) y[i] = Math.min(Math.max(x[i], 0f), 6f); }
            case HARD_TANH -> { for (int i = from; i < to; i++) y[i] = Math.min(Math.max(x[i], -1f), 1f); }
            case HARD_SIGMOID -> { for (int i = from; i < to; i++) y[i] = hardSigmoid(x[i]); }
            case HARD_SWISH -> { for (int i = from; i < to; i++) y[i] = x[i] * hardSigmoid(x[i]); }
            case SOFTSIGN -> { for (int i = from; i < to; i++) y[i] = x[i] / (1f + Math.abs(x[i])); }
            default -> { for (int i = from; i < to; i++) y[i] = (float) value(op, p, x[i]); }
        }
    }

    static void backward(ActivationOp op, float p, float[] x, float[] g, float[] z, int from, int to) {
        switch (op) {
            case RELU -> { for (int i = from; i < to; i++) z[i] = x[i] > 0f ? g[i] : 0f; }
            case LEAKY_RELU -> { for (int i = from; i < to; i++) z[i] = x[i] > 0f ? g[i] : g[i] * p; }
            case RELU6 -> { for (int i = from; i < to; i++) z[i] = x[i] > 0f && x[i] < 6f ? g[i] : 0f; }
            case HARD_TANH -> { for (int i = from; i < to; i++) z[i] = x[i] > -1f && x[i] < 1f ? g[i] : 0f; }
            case HARD_SIGMOID -> { for (int i = from; i < to; i++) z[i] = x[i] > -3f && x[i] < 3f ? g[i] / 6f : 0f; }
            case HARD_SWISH -> { for (int i = from; i < to; i++) z[i] = g[i] * hardSwishSlope(x[i]); }
            case SOFTSIGN -> {
                for (int i = from; i < to; i++) {
                    float d = 1f + Math.abs(x[i]);
                    z[i] = g[i] / (d * d);
                }
            }
            default -> { for (int i = from; i < to; i++) z[i] = (float) (g[i] * derivative(op, p, x[i])); }
        }
    }

    static float hardSigmoid(float x) {
        return Math.min(Math.max(x / 6f + 0.5f, 0f), 1f);
    }

    static float hardSwishSlope(float x) {
        return x <= -3f ? 0f : x >= 3f ? 1f : (2f * x + 3f) / 6f;
    }

    private static double sigmoid(double x) {
        if (x >= 0) {
            return 1 / (1 + Math.exp(-x));
        }
        double e = Math.exp(x);
        return e / (1 + e);
    }

    private static double softplus(double x) {
        return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
    }

    private static double phi(double x) {
        return 0.5 * (1 + Kernels.erf(x * INV_SQRT_2));
    }

    /** {@code f(x)} for the transcendental activations. */
    static double value(ActivationOp op, double p, double x) {
        return switch (op) {
            case SIGMOID -> sigmoid(x);
            case TANH -> Math.tanh(x);
            case SOFTPLUS -> softplus(x);
            case SILU -> x * sigmoid(x);
            case GELU -> x * phi(x);
            case GELU_TANH -> 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)));
            case MISH -> x * Math.tanh(softplus(x));
            case ELU -> x > 0 ? x : p * Math.expm1(x);
            case SELU -> SELU_SCALE * (x > 0 ? x : SELU_ALPHA * Math.expm1(x));
            default -> throw new IllegalArgumentException("not a transcendental activation: " + op);
        };
    }

    /** {@code f'(x)} for the transcendental activations. */
    static double derivative(ActivationOp op, double p, double x) {
        return switch (op) {
            case SIGMOID -> {
                double s = sigmoid(x);
                yield s * (1 - s);
            }
            case TANH -> {
                double t = Math.tanh(x);
                yield 1 - t * t;
            }
            case SOFTPLUS -> sigmoid(x);
            case SILU -> {
                double s = sigmoid(x);
                yield s * (1 + x * (1 - s));
            }
            case GELU -> {
                double e = Math.exp(-0.5 * x * x);
                yield 0.5 * (1 + Kernels.erf(x * INV_SQRT_2, e)) + x * INV_SQRT_2PI * e;
            }
            case GELU_TANH -> {
                double t = Math.tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
                yield 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * SQRT_2_OVER_PI * (1 + 3 * GELU_CUBIC * x * x);
            }
            case MISH -> {
                double t = Math.tanh(softplus(x));
                yield t + x * (1 - t * t) * sigmoid(x);
            }
            case ELU -> x > 0 ? 1 : p * Math.exp(x);
            case SELU -> SELU_SCALE * (x > 0 ? 1 : SELU_ALPHA * Math.exp(x));
            default -> throw new IllegalArgumentException("not a transcendental activation: " + op);
        };
    }
}
