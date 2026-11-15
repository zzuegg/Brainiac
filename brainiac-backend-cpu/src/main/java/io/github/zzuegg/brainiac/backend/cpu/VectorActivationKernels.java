package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD versions of the element-wise loops that use only exactly rounded arithmetic, so
 * they give the same bits as {@link Kernels} and {@link ActivationKernels}, whatever the
 * vector width and the thread split. Only loaded when the {@code jdk.incubator.vector}
 * module is present.
 */
final class VectorActivationKernels {

    private static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;
    private static final int L = S.length();

    private VectorActivationKernels() {
    }

    /** @return whether {@link #forward} and {@link #backward} handle the activation */
    static boolean supports(ActivationOp op) {
        return !op.transcendental();
    }

    /** @return whether {@link #binary} handles the operation */
    static boolean supports(BinaryOp op) {
        return switch (op) {
            case ADD, SUB, MUL, DIV, MAX, MIN -> true;
            default -> false;
        };
    }

    static void forward(ActivationOp op, float p, float[] x, float[] y, int from, int to) {
        int i = from;
        int upper = from + S.loopBound(to - from);
        for (; i < upper; i += L) {
            forward(op, p, FloatVector.fromArray(S, x, i)).intoArray(y, i);
        }
        ActivationKernels.forward(op, p, x, y, i, to);
    }

    private static FloatVector forward(ActivationOp op, float p, FloatVector v) {
        return switch (op) {
            case RELU -> FloatVector.zero(S).blend(v, v.compare(VectorOperators.GT, 0f));
            case LEAKY_RELU -> v.blend(v.mul(p), v.compare(VectorOperators.GT, 0f).not());
            case RELU6 -> v.max(0f).min(6f);
            case HARD_TANH -> v.max(-1f).min(1f);
            case HARD_SIGMOID -> v.div(6f).add(0.5f).max(0f).min(1f);
            case HARD_SWISH -> v.mul(v.div(6f).add(0.5f).max(0f).min(1f));
            case SOFTSIGN -> v.div(v.abs().add(1f));
            default -> throw new IllegalArgumentException("no SIMD kernel for " + op);
        };
    }

    static void backward(ActivationOp op, float p, float[] x, float[] g, float[] z, int from, int to) {
        int i = from;
        int upper = from + S.loopBound(to - from);
        for (; i < upper; i += L) {
            FloatVector v = FloatVector.fromArray(S, x, i);
            FloatVector gv = FloatVector.fromArray(S, g, i);
            backward(op, p, v, gv).intoArray(z, i);
        }
        ActivationKernels.backward(op, p, x, g, z, i, to);
    }

    private static FloatVector backward(ActivationOp op, float p, FloatVector v, FloatVector g) {
        FloatVector zero = FloatVector.zero(S);
        return switch (op) {
            case RELU -> zero.blend(g, v.compare(VectorOperators.GT, 0f));
            case LEAKY_RELU -> g.mul(p).blend(g, v.compare(VectorOperators.GT, 0f));
            case RELU6 -> zero.blend(g, inside(v, 0f, 6f));
            case HARD_TANH -> zero.blend(g, inside(v, -1f, 1f));
            case HARD_SIGMOID -> zero.blend(g.div(6f), inside(v, -3f, 3f));
            case HARD_SWISH -> {
                FloatVector slope = v.mul(2f).add(3f).div(6f)
                        .blend(0f, v.compare(VectorOperators.LE, -3f))
                        .blend(1f, v.compare(VectorOperators.GE, 3f));
                yield g.mul(slope);
            }
            case SOFTSIGN -> {
                FloatVector d = v.abs().add(1f);
                yield g.div(d.mul(d));
            }
            default -> throw new IllegalArgumentException("no SIMD kernel for " + op);
        };
    }

    private static VectorMask<Float> inside(FloatVector v, float low, float high) {
        return v.compare(VectorOperators.GT, low).and(v.compare(VectorOperators.LT, high));
    }

    /** {@code z[i] = x[i] op y[i]} for {@code i} in {@code [from, to)}. */
    static void binary(BinaryOp op, float[] x, float[] y, float[] z, int from, int to) {
        int i = from;
        int upper = from + S.loopBound(to - from);
        for (; i < upper; i += L) {
            apply(op, FloatVector.fromArray(S, x, i), FloatVector.fromArray(S, y, i)).intoArray(z, i);
        }
        Kernels.binary(op, x, y, z, i, to);
    }

    /** {@code z[r, j] = x[r, j] op y[j]} for rows {@code [from, to)} of length {@code n}. */
    static void binaryRows(BinaryOp op, float[] x, float[] y, float[] z, int n, int from, int to) {
        int upper = S.loopBound(n);
        for (int r = from; r < to; r++) {
            int base = r * n;
            int j = 0;
            for (; j < upper; j += L) {
                apply(op, FloatVector.fromArray(S, x, base + j), FloatVector.fromArray(S, y, j))
                        .intoArray(z, base + j);
            }
            for (; j < n; j++) {
                z[base + j] = Kernels.apply(op, x[base + j], y[j]);
            }
        }
    }

    /** {@code z[i] = x[i] op s}. */
    static void binaryScalarRight(BinaryOp op, float[] x, float s, float[] z, int from, int to) {
        int i = from;
        int upper = from + S.loopBound(to - from);
        FloatVector sv = FloatVector.broadcast(S, s);
        for (; i < upper; i += L) {
            apply(op, FloatVector.fromArray(S, x, i), sv).intoArray(z, i);
        }
        Kernels.binaryScalarRight(op, x, s, z, i, to);
    }

    private static FloatVector apply(BinaryOp op, FloatVector a, FloatVector b) {
        return switch (op) {
            case ADD -> a.add(b);
            case SUB -> a.sub(b);
            case MUL -> a.mul(b);
            case DIV -> a.div(b);
            case MAX -> a.max(b);
            case MIN -> a.min(b);
            default -> throw new IllegalArgumentException("no SIMD kernel for " + op);
        };
    }
}
