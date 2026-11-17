package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

/**
 * The kernels for every type but float32, in double precision. Float64 is computed
 * exactly in its own precision; float16 and bfloat16 are widened, computed in double and
 * rounded once when stored, which is at least as accurate as hardware that accumulates
 * in float32. Simple loops without SIMD: the float32 kernels are the fast path.
 */
final class WideKernels {

    private WideKernels() {
    }

    static void unary(UnaryOp op, double[] x, double[] y) {
        long cost = switch (op) {
            case EXP, LOG, TANH, SIGMOID, ERF, SOFTPLUS, SIN, COS -> 20;
            default -> 1;
        };
        Parallel.forRange(x.length, cost, (from, to) -> {
            for (int i = from; i < to; i++) {
                y[i] = unary(op, x[i]);
            }
        });
    }

    static double unary(UnaryOp op, double v) {
        return switch (op) {
            case NEG -> -v;
            case ABS -> Math.abs(v);
            case EXP -> Math.exp(v);
            case LOG -> Math.log(v);
            case SQRT -> Math.sqrt(v);
            case SQUARE -> v * v;
            case TANH -> Math.tanh(v);
            case SIGMOID -> v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v));
            case RELU -> v > 0 ? v : 0;
            case STEP -> v > 0 ? 1 : 0;
            case SIGN -> Math.signum(v);
            case ERF -> Kernels.erf(v);
            case SOFTPLUS -> Math.max(v, 0) + Math.log1p(Math.exp(-Math.abs(v)));
            case ROUND -> Math.rint(v);
            case SIN -> Math.sin(v);
            case COS -> Math.cos(v);
        };
    }

    static void activation(ActivationOp op, double p, double[] x, double[] y) {
        Parallel.forRange(x.length, ActivationKernels.cost(op), (from, to) -> {
            for (int i = from; i < to; i++) {
                y[i] = activation(op, p, x[i]);
            }
        });
    }

    static void activationBackward(ActivationOp op, double p, double[] x, double[] g, double[] z) {
        Parallel.forRange(x.length, ActivationKernels.cost(op), (from, to) -> {
            for (int i = from; i < to; i++) {
                z[i] = g[i] * slope(op, p, x[i]);
            }
        });
    }

    private static double activation(ActivationOp op, double p, double x) {
        return switch (op) {
            case RELU -> x > 0 ? x : 0;
            case LEAKY_RELU -> x > 0 ? x : p * x;
            case RELU6 -> Math.min(Math.max(x, 0), 6);
            case HARD_TANH -> Math.min(Math.max(x, -1), 1);
            case HARD_SIGMOID -> Math.min(Math.max(x / 6 + 0.5, 0), 1);
            case HARD_SWISH -> x * Math.min(Math.max(x / 6 + 0.5, 0), 1);
            case SOFTSIGN -> x / (1 + Math.abs(x));
            default -> ActivationKernels.value(op, p, x);
        };
    }

    private static double slope(ActivationOp op, double p, double x) {
        return switch (op) {
            case RELU -> x > 0 ? 1 : 0;
            case LEAKY_RELU -> x > 0 ? 1 : p;
            case RELU6 -> x > 0 && x < 6 ? 1 : 0;
            case HARD_TANH -> x > -1 && x < 1 ? 1 : 0;
            case HARD_SIGMOID -> x > -3 && x < 3 ? 1.0 / 6 : 0;
            case HARD_SWISH -> x <= -3 ? 0 : x >= 3 ? 1 : (2 * x + 3) / 6;
            case SOFTSIGN -> {
                double d = 1 + Math.abs(x);
                yield 1 / (d * d);
            }
            default -> ActivationKernels.derivative(op, p, x);
        };
    }

    static double apply(BinaryOp op, double a, double b) {
        return switch (op) {
            case ADD -> a + b;
            case SUB -> a - b;
            case MUL -> a * b;
            case DIV -> a / b;
            case POW -> Math.pow(a, b);
            case MAX -> Math.max(a, b);
            case MIN -> Math.min(a, b);
            case GREATER -> a > b ? 1 : 0;
            case EQUAL -> a == b ? 1 : 0;
            case MASK -> b != 0 ? a : 0;
        };
    }

    static void binary(BinaryOp op, double[] x, long[] aStrides, double[] y, long[] bStrides, int[] shape,
            double[] z) {
        int rank = shape.length;
        Parallel.forRange(z.length, op == BinaryOp.POW ? 24 : 4, (from, to) -> {
            int[] index = new int[rank];
            long ia = 0;
            long ib = 0;
            long rest = from;
            for (int d = rank - 1; d >= 0; d--) {
                index[d] = (int) (rest % shape[d]);
                rest /= shape[d];
                ia += index[d] * aStrides[d];
                ib += index[d] * bStrides[d];
            }
            for (int i = from; i < to; i++) {
                z[i] = apply(op, x[(int) ia], y[(int) ib]);
                for (int d = rank - 1; d >= 0; d--) {
                    index[d]++;
                    ia += aStrides[d];
                    ib += bStrides[d];
                    if (index[d] < shape[d]) {
                        break;
                    }
                    ia -= (long) shape[d] * aStrides[d];
                    ib -= (long) shape[d] * bStrides[d];
                    index[d] = 0;
                }
            }
        });
    }

    static void reduce(ReduceOp op, double[] x, int outer, int len, int inner, double[] y) {
        Parallel.forRange(outer * inner, len, (from, to) -> {
            for (int r = from; r < to; r++) {
                int o = r / inner;
                int j = r % inner;
                int start = o * len * inner + j;
                double acc = switch (op) {
                    case SUM, MEAN -> 0;
                    case MAX -> Double.NEGATIVE_INFINITY;
                    case MIN -> Double.POSITIVE_INFINITY;
                    case ARGMAX -> 0;
                };
                int best = 0;
                for (int i = 0; i < len; i++) {
                    double v = x[start + i * inner];
                    switch (op) {
                        case SUM, MEAN -> acc += v;
                        case MAX -> acc = Math.max(acc, v);
                        case MIN -> acc = Math.min(acc, v);
                        case ARGMAX -> {
                            if (v > x[start + best * inner]) {
                                best = i;
                            }
                        }
                    }
                }
                y[r] = switch (op) {
                    case MEAN -> acc / len;
                    case ARGMAX -> best;
                    default -> acc;
                };
            }
        });
    }

    /** {@code z = a b} for {@code batch} pairs of matrices {@code [m, k] x [k, n]}. */
    static void matmul(double[] a, double[] b, double[] z, int batch, int m, int k, int n, boolean sharedB) {
        Parallel.forRange(batch * m, 2L * k * n, (from, to) -> {
            for (int row = from; row < to; row++) {
                int bt = row / m;
                int aBase = row * k;
                int bBase = sharedB ? 0 : bt * k * n;
                int zBase = row * n;
                java.util.Arrays.fill(z, zBase, zBase + n, 0);
                for (int kk = 0; kk < k; kk++) {
                    double av = a[aBase + kk];
                    if (av == 0) {
                        continue;
                    }
                    int bRow = bBase + kk * n;
                    for (int j = 0; j < n; j++) {
                        z[zBase + j] += av * b[bRow + j];
                    }
                }
            }
        });
    }

    static void im2col(double[] x, WindowKernels.Plan plan, int batch, double pad, double[] out) {
        int[] table = plan.table();
        int c = plan.channels;
        int ks = plan.kernelSize;
        int img = plan.imageSize;
        Parallel.forRange(batch * plan.positions, (long) c * ks, (from, to) -> {
            for (int row = from; row < to; row++) {
                int n = row / plan.positions;
                int p = row % plan.positions;
                for (int ch = 0; ch < c; ch++) {
                    int inBase = (n * c + ch) * img;
                    int o = row * c * ks + ch * ks;
                    for (int k = 0; k < ks; k++) {
                        int cell = table[p * ks + k];
                        out[o + k] = cell < 0 ? pad : x[inBase + cell];
                    }
                }
            }
        });
    }

    static void col2im(double[] cols, WindowKernels.Plan plan, int batch, double[] out) {
        int[] table = plan.table();
        int c = plan.channels;
        int ks = plan.kernelSize;
        int img = plan.imageSize;
        Parallel.forRange(batch * c, (long) plan.positions * ks, (from, to) -> {
            for (int plane = from; plane < to; plane++) {
                int n = plane / c;
                int ch = plane % c;
                int outBase = plane * img;
                java.util.Arrays.fill(out, outBase, outBase + img, 0);
                for (int p = 0; p < plan.positions; p++) {
                    int src = (n * plan.positions + p) * c * ks + ch * ks;
                    for (int k = 0; k < ks; k++) {
                        int cell = table[p * ks + k];
                        if (cell >= 0) {
                            out[outBase + cell] += cols[src + k];
                        }
                    }
                }
            }
        });
    }

    static void gather(double[] x, int outer, int length, int inner, int[] idx, double[] out) {
        int count = idx.length;
        for (int r = 0; r < outer * count; r++) {
            System.arraycopy(x, ((r / count) * length + idx[r % count]) * inner, out, r * inner, inner);
        }
    }

    static void scatterAdd(double[] src, int outer, int length, int inner, int[] idx, double[] out) {
        int count = idx.length;
        java.util.Arrays.fill(out, 0, outer * length * inner, 0);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < count; i++) {
                int s = (o * count + i) * inner;
                int d = (o * length + idx[i]) * inner;
                for (int j = 0; j < inner; j++) {
                    out[d + j] += src[s + j];
                }
            }
        }
    }

    /** A strided copy, one element at a time. */
    static void copy(double[] x, long inOffset, long[] inStrides, int[] shape, double[] z, long outOffset,
            long[] outStrides) {
        int rank = shape.length;
        long count = 1;
        for (int d : shape) {
            count *= d;
        }
        if (count == 0) {
            return;
        }
        int[] index = new int[rank];
        long ia = inOffset;
        long iz = outOffset;
        for (long i = 0; i < count; i++) {
            z[(int) iz] = x[(int) ia];
            for (int d = rank - 1; d >= 0; d--) {
                index[d]++;
                ia += inStrides[d];
                iz += outStrides[d];
                if (index[d] < shape[d]) {
                    break;
                }
                ia -= (long) shape[d] * inStrides[d];
                iz -= (long) shape[d] * outStrides[d];
                index[d] = 0;
            }
        }
    }
}
