package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * HQQ, half-quadratic quantization: calibration-free weight quantization that fits each
 * group's zero point by minimizing a sparsity-promoting {@code l_p} norm ({@code p < 1}) of
 * the weight error rather than its square. Outlier weights then count as a few large errors
 * the norm tolerates, and the zero point settles where most weights round well. The
 * non-convex problem is split with an auxiliary error variable {@code E} (half-quadratic
 * splitting) and solved by alternating closed forms, a few dozen cheap passes in all:
 * <pre>
 *   Q = clamp(round(W / s + z)),  E = shrink_p(W - s (Q - z), beta),
 *   z = mean(Q - (W - E) / s),     beta = kappa * beta
 * </pre>
 * with the generalized soft threshold {@code shrink_p(x) = sign(x) relu(|x| - |x|^(p-1) /
 * beta)}. No data is needed, so it quantizes in milliseconds where GPTQ and AWQ need
 * calibration inputs and solves.
 *
 * <p>Weights {@code [in, out]} are grouped along the input axis per output column, as the
 * core {@code Tensor.quantized(bits, group)} groups them, with an asymmetric grid of
 * {@code 2^bits} levels; {@link #roundToNearest} is that grid with its min/max zero point,
 * HQQ's starting point.
 *
 * <p>Badri and Shaji, "Half-Quadratic Quantization of Large Machine Learning Models" (2023),
 * <a href="https://mobiusml.github.io/hqq_blog/">mobiusml.github.io/hqq_blog</a>.
 */
public final class Hqq {

    private final int bits;
    private final int group;
    private int iterations = 20;
    private double p = 0.7;
    private double beta = 10;
    private double kappa = 1.01;

    /**
     * @param bits bits per weight, 2 to 8
     * @param group input rows per scale and zero point; 0 for whole columns
     */
    public Hqq(int bits, int group) {
        if (bits < 1 || bits > 8 || group < 0) {
            throw new IllegalArgumentException("bad bits or group");
        }
        this.bits = bits;
        this.group = group;
    }

    /**
     * @param n half-quadratic iterations, 20 by default
     * @return this
     */
    public Hqq iterations(int n) {
        this.iterations = n;
        return this;
    }

    /**
     * @param norm the {@code p} of the error norm, 0.7 by default
     * @param startBeta the starting penalty, 10 by default
     * @param growth the penalty's growth per iteration, 1.01 by default
     * @return this
     */
    public Hqq solver(double norm, double startBeta, double growth) {
        this.p = norm;
        this.beta = startBeta;
        this.kappa = growth;
        return this;
    }

    /**
     * Asymmetric round-to-nearest on the same grid, zero point from each group's minimum.
     *
     * @param weight {@code [in, out]}
     * @return the dequantized weight
     */
    public Tensor roundToNearest(Tensor weight) {
        return run(weight, 0);
    }

    /**
     * Quantizes with optimized zero points.
     *
     * @param weight {@code [in, out]}
     * @return the dequantized weight
     */
    public Tensor quantize(Tensor weight) {
        return run(weight, iterations);
    }

    private Tensor run(Tensor weight, int iters) {
        int rows = weight.shape().dim(0);
        int cols = weight.shape().dim(1);
        int g = group == 0 ? rows : group;
        if (rows % g != 0) {
            throw new IllegalArgumentException(rows + " rows are not a multiple of the group " + g);
        }
        float[] w = weight.toArray();
        float[] out = new float[w.length];
        int top = (1 << bits) - 1;
        double[] x = new double[g];
        double[] q = new double[g];
        double[] e = new double[g];
        for (int c = 0; c < cols; c++) {
            for (int r0 = 0; r0 < rows; r0 += g) {
                double lo = Double.MAX_VALUE;
                double hi = -Double.MAX_VALUE;
                for (int i = 0; i < g; i++) {
                    x[i] = w[(r0 + i) * cols + c];
                    lo = Math.min(lo, x[i]);
                    hi = Math.max(hi, x[i]);
                }
                double s = hi > lo ? (hi - lo) / top : 1;
                double z = -lo / s;
                double b = beta;
                double bestZ = z;
                double bestError = Double.MAX_VALUE;
                for (int it = 0; it <= iters; it++) {
                    double sum = 0;
                    double error = 0;
                    for (int i = 0; i < g; i++) {
                        q[i] = Math.max(0, Math.min(top, Math.rint(x[i] / s + z)));
                        double residual = x[i] - s * (q[i] - z);
                        error += Math.abs(residual);
                        e[i] = shrink(residual, b);
                        sum += q[i] - (x[i] - e[i]) / s;
                    }
                    if (error < bestError) {   // as the reference code, keep the best zero point seen
                        bestError = error;
                        bestZ = z;
                    }
                    if (it == iters) {
                        break;
                    }
                    z = sum / g;
                    b *= kappa;
                }
                z = bestZ;
                for (int i = 0; i < g; i++) {
                    double level = Math.max(0, Math.min(top, Math.rint(x[i] / s + z)));
                    out[(r0 + i) * cols + c] = (float) (s * (level - z));
                }
            }
        }
        return Tensor.of(weight.shape(), out, weight.device());
    }

    /** The generalized soft threshold for the {@code l_p} norm. */
    private double shrink(double v, double b) {
        double a = Math.abs(v);
        if (a == 0) {
            return 0;
        }
        return Math.signum(v) * Math.max(0, a - Math.pow(a, p - 1) / b);
    }
}
