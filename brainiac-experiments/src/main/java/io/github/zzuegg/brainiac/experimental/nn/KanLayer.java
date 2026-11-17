package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A Kolmogorov-Arnold network layer (KAN; Liu et al., <a href="https://arxiv.org/abs/2404.19756">2404.19756</a>):
 * every edge {@code i -> j} carries its own learnable univariate function, and outputs sum
 * their incoming edges:
 *
 * <pre>
 * y_j = sum_i  w_ij silu(x_i) + sum_k c_ijk B_k(x_i)
 * </pre>
 *
 * <p>The {@code B_k} are B-splines of a given order on a uniform grid over
 * {@code [low, high]}, built by the Cox-de Boor recursion out of autograd ops, so gradients
 * reach both the coefficients and the inputs (layers stack). Splines vanish outside the grid,
 * where only the silu base term acts. {@link #refine} implements the paper's grid extension:
 * a finer grid is fitted by least squares to the current functions, so training can start
 * coarse and continue on more intervals without losing what was learned.
 */
public final class KanLayer implements Layer {
    private final int inputs;
    private final int outputs;
    private final int intervals;
    private final int order;
    private final float low;
    private final float high;
    private final Var base;
    private final Var coefficients;

    private KanLayer(int inputs, int outputs, int intervals, int order, float low, float high, Tensor base,
            Tensor coefficients) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.intervals = intervals;
        this.order = order;
        this.low = low;
        this.high = high;
        this.base = Var.param(base);
        this.coefficients = Var.param(coefficients);
    }

    private static KanLayer random(int inputs, int outputs, int intervals, int order, float low, float high,
            SplittableRandom random) {
        float std = (float) (1 / Math.sqrt(inputs));
        Tensor b = Tensor.normal(Shape.of(inputs, outputs), 0, std, random);
        Tensor c = Tensor.normal(Shape.of(inputs * (intervals + order), outputs), 0, 0.1f * std, random);
        return new KanLayer(inputs, outputs, intervals, order, low, high, b, c);
    }

    /**
     * Creates a layer with cubic splines.
     *
     * @param inputs the input width
     * @param outputs the output width
     * @param intervals the grid intervals G
     * @param low the grid start
     * @param high the grid end
     * @param seed the seed
     * @return the layer
     */
    public static KanLayer of(int inputs, int outputs, int intervals, float low, float high, long seed) {
        if (intervals < 1 || !(high > low)) {
            throw new IllegalArgumentException("need intervals >= 1 and high > low");
        }
        return random(inputs, outputs, intervals, 3, low, high, new SplittableRandom(seed));
    }

    /**
     * Describes a cubic-spline layer on {@code [-1, 1]} for model builders.
     *
     * @param outputs the output width
     * @param intervals the grid intervals
     * @return the spec
     */
    public static LayerSpec spec(int outputs, int intervals) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return random(input.dim(-1), outputs, intervals, 3, -1, 1, random);
            }
        };
    }

    /** @return the grid intervals */
    public int intervals() {
        return intervals;
    }

    /** @return the number of basis functions per edge, intervals + order */
    public int basisSize() {
        return intervals + order;
    }

    private float knot(int m) {
        return low + (m - order) * (high - low) / intervals;
    }

    /**
     * Evaluates all B-spline basis functions at the inputs.
     *
     * @param x {@code [batch, inputs]}
     * @return {@code [batch, inputs, intervals + order]}
     */
    public Var basis(Var x) {
        int b = x.shape().dim(0);
        int knots = intervals + 2 * order + 1;
        float[] t = new float[knots];
        for (int m = 0; m < knots; m++) {
            t[m] = knot(m);
        }
        Tensor grid = Tensor.of(Shape.of(1, 1, knots), t, x.value().device());
        Var x3 = x.reshape(b, inputs, 1);
        // order 0: indicator of [t_m, t_m+1)
        Tensor xv = x3.value();
        Tensor ge = xv.sub(grid.slice(2, 0, knots - 1)).add(1e-12f).step();
        Tensor lt = grid.slice(2, 1, knots).sub(xv).step();
        Var bases = Var.of(ge.mul(lt));
        float h = (high - low) / intervals;
        for (int p = 1; p <= order; p++) {
            int n = knots - 1 - p;
            Var left = x3.sub(grid.slice(2, 0, n)).mul(bases.slice(2, 0, n));
            Var right = Var.of(grid.slice(2, p + 1, p + 1 + n)).sub(x3).mul(bases.slice(2, 1, n + 1));
            bases = left.add(right).div(p * h);
        }
        return bases;
    }

    @Override
    public Var forward(Var x) {
        if (x.shape().rank() != 2 || x.shape().dim(1) != inputs) {
            throw new IllegalArgumentException("KanLayer expects [batch, " + inputs + "], got " + x.shape());
        }
        int b = x.shape().dim(0);
        Var silu = x.mul(x.sigmoid());
        Var splines = basis(x).reshape(b, inputs * basisSize()).matmul(coefficients);
        return silu.matmul(base).add(splines);
    }

    /**
     * Evaluates one edge function, base plus spline, for plotting or inspection.
     *
     * @param input the source input i
     * @param output the target output j
     * @param x the points
     * @return {@code phi_ij(x)} at each point
     */
    public float[] edge(int input, int output, float[] x) {
        float[] in = new float[x.length * inputs];
        for (int n = 0; n < x.length; n++) {
            in[n * inputs + input] = x[n];
        }
        Tensor bv = basis(Var.of(Tensor.of(Shape.of(x.length, inputs), in))).value()
                .slice(1, input, input + 1).reshape(x.length, basisSize());
        Tensor c = coefficients.value().slice(0, input * basisSize(), (input + 1) * basisSize())
                .slice(1, output, output + 1);
        float[] spline = bv.matmul(c).toArray();
        float w = base.value().slice(0, input, input + 1).slice(1, output, output + 1).item();
        float[] out = new float[x.length];
        for (int n = 0; n < x.length; n++) {
            out[n] = (float) (w * x[n] / (1 + Math.exp(-x[n])) + spline[n]);
        }
        return out;
    }

    /**
     * Grid extension: a layer with a finer grid (and possibly a new range) whose splines are
     * least-squares fits of this layer's splines at the given sample inputs. Base weights are
     * copied. The new layer has fresh parameters, so give it to the optimizer anew.
     *
     * @param newIntervals the new number of intervals
     * @param newLow the new grid start
     * @param newHigh the new grid end
     * @param samples representative inputs {@code [n, inputs]}, n well above the basis size
     * @return the refined layer
     */
    public KanLayer refine(int newIntervals, float newLow, float newHigh, Tensor samples) {
        KanLayer fine = new KanLayer(inputs, outputs, newIntervals, order, newLow, newHigh, base.value(),
                Tensor.zeros(Shape.of(inputs * (newIntervals + order), outputs)));
        int n = samples.shape().dim(0);
        int oldK = basisSize();
        int newK = fine.basisSize();
        float[] oldB = basis(Var.of(samples)).value().toArray();
        float[] newB = fine.basis(Var.of(samples)).value().toArray();
        float[] c = coefficients.value().toArray();
        float[] fitted = new float[inputs * newK * outputs];
        for (int i = 0; i < inputs; i++) {
            RidgeRegression r = RidgeRegression.of(newK, outputs);
            for (int s = 0; s < n; s++) {
                double[] feature = new double[newK];
                for (int k = 0; k < newK; k++) {
                    feature[k] = newB[(s * inputs + i) * newK + k];
                }
                double[] target = new double[outputs];
                for (int k = 0; k < oldK; k++) {
                    float bk = oldB[(s * inputs + i) * oldK + k];
                    if (bk != 0) {
                        for (int j = 0; j < outputs; j++) {
                            target[j] += bk * c[(i * oldK + k) * outputs + j];
                        }
                    }
                }
                r.add(feature, target);
            }
            double[][] w = r.solve(1e-6);
            for (int k = 0; k < newK; k++) {
                for (int j = 0; j < outputs; j++) {
                    fitted[(i * newK + k) * outputs + j] = (float) w[k][j];
                }
            }
        }
        fine.coefficients.assign(Tensor.of(Shape.of(inputs * newK, outputs), fitted));
        return fine;
    }

    @Override
    public List<Var> parameters() {
        return List.of(base, coefficients);
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }
}
