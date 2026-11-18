package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Networks that are non-decreasing in chosen inputs by construction, plus a verifier:
 *
 * <ul>
 *   <li>{@link MinMax} (Sill, NeurIPS 1998): {@code min_k max_j (w_kj · x + b_kj)} with
 *       {@code |w|} on the monotone inputs, a universal approximator of monotone functions;</li>
 *   <li>{@link Lattice} (after You et al., NeurIPS 2017,
 *       <a href="https://arxiv.org/abs/1709.06680">arXiv:1709.06680</a>): per-feature
 *       piecewise-linear calibrators into {@code [0, 1]} (increasing on the monotone features)
 *       feeding one multilinear lattice with two vertices per feature. Vertex values are
 *       {@code θ(v) = Σ g(u)} over the vertices {@code u} below {@code v} in the monotone
 *       coordinates and equal elsewhere, with {@code g(u) >= 0} whenever {@code u} has a monotone
 *       bit, so every lattice edge along a monotone feature rises (a single-layer lattice, not the
 *       paper's deep ensemble of lattices);</li>
 *   <li>{@link ConstrainedMlp} (Runje &amp; Shankaranarayana, ICML 2023,
 *       <a href="https://arxiv.org/abs/2205.11775">arXiv:2205.11775</a>): {@code |W|} on the
 *       monotone inputs and in every later layer, with hidden units split into convex
 *       {@code relu(x)}, concave {@code -relu(-x)} and saturated {@code clamp(x, -1, 1)}
 *       activations, all non-decreasing.</li>
 * </ul>
 *
 * Every layer maps {@code [n, d]} to {@code [n, 1]}. {@link #violations} counts decreases along
 * grid lines of each monotone feature.
 */
public final class MonotonicLayers {

    /** The monotone architectures. */
    public enum Kind {
        /** Sill's min-max network */
        MIN_MAX,
        /** calibrators and a multilinear lattice */
        LATTICE,
        /** constrained MLP with convex, concave and saturated units */
        CONSTRAINED_MLP
    }

    private MonotonicLayers() {
    }

    /**
     * Creates a layer of a kind with default sizes.
     *
     * @param kind the architecture
     * @param monotone which inputs the output must not decrease in
     * @param low the inputs' lower range (the lattice's calibrator keypoints)
     * @param high the inputs' upper range
     * @param seed the seed
     * @return the layer: 4 groups of 4 units, 6 calibrator keypoints, or a 32-32 MLP
     */
    public static Layer create(Kind kind, boolean[] monotone, double[] low, double[] high, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return switch (kind) {
            case MIN_MAX -> new MinMax(monotone, 4, 4, r);
            case LATTICE -> new Lattice(monotone, low, high, 6, r);
            case CONSTRAINED_MLP -> new ConstrainedMlp(monotone, new int[] {32, 32}, r);
        };
    }

    private static Tensor rowMask(boolean[] monotone, int cols) {
        float[] m = new float[monotone.length * cols];
        for (int i = 0; i < monotone.length; i++) {
            for (int j = 0; j < cols; j++) {
                m[i * cols + j] = monotone[i] ? 1 : 0;
            }
        }
        return Tensor.of(Shape.of(monotone.length, cols), m);
    }

    /** {@code w} where the mask is 0, {@code |w|} where it is 1. */
    private static Var constrain(Var w, Tensor mask) {
        return w.add(w.abs().sub(w).mul(mask));
    }

    private static Var normal(SplittableRandom r, double std, int rows, int cols) {
        float[] v = new float[rows * cols];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextGaussian() * std);
        }
        return Var.param(Tensor.of(Shape.of(rows, cols), v));
    }

    /** Sill's min-max network. */
    public static final class MinMax implements Layer {
        private final int groups;
        private final int units;
        private final Tensor mask;
        private final Var weight;
        private final Var bias;

        /**
         * Creates the network.
         *
         * @param monotone the monotone inputs
         * @param groups the groups {@code k} (the min)
         * @param units linear units per group {@code j} (the max)
         * @param random the initializer
         */
        public MinMax(boolean[] monotone, int groups, int units, SplittableRandom random) {
            this.groups = groups;
            this.units = units;
            this.mask = rowMask(monotone, groups * units);
            this.weight = normal(random, Math.sqrt(1.0 / monotone.length), monotone.length, groups * units);
            this.bias = normal(random, 0.5, 1, groups * units);
        }

        @Override
        public Var forward(Var input) {
            int n = input.shape().dim(0);
            Var z = input.matmul(constrain(weight, mask)).add(bias);
            Var max = z.reshape(n, groups, units).max(2, false);
            return max.neg().max(1, true).neg();
        }

        @Override
        public List<Var> parameters() {
            return List.of(weight, bias);
        }

        @Override
        public int outputSize(int inputSize) {
            return 1;
        }
    }

    /** Monotone piecewise-linear calibrators feeding a monotone multilinear lattice. */
    public static final class Lattice implements Layer {
        private final int d;
        private final Tensor keypoints;
        private final Tensor widths;
        private final Tensor calibratorMask;
        private final Var delta;
        private final Var offset;
        private final Tensor zeta;
        private final Tensor vertexMask;
        private final Var g;

        /**
         * Creates the layer ({@code 2^d} vertices, so keep {@code d <= 10}).
         *
         * @param monotone the monotone inputs
         * @param low the inputs' lower range
         * @param high the inputs' upper range
         * @param keypoints calibrator keypoints per input (2 or more)
         * @param random the initializer
         */
        public Lattice(boolean[] monotone, double[] low, double[] high, int keypoints, SplittableRandom random) {
            this.d = monotone.length;
            if (d > 10) {
                throw new IllegalArgumentException("a lattice over more than 10 inputs is too large");
            }
            int segments = keypoints - 1;
            float[] kp = new float[d * segments];
            float[] w = new float[d * segments];
            float[] init = new float[d * segments];
            // softplus(init) = 1 / segments, so each calibrator starts as the identity onto [0, 1]
            float raw = (float) Math.log(Math.expm1(1.0 / segments));
            for (int i = 0; i < d; i++) {
                double step = (high[i] - low[i]) / segments;
                for (int s = 0; s < segments; s++) {
                    kp[i * segments + s] = (float) (low[i] + s * step);
                    w[i * segments + s] = (float) step;
                    init[i * segments + s] = monotone[i] ? raw : 1f / segments;
                }
            }
            this.keypoints = Tensor.of(Shape.of(d, segments), kp);
            this.widths = Tensor.of(Shape.of(d, segments), w);
            this.calibratorMask = rowMask(monotone, segments);
            this.delta = Var.param(Tensor.of(Shape.of(d, segments), init));
            this.offset = Var.param(Tensor.zeros(Shape.of(d)));
            int monoBits = 0;
            for (int i = 0; i < d; i++) {
                monoBits |= monotone[i] ? 1 << i : 0;
            }
            int vertices = 1 << d;
            float[] z = new float[vertices * vertices];
            float[] vm = new float[vertices];
            float[] g0 = new float[vertices];
            for (int v = 0; v < vertices; v++) {
                vm[v] = (v & monoBits) != 0 ? 1 : 0;
                g0[v] = (float) (0.1 * random.nextGaussian());
                for (int u = 0; u < vertices; u++) {
                    boolean sameFree = (u & ~monoBits) == (v & ~monoBits);
                    boolean below = (u & monoBits & ~v) == 0;
                    z[v * vertices + u] = sameFree && below ? 1 : 0;
                }
            }
            this.zeta = Tensor.of(Shape.of(vertices, vertices), z);
            this.vertexMask = Tensor.of(Shape.of(vertices, 1), vm);
            this.g = Var.param(Tensor.of(Shape.of(vertices, 1), g0));
        }

        /** @return the calibrated inputs {@code [n, d]} in {@code [0, 1]} */
        Var calibrate(Var input) {
            int n = input.shape().dim(0);
            Var increments = delta.add(delta.softplus().sub(delta).mul(calibratorMask));
            Var seg = input.reshape(n, d, 1).sub(keypoints).div(widths).clamp(0f, 1f);
            return seg.mul(increments).sum(2, false).add(offset).clamp(0f, 1f);
        }

        @Override
        public Var forward(Var input) {
            int n = input.shape().dim(0);
            Var u = calibrate(input);
            Var weights = Var.of(Tensor.full(Shape.of(n, 1), 1f));
            for (int i = 0; i < d; i++) {
                Var ui = u.slice(1, i, i + 1);
                weights = Var.concat(1, weights.mul(ui.neg().add(1f)), weights.mul(ui));
            }
            Var gEff = g.add(g.softplus().sub(g).mul(vertexMask));
            Var theta = Var.of(zeta).matmul(gEff);
            return weights.matmul(theta);
        }

        @Override
        public List<Var> parameters() {
            return List.of(delta, offset, g);
        }

        @Override
        public int outputSize(int inputSize) {
            return 1;
        }
    }

    /** The constrained monotonic MLP. */
    public static final class ConstrainedMlp implements Layer {
        private final List<Var> weights = new ArrayList<>();
        private final List<Var> biases = new ArrayList<>();
        private final Tensor firstMask;
        private final List<Tensor[]> activationMasks = new ArrayList<>();

        /**
         * Creates the network.
         *
         * @param monotone the monotone inputs
         * @param hidden hidden widths
         * @param random the initializer
         */
        public ConstrainedMlp(boolean[] monotone, int[] hidden, SplittableRandom random) {
            int in = monotone.length;
            this.firstMask = rowMask(monotone, hidden[0]);
            for (int h : hidden) {
                weights.add(normal(random, Math.sqrt(1.0 / in), in, h));
                biases.add(Var.param(Tensor.zeros(Shape.of(h))));
                float[][] m = new float[3][h];
                for (int j = 0; j < h; j++) {
                    m[j * 3 / h][j] = 1;
                }
                activationMasks.add(new Tensor[] {Tensor.of(Shape.of(h), m[0]), Tensor.of(Shape.of(h), m[1]),
                    Tensor.of(Shape.of(h), m[2])});
                in = h;
            }
            weights.add(normal(random, Math.sqrt(1.0 / in), in, 1));
            biases.add(Var.param(Tensor.zeros(Shape.of(1))));
        }

        @Override
        public Var forward(Var input) {
            Var x = input;
            for (int l = 0; l < weights.size(); l++) {
                Var w = weights.get(l);
                Var eff = l == 0 ? constrain(w, firstMask) : w.abs();
                x = x.matmul(eff).add(biases.get(l));
                if (l < activationMasks.size()) {
                    Tensor[] m = activationMasks.get(l);
                    x = x.relu().mul(m[0]).add(x.neg().relu().neg().mul(m[1])).add(x.clamp(-1f, 1f).mul(m[2]));
                }
            }
            return x;
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(weights);
            p.addAll(biases);
            return p;
        }

        @Override
        public int outputSize(int inputSize) {
            return 1;
        }
    }

    /**
     * The share of monotonicity violations: from random base points in the box, each monotone
     * feature is swept over a grid with the others fixed, and every adjacent pair whose output
     * drops by more than {@code 1e-5} counts.
     *
     * @param f the model, rows to outputs
     * @param low the box's lower corner
     * @param high the box's upper corner
     * @param monotone the features to check
     * @param grid grid points per sweep
     * @param points base points
     * @param seed the seed
     * @return violating pairs over checked pairs
     */
    public static double violations(Function<double[][], double[]> f, double[] low, double[] high,
            boolean[] monotone, int grid, int points, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = low.length;
        long bad = 0;
        long pairs = 0;
        for (int p = 0; p < points; p++) {
            double[] base = new double[d];
            for (int i = 0; i < d; i++) {
                base[i] = low[i] + (high[i] - low[i]) * r.nextDouble();
            }
            for (int i = 0; i < d; i++) {
                if (!monotone[i]) {
                    continue;
                }
                double[][] sweep = new double[grid][];
                for (int g = 0; g < grid; g++) {
                    sweep[g] = base.clone();
                    sweep[g][i] = low[i] + (high[i] - low[i]) * g / (grid - 1);
                }
                double[] y = f.apply(sweep);
                for (int g = 1; g < grid; g++) {
                    bad += y[g] < y[g - 1] - 1e-5 ? 1 : 0;
                    pairs++;
                }
            }
        }
        return pairs == 0 ? 0 : bad / (double) pairs;
    }

    /**
     * A layer's outputs on rows, without gradients.
     *
     * @param layer a layer with one output
     * @param x rows
     * @return the outputs
     */
    public static double[] predict(Layer layer, double[][] x) {
        float[] v = new float[x.length * x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                v[i * x[0].length + j] = (float) x[i][j];
            }
        }
        float[] y = layer.forward(Var.of(Tensor.of(Shape.of(x.length, x[0].length), v))).value().toArray();
        double[] out = new double[x.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = y[i];
        }
        return out;
    }
}
