package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The fixed random recurrent core of an echo state network (Jaeger, 2001; Jaeger and Haas,
 * <a href="https://doi.org/10.1126/science.1091277">Science 2004</a>): a sparse random
 * matrix rescaled to a chosen spectral radius, driven by the input through random input
 * weights, with leaky tanh units:
 *
 * <pre>
 * h_t = (1 - a) h_(t-1) + a tanh(W h_(t-1) + W_in x_t + b)
 * </pre>
 *
 * <p>Nothing here is trained: {@link #parameters()} is empty and a linear readout on the
 * states (for example {@code experimental.train.RidgeRegression}) does all the learning.
 * As a {@link Layer} it maps {@code [batch, steps, inputs]} to the states
 * {@code [batch, steps, units]} from a zero start; {@link #step} runs it one input at a
 * time in double precision, for closed-loop forecasting.
 */
public final class EchoStateReservoir implements Layer {
    private final int inputs;
    private final int units;
    private final double leak;
    private final double[][] recurrent;
    private final double[][] input;
    private final double[] bias;
    private final double[] state;

    private EchoStateReservoir(int inputs, int units, double spectralRadius, double inputScale, double leak,
            double density, SplittableRandom random) {
        this.inputs = inputs;
        this.units = units;
        this.leak = leak;
        this.recurrent = new double[units][units];
        for (int i = 0; i < units; i++) {
            for (int j = 0; j < units; j++) {
                if (random.nextDouble() < density) {
                    recurrent[i][j] = random.nextDouble(-1, 1);
                }
            }
        }
        double radius = spectralRadius(recurrent, random);
        for (double[] row : recurrent) {
            for (int j = 0; j < units; j++) {
                row[j] *= spectralRadius / radius;
            }
        }
        this.input = new double[units][inputs];
        this.bias = new double[units];
        for (int i = 0; i < units; i++) {
            for (int j = 0; j < inputs; j++) {
                input[i][j] = random.nextDouble(-inputScale, inputScale);
            }
            bias[i] = random.nextDouble(-inputScale, inputScale);
        }
        this.state = new double[units];
    }

    /**
     * Creates a reservoir.
     *
     * @param inputs the input width
     * @param units the reservoir size
     * @param spectralRadius the largest eigenvalue magnitude of the recurrent matrix, typically near 1
     * @param inputScale the range of the uniform input weights and biases
     * @param leak the leak rate a in (0, 1]
     * @param density the share of nonzero recurrent weights
     * @param seed the seed
     * @return the reservoir
     */
    public static EchoStateReservoir of(int inputs, int units, double spectralRadius, double inputScale, double leak,
            double density, long seed) {
        if (!(leak > 0 && leak <= 1) || !(density > 0 && density <= 1)) {
            throw new IllegalArgumentException("need 0 < leak <= 1 and 0 < density <= 1");
        }
        return new EchoStateReservoir(inputs, units, spectralRadius, inputScale, leak, density,
                new SplittableRandom(seed));
    }

    /**
     * Describes a reservoir for model builders; input {@code [steps, inputs]}.
     *
     * @param units the reservoir size
     * @param spectralRadius the spectral radius
     * @param inputScale the input weight range
     * @param leak the leak rate
     * @return the spec
     */
    public static LayerSpec spec(int units, double spectralRadius, double inputScale, double leak) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape in) {
                return Shape.of(in.dim(0), units);
            }

            @Override
            public Layer create(Shape in, SplittableRandom random) {
                return new EchoStateReservoir(in.dim(-1), units, spectralRadius, inputScale, leak,
                        Math.min(1, 10.0 / units), random);
            }
        };
    }

    /**
     * Estimates the spectral radius by power iteration on a complex-safe iterate: the growth
     * rate of {@code ||W^k v||} over many steps converges to the largest |eigenvalue| even when
     * that eigenvalue is complex.
     */
    private static double spectralRadius(double[][] w, SplittableRandom random) {
        int n = w.length;
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = random.nextDouble(-1, 1);
        }
        double logGrowth = 0;
        int iterations = 300;
        int burn = 100;
        for (int it = 0; it < iterations; it++) {
            double[] next = new double[n];
            for (int i = 0; i < n; i++) {
                double s = 0;
                double[] row = w[i];
                for (int j = 0; j < n; j++) {
                    s += row[j] * v[j];
                }
                next[i] = s;
            }
            double norm = 0;
            for (double x : next) {
                norm += x * x;
            }
            norm = Math.sqrt(norm);
            if (norm == 0) {
                return 1e-12;
            }
            if (it >= burn) {
                logGrowth += Math.log(norm);
            }
            for (int i = 0; i < n; i++) {
                v[i] = next[i] / norm;
            }
        }
        return Math.exp(logGrowth / (iterations - burn));
    }

    /** @return the reservoir size */
    public int units() {
        return units;
    }

    /** Sets the state to zero. */
    public void reset() {
        java.util.Arrays.fill(state, 0);
    }

    /**
     * Advances one step.
     *
     * @param x the input
     * @return a copy of the new state
     */
    public double[] step(double[] x) {
        double[] pre = new double[units];
        for (int i = 0; i < units; i++) {
            double s = bias[i];
            double[] row = recurrent[i];
            for (int j = 0; j < units; j++) {
                s += row[j] * state[j];
            }
            for (int j = 0; j < inputs; j++) {
                s += input[i][j] * x[j];
            }
            pre[i] = s;
        }
        for (int i = 0; i < units; i++) {
            state[i] = (1 - leak) * state[i] + leak * Math.tanh(pre[i]);
        }
        return state.clone();
    }

    private Tensor matrix(double[][] m, boolean transpose, Tensor like) {
        int r = transpose ? m[0].length : m.length;
        int c = transpose ? m.length : m[0].length;
        float[] f = new float[r * c];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++) {
                f[transpose ? j * c + i : i * c + j] = (float) m[i][j];
            }
        }
        return Tensor.of(Shape.of(r, c), f, like.device());
    }

    @Override
    public Var forward(Var x) {
        Shape s = x.shape();
        if (s.rank() != 3 || s.dim(2) != inputs) {
            throw new IllegalArgumentException("EchoStateReservoir expects [batch, steps, " + inputs + "], got " + s);
        }
        int b = s.dim(0);
        Var win = Var.of(matrix(input, true, x.value()));
        Var w = Var.of(matrix(recurrent, true, x.value()));
        float[] bf = new float[units];
        for (int i = 0; i < units; i++) {
            bf[i] = (float) bias[i];
        }
        Var drive = x.matmul(win).add(Var.of(Tensor.of(Shape.of(units), bf, x.value().device())));
        Var h = Var.of(Tensor.zeros(Shape.of(b, units), x.value().device()));
        List<Var> out = new ArrayList<>();
        for (int t = 0; t < s.dim(1); t++) {
            Var pre = h.matmul(w).add(drive.slice(1, t, t + 1).reshape(b, units));
            h = h.mul((float) (1 - leak)).add(pre.tanh().mul((float) leak));
            out.add(h.reshape(b, 1, units));
        }
        return Var.concat(1, out.toArray(Var[]::new));
    }

    @Override
    public List<Var> parameters() {
        return List.of();
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }
}
