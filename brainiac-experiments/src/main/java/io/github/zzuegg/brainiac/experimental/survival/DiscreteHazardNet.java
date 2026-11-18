package io.github.zzuegg.brainiac.experimental.survival;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A discrete-time neural survival model (Gensheimer and Narasimhan, PeerJ 2019,
 * <a href="https://arxiv.org/abs/1805.00917">arXiv:1805.00917</a>): time is cut into bins and an
 * MLP predicts, per bin, the conditional hazard {@code h_k(x) = P(event in bin k | alive at its
 * start)} through a sigmoid. The likelihood uses every unit up to its own bin: survival through the
 * earlier bins, the event in its bin if it had one, and survival through its bin if it was censored
 * in the bin's second half. Nothing ties the bins together, so the hazards of two units may cross
 * freely, unlike under proportional hazards.
 *
 * <p>As a {@link Layer} it maps {@code [batch, features]} to {@code [batch, bins]} hazard logits;
 * {@link #loss} is the negative log-likelihood and {@link #survival} the survival curve
 * {@code S(t) = prod over bins before t of (1 - h_k)}, interpolated inside a bin.
 */
public final class DiscreteHazardNet implements Layer {

    private final double[] cuts;
    private final Model network;

    /**
     * Creates the network.
     *
     * @param features input covariates
     * @param hidden the width of its two hidden layers
     * @param cuts bin edges {@code 0 = c_0 < c_1 < ... < c_B}; times past {@code c_B} fall in the last bin
     * @param seed the seed
     */
    public DiscreteHazardNet(int features, int hidden, double[] cuts, long seed) {
        this.cuts = cuts.clone();
        this.network = Model.sequential().input(features).dense(hidden, Activation.RELU)
                .dense(hidden, Activation.RELU).dense(cuts.length - 1).seed(seed).build();
    }

    /**
     * @param features input covariates
     * @param hidden hidden width
     * @param cuts bin edges
     * @return a spec for model builders
     */
    public static LayerSpec spec(int features, int hidden, double[] cuts) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(cuts.length - 1);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new DiscreteHazardNet(input.dim(-1), hidden, cuts, random.nextLong());
            }
        };
    }

    /**
     * Equal-count bin edges from the observed event times.
     *
     * @param time observed times
     * @param bins the number of bins
     * @return {@code bins + 1} edges starting at 0
     */
    public static double[] quantileCuts(double[] time, int bins) {
        double[] s = time.clone();
        java.util.Arrays.sort(s);
        double[] c = new double[bins + 1];
        for (int k = 1; k < bins; k++) {
            c[k] = s[(int) ((long) k * (s.length - 1) / bins)];
        }
        c[bins] = s[s.length - 1] * (1 + 1e-9) + 1e-9;
        return c;
    }

    /**
     * @param t a time
     * @return its bin index
     */
    public int bin(double t) {
        int bins = cuts.length - 1;
        for (int k = 0; k < bins; k++) {
            if (t < cuts[k + 1]) {
                return k;
            }
        }
        return bins - 1;
    }

    /** @return the bin edges */
    public double[] cuts() {
        return cuts.clone();
    }

    @Override
    public Var forward(Var input) {
        return network.forward(input);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return network.forward(input, training);
    }

    @Override
    public List<Var> parameters() {
        return network.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return cuts.length - 1;
    }

    /**
     * The mean negative log-likelihood of a batch.
     *
     * @param logits {@code [batch, bins]} hazard logits from {@link #forward}
     * @param time observed times
     * @param event 0 if censored
     * @return the scalar loss
     */
    public Var loss(Var logits, double[] time, int[] event) {
        int n = time.length;
        int bins = cuts.length - 1;
        float[] survive = new float[n * bins];
        float[] fail = new float[n * bins];
        for (int i = 0; i < n; i++) {
            int j = bin(time[i]);
            for (int k = 0; k < j; k++) {
                survive[i * bins + k] = 1;
            }
            if (event[i] != 0) {
                fail[i * bins + j] = 1;
            } else if (time[i] >= (cuts[j] + cuts[j + 1]) / 2) {
                survive[i * bins + j] = 1;
            }
        }
        Shape s = Shape.of(n, bins);
        // -log(1 - h) = softplus(l), -log h = softplus(-l)
        return logits.softplus().mul(Tensor.of(s, survive)).add(logits.neg().softplus().mul(Tensor.of(s, fail))).sum()
                .div(n);
    }

    /**
     * Trains by Adam on random mini-batches.
     *
     * @param x covariates
     * @param time observed times
     * @param event 0 if censored
     * @param steps Adam steps
     * @param batch rows per step
     * @param lr the learning rate
     * @param seed the batch seed
     * @return this
     */
    public DiscreteHazardNet fit(double[][] x, double[] time, int[] event, int steps, int batch, float lr,
            long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int d = x[0].length;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[batch * d];
            double[] bt = new double[batch];
            int[] be = new int[batch];
            for (int i = 0; i < batch; i++) {
                int k = r.nextInt(x.length);
                for (int j = 0; j < d; j++) {
                    bx[i * d + j] = (float) x[k][j];
                }
                bt[i] = time[k];
                be[i] = event[k];
            }
            loss(forward(Var.of(Tensor.of(Shape.of(batch, d), bx)), true), bt, be).backward();
            opt.step(parameters());
        }
        return this;
    }

    /**
     * Predicted survival curves.
     *
     * @param x covariates
     * @param grid times
     * @return {@code [n][grid]} values of {@code S(t | x)}
     */
    public double[][] survival(double[][] x, double[] grid) {
        int n = x.length;
        int d = x[0].length;
        int bins = cuts.length - 1;
        float[] f = new float[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                f[i * d + j] = (float) x[i][j];
            }
        }
        float[] logits = Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(n, d), f)), false)).value().toArray();
        double[][] out = new double[n][grid.length];
        for (int i = 0; i < n; i++) {
            for (int g = 0; g < grid.length; g++) {
                double t = grid[g];
                int j = bin(t);
                double s = 1;
                for (int k = 0; k < j; k++) {
                    s *= 1 - sigmoid(logits[i * bins + k]);
                }
                double frac = Math.clamp((t - cuts[j]) / (cuts[j + 1] - cuts[j]), 0, 1);
                s *= Math.pow(1 - sigmoid(logits[i * bins + j]), frac);
                out[i][g] = s;
            }
        }
        return out;
    }

    private static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    @Override
    public String toString() {
        return "DiscreteHazardNet(" + (cuts.length - 1) + " bins)";
    }
}
