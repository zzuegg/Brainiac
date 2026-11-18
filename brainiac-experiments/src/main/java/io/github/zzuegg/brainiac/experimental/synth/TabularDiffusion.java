package io.github.zzuegg.brainiac.experimental.synth;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * TabDDPM (Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models", ICML 2023,
 * <a href="https://arxiv.org/abs/2209.15421">arXiv:2209.15421</a>): Gaussian diffusion for the
 * numeric columns and multinomial diffusion (Hoogeboom et al., 2021) for the categorical ones,
 * with one MLP denoiser over all columns.
 * <ul>
 *   <li>numeric columns are standardised (TabDDPM uses a quantile transform) and noised as
 *       {@code x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε}; the network predicts {@code ε};</li>
 *   <li>a categorical column with {@code K} categories is noised as
 *       {@code q(x_t | x_0) = Cat(ᾱ_t x_0 + (1 - ᾱ_t)/K)}; the network predicts {@code x_0}'s
 *       distribution, trained by cross-entropy (an {@code x_0}-parameterised simplification of
 *       TabDDPM's KL term), and sampling uses the exact posterior
 *       {@code q(x_{t-1} | x_t, x̂_0) ∝ [α_t x_t + (1-α_t)/K] ⊙ [ᾱ_{t-1} x̂_0 + (1-ᾱ_{t-1})/K]};</li>
 *   <li>the denoiser reads the noisy numeric values, the noisy one-hot categories and a
 *       sinusoidal embedding of {@code t}; linear β from 1e-4 to 0.02.</li>
 * </ul>
 */
public final class TabularDiffusion {

    private static final int TIME_FEATURES = 8;

    private final int steps;
    private final int[] categories;
    private final int numeric;
    private final int oneHot;
    private final double[] beta;
    private final double[] alphaBar;
    private final SimpleMlp net;
    private final Optimizer optimizer;
    private double[][] stats;

    /**
     * Creates an untrained model.
     *
     * @param numeric numeric columns
     * @param categories categories per categorical column
     * @param steps diffusion steps, for example 100
     * @param hidden the denoiser width
     * @param lr the Adam rate
     * @param seed the initialisation seed
     */
    public TabularDiffusion(int numeric, int[] categories, int steps, int hidden, float lr, long seed) {
        this.steps = steps;
        this.categories = categories.clone();
        this.numeric = numeric;
        int k = 0;
        for (int c : categories) {
            k += c;
        }
        this.oneHot = k;
        this.beta = new double[steps + 1];
        this.alphaBar = new double[steps + 1];
        alphaBar[0] = 1;
        for (int t = 1; t <= steps; t++) {
            beta[t] = 1e-4 + (0.02 - 1e-4) * (t - 1) / Math.max(1, steps - 1);
            alphaBar[t] = alphaBar[t - 1] * (1 - beta[t]);
        }
        this.net = SimpleMlp.relu(seed, numeric + oneHot + TIME_FEATURES, hidden, hidden, numeric + oneHot);
        this.optimizer = Optimizer.adam(lr);
    }

    private float[] timeFeatures(int t) {
        float[] f = new float[TIME_FEATURES];
        double x = t / (double) steps;
        for (int i = 0; i < TIME_FEATURES / 2; i++) {
            double w = Math.pow(10, i) * Math.PI;
            f[2 * i] = (float) Math.sin(w * x);
            f[2 * i + 1] = (float) Math.cos(w * x);
        }
        return f;
    }

    private int sampleCategory(double[] p, SplittableRandom r) {
        double u = r.nextDouble();
        for (int k = 0; k < p.length - 1; k++) {
            u -= p[k];
            if (u < 0) {
                return k;
            }
        }
        return p.length - 1;
    }

    /**
     * Trains on a table.
     *
     * @param table the real table
     * @param iterations Adam steps
     * @param batch rows per step
     * @param seed the seed
     * @return the final loss
     */
    public double fit(MixedTable table, int iterations, int batch, long seed) {
        stats = table.numericStats();
        SplittableRandom r = new SplittableRandom(seed);
        int width = numeric + oneHot + TIME_FEATURES;
        double last = 0;
        for (int it = 0; it < iterations; it++) {
            float[] in = new float[batch * width];
            float[] eps = new float[batch * numeric];
            float[] targetMask = new float[batch * oneHot];
            for (int b = 0; b < batch; b++) {
                int i = r.nextInt(table.rows());
                int t = 1 + r.nextInt(steps);
                double sa = Math.sqrt(alphaBar[t]);
                double sn = Math.sqrt(1 - alphaBar[t]);
                for (int j = 0; j < numeric; j++) {
                    double e = r.nextGaussian();
                    eps[b * numeric + j] = (float) e;
                    double x0 = (table.numeric()[i][j] - stats[0][j]) / stats[1][j];
                    in[b * width + j] = (float) (sa * x0 + sn * e);
                }
                int off = 0;
                for (int c = 0; c < categories.length; c++) {
                    int k = categories[c];
                    int x0 = table.categorical()[i][c];
                    double[] p = new double[k];
                    for (int q = 0; q < k; q++) {
                        p[q] = (1 - alphaBar[t]) / k + (q == x0 ? alphaBar[t] : 0);
                    }
                    in[b * width + numeric + off + sampleCategory(p, r)] = 1;
                    targetMask[b * oneHot + off + x0] = 1;
                    off += k;
                }
                System.arraycopy(timeFeatures(t), 0, in, b * width + numeric + oneHot, TIME_FEATURES);
            }
            Var out = net.forward(Var.of(Tensor.of(Shape.of(batch, width), in)));
            Var loss = out.slice(1, 0, numeric).sub(Var.of(Tensor.of(Shape.of(batch, numeric), eps))).square().mean();
            int off = 0;
            Tensor mask = Tensor.of(Shape.of(batch, oneHot), targetMask);
            for (int k : categories) {
                Var logits = out.slice(1, numeric + off, numeric + off + k);
                Var picked = logits.mul(Var.of(mask.slice(1, off, off + k))).sum(1, false);
                loss = loss.add(logits.logsumexp(1, false).sub(picked).mean().div(categories.length));
                off += k;
            }
            loss.backward();
            optimizer.step(net.parameters());
            last = loss.value().item();
        }
        return last;
    }

    /**
     * Samples rows by ancestral sampling.
     *
     * @param rows rows to generate
     * @param seed the seed
     * @return the synthetic table
     */
    public MixedTable sample(int rows, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int width = numeric + oneHot + TIME_FEATURES;
        double[][] x = new double[rows][numeric];
        int[][] cat = new int[rows][categories.length];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < numeric; j++) {
                x[i][j] = r.nextGaussian();
            }
            for (int c = 0; c < categories.length; c++) {
                cat[i][c] = r.nextInt(categories[c]);
            }
        }
        for (int t = steps; t >= 1; t--) {
            float[] in = new float[rows * width];
            float[] tf = timeFeatures(t);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < numeric; j++) {
                    in[i * width + j] = (float) x[i][j];
                }
                int off = 0;
                for (int c = 0; c < categories.length; c++) {
                    in[i * width + numeric + off + cat[i][c]] = 1;
                    off += categories[c];
                }
                System.arraycopy(tf, 0, in, i * width + numeric + oneHot, TIME_FEATURES);
            }
            float[] out = net.predict(Tensor.of(Shape.of(rows, width), in)).toArray();
            double alpha = 1 - beta[t];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < numeric; j++) {
                    double e = out[i * (numeric + oneHot) + j];
                    double mean = (x[i][j] - beta[t] / Math.sqrt(1 - alphaBar[t]) * e) / Math.sqrt(alpha);
                    double var = t > 1 ? beta[t] * (1 - alphaBar[t - 1]) / (1 - alphaBar[t]) : 0;
                    x[i][j] = mean + Math.sqrt(var) * r.nextGaussian();
                }
                int off = 0;
                for (int c = 0; c < categories.length; c++) {
                    int k = categories[c];
                    double max = Double.NEGATIVE_INFINITY;
                    for (int q = 0; q < k; q++) {
                        max = Math.max(max, out[i * (numeric + oneHot) + numeric + off + q]);
                    }
                    double[] p0 = new double[k];
                    double s0 = 0;
                    for (int q = 0; q < k; q++) {
                        p0[q] = Math.exp(out[i * (numeric + oneHot) + numeric + off + q] - max);
                        s0 += p0[q];
                    }
                    double[] post = new double[k];
                    double sp = 0;
                    for (int q = 0; q < k; q++) {
                        double fromT = alpha * (q == cat[i][c] ? 1 : 0) + (1 - alpha) / k;
                        double fromZero = alphaBar[t - 1] * p0[q] / s0 + (1 - alphaBar[t - 1]) / k;
                        post[q] = fromT * fromZero;
                        sp += post[q];
                    }
                    for (int q = 0; q < k; q++) {
                        post[q] /= sp;
                    }
                    cat[i][c] = t > 1 ? sampleCategory(post, r) : argmax(p0);
                    off += k;
                }
            }
        }
        for (double[] row : x) {
            for (int j = 0; j < numeric; j++) {
                row[j] = row[j] * stats[1][j] + stats[0][j];
            }
        }
        return new MixedTable(x, cat, categories.clone());
    }

    private static int argmax(double[] p) {
        int b = 0;
        for (int i = 1; i < p.length; i++) {
            b = p[i] > p[b] ? i : b;
        }
        return b;
    }

    /** @return the denoiser's parameters */
    public List<Var> parameters() {
        return new ArrayList<>(net.parameters());
    }

    @Override
    public String toString() {
        return "TabularDiffusion(" + numeric + " numeric, " + categories.length + " categorical, " + steps + " steps)";
    }
}
