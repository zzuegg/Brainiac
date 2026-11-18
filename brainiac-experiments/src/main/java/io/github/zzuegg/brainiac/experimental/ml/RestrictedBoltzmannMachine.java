package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * A restricted Boltzmann machine: a bipartite Boltzmann machine whose hidden units are
 * conditionally independent given the visible ones and vice versa, so block Gibbs sampling is
 * cheap. Energy {@code E(v, h) = -a·v - b·h - vᵀ W h} for Bernoulli visibles, and
 * {@code ½|v - a|² - b·h - vᵀ W h} for unit-variance Gaussian visibles.
 *
 * <ul>
 *   <li>{@link Training#CD_K}: contrastive divergence, the negative phase from {@code k} Gibbs
 *       steps started at the data (Hinton, Neural Computation 2002);</li>
 *   <li>{@link Training#PCD}: persistent CD, negative chains kept between updates (Tieleman,
 *       ICML 2008);</li>
 *   <li>{@link Training#PARALLEL_TEMPERING}: persistent chains at several inverse temperatures
 *       {@code β ∈ (0, 1]} with swaps between neighbours, so the {@code β = 1} chain that feeds
 *       the negative phase mixes between modes (Desjardins et al., AISTATS 2010);</li>
 *   <li>{@link #aisLogPartition}: annealed importance sampling of {@code log Z} from a base RBM
 *       with the data's visible biases and no couplings (Salakhutdinov &amp; Murray, ICML 2008),
 *       on {@link AnnealedImportanceSampler#sigmoidSchedule}; {@link #exactLogPartition} enumerates
 *       the hidden states of small machines, to check it.</li>
 * </ul>
 */
public final class RestrictedBoltzmannMachine {

    /** The visible units. */
    public enum Visible {
        /** binary visibles */
        BERNOULLI,
        /** unit-variance Gaussian visibles */
        GAUSSIAN
    }

    /** The training algorithm. */
    public enum Training {
        /** contrastive divergence */
        CD_K,
        /** persistent contrastive divergence */
        PCD,
        /** persistent chains with parallel tempering */
        PARALLEL_TEMPERING
    }

    /**
     * Training settings.
     *
     * @param training the algorithm
     * @param k Gibbs steps per update
     * @param chains persistent chains (per temperature)
     * @param temperatures the number of tempered chains for parallel tempering
     * @param learningRate the step size
     * @param batch rows per update
     * @param seed the seed
     */
    public record Settings(Training training, int k, int chains, int temperatures, double learningRate, int batch,
            long seed) {
    }

    private final Visible visible;
    private final int nv;
    private final int nh;
    private final double[][] w;
    private final double[] a;
    private final double[] b;

    private RestrictedBoltzmannMachine(Visible visible, int nv, int nh, long seed) {
        this.visible = visible;
        this.nv = nv;
        this.nh = nh;
        this.w = new double[nv][nh];
        this.a = new double[nv];
        this.b = new double[nh];
        SplittableRandom r = new SplittableRandom(seed);
        for (double[] row : w) {
            for (int j = 0; j < nh; j++) {
                row[j] = 0.01 * r.nextGaussian();
            }
        }
    }

    /**
     * Creates a machine with small random couplings and zero biases.
     *
     * @param visible the visible type
     * @param visibleUnits visible units
     * @param hiddenUnits hidden units
     * @param seed the seed
     * @return the machine
     */
    public static RestrictedBoltzmannMachine create(Visible visible, int visibleUnits, int hiddenUnits, long seed) {
        return new RestrictedBoltzmannMachine(visible, visibleUnits, hiddenUnits, seed);
    }

    /** @return the number of visible units */
    public int visibleUnits() {
        return nv;
    }

    /** @return the number of hidden units */
    public int hiddenUnits() {
        return nh;
    }

    static double sigmoid(double z) {
        return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
    }

    static double softplus(double z) {
        return Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
    }

    /**
     * @param v a visible state
     * @param beta the inverse temperature
     * @return {@code p(h_j = 1 | v)} at {@code β}
     */
    public double[] hiddenProbabilities(double[] v, double beta) {
        double[] p = new double[nh];
        for (int j = 0; j < nh; j++) {
            double s = b[j];
            for (int i = 0; i < nv; i++) {
                s += v[i] * w[i][j];
            }
            p[j] = sigmoid(beta * s);
        }
        return p;
    }

    /**
     * @param v a visible state
     * @return {@code p(h = 1 | v)}, the hidden features
     */
    public double[] hiddenProbabilities(double[] v) {
        return hiddenProbabilities(v, 1);
    }

    private double[] visibleMean(double[] h, double beta, double[] baseBias) {
        double[] m = new double[nv];
        for (int i = 0; i < nv; i++) {
            double s = a[i];
            for (int j = 0; j < nh; j++) {
                s += w[i][j] * h[j];
            }
            m[i] = visible == Visible.BERNOULLI
                    ? sigmoid(beta * s + (baseBias == null ? 0 : (1 - beta) * baseBias[i])) : beta * s;
        }
        return m;
    }

    private double[] sampleBernoulli(double[] p, SplittableRandom r) {
        double[] s = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            s[i] = r.nextDouble() < p[i] ? 1 : 0;
        }
        return s;
    }

    private double[] sampleVisible(double[] h, double beta, double[] baseBias, SplittableRandom r) {
        double[] m = visibleMean(h, beta, baseBias);
        if (visible == Visible.BERNOULLI) {
            return sampleBernoulli(m, r);
        }
        double sd = 1 / Math.sqrt(Math.max(beta, 1e-3));
        for (int i = 0; i < nv; i++) {
            m[i] = m[i] / Math.max(beta, 1e-3) + sd * r.nextGaussian();
        }
        return m;
    }

    private double[] gibbs(double[] v, double beta, SplittableRandom r) {
        return sampleVisible(sampleBernoulli(hiddenProbabilities(v, beta), r), beta, null, r);
    }

    /**
     * The free energy {@code F(v) = -log Σ_h exp(-E(v, h))}.
     *
     * @param v a visible state
     * @return {@code F(v)}
     */
    public double freeEnergy(double[] v) {
        return freeEnergy(v, 1);
    }

    private double freeEnergy(double[] v, double beta) {
        double f = 0;
        for (int i = 0; i < nv; i++) {
            f += visible == Visible.BERNOULLI ? -a[i] * v[i] : 0.5 * (v[i] - a[i]) * (v[i] - a[i]);
        }
        f *= beta;
        for (int j = 0; j < nh; j++) {
            double s = b[j];
            for (int i = 0; i < nv; i++) {
                s += v[i] * w[i][j];
            }
            f -= softplus(beta * s);
        }
        return f;
    }

    /**
     * Trains for a number of updates.
     *
     * @param data rows (0/1 for Bernoulli visibles)
     * @param settings the settings
     * @param updates the number of parameter updates
     */
    public void train(double[][] data, Settings settings, int updates) {
        SplittableRandom r = new SplittableRandom(settings.seed());
        int temps = settings.training() == Training.PARALLEL_TEMPERING ? Math.max(2, settings.temperatures()) : 1;
        double[] betas = new double[temps];
        for (int t = 0; t < temps; t++) {
            betas[t] = 1 - t / (double) temps;   // 1, ..., 1/temps
        }
        double[][][] chains = new double[temps][settings.chains()][];
        for (int t = 0; t < temps; t++) {
            for (int c = 0; c < settings.chains(); c++) {
                chains[t][c] = data[r.nextInt(data.length)].clone();
            }
        }
        double[][] gw = new double[nv][nh];
        double[] ga = new double[nv];
        double[] gb = new double[nh];
        for (int u = 0; u < updates; u++) {
            for (double[] row : gw) {
                java.util.Arrays.fill(row, 0);
            }
            java.util.Arrays.fill(ga, 0);
            java.util.Arrays.fill(gb, 0);
            int bs = settings.batch();
            for (int s = 0; s < bs; s++) {
                double[] v = data[r.nextInt(data.length)];
                accumulate(v, hiddenProbabilities(v), gw, ga, gb, 1.0 / bs);
                if (settings.training() == Training.CD_K) {
                    double[] neg = v;
                    for (int k = 0; k < settings.k(); k++) {
                        neg = gibbs(neg, 1, r);
                    }
                    accumulate(neg, hiddenProbabilities(neg), gw, ga, gb, -1.0 / bs);
                }
            }
            if (settings.training() != Training.CD_K) {
                for (int t = 0; t < temps; t++) {
                    for (int c = 0; c < settings.chains(); c++) {
                        for (int k = 0; k < settings.k(); k++) {
                            chains[t][c] = gibbs(chains[t][c], betas[t], r);
                        }
                    }
                }
                if (temps > 1) {
                    for (int c = 0; c < settings.chains(); c++) {
                        for (int t = 0; t + 1 < temps; t++) {
                            double[] x = chains[t][c];
                            double[] y = chains[t + 1][c];
                            double logAccept = freeEnergy(x, betas[t]) + freeEnergy(y, betas[t + 1])
                                    - freeEnergy(y, betas[t]) - freeEnergy(x, betas[t + 1]);
                            if (Math.log(r.nextDouble()) < logAccept) {
                                chains[t][c] = y;
                                chains[t + 1][c] = x;
                            }
                        }
                    }
                }
                for (double[] neg : chains[0]) {
                    accumulate(neg, hiddenProbabilities(neg), gw, ga, gb, -1.0 / settings.chains());
                }
            }
            double lr = settings.learningRate();
            for (int i = 0; i < nv; i++) {
                a[i] += lr * ga[i];
                for (int j = 0; j < nh; j++) {
                    w[i][j] += lr * gw[i][j];
                }
            }
            for (int j = 0; j < nh; j++) {
                b[j] += lr * gb[j];
            }
        }
    }

    private void accumulate(double[] v, double[] ph, double[][] gw, double[] ga, double[] gb, double scale) {
        for (int i = 0; i < nv; i++) {
            ga[i] += scale * (visible == Visible.BERNOULLI ? v[i] : v[i] - a[i]);
            if (v[i] != 0) {
                for (int j = 0; j < nh; j++) {
                    gw[i][j] += scale * v[i] * ph[j];
                }
            }
        }
        for (int j = 0; j < nh; j++) {
            gb[j] += scale * ph[j];
        }
    }

    /**
     * Samples by running independent Gibbs chains from uniform (or standard normal) noise.
     *
     * @param n the number of samples
     * @param sweeps Gibbs sweeps per chain
     * @param seed the seed
     * @return the samples
     */
    public double[][] sample(int n, int sweeps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] out = new double[n][];
        for (int s = 0; s < n; s++) {
            double[] v = new double[nv];
            for (int i = 0; i < nv; i++) {
                v[i] = visible == Visible.BERNOULLI ? (r.nextBoolean() ? 1 : 0) : r.nextGaussian();
            }
            for (int k = 0; k < sweeps; k++) {
                v = gibbs(v, 1, r);
            }
            out[s] = v;
        }
        return out;
    }

    /**
     * The exact {@code log Z} by enumerating the hidden states ({@code hidden <= 22}).
     *
     * @return {@code log Z}
     */
    public double exactLogPartition() {
        if (nh > 22) {
            throw new IllegalStateException("too many hidden units to enumerate: " + nh);
        }
        int states = 1 << nh;
        double[] terms = new double[states];
        double[] h = new double[nh];
        for (int s = 0; s < states; s++) {
            double t = 0;
            for (int j = 0; j < nh; j++) {
                h[j] = (s >> j) & 1;
                t += b[j] * h[j];
            }
            double sq = 0;
            for (int i = 0; i < nv; i++) {
                double z = a[i];
                for (int j = 0; j < nh; j++) {
                    z += w[i][j] * h[j];
                }
                if (visible == Visible.BERNOULLI) {
                    t += softplus(z);
                } else {
                    sq += 0.5 * z * z - 0.5 * a[i] * a[i];
                }
            }
            terms[s] = visible == Visible.BERNOULLI ? t : t + sq + 0.5 * nv * Math.log(2 * Math.PI);
        }
        return McmcDiagnostics.logMeanExp(terms) + Math.log(states);
    }

    /**
     * AIS estimate of {@code log Z} (Bernoulli visibles): from the base model with visible biases
     * {@code base} and no couplings, through {@code p_β(v) ∝ exp((1-β) base·v + β a·v) Π_j (1 +
     * exp(β (b_j + W_j·v)))}, one Gibbs transition per stage.
     *
     * @param base the base model's visible biases (such as the data's logit marginals)
     * @param particles AIS runs
     * @param stages annealing stages
     * @param seed the seed
     * @return the estimate of {@code log Z}
     */
    public double aisLogPartition(double[] base, int particles, int stages, long seed) {
        if (visible != Visible.BERNOULLI) {
            throw new IllegalStateException("AIS is implemented for Bernoulli visibles");
        }
        double[] beta = AnnealedImportanceSampler.sigmoidSchedule(stages);
        SplittableRandom r = new SplittableRandom(seed);
        double logZ0 = nh * Math.log(2);
        double[] p0 = new double[nv];
        for (int i = 0; i < nv; i++) {
            logZ0 += softplus(base[i]);
            p0[i] = sigmoid(base[i]);
        }
        double[] lw = new double[particles];
        for (int p = 0; p < particles; p++) {
            double[] v = sampleBernoulli(p0, r);
            for (int k = 1; k <= stages; k++) {
                lw[p] += logUnnormalised(v, beta[k], base) - logUnnormalised(v, beta[k - 1], base);
                if (k < stages) {
                    double[] h = sampleBernoulli(hiddenProbabilities(v, beta[k]), r);
                    v = sampleBernoulli(visibleMean(h, beta[k], base), r);
                }
            }
        }
        return logZ0 + McmcDiagnostics.logMeanExp(lw);
    }

    private double logUnnormalised(double[] v, double beta, double[] base) {
        double s = 0;
        for (int i = 0; i < nv; i++) {
            s += ((1 - beta) * base[i] + beta * a[i]) * v[i];
        }
        for (int j = 0; j < nh; j++) {
            double z = b[j];
            for (int i = 0; i < nv; i++) {
                z += v[i] * w[i][j];
            }
            s += softplus(beta * z);
        }
        return s;
    }

    /**
     * The mean log-likelihood of rows given {@code log Z}.
     *
     * @param data rows
     * @param logZ the log partition function
     * @return {@code mean(-F(v)) - log Z}
     */
    public double logLikelihood(double[][] data, double logZ) {
        double s = 0;
        for (double[] v : data) {
            s -= freeEnergy(v);
        }
        return s / data.length - logZ;
    }

    /**
     * The base biases of {@link #aisLogPartition}: the logits of the data's visible marginals,
     * smoothed.
     *
     * @param data binary rows
     * @return per-visible biases
     */
    public static double[] marginalBiases(double[][] data) {
        int nv = data[0].length;
        double[] out = new double[nv];
        for (int i = 0; i < nv; i++) {
            double m = 0;
            for (double[] v : data) {
                m += v[i];
            }
            m = (m + 1) / (data.length + 2);
            out[i] = Math.log(m / (1 - m));
        }
        return out;
    }
}
