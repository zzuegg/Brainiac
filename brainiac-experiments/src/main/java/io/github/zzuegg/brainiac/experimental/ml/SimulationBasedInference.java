package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Simulation-based (likelihood-free) inference: posteriors {@code p(theta | x_obs)} for models
 * that can be simulated but whose likelihood cannot be evaluated (Cranmer et al. 2020,
 * <a href="https://arxiv.org/abs/1911.01429">arXiv:1911.01429</a>).
 *
 * <ul>
 *   <li>{@link #rejectionAbc}: simulate from the prior, keep the parameters whose simulated
 *       summaries land closest to the observed ones (the baseline);</li>
 *   <li>{@link #smcAbc}: ABC population Monte Carlo (Sisson et al. 2007,
 *       <a href="https://www.pnas.org/doi/10.1073/pnas.0607208104">PNAS</a>; Beaumont et al.
 *       2009): the tolerance shrinks to a quantile of the previous round's distances, particles
 *       are perturbed by a Gaussian kernel with twice the weighted covariance and reweighted by
 *       prior over proposal, so simulations concentrate near the posterior;</li>
 *   <li>{@link #neuralPosterior}: neural posterior estimation (Papamakarios and Murray 2016;
 *       Greenberg et al. 2019, <a href="https://arxiv.org/abs/1905.07488">arXiv:1905.07488</a>):
 *       train a conditional density estimator {@code q(theta | x)} on prior simulations by
 *       maximum likelihood. Here it is a mixture density network (diagonal Gaussian components)
 *       built with autograd. The result is amortised: it gives the posterior of any new
 *       observation without new simulations.</li>
 * </ul>
 * {@link #c2st} scores a sample set against reference posterior samples by the
 * cross-validated accuracy of a k-NN classifier telling them apart (0.5 is perfect).
 */
public final class SimulationBasedInference {

    private SimulationBasedInference() {
    }

    /** A prior over parameters. */
    public interface Prior {
        /**
         * Draws parameters.
         *
         * @param random the random source
         * @return {@code theta}
         */
        double[] sample(SplittableRandom random);

        /**
         * The prior log density (up to a constant).
         *
         * @param theta the parameters
         * @return {@code log p(theta)}, negative infinity outside the support
         */
        double logDensity(double[] theta);
    }

    /** A stochastic simulator returning summary statistics. */
    @FunctionalInterface
    public interface Simulator {
        /**
         * Simulates data.
         *
         * @param theta the parameters
         * @param random the random source
         * @return the summary statistics
         */
        double[] simulate(double[] theta, SplittableRandom random);
    }

    /**
     * Posterior samples.
     *
     * @param samples the parameter samples {@code [n][d]}
     * @param simulations the simulator calls spent
     */
    public record Posterior(double[][] samples, long simulations) {
    }

    /** The per-coordinate prior-predictive standard deviation of the summaries, for distances. */
    private static double[] summaryScale(Prior prior, Simulator sim, SplittableRandom r, int pilot) {
        double[][] s = new double[pilot][];
        for (int i = 0; i < pilot; i++) {
            s[i] = sim.simulate(prior.sample(r), r);
        }
        double[] mean = MlMatrices.columnMeans(s);
        double[] sd = new double[mean.length];
        for (double[] v : s) {
            for (int j = 0; j < sd.length; j++) {
                sd[j] += (v[j] - mean[j]) * (v[j] - mean[j]) / pilot;
            }
        }
        for (int j = 0; j < sd.length; j++) {
            sd[j] = Math.sqrt(sd[j]) + 1e-12;
        }
        return sd;
    }

    private static double distance(double[] a, double[] b, double[] scale) {
        double s = 0;
        for (int j = 0; j < a.length; j++) {
            double d = (a[j] - b[j]) / scale[j];
            s += d * d;
        }
        return Math.sqrt(s);
    }

    /**
     * Rejection ABC: keeps the {@code keep} closest of {@code simulations} prior draws.
     *
     * @param prior the prior
     * @param sim the simulator
     * @param observed the observed summaries
     * @param simulations the simulation budget
     * @param keep the number of samples kept
     * @param random the random source
     * @return the accepted parameters
     */
    public static Posterior rejectionAbc(Prior prior, Simulator sim, double[] observed, int simulations, int keep,
            SplittableRandom random) {
        double[] scale = summaryScale(prior, sim, random, 200);
        double[][] thetas = new double[simulations][];
        double[] dist = new double[simulations];
        Integer[] order = new Integer[simulations];
        for (int i = 0; i < simulations; i++) {
            thetas[i] = prior.sample(random);
            dist[i] = distance(sim.simulate(thetas[i], random), observed, scale);
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(dist[a], dist[b]));
        double[][] out = new double[keep][];
        for (int i = 0; i < keep; i++) {
            out[i] = thetas[order[i]];
        }
        return new Posterior(out, simulations + 200L);
    }

    /**
     * ABC population Monte Carlo with an adaptive tolerance.
     *
     * @param prior the prior
     * @param sim the simulator
     * @param observed the observed summaries
     * @param particles the population size
     * @param budget the total simulation budget
     * @param quantile the tolerance quantile of the previous round's distances, for example 0.5
     * @param random the random source
     * @return the final population (resampled by weight)
     */
    public static Posterior smcAbc(Prior prior, Simulator sim, double[] observed, int particles, int budget,
            double quantile, SplittableRandom random) {
        double[] scale = summaryScale(prior, sim, random, 200);
        long sims = 200;
        int d = prior.sample(random).length;
        double[][] theta = new double[particles][];
        double[] dist = new double[particles];
        for (int i = 0; i < particles; i++) {
            theta[i] = prior.sample(random);
            dist[i] = distance(sim.simulate(theta[i], random), observed, scale);
            sims++;
        }
        double[] w = new double[particles];
        Arrays.fill(w, 1.0 / particles);
        while (sims < budget) {
            double[] sorted = dist.clone();
            Arrays.sort(sorted);
            double eps = sorted[(int) (quantile * (particles - 1))];
            // perturbation kernel: N(0, 2 * weighted covariance)
            double[] mean = new double[d];
            for (int i = 0; i < particles; i++) {
                for (int j = 0; j < d; j++) {
                    mean[j] += w[i] * theta[i][j];
                }
            }
            double[][] cov = new double[d][d];
            for (int i = 0; i < particles; i++) {
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        cov[a][b] += 2 * w[i] * (theta[i][a] - mean[a]) * (theta[i][b] - mean[b]);
                    }
                }
            }
            for (int a = 0; a < d; a++) {
                cov[a][a] += 1e-9;
            }
            double[][] chol = MlMatrices.cholesky(cov);
            double logDetHalf = 0.5 * MlMatrices.logDet(chol);
            double[][] next = new double[particles][];
            double[] nextDist = new double[particles];
            double[] nextW = new double[particles];
            double[] cdf = new double[particles];
            double c = 0;
            for (int i = 0; i < particles; i++) {
                c += w[i];
                cdf[i] = c;
            }
            for (int i = 0; i < particles && sims < budget; i++) {
                while (sims < budget) {
                    double u = random.nextDouble() * c;
                    int k = 0;
                    while (k < particles - 1 && cdf[k] < u) {
                        k++;
                    }
                    double[] z = new double[d];
                    for (int a = 0; a < d; a++) {
                        z[a] = random.nextGaussian();
                    }
                    double[] prop = theta[k].clone();
                    for (int a = 0; a < d; a++) {
                        for (int b = 0; b <= a; b++) {
                            prop[a] += chol[a][b] * z[b];
                        }
                    }
                    if (!Double.isFinite(prior.logDensity(prop))) {
                        continue;
                    }
                    double dd = distance(sim.simulate(prop, random), observed, scale);
                    sims++;
                    if (dd <= eps) {
                        next[i] = prop;
                        nextDist[i] = dd;
                        break;
                    }
                }
            }
            if (next[particles - 1] == null) {
                break;   // budget ran out mid-round: keep the last complete population
            }
            double total = 0;
            for (int i = 0; i < particles; i++) {
                double mix = 0;
                for (int k = 0; k < particles; k++) {
                    double[] diff = new double[d];
                    for (int a = 0; a < d; a++) {
                        diff[a] = next[i][a] - theta[k][a];
                    }
                    double[] y = MlMatrices.forwardSolve(chol, diff);
                    mix += w[k] * Math.exp(-0.5 * MlMatrices.dot(y, y) - logDetHalf);
                }
                nextW[i] = Math.exp(prior.logDensity(next[i])) / Math.max(mix, 1e-300);
                total += nextW[i];
            }
            for (int i = 0; i < particles; i++) {
                nextW[i] /= total;
            }
            theta = next;
            dist = nextDist;
            w = nextW;
        }
        double[] lw = new double[particles];
        for (int i = 0; i < particles; i++) {
            lw[i] = Math.log(Math.max(w[i], 1e-300));
        }
        int[] pick = ProbabilisticProgram.systematic(lw, random);
        double[][] out = new double[particles][];
        for (int i = 0; i < particles; i++) {
            out[i] = theta[pick[i]];
        }
        return new Posterior(out, sims);
    }

    /** A trained amortised posterior {@code q(theta | x)}: a mixture density network. */
    public static final class NeuralPosterior {
        private final int dTheta;
        private final int components;
        private final double[] thetaMean;
        private final double[] thetaSd;
        private final double[] xMean;
        private final double[] xSd;
        private final Var w1;
        private final Var b1;
        private final Var w2;
        private final Var b2;
        private final Var w3;
        private final Var b3;
        private final long simulations;

        private NeuralPosterior(int dTheta, int dx, int components, int hidden, double[][] thetas, double[][] xs,
                long simulations, SplittableRandom r) {
            this.dTheta = dTheta;
            this.components = components;
            this.simulations = simulations;
            thetaMean = MlMatrices.columnMeans(thetas);
            thetaSd = sd(thetas, thetaMean);
            xMean = MlMatrices.columnMeans(xs);
            xSd = sd(xs, xMean);
            int out = components * (1 + 2 * dTheta);
            w1 = Var.param(Tensor.normal(Shape.of(dx, hidden), 0, (float) Math.sqrt(1.0 / dx), r));
            b1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
            w2 = Var.param(Tensor.normal(Shape.of(hidden, hidden), 0, (float) Math.sqrt(1.0 / hidden), r));
            b2 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
            w3 = Var.param(Tensor.normal(Shape.of(hidden, out), 0, 0.01f, r));
            float[] bias = new float[out];
            for (int k = 0; k < components; k++) {
                for (int j = 0; j < dTheta; j++) {
                    bias[components + k * dTheta + j] = (float) r.nextGaussian();   // spread the means
                }
            }
            b3 = Var.param(Tensor.of(Shape.of(1, out), bias));
        }

        private static double[] sd(double[][] x, double[] mean) {
            double[] s = new double[mean.length];
            for (double[] v : x) {
                for (int j = 0; j < s.length; j++) {
                    s[j] += (v[j] - mean[j]) * (v[j] - mean[j]) / x.length;
                }
            }
            for (int j = 0; j < s.length; j++) {
                s[j] = Math.sqrt(s[j]) + 1e-9;
            }
            return s;
        }

        private static Tensor standardise(double[][] rows, double[] mean, double[] sd) {
            int d = mean.length;
            float[] v = new float[rows.length * d];
            for (int i = 0; i < rows.length; i++) {
                for (int j = 0; j < d; j++) {
                    v[i * d + j] = (float) ((rows[i][j] - mean[j]) / sd[j]);
                }
            }
            return Tensor.of(Shape.of(rows.length, d), v);
        }

        /** The network output split as {logits [B,K], means [B,K*d], log stds [B,K*d]}. */
        private Var[] head(Tensor x) {
            Var h = Var.of(x).matmul(w1).add(b1).tanh().matmul(w2).add(b2).tanh();
            Var o = h.matmul(w3).add(b3);
            int k = components;
            int kd = components * dTheta;
            return new Var[] {o.slice(1, 0, k), o.slice(1, k, k + kd), o.slice(1, k + kd, k + 2 * kd).clamp(-7f, 3f)};
        }

        private Var nll(Tensor x, Tensor theta) {
            Var[] h = head(x);
            int b = x.shape().dim(0);
            Var logPi = h[0].sub(h[0].logsumexp(1, true));
            Var mu = h[1].reshape(b, components, dTheta);
            Var ls = h[2].reshape(b, components, dTheta);
            Var t = Var.of(theta.reshape(b, 1, dTheta));
            Var z = t.sub(mu).mul(ls.neg().exp());
            Var comp = z.square().mul(-0.5f).sub(ls).sum(2, false)
                    .sub((float) (0.5 * dTheta * Math.log(2 * Math.PI)));
            return logPi.add(comp).logsumexp(1, false).mean().neg();
        }

        /**
         * Draws posterior samples for an observation.
         *
         * @param x the observed summaries
         * @param n the number of samples
         * @param random the random source
         * @return samples {@code [n][d]}
         */
        public double[][] sample(double[] x, int n, SplittableRandom random) {
            Var[] h = Var.noGrad(() -> head(standardise(new double[][] {x}, xMean, xSd)));
            double[] logits = h[0].value().toDoubleArray();
            double[] mu = h[1].value().toDoubleArray();
            double[] ls = h[2].value().toDoubleArray();
            double max = Arrays.stream(logits).max().orElse(0);
            double[] p = new double[components];
            double z = 0;
            for (int k = 0; k < components; k++) {
                p[k] = Math.exp(logits[k] - max);
                z += p[k];
            }
            double[][] out = new double[n][dTheta];
            for (int i = 0; i < n; i++) {
                double u = random.nextDouble() * z;
                int k = 0;
                while (k < components - 1 && u > p[k]) {
                    u -= p[k];
                    k++;
                }
                for (int j = 0; j < dTheta; j++) {
                    double v = mu[k * dTheta + j] + Math.exp(ls[k * dTheta + j]) * random.nextGaussian();
                    out[i][j] = thetaMean[j] + thetaSd[j] * v;
                }
            }
            return out;
        }

        /**
         * The simulations used for training.
         *
         * @return the simulation count
         */
        public long simulations() {
            return simulations;
        }
    }

    /**
     * Trains an amortised neural posterior on prior simulations.
     *
     * @param prior the prior
     * @param sim the simulator
     * @param simulations the simulation budget
     * @param components mixture components, for example 5
     * @param hidden the hidden width, for example 64
     * @param iterations Adam steps on minibatches of 256
     * @param seed the seed
     * @return the posterior estimator
     */
    public static NeuralPosterior neuralPosterior(Prior prior, Simulator sim, int simulations, int components,
            int hidden, int iterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] thetas = new double[simulations][];
        double[][] xs = new double[simulations][];
        for (int i = 0; i < simulations; i++) {
            thetas[i] = prior.sample(r);
            xs[i] = sim.simulate(thetas[i], r);
        }
        NeuralPosterior q = new NeuralPosterior(thetas[0].length, xs[0].length, components, hidden, thetas, xs,
                simulations, r);
        Tensor tx = NeuralPosterior.standardise(xs, q.xMean, q.xSd);
        Tensor tt = NeuralPosterior.standardise(thetas, q.thetaMean, q.thetaSd);
        List<Var> ps = new ArrayList<>(List.of(q.w1, q.b1, q.w2, q.b2, q.w3, q.b3));
        Optimizer opt = Optimizer.adam(3e-3f);
        int batch = Math.min(256, simulations);
        for (int it = 0; it < iterations; it++) {
            int from = r.nextInt(simulations - batch + 1);
            q.nll(tx.slice(0, from, from + batch), tt.slice(0, from, from + batch)).backward();
            opt.step(ps);
        }
        return q;
    }

    /**
     * The classifier two-sample test: leave-one-out 5-NN accuracy at telling the two sample
     * sets apart, after standardising by the reference. 0.5 means indistinguishable.
     *
     * @param reference reference posterior samples {@code [n][d]}
     * @param estimate estimated posterior samples {@code [n][d]}
     * @return the classification accuracy
     */
    public static double c2st(double[][] reference, double[][] estimate) {
        int n = Math.min(reference.length, estimate.length);
        double[] mean = MlMatrices.columnMeans(reference);
        double[] sd = NeuralPosterior.sd(reference, mean);
        double[][] x = new double[2 * n][];
        int[] labels = new int[2 * n];
        for (int i = 0; i < n; i++) {
            x[i] = new double[mean.length];
            x[n + i] = new double[mean.length];
            for (int j = 0; j < mean.length; j++) {
                x[i][j] = (reference[i][j] - mean[j]) / sd[j];
                x[n + i][j] = (estimate[i][j] - mean[j]) / sd[j];
            }
            labels[n + i] = 1;
        }
        return MlMetrics.knnAccuracy(x, labels, 5);
    }
}
