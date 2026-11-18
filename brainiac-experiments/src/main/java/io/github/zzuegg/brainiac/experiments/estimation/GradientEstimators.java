package io.github.zzuegg.brainiac.experiments.estimation;

import io.github.zzuegg.brainiac.experimental.estimators.DiscreteGradients;
import io.github.zzuegg.brainiac.experimental.estimators.DiscreteGradients.Estimator;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Discrete gradient estimators (12.5.1) at an equal budget of function evaluations.
 * <ul>
 *   <li>{@link #toy}: {@code E[mean_i (b_i - 0.499)^2]} over four Bernoulli variables, where the
 *       exact gradient is tiny and positive; at {@code p = sigmoid(-1)} the straight-through
 *       estimators point the wrong way, while the unbiased ones only differ in variance;</li>
 *   <li>{@link #binaryVae}: a linear VAE with 16 Bernoulli latents on binarised MNIST-like
 *       digits (the paper's 200 latents scaled down), trained with each encoder estimator for a
 *       fixed number of steps and scored by its ELBO;</li>
 *   <li>{@link #categoricalVae}: the same with four 10-way categorical latents.</li>
 * </ul>
 */
public final class GradientEstimators {

    private GradientEstimators() {
    }

    /**
     * One estimator on the toy objective.
     *
     * @param logVariance {@code log10} of the summed per-coordinate variance of single estimates
     * @param relativeBias {@code |mean - exact| / |exact|}
     * @param signCorrect the share of coordinates whose mean estimate has the exact sign
     */
    public record ToyRow(double logVariance, double relativeBias, double signCorrect) {
    }

    /** The draws per gradient that spend {@code budget} evaluations. */
    static int draws(Estimator e, int budget) {
        return Math.max(e == Estimator.RLOO || e == Estimator.RODEO ? 2 : 1, budget / e.evaluationsPerDraw());
    }

    /**
     * The toy objective {@code mean_i (b_i - t)^2}.
     *
     * @param d the dimension
     * @param t the target
     * @return the objective
     */
    static DiscreteGradients.Objective toyObjective(int d, double t) {
        return new DiscreteGradients.Objective() {
            @Override
            public double value(double[] b) {
                double s = 0;
                for (double v : b) {
                    s += (v - t) * (v - t);
                }
                return s / d;
            }

            @Override
            public double[] gradient(double[] z) {
                double[] g = new double[d];
                for (int i = 0; i < d; i++) {
                    g[i] = 2 * (z[i] - t) / d;
                }
                return g;
            }
        };
    }

    /**
     * Bias and variance of every estimator on the toy objective.
     *
     * @param logit the logit of every variable, for example -1
     * @param budget function evaluations per gradient, for example 12
     * @param repeats estimates to average, for example 4000
     * @param seed the seed
     * @return estimator name to its row
     */
    public static Map<String, ToyRow> toy(double logit, int budget, int repeats, long seed) {
        int d = 4;
        double[] theta = new double[d];
        java.util.Arrays.fill(theta, logit);
        return measure(theta, toyObjective(d, 0.499), budget, repeats, seed);
    }

    /**
     * Bias and variance on a non-separable objective, {@code (4 mean(b) - 1.3)^2 + t . b} over
     * eight variables with random logits and offsets, where no constant baseline removes the
     * noise and estimators differ in variance.
     *
     * @param budget function evaluations per gradient
     * @param repeats estimates to average
     * @param seed the seed
     * @return estimator name to its row
     */
    public static Map<String, ToyRow> interacting(int budget, int repeats, long seed) {
        int d = 8;
        SplittableRandom r = new SplittableRandom(seed);
        double[] theta = new double[d];
        double[] t = new double[d];
        for (int i = 0; i < d; i++) {
            theta[i] = r.nextGaussian();
            t[i] = r.nextGaussian() * 0.3;
        }
        DiscreteGradients.Objective f = new DiscreteGradients.Objective() {
            @Override
            public double value(double[] b) {
                double m = 0;
                double lin = 0;
                for (int i = 0; i < d; i++) {
                    m += b[i] / d;
                    lin += t[i] * b[i];
                }
                return (4 * m - 1.3) * (4 * m - 1.3) + lin;
            }

            @Override
            public double[] gradient(double[] z) {
                double m = 0;
                for (int i = 0; i < d; i++) {
                    m += z[i] / d;
                }
                double[] g = new double[d];
                for (int i = 0; i < d; i++) {
                    g[i] = 2 * (4 * m - 1.3) * 4 / d + t[i];
                }
                return g;
            }
        };
        return measure(theta, f, budget, repeats, seed);
    }

    private static Map<String, ToyRow> measure(double[] theta, DiscreteGradients.Objective f, int budget,
            int repeats, long seed) {
        int d = theta.length;
        double[] exact = DiscreteGradients.exactBernoulli(theta, f);
        Map<String, ToyRow> out = new LinkedHashMap<>();
        for (Estimator e : Estimator.values()) {
            DiscreteGradients est = DiscreteGradients.of(e, draws(e, budget), 0.5, 0.01);
            SplittableRandom r = new SplittableRandom(seed);
            for (int w = 0; w < 300; w++) {   // let adaptive control variates settle
                est.bernoulli(theta, f, r);
            }
            double[] mean = new double[d];
            double[] sq = new double[d];
            for (int k = 0; k < repeats; k++) {
                double[] g = est.bernoulli(theta, f, r);
                for (int i = 0; i < d; i++) {
                    mean[i] += g[i] / repeats;
                    sq[i] += g[i] * g[i] / repeats;
                }
            }
            double var = 0;
            double biasNum = 0;
            double exactNorm = 0;
            int sign = 0;
            for (int i = 0; i < d; i++) {
                var += sq[i] - mean[i] * mean[i];
                biasNum += (mean[i] - exact[i]) * (mean[i] - exact[i]);
                exactNorm += exact[i] * exact[i];
                sign += Math.signum(mean[i]) == Math.signum(exact[i]) ? 1 : 0;
            }
            out.put(e.name(), new ToyRow(Math.log10(var), Math.sqrt(biasNum / exactNorm), sign / (double) d));
        }
        return out;
    }

    /** Plain Adam on a flat double array (ascent when the gradient is of an objective to maximise). */
    static final class Adam {
        private final double[] m;
        private final double[] v;
        private final double lr;
        private int t;

        Adam(int size, double lr) {
            this.m = new double[size];
            this.v = new double[size];
            this.lr = lr;
        }

        void ascend(double[] param, double[] grad) {
            t++;
            for (int j = 0; j < param.length; j++) {
                m[j] = 0.9 * m[j] + 0.1 * grad[j];
                v[j] = 0.999 * v[j] + 0.001 * grad[j] * grad[j];
                double mh = m[j] / (1 - Math.pow(0.9, t));
                double vh = v[j] / (1 - Math.pow(0.999, t));
                param[j] += lr * mh / (Math.sqrt(vh) + 1e-8);
            }
        }
    }

    static double[][] binarisedDigits(int perClass, long seed) {
        ToyTasks.Data data = ToyTasks.digits(10, perClass, 8, 0.05, seed);
        double[][] x = new double[data.size()][64];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < 64; j++) {
                x[i][j] = data.x()[i][j] > 0.5 ? 1 : 0;
            }
        }
        return x;
    }

    private static double sigmoid(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    /** log p(x | latent) for a Bernoulli decoder with logits {@code V z + a}. */
    private static double logLik(double[] x, double[] z, double[] dec, double[] bias, int latent) {
        double s = 0;
        int px = x.length;
        for (int j = 0; j < px; j++) {
            double l = bias[j];
            for (int k = 0; k < latent; k++) {
                l += dec[j * latent + k] * z[k];
            }
            // log sigmoid(l) for x = 1, log sigmoid(-l) for x = 0
            double sl = x[j] == 1 ? l : -l;
            s += -Math.log1p(Math.exp(-Math.abs(sl))) + Math.min(sl, 0);
        }
        return s;
    }

    /** d log p(x | z) / dz. */
    private static double[] logLikGrad(double[] x, double[] z, double[] dec, double[] bias, int latent) {
        double[] g = new double[latent];
        for (int j = 0; j < x.length; j++) {
            double l = bias[j];
            for (int k = 0; k < latent; k++) {
                l += dec[j * latent + k] * z[k];
            }
            double r = x[j] - sigmoid(l);
            for (int k = 0; k < latent; k++) {
                g[k] += r * dec[j * latent + k];
            }
        }
        return g;
    }

    /**
     * A linear Bernoulli-latent VAE trained with each encoder estimator.
     *
     * @param estimators the estimators to compare
     * @param latent latent bits, for example 16
     * @param steps training steps, for example 300
     * @param budget function evaluations per example and step, for example 4
     * @param seed the seed
     * @return estimator name to the final training ELBO per image (nats)
     */
    public static Map<String, Double> binaryVae(List<Estimator> estimators, int latent, int steps, int budget,
            long seed) {
        double[][] x = binarisedDigits(30, seed);
        int px = 64;
        Map<String, Double> out = new LinkedHashMap<>();
        for (Estimator e : estimators) {
            SplittableRandom r = new SplittableRandom(seed);
            double[] enc = new double[latent * px];
            double[] encB = new double[latent];
            double[] dec = new double[px * latent];
            double[] decB = new double[px];
            for (int j = 0; j < enc.length; j++) {
                enc[j] = r.nextGaussian() * 0.05;
                dec[j] = r.nextGaussian() * 0.05;
            }
            Adam[] adam = {new Adam(enc.length, 0.01), new Adam(latent, 0.01), new Adam(dec.length, 0.01),
                new Adam(px, 0.01)};
            DiscreteGradients est = DiscreteGradients.of(e, draws(e, budget), 0.5, 0.01);
            int batch = 20;
            for (int s = 0; s < steps; s++) {
                double[] gEnc = new double[enc.length];
                double[] gEncB = new double[latent];
                double[] gDec = new double[dec.length];
                double[] gDecB = new double[px];
                for (int bi = 0; bi < batch; bi++) {
                    double[] xi = x[r.nextInt(x.length)];
                    double[] theta = new double[latent];
                    for (int k = 0; k < latent; k++) {
                        double t = encB[k];
                        for (int j = 0; j < px; j++) {
                            t += enc[k * px + j] * xi[j];
                        }
                        theta[k] = t;
                    }
                    DiscreteGradients.Objective f = new DiscreteGradients.Objective() {
                        @Override
                        public double value(double[] b) {
                            return logLik(xi, b, dec, decB, latent);
                        }

                        @Override
                        public double[] gradient(double[] z) {
                            return logLikGrad(xi, z, dec, decB, latent);
                        }
                    };
                    double[] gTheta = est.bernoulli(theta, f, r);
                    double[] b = new double[latent];
                    for (int k = 0; k < latent; k++) {
                        double p = sigmoid(theta[k]);
                        gTheta[k] += -theta[k] * p * (1 - p);   // the entropy term
                        b[k] = r.nextDouble() < p ? 1 : 0;
                        gEncB[k] += gTheta[k] / batch;
                        for (int j = 0; j < px; j++) {
                            gEnc[k * px + j] += gTheta[k] * xi[j] / batch;
                        }
                    }
                    for (int j = 0; j < px; j++) {
                        double l = decB[j];
                        for (int k = 0; k < latent; k++) {
                            l += dec[j * latent + k] * b[k];
                        }
                        double res = xi[j] - sigmoid(l);
                        gDecB[j] += res / batch;
                        for (int k = 0; k < latent; k++) {
                            gDec[j * latent + k] += res * b[k] / batch;
                        }
                    }
                }
                adam[0].ascend(enc, gEnc);
                adam[1].ascend(encB, gEncB);
                adam[2].ascend(dec, gDec);
                adam[3].ascend(decB, gDecB);
            }
            // the ELBO: E_q log p(x|b) + H(q) - latent log 2, by 10 samples per image
            SplittableRandom er = new SplittableRandom(seed + 99);
            double elbo = 0;
            for (double[] xi : x) {
                double[] p = new double[latent];
                double h = 0;
                for (int k = 0; k < latent; k++) {
                    double t = encB[k];
                    for (int j = 0; j < px; j++) {
                        t += enc[k * px + j] * xi[j];
                    }
                    p[k] = sigmoid(t);
                    double pc = Math.min(1 - 1e-12, Math.max(1e-12, p[k]));
                    h += -pc * Math.log(pc) - (1 - pc) * Math.log(1 - pc);
                }
                double ll = 0;
                for (int s = 0; s < 10; s++) {
                    double[] b = new double[latent];
                    for (int k = 0; k < latent; k++) {
                        b[k] = er.nextDouble() < p[k] ? 1 : 0;
                    }
                    ll += logLik(xi, b, dec, decB, latent) / 10;
                }
                elbo += (ll + h - latent * Math.log(2)) / x.length;
            }
            out.put(e.name(), elbo);
        }
        return out;
    }

    /**
     * A linear VAE with {@code m} categorical latents of 10 classes each.
     *
     * @param estimators REINFORCE, RLOO, STRAIGHT_THROUGH or GUMBEL_ST
     * @param m the number of categorical latents, for example 4
     * @param steps training steps
     * @param budget evaluations per example and step
     * @param seed the seed
     * @return estimator name to the final training ELBO per image (nats)
     */
    public static Map<String, Double> categoricalVae(List<Estimator> estimators, int m, int steps, int budget,
            long seed) {
        int k = 10;
        int latent = m * k;
        double[][] x = binarisedDigits(30, seed);
        int px = 64;
        Map<String, Double> out = new LinkedHashMap<>();
        for (Estimator e : estimators) {
            SplittableRandom r = new SplittableRandom(seed);
            double[] enc = new double[latent * px];
            double[] encB = new double[latent];
            double[] dec = new double[px * latent];
            double[] decB = new double[px];
            for (int j = 0; j < enc.length; j++) {
                enc[j] = r.nextGaussian() * 0.05;
                dec[j] = r.nextGaussian() * 0.05;
            }
            Adam[] adam = {new Adam(enc.length, 0.01), new Adam(latent, 0.01), new Adam(dec.length, 0.01),
                new Adam(px, 0.01)};
            DiscreteGradients est = DiscreteGradients.of(e, draws(e, budget), 0.5, 0.01);
            int batch = 20;
            for (int s = 0; s < steps; s++) {
                double[] gEnc = new double[enc.length];
                double[] gEncB = new double[latent];
                double[] gDec = new double[dec.length];
                double[] gDecB = new double[px];
                for (int bi = 0; bi < batch; bi++) {
                    double[] xi = x[r.nextInt(x.length)];
                    double[][] theta = new double[m][k];
                    for (int a = 0; a < m; a++) {
                        for (int c = 0; c < k; c++) {
                            int row = a * k + c;
                            double t = encB[row];
                            for (int j = 0; j < px; j++) {
                                t += enc[row * px + j] * xi[j];
                            }
                            theta[a][c] = t;
                        }
                    }
                    DiscreteGradients.CategoricalObjective f = new DiscreteGradients.CategoricalObjective() {
                        @Override
                        public double value(double[][] y) {
                            return logLik(xi, flat(y), dec, decB, latent);
                        }

                        @Override
                        public double[][] gradient(double[][] y) {
                            double[] g = logLikGrad(xi, flat(y), dec, decB, latent);
                            double[][] out2 = new double[m][k];
                            for (int a = 0; a < m; a++) {
                                System.arraycopy(g, a * k, out2[a], 0, k);
                            }
                            return out2;
                        }
                    };
                    double[][] gTheta = est.categorical(theta, f, r);
                    double[] z = new double[latent];
                    for (int a = 0; a < m; a++) {
                        double[] p = softmax(theta[a]);
                        // entropy gradient: dH/dtheta_c = -p_c (log p_c - sum_j p_j log p_j)
                        double plogp = 0;
                        for (int c = 0; c < k; c++) {
                            plogp += p[c] * Math.log(Math.max(1e-300, p[c]));
                        }
                        double u = r.nextDouble();
                        double cum = 0;
                        int pick = k - 1;
                        for (int c = 0; c < k; c++) {
                            cum += p[c];
                            if (u < cum) {
                                pick = c;
                                break;
                            }
                        }
                        z[a * k + pick] = 1;
                        for (int c = 0; c < k; c++) {
                            double g = gTheta[a][c] - p[c] * (Math.log(Math.max(1e-300, p[c])) - plogp);
                            int row = a * k + c;
                            gEncB[row] += g / batch;
                            for (int j = 0; j < px; j++) {
                                gEnc[row * px + j] += g * xi[j] / batch;
                            }
                        }
                    }
                    for (int j = 0; j < px; j++) {
                        double l = decB[j];
                        for (int q = 0; q < latent; q++) {
                            l += dec[j * latent + q] * z[q];
                        }
                        double res = xi[j] - sigmoid(l);
                        gDecB[j] += res / batch;
                        for (int q = 0; q < latent; q++) {
                            gDec[j * latent + q] += res * z[q] / batch;
                        }
                    }
                }
                adam[0].ascend(enc, gEnc);
                adam[1].ascend(encB, gEncB);
                adam[2].ascend(dec, gDec);
                adam[3].ascend(decB, gDecB);
            }
            SplittableRandom er = new SplittableRandom(seed + 99);
            double elbo = 0;
            for (double[] xi : x) {
                double ll = 0;
                double h = 0;
                double[][] p = new double[m][];
                for (int a = 0; a < m; a++) {
                    double[] th = new double[k];
                    for (int c = 0; c < k; c++) {
                        int row = a * k + c;
                        th[c] = encB[row];
                        for (int j = 0; j < px; j++) {
                            th[c] += enc[row * px + j] * xi[j];
                        }
                    }
                    p[a] = softmax(th);
                    for (double pc : p[a]) {
                        h -= pc * Math.log(Math.max(1e-300, pc));
                    }
                }
                for (int s = 0; s < 10; s++) {
                    double[] z = new double[latent];
                    for (int a = 0; a < m; a++) {
                        double u = er.nextDouble();
                        double cum = 0;
                        int pick = k - 1;
                        for (int c = 0; c < k; c++) {
                            cum += p[a][c];
                            if (u < cum) {
                                pick = c;
                                break;
                            }
                        }
                        z[a * k + pick] = 1;
                    }
                    ll += logLik(xi, z, dec, decB, latent) / 10;
                }
                elbo += (ll + h - m * Math.log(k)) / x.length;
            }
            out.put(e.name(), elbo);
        }
        return out;
    }

    private static double[] flat(double[][] y) {
        int k = y[0].length;
        double[] z = new double[y.length * k];
        for (int a = 0; a < y.length; a++) {
            System.arraycopy(y[a], 0, z, a * k, k);
        }
        return z;
    }

    private static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            max = Math.max(max, v);
        }
        double[] p = new double[x.length];
        double s = 0;
        for (int j = 0; j < x.length; j++) {
            p[j] = Math.exp(x[j] - max);
            s += p[j];
        }
        for (int j = 0; j < x.length; j++) {
            p[j] /= s;
        }
        return p;
    }
}
