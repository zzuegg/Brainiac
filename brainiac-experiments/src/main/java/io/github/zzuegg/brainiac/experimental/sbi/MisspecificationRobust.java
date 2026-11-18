package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.autograd.OptimalTransport;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Simulation-based inference when real data differ from simulations. All methods share an MDN
 * neural posterior {@code q(theta | x)} (a {@link ConditionalDensity}) trained on simulations.
 * <ul>
 *   <li>{@link #detect}: a kernel MMD two-sample test ({@link KernelTests#mmdTest}) between real
 *       observations and prior-predictive simulations flags misspecification;</li>
 *   <li>{@link #rope}: RoPE (Wehenkel et al., ICML 2025): a small calibration set of real
 *       {@code (theta, x)} pairs fits a map {@code g} from real data to the simulated data space
 *       (ridge regression onto the simulator mean at the calibration parameters); a batch of
 *       real observations is coupled to prior-predictive simulations by entropic optimal
 *       transport (core {@link OptimalTransport#sinkhorn}) on {@code |g(x_real) - x_sim|^2}, and
 *       each real observation's posterior is the mixture of NPE posteriors of the simulations it
 *       is coupled to, mixed with the prior at rate {@code tau} (the mass RoPE leaves
 *       unmatched), trading calibration against informativeness;</li>
 *   <li>{@link #noiseAugmented}: an RVNP-style learned error model (in the spirit of
 *       O'Callaghan, Mandel and Gilmore 2025, <a href="https://arxiv.org/abs/2509.05724">
 *       arXiv:2509.05724</a>): NPE is trained on simulations with additive Gaussian error of a
 *       random scale {@code sigma} that it also sees, and the scale is then chosen from the real
 *       observations alone by maximising their kernel marginal likelihood under the simulator,
 *       so the misspecification strength is learned from data rather than set by hand;</li>
 *   <li>{@link #rnpe}: robust NPE (Ward et al., NeurIPS 2022): a spike-and-slab error model
 *       on standardised data ({@code (1 - rho) N(0, 0.01^2) + rho Cauchy(0, 0.25)} per
 *       coordinate) with a kernel density of simulated data as the prior over the "clean"
 *       data; the clean data are sampled by slice sampling and pushed through NPE.</li>
 * </ul>
 */
public final class MisspecificationRobust {

    private final ConditionalDensity npe;
    private final double[][] simThetas;
    private final double[][] simXs;
    private final double[] xMean;
    private final double[] xSd;
    private final boolean augmented;
    private final double sigmaMax;

    private MisspecificationRobust(ConditionalDensity npe, double[][] simThetas, double[][] simXs, boolean augmented,
            double sigmaMax) {
        this.npe = npe;
        this.simThetas = simThetas;
        this.simXs = simXs;
        this.xMean = SbiArrays.mean(simXs);
        this.xSd = SbiArrays.sd(simXs, xMean);
        this.augmented = augmented;
        this.sigmaMax = sigmaMax;
    }

    /**
     * Trains the shared NPE on prior simulations.
     *
     * @param prior the prior
     * @param sim the simulator
     * @param simulations the budget
     * @param config MDN settings
     * @param seed the seed
     * @return the estimator
     */
    public static MisspecificationRobust train(Prior prior, Simulator sim, int simulations,
            ConditionalDensity.Config config, long seed) {
        return train(prior, sim, simulations, config, 0, seed);
    }

    /**
     * Trains the NPE on simulations with additive Gaussian error {@code sigma * sd(x) * N(0, I)},
     * {@code sigma ~ U(0, sigmaMax)}, conditioned on {@code (x, sigma)} (the noise-augmented
     * variant for {@link #noiseAugmented}).
     *
     * @param prior the prior
     * @param sim the simulator
     * @param simulations the budget
     * @param config MDN settings
     * @param sigmaMax the largest relative error scale, 0 for plain NPE
     * @param seed the seed
     * @return the estimator
     */
    public static MisspecificationRobust train(Prior prior, Simulator sim, int simulations,
            ConditionalDensity.Config config, double sigmaMax, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] t = new double[simulations][];
        double[][] x = new double[simulations][];
        for (int i = 0; i < simulations; i++) {
            t[i] = prior.sample(r);
            x[i] = sim.simulate(t[i], r);
        }
        if (sigmaMax <= 0) {
            return new MisspecificationRobust(ConditionalDensity.fit(ConditionalDensity.Kind.MDN, t, x, config, seed),
                    t, x, false, 0);
        }
        double[] m = SbiArrays.mean(x);
        double[] sd = SbiArrays.sd(x, m);
        double[][] cond = new double[simulations][];
        for (int i = 0; i < simulations; i++) {
            double s = sigmaMax * r.nextDouble();
            cond[i] = new double[x[i].length + 1];
            for (int j = 0; j < x[i].length; j++) {
                cond[i][j] = x[i][j] + s * sd[j] * r.nextGaussian();
            }
            cond[i][x[i].length] = s;
        }
        return new MisspecificationRobust(ConditionalDensity.fit(ConditionalDensity.Kind.MDN, t, cond, config, seed),
                t, x, true, sigmaMax);
    }

    /**
     * Posterior samples of the underlying NPE (no robustness).
     *
     * @param x the observation
     * @param n samples
     * @param random the random source
     * @return {@code [n][d]}
     */
    public double[][] posterior(double[] x, int n, SplittableRandom random) {
        return npe.sample(augmented ? SbiArrays.concat(x, new double[] {0}) : x, n, random);
    }

    /**
     * Fine-tunes the NPE on real calibration pairs (the naive use of real data).
     *
     * @param thetas calibration parameters
     * @param xs real observations
     * @param iterations Adam steps
     * @param seed the seed
     */
    public void fineTune(double[][] thetas, double[][] xs, int iterations, long seed) {
        npe.train(thetas, xs, iterations, seed);
    }

    /**
     * The MMD test between real observations and prior-predictive simulations, with the median
     * heuristic bandwidth on standardised data.
     *
     * @param real real observations
     * @param simulated simulated observations
     * @param permutations permutations
     * @param seed the seed
     * @return the statistic and p-value
     */
    public static KernelTests.Result detect(double[][] real, double[][] simulated, int permutations, long seed) {
        double[][] all = SbiArrays.stack(real, simulated);
        double[] m = SbiArrays.mean(all);
        double[] sd = SbiArrays.sd(all, m);
        double[][] a = SbiArrays.rows(SbiArrays.standardised(real, m, sd));
        double[][] b = SbiArrays.rows(SbiArrays.standardised(simulated, m, sd));
        return KernelTests.mmdTest(a, b, KernelTests.medianDistance(SbiArrays.stack(a, b)), permutations, seed);
    }

    /**
     * RoPE posteriors for a batch of real observations.
     *
     * @param sim the simulator (for the calibration map)
     * @param calibrationThetas real calibration parameters
     * @param calibrationXs real calibration observations
     * @param real the real observations to infer
     * @param coupled prior-predictive simulations to couple to (taken from the training pool)
     * @param epsilon the Sinkhorn entropy, relative to the median cost
     * @param tau the prior mixing rate
     * @param n samples per observation
     * @param random the random source
     * @return {@code [real.length][n][d]}
     */
    public double[][][] rope(Simulator sim, double[][] calibrationThetas, double[][] calibrationXs, double[][] real,
            int coupled, double epsilon, double tau, int n, SplittableRandom random) {
        int dx = xMean.length;
        // target: the simulator mean at each calibration parameter (average of 8 simulations)
        double[][] target = new double[calibrationThetas.length][dx];
        for (int i = 0; i < calibrationThetas.length; i++) {
            for (int k = 0; k < 8; k++) {
                double[] v = sim.simulate(calibrationThetas[i], random);
                for (int j = 0; j < dx; j++) {
                    target[i][j] += v[j] / 8;
                }
            }
        }
        double[][] map = ridge(calibrationXs, target, 1.0);
        double[][] gx = new double[real.length][];
        for (int i = 0; i < real.length; i++) {
            gx[i] = applyRidge(map, real[i]);
        }
        int m = Math.min(coupled, simXs.length);
        double[][] pool = Arrays.copyOf(simXs, m);
        float[] c = new float[real.length * m];
        double[] all = new double[c.length];
        for (int i = 0; i < real.length; i++) {
            for (int j = 0; j < m; j++) {
                double d = 0;
                for (int k = 0; k < dx; k++) {
                    double z = (gx[i][k] - pool[j][k]) / xSd[k];
                    d += z * z;
                }
                c[i * m + j] = (float) d;
                all[i * m + j] = d;
            }
        }
        double[] sorted = all.clone();
        Arrays.sort(sorted);
        float eps = (float) (epsilon * Math.max(sorted[sorted.length / 2], 1e-6));
        float[] a = new float[real.length];
        Arrays.fill(a, 1f / real.length);
        float[] b = new float[m];
        Arrays.fill(b, 1f / m);
        Tensor plan = Var.noGrad(() -> OptimalTransport.sinkhorn(Var.of(Tensor.of(Shape.of(real.length, m), c)),
                Tensor.of(Shape.of(real.length), a), Tensor.of(Shape.of(m), b), eps, 200)).value();
        double[] p = plan.toDoubleArray();
        double[][][] out = new double[real.length][n][];
        for (int i = 0; i < real.length; i++) {
            double[] cdf = new double[m];
            double tot = 0;
            for (int j = 0; j < m; j++) {
                tot += Double.isFinite(p[i * m + j]) ? p[i * m + j] : 0;
                cdf[j] = tot;
            }
            for (int s = 0; s < n; s++) {
                if (random.nextDouble() < tau || tot <= 0) {
                    out[i][s] = simThetas[m + random.nextInt(simThetas.length - m)].clone();   // a prior draw
                    continue;
                }
                double u = random.nextDouble() * tot;
                int k = Arrays.binarySearch(cdf, u);
                k = Math.min(k < 0 ? -k - 1 : k, m - 1);
                out[i][s] = posterior(pool[k], 1, random)[0];
            }
        }
        return out;
    }

    private static double[][] ridge(double[][] x, double[][] y, double lambda) {
        int p = x[0].length;
        int q = y[0].length;
        double[][] xtx = new double[p + 1][p + 1];
        double[][] xty = new double[q][p + 1];
        for (int i = 0; i < x.length; i++) {
            double[] z = SbiArrays.concat(new double[] {1}, x[i]);
            for (int a = 0; a <= p; a++) {
                for (int b = 0; b <= p; b++) {
                    xtx[a][b] += z[a] * z[b];
                }
                for (int c = 0; c < q; c++) {
                    xty[c][a] += z[a] * y[i][c];
                }
            }
        }
        for (int a = 1; a <= p; a++) {
            xtx[a][a] += lambda;
        }
        double[][] w = new double[q][];
        for (int c = 0; c < q; c++) {
            w[c] = Matrices.solve(xtx, xty[c]);
        }
        return w;
    }

    private static double[] applyRidge(double[][] w, double[] x) {
        double[] out = new double[w.length];
        for (int c = 0; c < w.length; c++) {
            double v = w[c][0];
            for (int j = 0; j < x.length; j++) {
                v += w[c][j + 1] * x[j];
            }
            out[c] = v;
        }
        return out;
    }

    /**
     * The error scale chosen for a batch of real observations: the grid value of {@code sigma}
     * maximising the Gaussian-kernel marginal likelihood
     * {@code sum_i log mean_j N(x_i; x_j, (sigma^2 + h^2) diag(sd^2))} over the simulation pool.
     *
     * @param real the real observations
     * @return the selected {@code sigma}
     */
    public double selectNoise(double[][] real) {
        int dx = xMean.length;
        int m = Math.min(2000, simXs.length);
        double h = Math.pow(m, -1.0 / (dx + 4));
        double best = 0;
        double bestL = Double.NEGATIVE_INFINITY;
        for (int g = 0; g <= 20; g++) {
            double s = sigmaMax * g / 20;
            double var = s * s + h * h;
            double l = 0;
            for (double[] x : real) {
                double[] lk = new double[m];
                for (int j = 0; j < m; j++) {
                    double d = 0;
                    for (int k = 0; k < dx; k++) {
                        double z = (x[k] - simXs[j][k]) / xSd[k];
                        d += z * z;
                    }
                    lk[j] = -0.5 * d / var - 0.5 * dx * Math.log(var);
                }
                l += SbiArrays.logSumExp(lk);
            }
            if (l > bestL) {
                bestL = l;
                best = s;
            }
        }
        return best;
    }

    /**
     * Posteriors under the learned error scale (requires training with {@code sigmaMax > 0}).
     *
     * @param real the real observations
     * @param n samples per observation
     * @param random the random source
     * @return {@code [real.length][n][d]}
     */
    public double[][][] noiseAugmented(double[][] real, int n, SplittableRandom random) {
        if (!augmented) {
            throw new IllegalStateException("train with sigmaMax > 0");
        }
        double s = selectNoise(real);
        double[][][] out = new double[real.length][][];
        for (int i = 0; i < real.length; i++) {
            out[i] = npe.sample(SbiArrays.concat(real[i], new double[] {s}), n, random);
        }
        return out;
    }

    /**
     * RNPE posteriors: denoise each observation under the spike-and-slab error model by slice
     * sampling, then draw from NPE at the denoised data.
     *
     * @param real the real observations
     * @param rho the slab (error) probability per coordinate
     * @param n samples per observation
     * @param random the random source
     * @return {@code [real.length][n][d]}
     */
    public double[][][] rnpe(double[][] real, double rho, int n, SplittableRandom random) {
        int dx = xMean.length;
        int m = Math.min(1000, simXs.length);
        double[][] pool = SbiArrays.rows(SbiArrays.standardised(Arrays.copyOf(simXs, m), xMean, xSd));
        double h = Math.pow(m, -1.0 / (dx + 4));
        double[][][] out = new double[real.length][][];
        for (int i = 0; i < real.length; i++) {
            double[] xo = SbiArrays.rows(SbiArrays.standardised(new double[][] {real[i]}, xMean, xSd))[0];
            java.util.function.ToDoubleFunction<double[]> logp = s -> {
                double[] lk = new double[m];
                for (int j = 0; j < m; j++) {
                    double d = 0;
                    for (int k = 0; k < dx; k++) {
                        double z = (s[k] - pool[j][k]) / h;
                        d += z * z;
                    }
                    lk[j] = -0.5 * d;
                }
                double l = SbiArrays.logSumExp(lk);
                for (int k = 0; k < dx; k++) {
                    double e = xo[k] - s[k];
                    double spike = Math.log(1 - rho) - 0.5 * (e / 0.01) * (e / 0.01)
                            - Math.log(0.01 * Math.sqrt(2 * Math.PI));
                    double slab = Math.log(rho) - Math.log(Math.PI * 0.25 * (1 + (e / 0.25) * (e / 0.25)));
                    l += Math.max(spike, slab) + Math.log1p(Math.exp(-Math.abs(spike - slab)));
                }
                return l;
            };
            double[][] clean = SliceSampler.chain(logp, nearest(pool, xo), 30, 40, 0.5, random);
            out[i] = new double[n][];
            for (int s = 0; s < n; s++) {
                double[] c = clean[s % clean.length];
                double[] raw = new double[dx];
                for (int k = 0; k < dx; k++) {
                    raw[k] = xMean[k] + xSd[k] * c[k];
                }
                out[i][s] = posterior(raw, 1, random)[0];
            }
        }
        return out;
    }

    private static double[] nearest(double[][] pool, double[] x) {
        double best = Double.POSITIVE_INFINITY;
        double[] arg = pool[0];
        for (double[] p : pool) {
            double d = 0;
            for (int k = 0; k < x.length; k++) {
                d += (p[k] - x[k]) * (p[k] - x[k]);
            }
            if (d < best) {
                best = d;
                arg = p;
            }
        }
        return arg.clone();
    }
}
