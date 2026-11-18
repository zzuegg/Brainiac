package io.github.zzuegg.brainiac.experiments.imaging;

import io.github.zzuegg.brainiac.experimental.data.InverseProblems;
import io.github.zzuegg.brainiac.experimental.imaging.ImageMath;
import io.github.zzuegg.brainiac.experimental.imaging.LinearOperator;
import io.github.zzuegg.brainiac.experimental.imaging.PosteriorSamplers;
import io.github.zzuegg.brainiac.experimental.ml.AnnealedImportanceSampler;
import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Posterior sampling for inverse problems (wave 13, items 13.4.7 and 13.4.8): {@link #exact}
 * checks DPS, PI-GDM, DDRM and DAPS against exact posterior samples on Gaussian-mixture priors,
 * with HMC, an SMC sampler and multi-start MAP as the sampler baselines.
 */
public final class PosteriorSampling {

    private PosteriorSampling() {
    }

    /** The measurement families of the exact table. */
    private enum Kind {
        INPAINT, PROJECTION, MAGNITUDE
    }

    /**
     * The exact table: for each scenario (dimension, measurement, noise) and sampler, the sliced
     * Wasserstein-2 distance and the MMD to exact posterior samples, the mode coverage
     * ({@code sum_c min(q_c, q*_c)} over prior components, {@code q*} from exact samples) and the
     * score (or gradient) evaluations per sample. Exact samples come from the closed-form mixture
     * posterior for linear measurements and from a fine grid for the 2-D magnitude measurement.
     *
     * @param samples the samples per method
     * @param steps the diffusion samplers' noise levels
     * @param sigmas the measurement noise levels
     * @param seed the seed
     * @return scenario / method to {SW2, MMD, coverage, evaluations per sample}
     */
    public static Map<String, double[]> exact(int samples, int steps, double[] sigmas, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        int[][] setups = {{2, 6, 0}, {2, 6, 1}, {8, 8, 0}, {8, 8, 1}, {2, 6, 2}};
        int scenario = 0;
        for (int[] setup : setups) {
            int d = setup[0];
            Kind kind = Kind.values()[setup[2]];
            InverseProblems.GaussianMixture prior = InverseProblems.mixturePrior(d, setup[1], 2, seed + d);
            for (double sy : sigmas) {
                scenario++;
                SplittableRandom r = new SplittableRandom(seed + 100 * scenario);
                double[][] a = measurementMatrix(kind, d, r);
                double[] truth = prior.sample(1, r)[0];
                double[] y = matVec(a, truth);
                if (kind == Kind.MAGNITUDE) {
                    for (int i = 0; i < y.length; i++) {
                        y[i] = Math.abs(y[i]);
                    }
                }
                for (int i = 0; i < y.length; i++) {
                    y[i] += sy * gaussian(r);
                }
                LinearOperator op = LinearOperator.matrix(a);
                PosteriorSamplers.Forward fwd = kind == Kind.MAGNITUDE ? PosteriorSamplers.Forward.magnitude(op)
                        : PosteriorSamplers.Forward.linear(op);
                double norm2 = op.normSquared(50);
                PosteriorSamplers.Problem problem = new PosteriorSamplers.Problem(fwd, y, sy, d, norm2);
                double[][] ref = kind == Kind.MAGNITUDE ? gridPosterior(prior, a, y, sy, samples, r)
                        : prior.posterior(a, sy, y).sample(samples, r);
                double[][] ref2 = kind == Kind.MAGNITUDE ? gridPosterior(prior, a, y, sy, samples, r)
                        : prior.posterior(a, sy, y).sample(samples, r);
                double variance = 0;
                for (double[] p : prior.sample(500, r)) {
                    variance += ImageMath.dot(p, p) / (500.0 * d);
                }
                PosteriorSamplers.Settings settings = PosteriorSamplers.Settings.defaults(steps, 10, 0.01, variance);
                String name = d + "-D " + kind.toString().toLowerCase(java.util.Locale.ROOT) + ", sigma_y " + sy;
                put(out, name + " / exact (second draw)", ref2, ref, prior, 0, r);
                for (PosteriorSamplers.Method m : PosteriorSamplers.Method.values()) {
                    if (kind == Kind.MAGNITUDE && (m == PosteriorSamplers.Method.PI_GDM
                            || m == PosteriorSamplers.Method.DDRM)) {
                        continue;
                    }
                    double[][] xs = new double[samples][];
                    long evals = 0;
                    for (int i = 0; i < samples; i++) {
                        PosteriorSamplers.Sample s = PosteriorSamplers.sample(m, prior::denoise, problem, settings, r);
                        xs[i] = s.x();
                        evals += s.scoreEvaluations();
                    }
                    put(out, name + " / " + m, xs, ref, prior, evals / (double) samples, r);
                }
                LogDensity post = (x, g) -> {
                    double v = prior.logDensity(x, g);
                    double[] f = fwd.apply(x);
                    double[] res = new double[f.length];
                    double q = 0;
                    for (int i = 0; i < f.length; i++) {
                        res[i] = (f[i] - y[i]) / (sy * sy);
                        q += (f[i] - y[i]) * (f[i] - y[i]);
                    }
                    if (g != null) {
                        double[] gl = fwd.vjp(x, res);
                        for (int i = 0; i < g.length; i++) {
                            g[i] -= gl[i];
                        }
                    }
                    return v - q / (2 * sy * sy);
                };
                HamiltonianMonteCarlo.Result hmc = HamiltonianMonteCarlo.hmc(post, prior.sample(1, r)[0], 0, 10,
                        samples, samples * 5, 0.8, r);
                double[][] thinned = new double[samples][];
                for (int i = 0; i < samples; i++) {
                    thinned[i] = hmc.samples()[5 * i];
                }
                put(out, name + " / HMC (one chain)", thinned, ref, prior,
                        hmc.gradientEvaluations() / (double) samples, r);
                AnnealedImportanceSampler.Estimate smc = AnnealedImportanceSampler.anneal(prior::logDensity,
                        rr -> prior.sample(1, rr)[0], post, samples, 30, 5, 0.05, true, r);
                put(out, name + " / SMC sampler", smc.particles(), ref, prior,
                        smc.gradientEvaluations() / (double) samples, r);
                double[][] map = new double[samples][];
                for (int i = 0; i < samples; i++) {
                    map[i] = mapAscent(post, prior.sample(1, r)[0], 1 / (1 / 0.04 + norm2 / (sy * sy)), 300);
                }
                put(out, name + " / MAP (multi-start)", map, ref, prior, 300, r);
            }
        }
        return out;
    }

    private static double[][] measurementMatrix(Kind kind, int d, SplittableRandom r) {
        int m = d == 2 ? 1 : d / 2;
        double[][] a = new double[m][d];
        for (int i = 0; i < m; i++) {
            if (kind == Kind.INPAINT) {
                a[i][i] = 1;
            } else {
                double n = 0;
                for (int j = 0; j < d; j++) {
                    a[i][j] = gaussian(r);
                    n += a[i][j] * a[i][j];
                }
                for (int j = 0; j < d; j++) {
                    a[i][j] /= Math.sqrt(n);
                }
            }
        }
        return a;
    }

    private static double[] mapAscent(LogDensity f, double[] start, double step, int iterations) {
        double[] x = start.clone();
        double[] g = new double[x.length];
        for (int it = 0; it < iterations; it++) {
            f.evaluate(x, g);
            for (int i = 0; i < x.length; i++) {
                x[i] += step * g[i];
            }
        }
        return x;
    }

    private static double[][] gridPosterior(InverseProblems.GaussianMixture prior, double[][] a, double[] y,
            double sy, int n, SplittableRandom r) {
        double lo = Double.MAX_VALUE;
        double hi = -Double.MAX_VALUE;
        for (double[] p : prior.sample(2000, new SplittableRandom(7))) {
            for (double v : p) {
                lo = Math.min(lo, v);
                hi = Math.max(hi, v);
            }
        }
        lo -= 1;
        hi += 1;
        int g = 400;
        double h = (hi - lo) / g;
        double[] cdf = new double[g * g];
        double max = Double.NEGATIVE_INFINITY;
        double[] lp = new double[g * g];
        for (int i = 0; i < g; i++) {
            for (int j = 0; j < g; j++) {
                double[] x = {lo + (i + 0.5) * h, lo + (j + 0.5) * h};
                double f = Math.abs(a[0][0] * x[0] + a[0][1] * x[1]) - y[0];
                lp[i * g + j] = prior.logDensity(x, null) - f * f / (2 * sy * sy);
                max = Math.max(max, lp[i * g + j]);
            }
        }
        double s = 0;
        for (int c = 0; c < lp.length; c++) {
            s += Math.exp(lp[c] - max);
            cdf[c] = s;
        }
        double[][] out = new double[n][];
        for (int k = 0; k < n; k++) {
            double u = r.nextDouble() * s;
            int c = Arrays.binarySearch(cdf, u);
            c = c < 0 ? -c - 1 : c;
            c = Math.min(c, cdf.length - 1);
            out[k] = new double[] {lo + (c / g + r.nextDouble()) * h, lo + (c % g + r.nextDouble()) * h};
        }
        return out;
    }

    private static void put(Map<String, double[]> out, String key, double[][] xs, double[][] ref,
            InverseProblems.GaussianMixture prior, double evals, SplittableRandom r) {
        out.put(key, new double[] {slicedWasserstein(xs, ref, 64, r), mmd(xs, ref), coverage(xs, ref, prior),
            evals});
    }

    /**
     * The sliced Wasserstein-2 distance between equal-size samples.
     *
     * @param a the first sample
     * @param b the second
     * @param directions the random directions
     * @param r the random source
     * @return the distance
     */
    static double slicedWasserstein(double[][] a, double[][] b, int directions, SplittableRandom r) {
        int d = a[0].length;
        int n = Math.min(a.length, b.length);
        double total = 0;
        for (int k = 0; k < directions; k++) {
            double[] u = new double[d];
            for (int j = 0; j < d; j++) {
                u[j] = gaussian(r);
            }
            double un = ImageMath.norm(u);
            double[] pa = new double[n];
            double[] pb = new double[n];
            for (int i = 0; i < n; i++) {
                pa[i] = ImageMath.dot(a[i], u) / un;
                pb[i] = ImageMath.dot(b[i], u) / un;
            }
            Arrays.sort(pa);
            Arrays.sort(pb);
            for (int i = 0; i < n; i++) {
                total += (pa[i] - pb[i]) * (pa[i] - pb[i]) / n;
            }
        }
        return Math.sqrt(total / directions);
    }

    /**
     * The (biased, V-statistic) maximum mean discrepancy with a Gaussian kernel whose bandwidth is
     * the median distance within the reference sample.
     *
     * @param a the sample
     * @param ref the reference
     * @return the MMD
     */
    static double mmd(double[][] a, double[][] ref) {
        double[] dists = new double[ref.length * (ref.length - 1) / 2];
        int k = 0;
        for (int i = 0; i < ref.length; i++) {
            for (int j = 0; j < i; j++) {
                dists[k++] = dist2(ref[i], ref[j]);
            }
        }
        Arrays.sort(dists);
        double bw = Math.max(dists[dists.length / 2], 1e-12);
        double xx = kernelMean(a, a, bw);
        double yy = kernelMean(ref, ref, bw);
        double xy = kernelMean(a, ref, bw);
        return Math.sqrt(Math.max(xx + yy - 2 * xy, 0));
    }

    private static double kernelMean(double[][] a, double[][] b, double bw) {
        double s = 0;
        for (double[] x : a) {
            for (double[] y : b) {
                s += Math.exp(-dist2(x, y) / bw);
            }
        }
        return s / ((double) a.length * b.length);
    }

    private static double dist2(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s;
    }

    /**
     * Mode coverage: assigns each sample to its most responsible prior component and sums
     * {@code min(q_c, q*_c)} against the reference's assignment shares.
     *
     * @param xs the sample
     * @param ref the reference
     * @param prior the prior
     * @return the coverage in {@code [0, 1]}
     */
    static double coverage(double[][] xs, double[][] ref, InverseProblems.GaussianMixture prior) {
        double[] q = shares(xs, prior);
        double[] qr = shares(ref, prior);
        double s = 0;
        for (int c = 0; c < q.length; c++) {
            s += Math.min(q[c], qr[c]);
        }
        return s;
    }

    private static double[] shares(double[][] xs, InverseProblems.GaussianMixture prior) {
        double[] q = new double[prior.components()];
        for (double[] x : xs) {
            double[] resp = prior.responsibilities(x);
            int best = 0;
            for (int c = 1; c < resp.length; c++) {
                if (resp[c] > resp[best]) {
                    best = c;
                }
            }
            q[best] += 1.0 / xs.length;
        }
        return q;
    }

    private static double[] matVec(double[][] a, double[] x) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = ImageMath.dot(a[i], x);
        }
        return y;
    }

    static double gaussian(SplittableRandom r) {
        double u = 1 - r.nextDouble();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r.nextDouble());
    }
}
