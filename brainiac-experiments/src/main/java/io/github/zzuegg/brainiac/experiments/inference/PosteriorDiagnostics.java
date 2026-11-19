package io.github.zzuegg.brainiac.experiments.inference;

import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite;
import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite.Problem;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.Classifier;
import io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.Coverage;
import io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.LocalTest;
import io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.PosteriorSampler;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Posterior diagnostics without reference posteriors (13.5.6): SBC, expected coverage, TARP
 * and local C2ST from {@link io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics},
 * run on exact posteriors with planted failures, so their detection rates at a 5% level are
 * known. Two tasks: the 2-D Gaussian-linear model (failures: over- and under-dispersion, a
 * mean shift) and a "squared" model {@code x_j = theta_j^2 + N(0, 0.3^2)} whose posterior is
 * bimodal in each coordinate (failure: dropping the negative mode). Local C2ST is also compared
 * with the oracle C2ST (against exact samples) for an MDN NPE across observations.
 */
public final class PosteriorDiagnostics {

    private PosteriorDiagnostics() {
    }

    /** The planted failures. */
    enum Failure {
        EXACT, OVERDISPERSED, UNDERDISPERSED, SHIFTED, MODE_DROPPED
    }

    private static PosteriorSampler gaussianExact(Failure f) {
        double prec = 1 + 1 / 0.25;
        double sd = 1 / Math.sqrt(prec);
        double scale = f == Failure.OVERDISPERSED ? 1.6 : f == Failure.UNDERDISPERSED ? 0.6 : 1;
        double shift = f == Failure.SHIFTED ? 0.6 * sd : 0;
        return (x, n, r) -> {
            double[][] out = new double[n][2];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < 2; j++) {
                    out[i][j] = x[j] / 0.25 / prec + shift + scale * sd * r.nextGaussian();
                }
            }
            return out;
        };
    }

    static Problem squared() {
        Prior prior = SimulatorSuite.gaussian(new double[2], 1);
        Simulator sim = (t, r) -> new double[] {t[0] * t[0] + 0.3 * r.nextGaussian(),
            t[1] * t[1] + 0.3 * r.nextGaussian()};
        return new Problem("squared", 2, prior, sim, null, null);
    }

    /** Exact posterior of {@link #squared} by inverse-CDF sampling on a grid per coordinate. */
    private static PosteriorSampler squaredExact(boolean dropNegative) {
        int grid = 801;
        return (x, n, r) -> {
            double[][] out = new double[n][2];
            for (int j = 0; j < 2; j++) {
                double[] cdf = new double[grid];
                double tot = 0;
                for (int g = 0; g < grid; g++) {
                    double t = -4 + 8.0 * g / (grid - 1);
                    double z = (x[j] - t * t) / 0.3;
                    tot += (dropNegative && t < 0) ? 0 : Math.exp(-0.5 * t * t - 0.5 * z * z);
                    cdf[g] = tot;
                }
                for (int i = 0; i < n; i++) {
                    double u = r.nextDouble() * tot;
                    int k = Arrays.binarySearch(cdf, u);
                    k = k < 0 ? -k - 1 : k;
                    out[i][j] = -4 + 8.0 * Math.min(k, grid - 1) / (grid - 1) + 0.005 * (r.nextDouble() - 0.5);
                }
            }
            return out;
        };
    }

    /**
     * Detection rates of the planted failures at the 5% level over {@code replicates} runs of
     * {@code draws} prior-predictive checks each.
     *
     * @param draws checks per replicate
     * @param replicates replicates
     * @param seed the seed
     * @return rows (task and failure) to {@code {SBC, expected coverage, TARP, local C2ST}}
     *     rejection rates
     */
    public static Map<String, double[]> detection(int draws, int replicates, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        Problem lin = SimulatorSuite.gaussianLinear(2);
        Problem sq = squared();
        for (Failure f : Failure.values()) {
            boolean squaredTask = f == Failure.MODE_DROPPED;
            out.put((squaredTask ? "squared " : "linear ") + f.name().toLowerCase(java.util.Locale.ROOT),
                    rates(squaredTask ? sq : lin, squaredTask ? squaredExact(true) : gaussianExact(f), draws,
                            replicates, seed + f.ordinal()));
        }
        out.put("squared exact", rates(sq, squaredExact(false), draws, replicates, seed + 77));
        return out;
    }

    private static double[] rates(Problem p, PosteriorSampler q, int draws, int replicates, long seed) {
        double[] rate = new double[4];
        for (int rep = 0; rep < replicates; rep++) {
            long s = seed + 1000L * rep;
            double[][][] c = io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.simulateChecks(
                    p.prior(), p.simulator(), q, draws, 100, s);
            double sbc = io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.sbc(c[0], c[2], 100, 10)
                    .minPValue();
            Coverage cov = io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.expectedCoverage(c[0],
                    c[2], 100);
            Coverage tarp = io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.tarp(c[0], c[2], 100,
                    p.prior(), s + 1);
            double[] xo = p.simulator().simulate(new double[] {0.8, -0.5}, new SplittableRandom(s + 2));
            LocalTest lc = io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.localC2st(
                    Classifier.LOGISTIC_QUADRATIC, c[0], c[1], q, new double[][] {xo}, 200, 19, s + 3)[0];
            rate[0] += sbc < 0.05 ? 1.0 / replicates : 0;
            rate[1] += cov.pValue() < 0.05 ? 1.0 / replicates : 0;
            rate[2] += tarp.pValue() < 0.05 ? 1.0 / replicates : 0;
            rate[3] += lc.pValue() <= 0.05 ? 1.0 / replicates : 0;   // 19 nulls: the smallest p is 0.05
        }
        return rate;
    }

    /**
     * Local C2ST against the oracle C2ST for an MDN NPE on the Gaussian-linear task: both are
     * computed at {@code observations} observations and compared by Spearman rank correlation.
     *
     * @param budget NPE simulations (small, so the estimate is imperfect)
     * @param observations observations
     * @param seed the seed
     * @return {@code {Spearman correlation, mean oracle C2ST, mean local statistic}}
     */
    public static double[] localAgainstOracle(int budget, int observations, long seed) {
        Problem lin = SimulatorSuite.gaussianLinear(2);
        SimulationBasedInference.NeuralPosterior npe = SimulationBasedInference.neuralPosterior(lin.prior(),
                lin.simulator(), budget, 2, 8, 60, seed);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] calT = new double[400][];
        double[][] calX = new double[400][];
        for (int i = 0; i < calT.length; i++) {
            calT[i] = lin.prior().sample(r);
            calX[i] = lin.simulator().simulate(calT[i], r);
        }
        double[][] xs = new double[observations][];
        double[] oracle = new double[observations];
        for (int i = 0; i < observations; i++) {
            double[] t = {5 * (r.nextDouble() - 0.5), 5 * (r.nextDouble() - 0.5)};
            xs[i] = lin.simulator().simulate(t, r);
            oracle[i] = SimulationBasedInference.c2st(lin.reference().sample(xs[i], t, 300, r),
                    npe.sample(xs[i], 300, r));
        }
        LocalTest[] lc = io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics.localC2st(
                Classifier.LOGISTIC_QUADRATIC,
                calT, calX, npe::sample, xs, 300, 3, seed + 1);
        double[] stat = new double[observations];
        for (int i = 0; i < observations; i++) {
            stat[i] = lc[i].statistic();
        }
        return new double[] {spearman(oracle, stat), Arrays.stream(oracle).average().orElse(0),
            Arrays.stream(stat).average().orElse(0)};
    }

    static double spearman(double[] a, double[] b) {
        double[] ra = ranks(a);
        double[] rb = ranks(b);
        double ma = Arrays.stream(ra).average().orElse(0);
        double mb = Arrays.stream(rb).average().orElse(0);
        double num = 0;
        double da = 0;
        double db = 0;
        for (int i = 0; i < a.length; i++) {
            num += (ra[i] - ma) * (rb[i] - mb);
            da += (ra[i] - ma) * (ra[i] - ma);
            db += (rb[i] - mb) * (rb[i] - mb);
        }
        return num / Math.sqrt(da * db + 1e-300);
    }

    private static double[] ranks(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (x, y) -> Double.compare(v[x], v[y]));
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            r[idx[i]] = i;
        }
        return r;
    }
}
