package io.github.zzuegg.brainiac.experiments.statistics;

import io.github.zzuegg.brainiac.experimental.anomaly.Ecod;
import io.github.zzuegg.brainiac.experimental.data.SparseDesigns;
import io.github.zzuegg.brainiac.experimental.discovery.Distributions;
import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.ml.IsolationForest;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;
import io.github.zzuegg.brainiac.experimental.testing.ConformalNovelty;
import io.github.zzuegg.brainiac.experimental.testing.Knockoffs;
import io.github.zzuegg.brainiac.experimental.testing.OnlineFdr;
import io.github.zzuegg.brainiac.experimental.testing.StabilitySelection;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Error-controlled discovery (items 12.1.4-12.1.6): batch multiple testing (Bonferroni, Holm,
 * BH, BY, Storey, e-BH) against uncorrected testing; online FDR (LOND, LORD++, SAFFRON, e-LORD,
 * e-SAFFRON) on p-value streams with known nulls; model-X knockoffs and stability selection
 * against lasso-CV support and BH on marginal/OLS p-values; and conformal novelty detection
 * with BH / e-BH against fixed-threshold anomaly scores (isolation forest, ECOD). Every table
 * reports the realised FDR and power averaged over replications.
 */
public final class MultipleTesting {

    private static final io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.Method[] BATCH =
            io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.Method.values();

    private MultipleTesting() {
    }

    /**
     * Batch procedures on z-test p-values (and likelihood-ratio e-values for e-BH).
     *
     * @param m hypotheses
     * @param pi0 the null share
     * @param mu the alternative mean
     * @param rho the within-block correlation of the statistics (blocks of 20)
     * @param reps replications
     * @param alpha the level
     * @param seed the seed
     * @return {@code {mean FDP, mean power}} per procedure
     */
    public static Map<String, double[]> batch(int m, double pi0, double mu, double rho, int reps, double alpha,
            long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (var method : BATCH) {
            out.put(method.name(), new double[2]);
        }
        for (int r = 0; r < reps; r++) {
            SparseDesigns.PValues pv = SparseDesigns.pValues(m, pi0, mu, rho, 20, SparseDesigns.Placement.UNIFORM,
                    seed + r);
            for (var method : BATCH) {
                boolean[] rej = io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.reject(method,
                        method == io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.Method.E_BH ? pv.e()
                                : pv.p(), alpha);
                accumulate(out.get(method.name()), rej, pv.isNull(), reps);
            }
        }
        return out;
    }

    /**
     * Online procedures on a stream.
     *
     * @param m hypotheses
     * @param pi0 the null share
     * @param mu the alternative mean
     * @param rho the within-block correlation (blocks of 20)
     * @param placement where the signals sit
     * @param reps replications
     * @param alpha the level
     * @param seed the seed
     * @return {@code {mean FDP, mean power}} per procedure, plus offline BH for reference
     */
    public static Map<String, double[]> online(int m, double pi0, double mu, double rho,
            SparseDesigns.Placement placement, int reps, double alpha, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (OnlineFdr.Method method : OnlineFdr.Method.values()) {
            out.put(method.name(), new double[2]);
        }
        out.put("offline BH", new double[2]);
        for (int r = 0; r < reps; r++) {
            SparseDesigns.PValues pv = SparseDesigns.pValues(m, pi0, mu, rho, 20, placement, seed + r);
            for (OnlineFdr.Method method : OnlineFdr.Method.values()) {
                boolean e = method == OnlineFdr.Method.E_LORD || method == OnlineFdr.Method.E_SAFFRON;
                accumulate(out.get(method.name()), OnlineFdr.run(method, e ? pv.e() : pv.p(), alpha), pv.isNull(),
                        reps);
            }
            accumulate(out.get("offline BH"), io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.reject(
                    io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.Method.BH, pv.p(), alpha),
                    pv.isNull(), reps);
        }
        return out;
    }

    /**
     * Variable selection with FDR control on a sparse linear design.
     *
     * @param n rows
     * @param p features
     * @param s true features
     * @param rho the Toeplitz correlation
     * @param reps replications
     * @param q the target FDR
     * @param seed the seed
     * @return {@code {mean FDP, mean power}} per method
     */
    public static Map<String, double[]> selection(int n, int p, int s, double rho, int reps, double q, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"knockoff+ (equi)", "knockoff+ (SDP)", "knockoff (equi)", "stability selection",
            "lasso CV support", "BH marginal", "BH OLS"};
        for (String k : names) {
            out.put(k, new double[2]);
        }
        for (int r = 0; r < reps; r++) {
            SparseDesigns.Regression d = SparseDesigns.linear(n, p, s, rho, 3, SparseDesigns.Noise.GAUSSIAN,
                    SparseDesigns.Coefficients.UNIFORM, seed + 31L * r);
            boolean[] nul = new boolean[p];
            for (int j = 0; j < p; j++) {
                nul[j] = d.beta()[j] == 0;
            }
            double[][] x = d.x();
            double[] y = d.y();
            accumulate(out.get(names[0]),
                    Knockoffs.select(x, y, q, Knockoffs.Construction.EQUICORRELATED, true, seed + r).selected(), nul,
                    reps);
            accumulate(out.get(names[1]), Knockoffs.select(x, y, q, Knockoffs.Construction.SDP, true, seed + r)
                    .selected(), nul, reps);
            accumulate(out.get(names[2]),
                    Knockoffs.select(x, y, q, Knockoffs.Construction.EQUICORRELATED, false, seed + r).selected(), nul,
                    reps);
            accumulate(out.get(names[3]), StabilitySelection.select(x, y, 30, 8, 0.05, 0.75, seed + r).selected(),
                    nul, reps);
            accumulate(out.get(names[4]), PenalizedRegression.of(PenalizedRegression.Penalty.LASSO)
                    .crossValidate(x, y, 30, 0.01, 5, seed + r).min().support(), nul, reps);
            accumulate(out.get(names[5]), io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.reject(
                    io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.Method.BH, marginalPValues(x, y), q),
                    nul, reps);
            if (n > p + 1) {
                accumulate(out.get(names[6]), io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.reject(
                        io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.Method.BH, olsPValues(x, y), q),
                        nul, reps);
            }
        }
        if (n <= p + 1) {
            out.remove(names[6]);
        }
        return out;
    }

    /**
     * Conformal novelty detection on Gaussian inliers with shifted outliers in the test batch.
     *
     * @param reference inliers in the reference set
     * @param m test batch size
     * @param outlierShare the outlier share of the batch
     * @param shift the outlier mean shift (per coordinate, on 3 of the dimensions)
     * @param reps replications
     * @param q the target FDR
     * @param seed the seed
     * @return {@code {mean FDP, mean power}} per method
     */
    public static Map<String, double[]> novelty(int reference, int m, double outlierShare, double shift, int reps,
            double q, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        int d = 8;
        ConformalNovelty.ScoreFunction knn = ConformalNovelty.knnDistance(5);
        for (int r = 0; r < reps; r++) {
            SplittableRandom rnd = new SplittableRandom(seed + 977L * r);
            double[][] ref = gaussian(reference, d, 0, rnd);
            boolean[] isNull = new boolean[m];
            double[][] test = new double[m][];
            for (int j = 0; j < m; j++) {
                boolean outlier = rnd.nextDouble() < outlierShare;
                isNull[j] = !outlier;
                test[j] = gaussian(1, d, 0, rnd)[0];
                if (outlier) {
                    for (int k = 0; k < 3; k++) {
                        test[j][k] += shift * (rnd.nextBoolean() ? 1 : -1);
                    }
                }
            }
            IsolationForest forest = IsolationForest.fit(ref, 100, Math.min(256, reference), seed + r);
            double[] iso = forest.score(test);
            double[] isoRef = forest.score(ref);
            accumulate(out.computeIfAbsent("IF top-5%", k -> new double[2]), topShare(iso, 0.05), isNull, reps);
            accumulate(out.computeIfAbsent("IF ref 95% quantile", k -> new double[2]), aboveQuantile(iso, isoRef, 0.95),
                    isNull, reps);
            Ecod ecod = Ecod.fit(ref);
            accumulate(out.computeIfAbsent("ECOD ref 95% quantile", k -> new double[2]),
                    aboveQuantile(ecod.score(test), ecod.score(ref), 0.95), isNull, reps);
            ToDoubleFunction<double[]> knnRef = knn.fit(ref);
            double[] kt = new double[m];
            for (int j = 0; j < m; j++) {
                kt[j] = knnRef.applyAsDouble(test[j]);
            }
            accumulate(out.computeIfAbsent("kNN top-5%", k -> new double[2]), topShare(kt, 0.05), isNull, reps);
            for (ConformalNovelty.Method method : ConformalNovelty.Method.values()) {
                accumulate(out.computeIfAbsent(method.name() + " (kNN)", k -> new double[2]),
                        ConformalNovelty.detect(method, knn, ref, test, q, 5, seed + r), isNull, reps);
            }
            accumulate(out.computeIfAbsent("SPLIT_BH (IF)", k -> new double[2]), ConformalNovelty.detect(
                    ConformalNovelty.Method.SPLIT_BH, ConformalNovelty.isolationForest(100, 256, seed + r), ref, test,
                    q, 1, seed + r), isNull, reps);
        }
        return out;
    }

    // ---------------------------------------------------------------- helpers

    private static void accumulate(double[] into, boolean[] rejected, boolean[] isNull, int reps) {
        double[] fp = io.github.zzuegg.brainiac.experimental.testing.MultipleTesting.fdpAndPower(rejected, isNull);
        into[0] += fp[0] / reps;
        into[1] += fp[1] / reps;
    }

    private static double[][] gaussian(int n, int d, double shift, SplittableRandom r) {
        double[][] x = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = r.nextGaussian() + shift;
            }
        }
        return x;
    }

    private static boolean[] topShare(double[] score, double share) {
        double[] sorted = score.clone();
        Arrays.sort(sorted);
        double cut = sorted[(int) Math.floor((1 - share) * (score.length - 1))];
        boolean[] out = new boolean[score.length];
        for (int i = 0; i < score.length; i++) {
            out[i] = score[i] > cut;
        }
        return out;
    }

    private static boolean[] aboveQuantile(double[] score, double[] reference, double level) {
        double[] sorted = reference.clone();
        Arrays.sort(sorted);
        double cut = sorted[(int) Math.floor(level * (sorted.length - 1))];
        boolean[] out = new boolean[score.length];
        for (int i = 0; i < score.length; i++) {
            out[i] = score[i] > cut;
        }
        return out;
    }

    /**
     * Two-sided p-values of the marginal correlation of each feature with the response.
     *
     * @param x the design
     * @param y the response
     * @return a p-value per feature
     */
    public static double[] marginalPValues(double[][] x, double[] y) {
        int n = y.length;
        int p = x[0].length;
        double[] out = new double[p];
        for (int j = 0; j < p; j++) {
            double c = LeastSquares.correlation(LeastSquares.column(x, j), y);
            double t = c * Math.sqrt((n - 2) / Math.max(1e-12, 1 - c * c));
            out[j] = Distributions.studentTwoSided(t, n - 2);
        }
        return out;
    }

    /**
     * Two-sided t-test p-values of OLS coefficients ({@code n > p}).
     *
     * @param x the design
     * @param y the response
     * @return a p-value per feature
     */
    public static double[] olsPValues(double[][] x, double[] y) {
        int n = y.length;
        int p = x[0].length;
        LeastSquares.Fit f = LeastSquares.fit(x, y, true, 0);
        double sigma2 = f.rss() / (n - p - 1);
        double[][] g = new double[p + 1][p + 1];
        for (double[] row : x) {
            double[] z = new double[p + 1];
            z[0] = 1;
            System.arraycopy(row, 0, z, 1, p);
            for (int a = 0; a <= p; a++) {
                for (int b = 0; b <= p; b++) {
                    g[a][b] += z[a] * z[b];
                }
            }
        }
        double[][] inv = MlMatrices.choleskyInverse(MlMatrices.cholesky(g));
        double[] out = new double[p];
        for (int j = 0; j < p; j++) {
            double t = f.beta()[j + 1] / Math.sqrt(sigma2 * inv[j + 1][j + 1]);
            out[j] = Distributions.studentTwoSided(t, n - p - 1);
        }
        return out;
    }
}
