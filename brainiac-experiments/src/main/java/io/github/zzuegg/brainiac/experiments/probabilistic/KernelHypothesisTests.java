package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.experimental.ml.KernelTests.Result;
import io.github.zzuegg.brainiac.experimental.ml.McmcDiagnostics;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Test power (and the type-I error under the null) of kernel tests against classical ones, at
 * level 0.05, estimated over repeated trials.
 *
 * <ul>
 *   <li>{@link #twoSample}: 4-D samples, {@code N(0, I)} against itself (null), against a
 *       Laplace distribution with the same mean and variance, against a mean shift of 0.4 in one
 *       coordinate and against a 1.5x scale in one coordinate. Tests: MMD with the median
 *       bandwidth, the deep-kernel MMD, Welch's t-test and Kolmogorov-Smirnov per coordinate
 *       (Bonferroni), and a sliced-Wasserstein permutation test.</li>
 *   <li>{@link #independence}: scalar {@code x, y}, independent (null), on a noisy ring
 *       (uncorrelated but dependent) and {@code y = sin(4x) + noise}. Tests: HSIC with the gamma
 *       null and by permutation, Pearson correlation and distance correlation.</li>
 * </ul>
 */
public final class KernelHypothesisTests {

    private KernelHypothesisTests() {
    }

    private static double[][] sample(String scenario, boolean second, int n, SplittableRandom r) {
        double[][] x = new double[n][4];
        for (double[] row : x) {
            for (int j = 0; j < 4; j++) {
                row[j] = r.nextGaussian();
            }
            if (!second) {
                continue;
            }
            switch (scenario) {
                case "Laplace" -> {
                    for (int j = 0; j < 4; j++) {
                        double u = r.nextDouble() - 0.5;
                        row[j] = -Math.signum(u) * Math.log(1 - 2 * Math.abs(u)) / Math.sqrt(2);
                    }
                }
                case "mean shift" -> row[0] += 0.4;
                case "variance shift" -> row[0] *= 1.5;
                default -> {
                }
            }
        }
        return x;
    }

    /**
     * Two-sample test power.
     *
     * @param n the size of each sample
     * @param trials the number of repeated trials (the deep-kernel test runs a third of them)
     * @param seed the seed
     * @return {@code "scenario: test"} to {@code {rejection rate}}
     */
    public static Map<String, double[]> twoSample(int n, int trials, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String scenario : new String[] {"null", "Laplace", "mean shift", "variance shift"}) {
            double[] rejections = new double[5];
            int deepTrials = Math.max(1, trials / 3);
            for (int t = 0; t < trials; t++) {
                SplittableRandom r = new SplittableRandom(seed * 1000 + t);
                double[][] x = sample(scenario, false, n, r);
                double[][] y = sample(scenario, true, n, r);
                rejections[0] += KernelTests.mmdTest(x, y, 0, 100, t).rejects(0.05) ? 1 : 0;
                rejections[1] += KernelTests.tTest(x, y).rejects(0.05) ? 1 : 0;
                rejections[2] += KernelTests.ksTest(x, y).rejects(0.05) ? 1 : 0;
                rejections[3] += slicedWassersteinTest(x, y, 100, t).rejects(0.05) ? 1 : 0;
                if (t < deepTrials) {
                    rejections[4] += KernelTests.deepKernelMmdTest(x, y, 16, 60, 100, t).rejects(0.05) ? 1 : 0;
                }
            }
            out.put(scenario + ": MMD (median)", new double[] {rejections[0] / trials});
            out.put(scenario + ": deep-kernel MMD", new double[] {rejections[4] / deepTrials});
            out.put(scenario + ": t-test", new double[] {rejections[1] / trials});
            out.put(scenario + ": KS", new double[] {rejections[2] / trials});
            out.put(scenario + ": sliced W", new double[] {rejections[3] / trials});
        }
        return out;
    }

    private static Result slicedWassersteinTest(double[][] x, double[][] y, int permutations, long seed) {
        double stat = McmcDiagnostics.slicedWasserstein(x, y, 10, seed);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] z = new double[x.length + y.length][];
        System.arraycopy(x, 0, z, 0, x.length);
        System.arraycopy(y, 0, z, x.length, y.length);
        int exceed = 0;
        for (int p = 0; p < permutations; p++) {
            for (int i = z.length - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                double[] tmp = z[i];
                z[i] = z[j];
                z[j] = tmp;
            }
            double[][] a = java.util.Arrays.copyOfRange(z, 0, x.length);
            double[][] b = java.util.Arrays.copyOfRange(z, x.length, z.length);
            if (McmcDiagnostics.slicedWasserstein(a, b, 10, seed) >= stat) {
                exceed++;
            }
        }
        return new Result(stat, (exceed + 1.0) / (permutations + 1.0));
    }

    /**
     * Independence test power.
     *
     * @param n the sample size
     * @param trials the number of repeated trials
     * @param seed the seed
     * @return {@code "scenario: test"} to {@code {rejection rate}}
     */
    public static Map<String, double[]> independence(int n, int trials, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String scenario : new String[] {"null", "ring", "sinusoid"}) {
            double[] rejections = new double[4];
            for (int t = 0; t < trials; t++) {
                SplittableRandom r = new SplittableRandom(seed * 1000 + t);
                double[][] x = new double[n][1];
                double[][] y = new double[n][1];
                for (int i = 0; i < n; i++) {
                    switch (scenario) {
                        case "ring" -> {
                            double a = 2 * Math.PI * r.nextDouble();
                            x[i][0] = Math.cos(a) + 0.1 * r.nextGaussian();
                            y[i][0] = Math.sin(a) + 0.1 * r.nextGaussian();
                        }
                        case "sinusoid" -> {
                            x[i][0] = 2 * r.nextDouble() - 1;
                            y[i][0] = Math.sin(4 * x[i][0]) + 0.5 * r.nextGaussian();
                        }
                        default -> {
                            x[i][0] = r.nextGaussian();
                            y[i][0] = r.nextGaussian();
                        }
                    }
                }
                rejections[0] += KernelTests.hsicTest(x, y, 0, t).rejects(0.05) ? 1 : 0;
                rejections[1] += KernelTests.hsicTest(x, y, 100, t).rejects(0.05) ? 1 : 0;
                rejections[2] += KernelTests.pearsonTest(McmcDiagnostics.column(x, 0), McmcDiagnostics.column(y, 0))
                        .rejects(0.05) ? 1 : 0;
                rejections[3] += KernelTests.distanceCorrelationTest(x, y, 100, t).rejects(0.05) ? 1 : 0;
            }
            out.put(scenario + ": HSIC (gamma)", new double[] {rejections[0] / trials});
            out.put(scenario + ": HSIC (permutation)", new double[] {rejections[1] / trials});
            out.put(scenario + ": Pearson", new double[] {rejections[2] / trials});
            out.put(scenario + ": distance correlation", new double[] {rejections[3] / trials});
        }
        return out;
    }
}
