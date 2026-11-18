package io.github.zzuegg.brainiac.experiments.inference;

import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite;
import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite.Misspecified;
import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite.Problem;
import io.github.zzuegg.brainiac.experimental.sbi.ConditionalDensity;
import io.github.zzuegg.brainiac.experimental.sbi.MisspecificationRobust;
import io.github.zzuegg.brainiac.experimental.sbi.NeuralRatio;
import io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Simulation-based inference under model misspecification (13.5.7) on
 * {@link SimulatorSuite#misspecified} tasks: the simulator is a 2-D Gaussian-linear model, the
 * real data have a bias and heavy-tailed noise. Every method is scored on {@code tests} real
 * observations with known parameters by the coverage of the true parameter at the 90% level
 * (Mahalanobis credible ellipsoids), the mean log density of the true parameter under a
 * Gaussian fit to the posterior samples, and the mean posterior standard deviation
 * (informativeness). A first row reports the MMD misspecification test.
 */
public final class Misspecification {

    private Misspecification() {
    }

    private static double[] score(double[][][] posts, double[][] truth) {
        int n = posts[0].length;
        double[][] flat = new double[truth.length * n][];
        double logp = 0;
        double sd = 0;
        int d = truth[0].length;
        for (int i = 0; i < truth.length; i++) {
            System.arraycopy(posts[i], 0, flat, i * n, n);
            for (int j = 0; j < d; j++) {
                double m = 0;
                double v = 0;
                for (double[] s : posts[i]) {
                    m += s[j] / n;
                }
                for (double[] s : posts[i]) {
                    v += (s[j] - m) * (s[j] - m) / n;
                }
                v = Math.max(v, 1e-8);
                logp += (-0.5 * (truth[i][j] - m) * (truth[i][j] - m) / v - 0.5 * Math.log(2 * Math.PI * v))
                        / truth.length;
                sd += Math.sqrt(v) / (d * truth.length);
            }
        }
        double cov = PosteriorDiagnostics.expectedCoverage(truth, flat, n).coverage()[17];
        return new double[] {cov, logp, sd};
    }

    /**
     * The comparison table.
     *
     * @param simulations simulation budget for the NPE
     * @param calibration real calibration pairs
     * @param tests real test observations
     * @param seed the seed
     * @return rows {@code {coverage at 0.9, mean log density of the truth, mean posterior sd}};
     *     the first row is {@code {MMD statistic, p-value, 0}}
     */
    public static Map<String, double[]> table(int simulations, int calibration, int tests, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        Misspecified task = SimulatorSuite.misspecified("gaussian-heavy-tails");
        Problem p = task.model();
        double[][][] cal = task.calibration(calibration, seed);
        double[][][] test = task.calibration(tests, seed + 1);
        double[][] truth = test[0];
        double[][] real = test[1];
        SplittableRandom r = new SplittableRandom(seed);
        double[][] sims = new double[200][];
        for (int i = 0; i < sims.length; i++) {
            sims[i] = p.simulator().simulate(p.prior().sample(r), r);
        }
        var mmd = MisspecificationRobust.detect(real, sims, 200, seed);
        out.put("MMD test {stat, p}", new double[] {mmd.statistic(), mmd.pValue(), 0});
        ConditionalDensity.Config cfg = new ConditionalDensity.Config(3, 32, 600, 128, 3e-3);
        int n = 200;

        MisspecificationRobust base = MisspecificationRobust.train(p.prior(), p.simulator(), simulations, cfg, seed);
        double[][][] npe = new double[tests][][];
        for (int i = 0; i < tests; i++) {
            npe[i] = base.posterior(real[i], n, r);
        }
        out.put("NPE (simulations only)", score(npe, truth));
        double[][][] rope = base.rope(p.simulator(), cal[0], cal[1], real, simulations / 2, 0.05, 0.05, n, r);
        out.put("RoPE tau=0.05", score(rope, truth));
        double[][][] rnpe = base.rnpe(real, 0.3, n, r);
        out.put("RNPE", score(rnpe, truth));

        NeuralRatio bnre = NeuralRatio.train(NeuralRatio.Variant.BNRE, p.prior(), p.simulator(), simulations, 1, 100,
                32, 500, seed);
        double[][][] b = new double[tests][][];
        for (int i = 0; i < tests; i++) {
            b[i] = bnre.posteriorSir(p.prior(), real[i], n, 2000, r);
        }
        out.put("BNRE", score(b, truth));

        MisspecificationRobust aug = MisspecificationRobust.train(p.prior(), p.simulator(), simulations, cfg, 3.0,
                seed + 1);
        out.put("RVNP-style learned error", score(aug.noiseAugmented(real, n, r), truth));

        base.fineTune(cal[0], cal[1], 300, seed + 2);
        double[][][] ft = new double[tests][][];
        for (int i = 0; i < tests; i++) {
            ft[i] = base.posterior(real[i], n, r);
        }
        out.put("NPE fine-tuned on calibration", score(ft, truth));
        return out;
    }
}
