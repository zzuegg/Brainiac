package io.github.zzuegg.brainiac.experiments.econometrics;

import io.github.zzuegg.brainiac.experimental.data.PanelStudies;
import io.github.zzuegg.brainiac.experimental.panel.RegressionDiscontinuity;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Regression discontinuity on the Lee (2008) design (wave 11, item 11.3.8).
 *
 * <ul>
 *   <li>{@link #sharp}: local linear estimates at the Imbens-Kalyanaraman bandwidth and at the
 *       Calonico-Cattaneo-Titiunik bandwidths, each with its conventional interval, and the CCT
 *       robust bias-corrected interval, against global polynomials of order 1 to 4 on the whole
 *       sample and the difference of means in a window of half-width 0.05. Scores: bias, RMSE and
 *       95% interval coverage (NaN for estimators without an interval).</li>
 *   <li>{@link #fuzzy}: the same with take-up 0.8 above and 0.2 below the cutoff, by the local
 *       Wald ratio (bandwidths chosen on the reduced form).</li>
 *   <li>{@link #density}: the McCrary log-density jump test's rejection rate at 5% with no
 *       manipulation (size) and with a share of the units just below the cutoff moved above it
 *       (power).</li>
 * </ul>
 */
public final class DiscontinuityDesigns {

    private DiscontinuityDesigns() {
    }

    private static void add(Map<String, double[]> out, String key, double est, double truth, double cover,
            int sims) {
        double[] acc = out.computeIfAbsent(key, s -> new double[3]);
        acc[0] += (est - truth) / sims;
        acc[1] += (est - truth) * (est - truth) / sims;
        acc[2] += cover / sims;
    }

    private static Map<String, double[]> run(boolean fuzzy, int n, int sims, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int s = 0; s < sims; s++) {
            PanelStudies.Discontinuity d = PanelStudies.discontinuity(n, fuzzy, 0, seed + s);
            double[] x = d.x();
            double[] y = d.y();
            double c = d.cutoff();
            double truth = d.effect();
            double hIk = RegressionDiscontinuity.ikBandwidth(x, y, c);
            double[] hb = RegressionDiscontinuity.cctBandwidths(x, y, c);
            RegressionDiscontinuity.Estimate[] ik = fuzzy
                    ? RegressionDiscontinuity.fuzzy(x, d.treated(), y, c, hIk, hIk)
                    : RegressionDiscontinuity.sharp(x, y, c, hIk, hIk);
            RegressionDiscontinuity.Estimate[] cct = fuzzy
                    ? RegressionDiscontinuity.fuzzy(x, d.treated(), y, c, hb[0], hb[1])
                    : RegressionDiscontinuity.sharp(x, y, c, hb[0], hb[1]);
            add(out, "local linear, IK h, conventional CI", ik[0].effect(), truth, ik[0].covers(truth) ? 1 : 0,
                    sims);
            add(out, "local linear, CCT h, conventional CI", cct[0].effect(), truth, cct[0].covers(truth) ? 1 : 0,
                    sims);
            add(out, "local linear, CCT h, robust bias-corrected CI", cct[1].effect(), truth,
                    cct[1].covers(truth) ? 1 : 0, sims);
            double jump = fuzzy ? takeUpJump(x, d.treated(), c) : 1;
            for (int order = 1; order <= 4; order++) {
                add(out, "global polynomial, order " + order,
                        RegressionDiscontinuity.globalPolynomial(x, y, c, order) / jump, truth, Double.NaN, sims);
            }
            add(out, "window means (|x| < 0.05)", RegressionDiscontinuity.windowMeans(x, y, c, 0.05) / jump, truth,
                    Double.NaN, sims);
        }
        out.values().forEach(v -> v[1] = Math.sqrt(v[1]));
        return out;
    }

    private static double takeUpJump(double[] x, int[] d, double c) {
        double[] dd = new double[d.length];
        for (int i = 0; i < d.length; i++) {
            dd[i] = d[i];
        }
        return RegressionDiscontinuity.windowMeans(x, dd, c, 0.2);
    }

    /**
     * The sharp design.
     *
     * @param n units per simulation
     * @param sims simulations
     * @param seed the seed
     * @return method to {@code {bias, RMSE, 95% coverage}}
     */
    public static Map<String, double[]> sharp(int n, int sims, long seed) {
        return run(false, n, sims, seed);
    }

    /**
     * The fuzzy design; the baselines divide their outcome jump by the take-up jump in a window
     * of half-width 0.2.
     *
     * @param n units per simulation
     * @param sims simulations
     * @param seed the seed
     * @return method to {@code {bias, RMSE, 95% coverage}}
     */
    public static Map<String, double[]> fuzzy(int n, int sims, long seed) {
        return run(true, n, sims, seed);
    }

    /**
     * The density test's rejection rates.
     *
     * @param n units per simulation
     * @param manipulation the share of units in {@code (-0.1, 0)} moved above the cutoff
     * @param sims simulations
     * @param seed the seed
     * @return {@code {size (no manipulation), power}}
     */
    public static double[] density(int n, double manipulation, int sims, long seed) {
        double size = 0;
        double power = 0;
        for (int s = 0; s < sims; s++) {
            double[] x0 = PanelStudies.discontinuity(n, false, 0, seed + s).x();
            double[] x1 = PanelStudies.discontinuity(n, false, manipulation, seed + s).x();
            size += (RegressionDiscontinuity.densityTest(x0, 0)[2] < 0.05 ? 1.0 : 0.0) / sims;
            power += (RegressionDiscontinuity.densityTest(x1, 0)[2] < 0.05 ? 1.0 : 0.0) / sims;
        }
        return new double[] {size, power};
    }

    /**
     * Mean selected bandwidths on the sharp design.
     *
     * @param n units per simulation
     * @param sims simulations
     * @param seed the seed
     * @return {@code {IK h, CCT h, CCT b}}
     */
    public static double[] bandwidths(int n, int sims, long seed) {
        double[] out = new double[3];
        for (int s = 0; s < sims; s++) {
            PanelStudies.Discontinuity d = PanelStudies.discontinuity(n, false, 0, seed + s);
            double[] hb = RegressionDiscontinuity.cctBandwidths(d.x(), d.y(), d.cutoff());
            out[0] += RegressionDiscontinuity.ikBandwidth(d.x(), d.y(), d.cutoff()) / sims;
            out[1] += hb[0] / sims;
            out[2] += hb[1] / sims;
        }
        return out;
    }
}
