package io.github.zzuegg.brainiac.experimental.panel;

/**
 * Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens and Wager, AER
 * 2021, <a href="https://arxiv.org/abs/1812.09970">arXiv:1812.09970</a>) for a block of treated
 * units adopting at one time.
 *
 * <ul>
 *   <li>Unit weights: {@code min over w0 and simplex w of
 *       sum_{t < T0} (w0 + sum_j w_j Y_jt - Ybar_tr,t)^2 + zeta^2 T0 ||w||^2}, with the
 *       regulariser {@code zeta = (N_tr T_post)^(1/4) sigma}, {@code sigma} the standard deviation
 *       of first differences of control pre-period outcomes. The intercept {@code w0} is what
 *       makes SDID immune to level differences that defeat synthetic control.</li>
 *   <li>Time weights: {@code min over l0 and simplex l of
 *       sum_{j control} (l0 + sum_{t < T0} l_t Y_jt - Ybar_j,post)^2} (with a tiny ridge).</li>
 *   <li>The estimate is the weighted DiD
 *       {@code (Ybar_tr,post - sum_t l_t Ybar_tr,t) - sum_j w_j (Ybar_j,post - sum_t l_t Y_jt)},
 *       which equals the weighted two-way fixed-effects regression of the paper.</li>
 * </ul>
 */
public final class SyntheticDid {

    private SyntheticDid() {
    }

    /**
     * An estimate.
     *
     * @param effect the ATT
     * @param unitWeights control-unit weights
     * @param timeWeights pre-period weights
     */
    public record Estimate(double effect, double[] unitWeights, double[] timeWeights) {
    }

    /**
     * Fits SDID.
     *
     * @param y outcomes {@code [units][periods]}
     * @param treatedUnits the treated units
     * @param start the first treated period
     * @return the estimate
     */
    public static Estimate fit(double[][] y, int[] treatedUnits, int start) {
        int periods = y[0].length;
        int post = periods - start;
        int[] controls = SyntheticControl.complement(y.length, treatedUnits);
        int nc = controls.length;
        double[] trMean = new double[periods];
        for (int i : treatedUnits) {
            for (int t = 0; t < periods; t++) {
                trMean[t] += y[i][t] / treatedUnits.length;
            }
        }
        // regulariser from first differences of control pre-period outcomes
        double s1 = 0;
        double s2 = 0;
        int c = 0;
        for (int j : controls) {
            for (int t = 1; t < start; t++) {
                double dlt = y[j][t] - y[j][t - 1];
                s1 += dlt;
                s2 += dlt * dlt;
                c++;
            }
        }
        double sigma = Math.sqrt(Math.max(0, s2 / c - (s1 / c) * (s1 / c)));
        double zeta = Math.pow(treatedUnits.length * (double) post, 0.25) * sigma;
        double[][] a = new double[start][nc];
        for (int t = 0; t < start; t++) {
            for (int k = 0; k < nc; k++) {
                a[t][k] = y[controls[k]][t];
            }
        }
        double[] target = java.util.Arrays.copyOf(trMean, start);
        double[] omega = SyntheticControl.simplexLeastSquares(a, target, zeta * zeta * start, true, 800);
        double[][] b = new double[nc][start];
        double[] postMean = new double[nc];
        for (int k = 0; k < nc; k++) {
            for (int t = 0; t < periods; t++) {
                if (t < start) {
                    b[k][t] = y[controls[k]][t];
                } else {
                    postMean[k] += y[controls[k]][t] / post;
                }
            }
        }
        double[] lambda = SyntheticControl.simplexLeastSquares(b, postMean, 1e-6 * sigma * sigma * nc, true, 800);
        double trPost = 0;
        for (int t = start; t < periods; t++) {
            trPost += trMean[t] / post;
        }
        double trPre = 0;
        for (int t = 0; t < start; t++) {
            trPre += lambda[t] * trMean[t];
        }
        double ctrl = 0;
        for (int k = 0; k < nc; k++) {
            double pre = 0;
            for (int t = 0; t < start; t++) {
                pre += lambda[t] * y[controls[k]][t];
            }
            ctrl += omega[k] * (postMean[k] - pre);
        }
        return new Estimate(trPost - trPre - ctrl, omega, lambda);
    }
}
