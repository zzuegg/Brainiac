package io.github.zzuegg.brainiac.experimental.survival;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Lbfgs;

import java.util.Arrays;
import java.util.List;

/**
 * Competing risks: when a subject can fail from one of several causes, the quantity of interest
 * is the cumulative incidence {@code F_k(t) = P(T <= t, cause = k)}.
 * <ul>
 *   <li>{@link #aalenJohansen} (Aalen and Johansen, Scand. J. Stat. 1978): the nonparametric
 *       estimate {@code F_k(t) = sum_{t_j <= t} S(t_j-) d_kj / n_j}, with {@code S} the all-cause
 *       Kaplan-Meier curve. One minus the Kaplan-Meier curve of cause {@code k} (treating the other
 *       causes as censoring) estimates a world without the other causes and overstates
 *       {@code F_k}.</li>
 *   <li>{@link FineGray} (Fine and Gray, JASA 1999): proportional hazards on the subdistribution
 *       hazard of cause {@code k}, {@code -d log(1 - F_k(t | x)) / dt = h_0(t) exp(beta x)}. Subjects
 *       who failed from another cause stay in the risk set with inverse probability of censoring
 *       weights {@code G(t) / G(min(T_j, t))} ({@code G} the censoring Kaplan-Meier curve); the
 *       weighted partial likelihood is maximised by L-BFGS ({@link Lbfgs}) and a weighted Breslow
 *       estimator gives the baseline, so {@code F_k(t | x) = 1 - exp(-H_0(t) exp(beta x))} and the
 *       coefficients describe effects on the cumulative incidence itself.</li>
 * </ul>
 */
public final class CompetingRisks {

    private CompetingRisks() {
    }

    /**
     * The Aalen-Johansen cumulative incidence of one cause on a grid.
     *
     * @param time observed times
     * @param event 0 if censored, else the cause {@code 1..}
     * @param cause the cause
     * @param grid increasing times
     * @return {@code F_cause(t)} on the grid
     */
    public static double[] aalenJohansen(double[] time, int[] event, int cause, double[] grid) {
        int n = time.length;
        Integer[] order = sortedOrder(time);
        double s = 1;
        double f = 0;
        double[] out = new double[grid.length];
        int g = 0;
        int k = 0;
        while (k < n) {
            double t = time[order[k]];
            int atRisk = n - k;
            int all = 0;
            int mine = 0;
            int m = k;
            while (m < n && time[order[m]] == t) {
                int e = event[order[m]];
                all += e != 0 ? 1 : 0;
                mine += e == cause ? 1 : 0;
                m++;
            }
            while (g < grid.length && grid[g] < t) {
                out[g++] = f;
            }
            f += s * mine / (double) atRisk;
            s *= 1 - all / (double) atRisk;
            k = m;
        }
        while (g < grid.length) {
            out[g++] = f;
        }
        return out;
    }

    /**
     * One minus the Kaplan-Meier curve of one cause, other causes treated as censoring (the biased
     * estimator).
     *
     * @param time observed times
     * @param event 0 if censored, else the cause
     * @param cause the cause
     * @param grid increasing times
     * @return {@code 1 - KM_cause(t)} on the grid
     */
    public static double[] oneMinusKaplanMeier(double[] time, int[] event, int cause, double[] grid) {
        int[] e = new int[event.length];
        for (int i = 0; i < e.length; i++) {
            e[i] = event[i] == cause ? 1 : 0;
        }
        SurvivalModels.KaplanMeier km = new SurvivalModels.KaplanMeier(time, e);
        double[] out = new double[grid.length];
        for (int g = 0; g < grid.length; g++) {
            out[g] = 1 - km.survival(grid[g]);
        }
        return out;
    }

    private static Integer[] sortedOrder(double[] time) {
        Integer[] order = new Integer[time.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(time[a], time[b]));
        return order;
    }

    /** The Fine-Gray subdistribution hazards model of one cause. */
    public static final class FineGray {

        private final double[] beta;
        private final double[] eventTimes;
        private final double[] baseline;

        private FineGray(double[] beta, double[] eventTimes, double[] baseline) {
            this.beta = beta;
            this.eventTimes = eventTimes;
            this.baseline = baseline;
        }

        /**
         * Fits the model.
         *
         * @param x covariates {@code [n][d]}
         * @param time observed times
         * @param event 0 if censored, else the cause
         * @param cause the cause modelled
         * @param l2 a ridge penalty, such as 1e-4
         * @return the model
         */
        public static FineGray fit(double[][] x, double[] time, int[] event, int cause, double l2) {
            int n = x.length;
            int d = x[0].length;
            int[] censored = new int[n];
            for (int i = 0; i < n; i++) {
                censored[i] = event[i] == 0 ? 1 : 0;
            }
            SurvivalModels.KaplanMeier g = new SurvivalModels.KaplanMeier(time, censored);
            // rows: subjects failing from the cause; columns: everyone, with their weight in that risk set
            int m = 0;
            for (int e : event) {
                m += e == cause ? 1 : 0;
            }
            int[] fails = new int[m];
            int k = 0;
            for (int i = 0; i < n; i++) {
                if (event[i] == cause) {
                    fails[k++] = i;
                }
            }
            float[] w = new float[m * n];
            for (int a = 0; a < m; a++) {
                double t = time[fails[a]];
                double gt = g.survival(t - 1e-12);
                for (int j = 0; j < n; j++) {
                    double wj;
                    if (time[j] >= t) {
                        wj = 1;
                    } else if (event[j] != 0 && event[j] != cause) {
                        wj = gt / Math.max(1e-6, g.survival(time[j] - 1e-12));
                    } else {
                        wj = 0;
                    }
                    w[a * n + j] = (float) wj;
                }
            }
            float[] xv = new float[n * d];
            float[] xf = new float[m * d];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    xv[i * d + j] = (float) x[i][j];
                }
            }
            for (int a = 0; a < m; a++) {
                for (int j = 0; j < d; j++) {
                    xf[a * d + j] = (float) x[fails[a]][j];
                }
            }
            Tensor xt = Tensor.of(Shape.of(n, d), xv);
            Tensor xft = Tensor.of(Shape.of(m, d), xf);
            Tensor wt = Tensor.of(Shape.of(m, n), w);
            Var b = Var.param(Tensor.zeros(Shape.of(d, 1)));
            new Lbfgs(10, 1e-7).minimize(List.of(b), () -> {
                Var denom = Var.of(wt).matmul(Var.of(xt).matmul(b).exp()).add(1e-12f).log();
                return Var.of(xft).matmul(b).sub(denom).sum().mul(-1f / n).add(b.square().sum().mul((float) l2));
            }, 200);
            float[] bv = b.value().toArray();
            double[] beta = new double[d];
            for (int j = 0; j < d; j++) {
                beta[j] = bv[j];
            }
            double[] r = new double[n];
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < d; j++) {
                    s += beta[j] * x[i][j];
                }
                r[i] = Math.exp(s);
            }
            Integer[] order = new Integer[m];
            for (int a = 0; a < m; a++) {
                order[a] = a;
            }
            Arrays.sort(order, (p, q) -> Double.compare(time[fails[p]], time[fails[q]]));
            double[] times = new double[m];
            double[] base = new double[m];
            double cum = 0;
            for (int k2 = 0; k2 < m; k2++) {
                int a = order[k2];
                double denom = 0;
                for (int j = 0; j < n; j++) {
                    denom += w[a * n + j] * r[j];
                }
                cum += 1 / Math.max(1e-12, denom);
                times[k2] = time[fails[a]];
                base[k2] = cum;
            }
            return new FineGray(beta, times, base);
        }

        /** @return the subdistribution log-hazard ratios */
        public double[] coefficients() {
            return beta.clone();
        }

        /**
         * The cumulative incidence of the cause.
         *
         * @param x covariates
         * @param t a time
         * @return {@code F(t | x)}
         */
        public double incidence(double[] x, double t) {
            int k = Arrays.binarySearch(eventTimes, t);
            int idx = k >= 0 ? k : -k - 2;
            while (k >= 0 && idx + 1 < eventTimes.length && eventTimes[idx + 1] == t) {
                idx++;
            }
            double h = idx < 0 ? 0 : baseline[idx];
            double eta = 0;
            for (int j = 0; j < beta.length; j++) {
                eta += beta[j] * x[j];
            }
            return 1 - Math.exp(-h * Math.exp(eta));
        }
    }
}
