package io.github.zzuegg.brainiac.experimental.survival;

/**
 * Metrics for right-censored survival predictions.
 * <ul>
 *   <li>{@link #harrellC} (Harrell et al. 1982): over comparable pairs (the earlier time is an
 *       event), the share where the earlier unit has the higher predicted risk; 0.5 is random.</li>
 *   <li>{@link #timeDependentC} (Antolini et al. 2005): the same with each pair compared through
 *       the predicted survival at the earlier unit's event time, so models whose risk ordering
 *       changes over time (crossing hazards) are scored fairly.</li>
 *   <li>{@link #integratedBrier} (Graf et al. 1999): the squared error of predicted survival
 *       against the observed status, averaged over a time grid, with inverse probability of
 *       censoring weights from a Kaplan-Meier estimate of the censoring distribution; lower is
 *       better and a model that always predicts 0.5 scores 0.25.</li>
 * </ul>
 */
public final class SurvivalMetrics {

    private SurvivalMetrics() {
    }

    /**
     * Harrell's concordance index.
     *
     * @param risk predicted risk per unit (higher means earlier event)
     * @param time observed times
     * @param event 0 if censored; with {@code cause > 0} only that cause counts as an event
     * @param cause the cause to score, or 0 for any event
     * @return the C-index
     */
    public static double harrellC(double[] risk, double[] time, int[] event, int cause) {
        double concordant = 0;
        long pairs = 0;
        int n = time.length;
        for (int i = 0; i < n; i++) {
            if (event[i] == 0 || (cause > 0 && event[i] != cause)) {
                continue;
            }
            for (int j = 0; j < n; j++) {
                if (time[i] < time[j]) {
                    pairs++;
                    if (risk[i] > risk[j]) {
                        concordant++;
                    } else if (risk[i] == risk[j]) {
                        concordant += 0.5;
                    }
                }
            }
        }
        return pairs == 0 ? 0.5 : concordant / pairs;
    }

    /**
     * Antolini's time-dependent concordance from predicted survival (or, for a cause, cumulative
     * incidence) curves.
     *
     * @param curve {@code [n][grid]} predicted survival {@code S(t | x)}, or predicted cumulative
     *     incidence when {@code incidence} is true
     * @param grid increasing times
     * @param time observed times
     * @param event 0 if censored
     * @param cause the cause to score, or 0 for any event
     * @param incidence whether {@code curve} holds cumulative incidence (higher = riskier) rather
     *     than survival (lower = riskier)
     * @return the time-dependent C-index
     */
    public static double timeDependentC(double[][] curve, double[] grid, double[] time, int[] event, int cause,
            boolean incidence) {
        double concordant = 0;
        long pairs = 0;
        int n = time.length;
        for (int i = 0; i < n; i++) {
            if (event[i] == 0 || (cause > 0 && event[i] != cause)) {
                continue;
            }
            int k = index(grid, time[i]);
            double si = curve[i][k];
            for (int j = 0; j < n; j++) {
                if (time[i] < time[j]) {
                    pairs++;
                    double sj = curve[j][k];
                    boolean right = incidence ? si > sj : si < sj;
                    if (right) {
                        concordant++;
                    } else if (si == sj) {
                        concordant += 0.5;
                    }
                }
            }
        }
        return pairs == 0 ? 0.5 : concordant / pairs;
    }

    /**
     * @param grid increasing times
     * @param t a time
     * @return the last grid index whose time is at most {@code t} (0 if none)
     */
    public static int index(double[] grid, double t) {
        int idx = java.util.Arrays.binarySearch(grid, t);
        int k = idx >= 0 ? idx : -idx - 2;
        return Math.max(0, Math.min(grid.length - 1, k));
    }

    /**
     * The IPCW integrated Brier score over a grid (trapezoidal rule, divided by the grid span).
     *
     * @param survival {@code [n][grid]} predicted {@code S(t | x)} for the test units
     * @param grid increasing times
     * @param time the test units' observed times
     * @param event the test units' event flags (0 censored)
     * @param trainTime training times, for the censoring distribution
     * @param trainEvent training event flags
     * @return the integrated Brier score
     */
    public static double integratedBrier(double[][] survival, double[] grid, double[] time, int[] event,
            double[] trainTime, int[] trainEvent) {
        int[] censored = new int[trainEvent.length];
        for (int i = 0; i < censored.length; i++) {
            censored[i] = trainEvent[i] == 0 ? 1 : 0;
        }
        SurvivalModels.KaplanMeier g = new SurvivalModels.KaplanMeier(trainTime, censored);
        double[] bs = new double[grid.length];
        int n = time.length;
        for (int k = 0; k < grid.length; k++) {
            double t = grid[k];
            double gt = Math.max(g.survival(t), 0.05);
            double s = 0;
            for (int i = 0; i < n; i++) {
                double p = survival[i][k];
                if (time[i] <= t && event[i] != 0) {
                    s += p * p / Math.max(g.survival(time[i] - 1e-12), 0.05);
                } else if (time[i] > t) {
                    s += (1 - p) * (1 - p) / gt;
                }
            }
            bs[k] = s / n;
        }
        double area = 0;
        for (int k = 1; k < grid.length; k++) {
            area += (bs[k] + bs[k - 1]) / 2 * (grid[k] - grid[k - 1]);
        }
        return area / (grid[grid.length - 1] - grid[0]);
    }

    /**
     * An evaluation grid: {@code points} times evenly spaced between the given quantiles of the
     * observed times.
     *
     * @param time observed times
     * @param lowQuantile the first quantile, such as 0.1
     * @param highQuantile the last quantile, such as 0.9
     * @param points grid points, at least 2
     * @return the grid
     */
    public static double[] grid(double[] time, double lowQuantile, double highQuantile, int points) {
        double[] sorted = time.clone();
        java.util.Arrays.sort(sorted);
        double lo = sorted[(int) (lowQuantile * (sorted.length - 1))];
        double hi = sorted[(int) (highQuantile * (sorted.length - 1))];
        double[] g = new double[points];
        for (int k = 0; k < points; k++) {
            g[k] = lo + (hi - lo) * k / (points - 1);
        }
        return g;
    }
}
