package io.github.zzuegg.brainiac.experimental.stats;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

/**
 * Streaming anomaly thresholds from extreme value theory (Siffer et al., "Anomaly detection in
 * streams with extreme value theory", KDD 2017). SPOT calibrates a peaks-over-threshold model
 * ({@link ExtremeValues}) on an initial batch, sets the alarm threshold
 * {@code z_q = t + sigma/xi ((q n / N_t)^-xi - 1)} for a target false-alarm probability
 * {@code q}, and updates the GPD as new peaks arrive (anomalies are not learned from). DSPOT
 * applies SPOT to the deviation from a moving average of the last {@code depth} normal values,
 * so the threshold follows drift.
 */
public final class SpotThreshold {

    private final double q;
    private final double initialQuantile;
    private final int depth;
    private double t;
    private final List<Double> peaks = new ArrayList<>();
    private long n;
    private double z;
    private final Deque<Double> window = new ArrayDeque<>();
    private double windowSum;
    private int sinceFit;

    private SpotThreshold(double q, double initialQuantile, int depth) {
        this.q = q;
        this.initialQuantile = initialQuantile;
        this.depth = depth;
    }

    /**
     * SPOT.
     *
     * @param q the target false-alarm probability, for example 1e-3
     * @param initialQuantile the quantile of the calibration batch used as {@code t}, for
     *     example 0.98
     * @return the detector (call {@link #calibrate} first)
     */
    public static SpotThreshold spot(double q, double initialQuantile) {
        return new SpotThreshold(q, initialQuantile, 0);
    }

    /**
     * DSPOT with a moving-average window.
     *
     * @param q the target false-alarm probability
     * @param initialQuantile the calibration quantile
     * @param depth the moving-average window
     * @return the detector
     */
    public static SpotThreshold dspot(double q, double initialQuantile, int depth) {
        return new SpotThreshold(q, initialQuantile, depth);
    }

    private double local(double x) {
        if (depth == 0) {
            return x;
        }
        return x - (window.isEmpty() ? 0 : windowSum / window.size());
    }

    private void remember(double x) {
        if (depth == 0) {
            return;
        }
        window.addLast(x);
        windowSum += x;
        if (window.size() > depth) {
            windowSum -= window.removeFirst();
        }
    }

    /**
     * Calibrates on an initial batch of normal data.
     *
     * @param batch the batch
     */
    public void calibrate(double[] batch) {
        double[] v = new double[batch.length - depth];
        for (int i = 0; i < batch.length; i++) {
            if (i >= depth) {
                v[i - depth] = local(batch[i]);
            }
            remember(batch[i]);
        }
        double[] s = v.clone();
        Arrays.sort(s);
        t = s[(int) Math.floor(initialQuantile * (s.length - 1))];
        for (double x : v) {
            if (x > t) {
                peaks.add(x - t);
            }
        }
        n = v.length;
        refit();
    }

    private void refit() {
        double[] y = peaks.stream().mapToDouble(Double::doubleValue).toArray();
        ExtremeValues.Gpd g = ExtremeValues.fitGpd(y);
        double r = q * n / peaks.size();
        z = t + (Math.abs(g.shape()) < 1e-9 ? -g.scale() * Math.log(r)
                : g.scale() / g.shape() * (Math.pow(r, -g.shape()) - 1));
        sinceFit = 0;
    }

    /**
     * Processes one value.
     *
     * @param x the value
     * @return whether it is an alarm
     */
    public boolean update(double x) {
        double v = local(x);
        if (v > z) {
            return true;
        }
        if (v > t) {
            peaks.add(v - t);
            n++;
            if (++sinceFit >= 10) {
                refit();
            }
        } else {
            n++;
        }
        remember(x);
        return false;
    }

    /**
     * The current alarm threshold (on the detrended scale for DSPOT).
     *
     * @return {@code z_q}
     */
    public double threshold() {
        return z;
    }
}
