package io.github.zzuegg.brainiac.experimental.calibrate;

import io.github.zzuegg.brainiac.experimental.train.Calibration;

/**
 * Structured matrix scaling (Berta et al., Nov 2025,
 * <a href="https://arxiv.org/abs/2511.03685">arXiv:2511.03685</a>): matrix scaling made safe
 * for small calibration sets. The logits are first temperature-scaled (so one penalty strength
 * fits confident and unconfident models alike), turned into log-probabilities clipped below at
 * {@code -30} and centred, and then mapped by
 * {@code softmax((I + diag(v) + offdiag(M)) s + b)}, with the paper's default ridge penalties
 * scaled by group size over sample count:
 * {@code k/n |b|² + k/n |v|² + k(k-1)/n |M|²}. With no data the map is temperature scaling;
 * with plenty it approaches full matrix scaling.
 *
 * <p>{@link Kind#LOW_RANK} is a cheaper variant with {@code M} replaced by a rank-{@code r}
 * product {@code U V^T} (same penalties on the factors), for many classes.
 */
public final class StructuredMatrixScaling implements Recalibrator {

    /** The parametrisation of the off-diagonal part. */
    public enum Kind {
        /** a full off-diagonal matrix {@code M} (the paper's default) */
        FULL,
        /** {@code U V^T} of a chosen rank */
        LOW_RANK
    }

    private final double temperature;
    private final AffineLogitFit map;

    private StructuredMatrixScaling(double temperature, AffineLogitFit map) {
        this.temperature = temperature;
        this.map = map;
    }

    /**
     * Fits the map with the paper's default penalties (all {@code λ = 1}).
     *
     * @param logits validation logits
     * @param labels validation labels
     * @param kind the parametrisation
     * @param rank the rank for {@link Kind#LOW_RANK}
     * @return the map
     */
    public static StructuredMatrixScaling fit(double[][] logits, int[] labels, Kind kind, int rank) {
        return fit(logits, labels, kind, rank, 1.0);
    }

    /**
     * Fits the map.
     *
     * @param logits validation logits
     * @param labels validation labels
     * @param kind the parametrisation
     * @param rank the rank for {@link Kind#LOW_RANK}
     * @param lambda a common multiplier of all penalties
     * @return the map
     */
    public static StructuredMatrixScaling fit(double[][] logits, int[] labels, Kind kind, int rank, double lambda) {
        int n = logits.length;
        int k = logits[0].length;
        double t = Calibration.fitTemperature(logits, labels);
        double[][] f = features(logits, t);
        double group = lambda * k / (double) n;
        double off = kind == Kind.FULL ? lambda * k * (k - 1.0) / n : lambda * 2.0 * k * rank / n;
        AffineLogitFit m = AffineLogitFit.fit(f, labels,
                kind == Kind.FULL ? AffineLogitFit.Structure.FULL : AffineLogitFit.Structure.LOW_RANK, rank, group,
                off, group, 500);
        return new StructuredMatrixScaling(t, m);
    }

    private static double[][] features(double[][] logits, double t) {
        double[][] f = AffineLogitFit.logProbabilities(logits, t, -30);
        for (double[] row : f) {
            double mean = 0;
            for (double v : row) {
                mean += v;
            }
            mean /= row.length;
            for (int i = 0; i < row.length; i++) {
                row[i] -= mean;
            }
        }
        return f;
    }

    /** @return the temperature of the preprocessing step */
    public double temperature() {
        return temperature;
    }

    @Override
    public double[][] apply(double[][] logits) {
        return map.probabilities(features(logits, temperature));
    }
}
