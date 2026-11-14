package io.github.zzuegg.brainiac.data;

/**
 * Scales each feature to mean 0 and standard deviation 1, which helps training. Fit it on
 * the training data and apply the same scaling to test data and later inputs.
 *
 * <pre>{@code
 * Standardizer s = Standardizer.fit(split.train());
 * Dataset train = split.train().standardize(s);
 * Dataset test = split.test().standardize(s);
 * float[] input = s.apply(new float[] {5.1f, 3.5f, 1.4f, 0.2f});
 * }</pre>
 */
public final class Standardizer {

    private final float[] mean;
    private final float[] std;

    private Standardizer(float[] mean, float[] std) {
        this.mean = mean;
        this.std = std;
    }

    /**
     * Computes each feature's mean and standard deviation. Features that never change
     * keep a scale of 1.
     *
     * @param data the data to measure
     * @return the standardizer
     */
    public static Standardizer fit(Dataset data) {
        int f = data.featureCount();
        int n = data.size();
        float[] x = data.rawFeatures();
        double[] sum = new double[f];
        double[] sumSq = new double[f];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < f; j++) {
                double v = x[i * f + j];
                sum[j] += v;
                sumSq[j] += v * v;
            }
        }
        float[] mean = new float[f];
        float[] std = new float[f];
        for (int j = 0; j < f; j++) {
            double m = sum[j] / n;
            double var = Math.max(0, sumSq[j] / n - m * m);
            mean[j] = (float) m;
            std[j] = var < 1e-12 ? 1f : (float) Math.sqrt(var);
        }
        return new Standardizer(mean, std);
    }

    /**
     * Scales one sample's features.
     *
     * @param features the raw features
     * @return the scaled features, as a new array
     */
    public float[] apply(float[] features) {
        if (features.length != mean.length) {
            throw new IllegalArgumentException("expected " + mean.length + " features, got " + features.length);
        }
        return apply(features, mean.length);
    }

    float[] apply(float[] rows, int width) {
        if (width != mean.length) {
            throw new IllegalArgumentException("standardizer was fitted on " + mean.length + " features, data has " + width);
        }
        float[] out = new float[rows.length];
        for (int i = 0; i < rows.length; i++) {
            int j = i % width;
            out[i] = (rows[i] - mean[j]) / std[j];
        }
        return out;
    }

    /** @return each feature's mean, as measured by {@link #fit} */
    public float[] mean() {
        return mean.clone();
    }

    /** @return each feature's standard deviation, as measured by {@link #fit} */
    public float[] std() {
        return std.clone();
    }
}
