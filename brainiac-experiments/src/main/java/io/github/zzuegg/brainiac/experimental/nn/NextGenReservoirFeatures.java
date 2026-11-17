package io.github.zzuegg.brainiac.experimental.nn;

/**
 * The feature vector of next generation reservoir computing (NG-RC; Gauthier, Bollt,
 * Griffith and Barbosa, <a href="https://arxiv.org/abs/2106.07688">2106.07688</a>): instead
 * of a random recurrent network, a constant, the last {@code k} inputs taken every
 * {@code skip} steps, and all unique quadratic monomials of those delayed inputs. A ridge
 * readout on these few features matches or beats a large echo state network on chaotic
 * forecasting, with tiny warm-up and training sets.
 */
public final class NextGenReservoirFeatures {
    private final int dimension;
    private final int delays;
    private final int skip;

    private NextGenReservoirFeatures(int dimension, int delays, int skip) {
        this.dimension = dimension;
        this.delays = delays;
        this.skip = skip;
    }

    /**
     * Creates the feature map.
     *
     * @param dimension the width of one input
     * @param delays the number of taps k
     * @param skip the distance between taps
     * @return the map
     */
    public static NextGenReservoirFeatures of(int dimension, int delays, int skip) {
        if (dimension < 1 || delays < 1 || skip < 1) {
            throw new IllegalArgumentException("dimension, delays and skip must be positive");
        }
        return new NextGenReservoirFeatures(dimension, delays, skip);
    }

    /** @return the earliest index that has a full history */
    public int warmup() {
        return (delays - 1) * skip;
    }

    /** @return the number of linear features */
    public int linearSize() {
        return dimension * delays;
    }

    /** @return the total number of features: 1 + linear + linear (linear + 1) / 2 */
    public int size() {
        int l = linearSize();
        return 1 + l + l * (l + 1) / 2;
    }

    /**
     * Builds the features at time {@code t} from {@code series[t]}, {@code series[t - skip]}, ….
     *
     * @param series the inputs, {@code [time][dimension]}
     * @param t the current time, at least {@link #warmup()}
     * @return the features
     */
    public double[] features(double[][] series, int t) {
        if (t < warmup()) {
            throw new IllegalArgumentException("need t >= " + warmup());
        }
        int l = linearSize();
        double[] lin = new double[l];
        for (int d = 0; d < delays; d++) {
            System.arraycopy(series[t - d * skip], 0, lin, d * dimension, dimension);
        }
        double[] f = new double[size()];
        f[0] = 1;
        System.arraycopy(lin, 0, f, 1, l);
        int k = 1 + l;
        for (int i = 0; i < l; i++) {
            for (int j = i; j < l; j++) {
                f[k++] = lin[i] * lin[j];
            }
        }
        return f;
    }
}
