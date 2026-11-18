package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;

/**
 * FTRL-Proximal online logistic regression over hashed sparse binary features, the algorithm
 * Google used for ad click prediction.
 *
 * <p>Follow-the-regularised-leader with per-coordinate learning rates: each coordinate keeps
 * {@code z} (gradients minus a proximal correction) and {@code n} (the sum of squared
 * gradients), and its weight is solved in closed form,
 * <pre>
 * w_i = 0                                                 if |z_i| &lt;= lambda1
 * w_i = -(z_i - sgn(z_i) lambda1) / ((beta + sqrt(n_i)) / alpha + lambda2)   otherwise
 * </pre>
 * so the L1 term makes most weights exactly zero while the AdaGrad-style rates keep accuracy.
 * The update after predicting {@code p} for label {@code y} is, per active coordinate,
 * {@code g = p - y}, {@code sigma = (sqrt(n + g^2) - sqrt(n)) / alpha},
 * {@code z += g - sigma w}, {@code n += g^2}.
 *
 * <p>The paper's memory savings are available: Bloom-filter feature inclusion (a feature enters
 * the model only after a counting Bloom filter has seen it {@code inclusionCount} times, so rare
 * features never take memory) and coefficients rounded to 16 bits (the q2.13 fixed-point format)
 * when predicting. {@link #nonZeros()} and {@link #bytes()} report the model size.
 *
 * <p>Paper: McMahan et al., "Ad Click Prediction: a View from the Trenches", KDD 2013,
 * <a href="https://doi.org/10.1145/2487575.2488200">doi:10.1145/2487575.2488200</a>.
 */
public final class FtrlProximal implements OnlineClassifier {

    private final double alpha;
    private final double beta;
    private final double lambda1;
    private final double lambda2;
    private final int space;
    private final float[] z;
    private final float[] n;
    private final short[] bloom;
    private final int inclusionCount;
    private final boolean sixteenBit;
    private int included;

    /**
     * Creates the learner.
     *
     * @param space the hashed feature space (a bias coordinate is added)
     * @param alpha the learning-rate scale, such as 0.1
     * @param beta the learning-rate offset, such as 1
     * @param lambda1 the L1 strength
     * @param lambda2 the L2 strength
     * @param inclusionCount Bloom-filter inclusion threshold; 0 or 1 includes every feature
     * @param sixteenBit whether coefficients are rounded to q2.13 when predicting
     */
    public FtrlProximal(int space, double alpha, double beta, double lambda1, double lambda2, int inclusionCount,
            boolean sixteenBit) {
        this.space = space;
        this.alpha = alpha;
        this.beta = beta;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        this.z = new float[space + 1];
        this.n = new float[space + 1];
        this.inclusionCount = inclusionCount;
        this.bloom = inclusionCount > 1 ? new short[Math.max(1024, space / 2)] : null;
        this.sixteenBit = sixteenBit;
    }

    private double weight(int i) {
        double zi = z[i];
        if (Math.abs(zi) <= lambda1) {
            return 0;
        }
        double w = -(zi - Math.signum(zi) * lambda1) / ((beta + Math.sqrt(n[i])) / alpha + lambda2);
        if (sixteenBit) {
            w = Math.max(-4, Math.min(4 - 1.0 / 8192, Math.rint(w * 8192) / 8192));
        }
        return w;
    }

    private boolean seen(int i) {
        return n[i] > 0 || z[i] != 0;
    }

    /** Counts a feature in the Bloom filter; true once it has been seen often enough. */
    private boolean admit(int i) {
        if (bloom == null || seen(i)) {
            return true;
        }
        int m = bloom.length;
        int a = DriftingStreams.hash(17, i) % m;
        int b = DriftingStreams.hash(31, i) % m;
        int c = DriftingStreams.hash(47, i) % m;
        int count = Math.min(bloom[a], Math.min(bloom[b], bloom[c]));
        if (count + 1 >= inclusionCount) {
            return true;
        }
        for (int h : new int[] {a, b, c}) {
            if (bloom[h] == count && bloom[h] < Short.MAX_VALUE) {
                bloom[h]++;   // conservative update: only the minimal counters grow
            }
        }
        return false;
    }

    private double probability(int[] active) {
        double s = weight(space);
        for (int i : active) {
            s += weight(i);
        }
        return 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, s))));
    }

    @Override
    public double[] predict(DriftingStreams.Sample sample) {
        double p = probability(sample.active());
        return new double[] {1 - p, p};
    }

    @Override
    public void learn(DriftingStreams.Sample sample) {
        int[] active = sample.active();
        double p = probability(active);
        double g = p - sample.label();
        update(space, g);
        for (int i : active) {
            if (admit(i)) {
                if (!seen(i)) {
                    included++;
                }
                update(i, g);
            }
        }
    }

    private void update(int i, double g) {
        double w = weight(i);
        double sigma = (Math.sqrt(n[i] + g * g) - Math.sqrt(n[i])) / alpha;
        z[i] += (float) (g - sigma * w);
        n[i] += (float) (g * g);
    }

    /** @return the number of non-zero weights */
    public int nonZeros() {
        int c = 0;
        for (int i = 0; i <= space; i++) {
            if (Math.abs(z[i]) > lambda1) {
                c++;
            }
        }
        return c;
    }

    /** @return the number of features admitted into the model */
    public int included() {
        return included;
    }

    /**
     * Memory of a sparse deployment: per stored coordinate its key (4 bytes), its coefficient
     * (2 bytes at 16 bits, else 4) and its squared-gradient sum (4 bytes), plus the Bloom filter.
     */
    @Override
    public long bytes() {
        long stored = 0;
        for (int i = 0; i <= space; i++) {
            if (seen(i)) {
                stored++;
            }
        }
        return stored * (4 + (sixteenBit ? 2 : 4) + 4) + (bloom == null ? 0 : 2L * bloom.length);
    }
}
