package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;

/**
 * Online logistic regression over hashed sparse binary features by plain stochastic gradient
 * descent (step {@code lr / sqrt(t)}) or AdaGrad (per-coordinate step
 * {@code lr / sqrt(sum g^2)}); the baselines FTRL-Proximal is compared with. Every coordinate
 * ever touched holds a non-zero weight.
 */
public final class OnlineLogistic implements OnlineClassifier {

    private final int space;
    private final double rate;
    private final boolean adagrad;
    private final double l2;
    private final float[] w;
    private final float[] g2;
    private long t;

    /**
     * Creates the learner.
     *
     * @param space the hashed feature space (a bias coordinate is added)
     * @param rate the learning rate
     * @param adagrad per-coordinate AdaGrad steps rather than a global {@code 1 / sqrt(t)} decay
     * @param l2 the L2 strength
     */
    public OnlineLogistic(int space, double rate, boolean adagrad, double l2) {
        this.space = space;
        this.rate = rate;
        this.adagrad = adagrad;
        this.l2 = l2;
        this.w = new float[space + 1];
        this.g2 = new float[space + 1];
    }

    private double probability(int[] active) {
        double s = w[space];
        for (int i : active) {
            s += w[i];
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
        t++;
        double g = probability(sample.active()) - sample.label();
        step(space, g);
        for (int i : sample.active()) {
            step(i, g);
        }
    }

    private void step(int i, double g) {
        double gi = g + l2 * w[i];
        if (adagrad) {
            g2[i] += (float) (gi * gi);
            w[i] -= (float) (rate * gi / Math.sqrt(g2[i] + 1e-8));
        } else {
            g2[i] = 1;   // marks the coordinate as touched
            w[i] -= (float) (rate * gi / Math.sqrt(t));
        }
    }

    /** @return the number of non-zero weights */
    public int nonZeros() {
        int c = 0;
        for (float v : w) {
            if (v != 0) {
                c++;
            }
        }
        return c;
    }

    @Override
    public long bytes() {
        long stored = 0;
        for (int i = 0; i <= space; i++) {
            if (g2[i] != 0) {
                stored++;
            }
        }
        return stored * (adagrad ? 12 : 8);
    }
}
