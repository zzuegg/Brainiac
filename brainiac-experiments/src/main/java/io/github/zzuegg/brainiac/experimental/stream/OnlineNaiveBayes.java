package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;

/**
 * Gaussian naive Bayes learned online: class counts and, per class and attribute, a running mean
 * and variance (Welford). A classic, constant-memory stream baseline that never forgets.
 */
public final class OnlineNaiveBayes implements OnlineClassifier {

    private final int features;
    private final int classes;
    private final double[] count;
    private final double[][] mean;
    private final double[][] m2;

    /**
     * Creates the model.
     *
     * @param features attributes
     * @param classes classes
     */
    public OnlineNaiveBayes(int features, int classes) {
        this.features = features;
        this.classes = classes;
        this.count = new double[classes];
        this.mean = new double[classes][features];
        this.m2 = new double[classes][features];
    }

    @Override
    public double[] predict(DriftingStreams.Sample sample) {
        double[] x = sample.x();
        double total = 0;
        for (double c : count) {
            total += c;
        }
        double[] lp = new double[classes];
        double best = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < classes; c++) {
            lp[c] = Math.log((count[c] + 1) / (total + classes));
            if (count[c] >= 2) {
                for (int a = 0; a < features; a++) {
                    double var = Math.max(m2[c][a] / count[c], 1e-6);
                    double d = x[a] - mean[c][a];
                    lp[c] += -0.5 * Math.log(2 * Math.PI * var) - d * d / (2 * var);
                }
            }
            best = Math.max(best, lp[c]);
        }
        double z = 0;
        for (int c = 0; c < classes; c++) {
            lp[c] = Math.exp(lp[c] - best);
            z += lp[c];
        }
        for (int c = 0; c < classes; c++) {
            lp[c] /= z;
        }
        return lp;
    }

    @Override
    public void learn(DriftingStreams.Sample sample) {
        int y = sample.label();
        double[] x = sample.x();
        count[y]++;
        for (int a = 0; a < features; a++) {
            double d = x[a] - mean[y][a];
            mean[y][a] += d / count[y];
            m2[y][a] += d * (x[a] - mean[y][a]);
        }
    }

    @Override
    public long bytes() {
        return 8L * classes * (1 + 2L * features);
    }
}
